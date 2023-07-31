import thumbs.config_logging  # must be first
import pandas as pd
import tensorflow as tf
import os
import cv2
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np
from thumbs.diff_augmentation import DiffAugmentLayer

from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_and_types, normalize_image, unnormalize_image
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import GanModel, BuiltModel

from tensorflow_addons.layers import InstanceNormalization
from keras.models import Sequential
from keras.activations import gelu
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU
from keras.layers import (
    Activation,
    StringLookup,
    Input,
    Activation,
    BatchNormalization,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    ReLU,
    LeakyReLU,
    LayerNormalization,
    Embedding,
    Multiply,
    Concatenate,
    # BatchNormalizationV2,
)

# from keras.layers.normalization.batch_normalization_v1 import (
#     BatchNormalization,
# )
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from thumbs.train import Train, TrainWassersteinGP


infinity = float("inf")

ngf = 64
ndf = 64


class PokemonModel(GanModel):
    def __init__(self, params: HyperParams, mparams: MutableHyperParams, vocab: List[str]) -> None:
        super().__init__(params, mparams)
        self.num_classes = len(vocab) + 1  # 18 pokemon types plus an OOV token

    def build_generator(self, z_dim):
        def block(input_x, f: int, layers: int = 0):
            x = Conv2DTranspose(ndf * f, kernel_size=5, strides=2, padding="same", use_bias=False, name=f"tconv_{f}")(input_x)
            x = InstanceNormalization(name=f"instance_norm_{f}")(x)
            x = Activation(gelu, name=f"gelu_{f}")(x)

            for i in range(1, layers + 1):
                x = Conv2DTranspose(ndf * f, kernel_size=5, strides=1, padding="same", use_bias=False, name=f"tconv_{f}_{i}")(x)
                x = InstanceNormalization(name=f"instance_norm_{f}_{i}")(x)
                x = Activation(gelu, name=f"gelu_{f}_{i}")(x)

            return x

        noise_input = Input(shape=(z_dim,))
        types_input = Input(shape=(self.num_classes,))

        x = Concatenate()([noise_input, types_input])
        x = Dense(4 * 4 * ngf * 8)(x)
        x = Reshape((4, 4, ngf * 8))(x)

        x = block(x, 8, 2)
        x = block(x, 4, 2)
        x = block(x, 2, 2)
        x = block(x, 1, 2)

        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = Activation("tanh")(x)

        model = tf.keras.Model([noise_input, types_input], x, name="generator")
        model.summary(line_length=200)
        f = "/mnt/e/experiments/generator.jpg"
        tf.keras.utils.plot_model(model, to_file=f, show_shapes=True, dpi=64)
        return model

    def build_discriminator(self, img_shape):
        def block(input_x, f: int, layers: int = 0, normalize_first: bool = True):
            x = Conv2D(ndf * f, kernel_size=5, strides=2, padding="same", use_bias=False, name=f"conv_{f}")(input_x)
            if normalize_first:
                x = InstanceNormalization(name=f"instance_norm_{f}")(x)
            x = Activation(gelu, name=f"gelu_{f}")(x)

            for i in range(1, layers + 1):
                x = Conv2D(ndf * f, kernel_size=5, strides=1, padding="same", use_bias=False, name=f"conv_{f}_{i}")(x)
                x = InstanceNormalization(name=f"instance_norm_{f}_{i}")(x)
                x = Activation(gelu, name=f"gelu_{f}_{i}")(x)

            return x

        image_input = Input(shape=img_shape, name="image_input")
        image = DiffAugmentLayer()(image_input)

        types_input = Input(shape=(self.num_classes,), name="types_input")
        types = Dense(img_shape[0] * img_shape[1] * 1)(types_input)
        types = Reshape((img_shape[0], img_shape[1], 1))(types)

        model_input = Concatenate()([image, types])

        x = block(model_input, f=1, layers=2, normalize_first=False)
        x = block(x, f=2, layers=2)
        x = block(x, f=4, layers=2)
        x = block(x, f=8, layers=2)

        x = Flatten()(x)
        x = Dense(1)(x)

        model = tf.keras.Model([image_input, types_input], x, name="discriminator")
        model.summary(line_length=200)
        f = "/mnt/e/experiments/discriminator.jpg"
        tf.keras.utils.plot_model(model, to_file=f, show_shapes=True, dpi=64)
        return model


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.augment_zooms = False
        self.augment_rotations = False
        # The paper says that flips seemed to be ok
        self.augment_flips = True

        data, types = get_pokemon_and_types(self.params.img_shape)
        self.vocab = types
        self.lookup = StringLookup(output_mode="multi_hot", name="string_lookup_gen", vocabulary=self.vocab)

        self.images = np.array([x[0] for x in data])
        self.labels = np.array([self.lookup(item[1]).numpy() for item in data])

        self.zoom_factor = 0.9

        # Can lookup the type info given a pokedex number
        df = pd.read_csv("/home/anthony/workspace/yt-data/data/pokemon/stats.csv")
        df = df.drop_duplicates(subset=["#"])
        self.df = df
        self.id_to_types = {}
        for index, row in df.iterrows():
            type1 = row["Type 1"].lower()
            type2 = row["Type 2"]
            both_types = [type1]
            if type2 == type2:
                both_types.append(type2.lower())

            self.id_to_types[row["#"]] = both_types

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.images, self.labels

    def get_random_labels(self, n: int):
        # get n random samples from self.labels
        return (self.labels[np.random.randint(0, len(self.labels), n)],)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams, self.get_random_labels)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=20,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=200,
            # gradient_penalty_factor=20,
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            name="pokemon_conditional_types_deep_2",
            similarity_threshold=0.0,
            similarity_penalty=20,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams, self.vocab)


if __name__ == "__main__":
    PokemonExperiment().start()