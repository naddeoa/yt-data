import thumbs.config_logging  # must be first
import pandas as pd
import tensorflow as tf
import os
import cv2
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np
from thumbs.diff_augmentation import DiffAugmentLayer

from thumbs.experiment import GanExperiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_and_types, normalize_image, unnormalize_image
from thumbs.params import HyperParams, GanHyperParams
from thumbs.model.model import GanModel, BuiltGANModel

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


class TileLayer(tf.keras.layers.Layer):
    def __init__(self, tiles):
        super(TileLayer, self).__init__()
        self.tiles = tiles

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.tile(inputs, self.tiles)


class OutlineLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(OutlineLayer, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, image):
        upper = tf.random.uniform(shape=[], minval=100, maxval=700, dtype=tf.int32)
        return self.create_outline_tensor(image, upper=upper)

    def create_outline(self, image: np.ndarray, upper=255, lower=None) -> np.ndarray:
        """
        image: numpy array that has been normalized to -1,1
        return: numpy array of an outline, noramlized to -1,1, with the same height and width
        """
        # Will result in outlines of varying detail
        image = unnormalize_image(image)
        # thresholds = [10, 100, 400, 800, 1000]
        if lower is None:
            lower = upper // 3

        edges: np.ndarray = cv2.bitwise_not(cv2.Canny(image, threshold1=lower, threshold2=upper))
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        assert edges.shape == image.shape
        return normalize_image(edges)

    def create_outline_tensor(self, image: tf.Tensor, upper=255, lower=None) -> tf.Tensor:
        outline = self.create_outline(image.numpy(), upper=upper.numpy(), lower=lower)
        tensor: tf.Tensor = tf.convert_to_tensor(outline, dtype=tf.float32)
        return tensor


class PokemonModel(GanModel):
    def __init__(self, params: HyperParams, mparams: GanHyperParams, vocab: List[str]) -> None:
        super().__init__(params, mparams)
        # self.vocab = vocab
        self.num_classes = len(vocab) + 1  # 18 pokemon types plus an OOV token

    def build_generator(self, z_dim):
        noise_input = Input(shape=(z_dim,))
        noise = Dense(8 * 8 * ngf * 8)(noise_input)
        noise = Reshape((8, 8, ngf * 8))(noise)

        types_input = Input(shape=(self.num_classes,))
        types = Dense(8 * 8 * ngf * 8)(types_input)
        types = Reshape((8, 8, ngf * 8))(types)

        x = Multiply()([noise, types])

        x = Conv2DTranspose(ngf * 4, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(ngf * 2, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(ngf, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = Activation("tanh")(x)

        model = tf.keras.Model([noise_input, types_input], x, name="generator")
        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape=(128, 128, 3)):
        image_input = Input(shape=img_shape)
        image_augment = DiffAugmentLayer()(image_input)

        types_input = Input(shape=(self.num_classes,))
        types_embedding = Dense(img_shape[0] * img_shape[1] * 1)(types_input)
        types_embedding = Reshape((img_shape[0], img_shape[1], 1))(types_embedding)

        model_input = Multiply()([image_augment, types_embedding])

        x = Conv2D(ndf, kernel_size=5, strides=2, padding="same", use_bias=False)(model_input)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(ndf * 2, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(ndf * 4, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(ndf * 8, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1)(x)

        model = tf.keras.Model([image_input, types_input], x, name="discriminator")
        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator):
        # Can't really do this easily because they both take multiple inputs
        return None


class PokemonExperiment(GanExperiment):
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

    def get_train(self, model: BuiltGANModel, mparams: GanHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams, self.get_random_labels)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = GanHyperParams(
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
            name = "pokemon_conditional_types",
            similarity_threshold=0.0,
            similarity_penalty=20,
        )

    def get_model(self, mparams: GanHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams, self.vocab)


if __name__ == "__main__":
    PokemonExperiment().start()
