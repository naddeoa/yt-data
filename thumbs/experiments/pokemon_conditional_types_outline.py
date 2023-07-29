import thumbs.config_logging  # must be first
from itertools import islice
from copy import copy
import pandas as pd
import os
import tensorflow as tf
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
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU
from keras.layers import (
    Activation,
    StringLookup,
    Input,
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
    def __init__(self, params: HyperParams, mparams: MutableHyperParams, vocab: List[str]) -> None:
        super().__init__(params, mparams)
        # self.vocab = vocab
        self.num_classes = len(vocab) + 1  # 18 pokemon types plus an OOV token

    def build_generator(self, z_dim):
        noise_input = Input(shape=(z_dim,), name="noise input")
        types_input = Input(shape=(self.num_classes), name="types input")
        outline_input = Input(shape=self.params.img_shape, name="outline input")

        # Encoder
        # Layer 1: input shape is (batch_size, 128, 128, 1)
        down1 = Conv2D(ngf, 4, strides=2, padding="same", use_bias=False)(outline_input)  # output shape: (batch_size, 64, 64, 64)
        down1 = LeakyReLU()(down1)

        # Layer 2: input shape is (batch_size, 64, 64, 64)
        down2 = Conv2D(ngf * 2, 4, strides=2, padding="same", use_bias=False)(down1)  # output shape: (batch_size, 32, 32, 128)
        down2 = BatchNormalization()(down2)
        down2 = LeakyReLU()(down2)

        # Layer 3: input shape is (batch_size, 32, 32, 128)
        down3 = Conv2D(ngf * 4, 4, strides=2, padding="same", use_bias=False)(down2)  # output shape: (batch_size, 16, 16, 256)
        down3 = BatchNormalization()(down3)
        down3 = LeakyReLU()(down3)

        # Layer 4: input shape is (batch_size, 16, 16, 256)
        down4 = Conv2D(ngf * 8, 4, strides=2, padding="same", use_bias=False)(down3)  # output shape: (batch_size, 8, 8, 512)
        down4 = BatchNormalization()(down4)
        down4 = LeakyReLU()(down4)

        # Mix in the noise and types
        noise_and_types = Concatenate()([noise_input, types_input])
        noise_and_types = Dense(8 * 8 * ngf * 8)(noise_and_types)
        noise_and_types = Reshape((8, 8, ngf * 8))(noise_and_types)
        bottleneck = Concatenate()([noise_and_types, down4])

        # Decoder with skip connections
        # Layer 1: input shape is (batch_size, 8, 8, 512)
        up1 = Conv2DTranspose(ngf * 4, 4, strides=2, padding="same", use_bias=False)(bottleneck)  # output shape: (batch_size, 16, 16, 256)
        up1 = BatchNormalization()(up1)
        up1 = Concatenate()([up1, down3])  # output shape: (batch_size, 16, 16, 512)
        up1 = LeakyReLU()(up1)

        # Layer 2: input shape is (batch_size, 16, 16, 512)
        up2 = Conv2DTranspose(ngf * 2, 4, strides=2, padding="same", use_bias=False)(up1)  # output shape: (batch_size, 32, 32, 128)
        up2 = BatchNormalization()(up2)
        up2 = Concatenate()([up2, down2])  # output shape: (batch_size, 32, 32, 256)
        up2 = LeakyReLU()(up2)

        # Layer 3: input shape is (batch_size, 32, 32, 256)
        up3 = Conv2DTranspose(ngf, 4, strides=2, padding="same", use_bias=False)(up2)  # output shape: (batch_size, 64, 64, 64)
        up3 = BatchNormalization()(up3)
        up3 = Concatenate()([up3, down1])  # output shape: (batch_size, 64, 64, 128)
        up3 = LeakyReLU()(up3)

        # Final layer: input shape is (batch_size, 64, 64, 128)
        last = Conv2DTranspose(3, 4, strides=2, padding="same", use_bias=False, activation="tanh")(
            up3
        )  # output shape: (batch_size, 128, 128, 3)

        model = tf.keras.Model([noise_input, types_input, outline_input], last, name="generator")

        model.summary(line_length=200)
        f = "/mnt/e/experiments/generator.jpg"
        tf.keras.utils.plot_model(model, to_file=f, show_shapes=True, dpi=64)
        return model

    def build_discriminator(self, img_shape):
        def convolutions(x):
            x = Conv2D(ndf, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
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
            return x

        types_input = Input(shape=(self.num_classes,), name="types input")
        types = Dense(img_shape[0] * img_shape[1] * img_shape[2])(types_input)
        types = Reshape(img_shape)(types)

        image_input = Input(shape=img_shape, name="image input")

        image = DiffAugmentLayer()(image_input)
        image = Concatenate()([image, types])
        image = convolutions(image)

        outline_input = Input(shape=self.params.img_shape, name="outline input")
        # outline = DiffAugmentLayer()(outline_input)
        # outline = convolutions(outline)


        x = Conv2D(ndf * 10, kernel_size=3, strides=1, padding="same", use_bias=False)(image)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1)(x)

        model = tf.keras.Model([image_input, types_input, outline_input], x, name="discriminator")
        model.summary(line_length=200)
        f = "/mnt/e/experiments/discriminator.jpg"
        tf.keras.utils.plot_model(model, to_file=f, show_shapes=True, dpi=64)
        return model

    def build_gan(self, generator, discriminator):
        # Can't really do this easily because they both take multiple inputs
        return None


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

    def lookup_inverted(self, multi_hot_encoded):
        return [self.vocab[i - 1] for i, is_present in enumerate(multi_hot_encoded) if is_present]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.images, self.labels

    def get_random_labels(self, n: int):
        params = copy(self.get_mutable_params()[0])
        params.batch_size = n
        dataset = tf.data.Dataset.from_tensor_slices(self.get_data())
        data_iterator = self.prepare_data(dataset, params).__iter__()
        items = data_iterator.get_next()
        return items[1:]

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams, self.get_random_labels)

    def custom_agumentation(self, images: tf.Tensor, type_labels: Optional[tf.Tensor] = None) -> Union[tf.Tensor, tuple]:
        assert type_labels is not None
        output = super().custom_agumentation(images, None)
        assert not isinstance(output, tuple)
        images = output

        # This seems to be a reasonable range for the outline. Higher and some images were totally blank
        upper = tf.random.uniform(shape=[], minval=100, maxval=700, dtype=tf.int32)
        outlines = tf.py_function(self.create_outline_tensor, [images, upper], tf.float32)
        return images, type_labels, outlines

    def create_outline_tensor(self, image: tf.Tensor, upper=255, lower=None) -> tf.Tensor:
        outline = self.create_outline(image.numpy(), upper=upper.numpy(), lower=lower)
        tensor: tf.Tensor = tf.convert_to_tensor(outline, dtype=tf.float32)
        return tensor

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

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=10,
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
            name="pokemon_conditional_types_outline2",
            similarity_threshold=0.0,
            similarity_penalty=20,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams, self.vocab)


if __name__ == "__main__":
    PokemonExperiment().start()
