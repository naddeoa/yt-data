import thumbs.config_logging  # must be first
import random
import cv2
import pandas as pd
import tensorflow as tf
import os
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data256, normalize_image, unnormalize_image
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import Model, BuiltModel

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

from thumbs.train import Train, TrainMSE, TrainBCE, TrainBCESimilarity, TrainWassersteinGP


infinity = float("inf")

ngf = 128
ndf = 128


class PokemonModel(Model):
    def build_generator(self, z_dim):
        noise_input = Input(shape=(z_dim,), name="noise input")

        # black/white only, same dimensions as image
        outline_shape = (self.params.img_shape[0], self.params.img_shape[1], 1)
        outline_input = Input(shape=outline_shape, name="outline input")
        # 128x128x1

        outline = Conv2D(64, kernel_size=5, strides=2, padding="same", use_bias=False)(outline_input)
        outline = LeakyReLU(alpha=0.2)(outline)
        # 64x64x64

        outline = Conv2D(128, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        outline = InstanceNormalization()(outline)
        outline = LeakyReLU(alpha=0.2)(outline)
        # 32x32x128
        
        outline = Conv2D(256, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        outline = InstanceNormalization()(outline)
        outline = LeakyReLU(alpha=0.2)(outline)
        # 16x16x256

        outline = Conv2D(512, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        outline = InstanceNormalization()(outline)
        outline = LeakyReLU(alpha=0.2)(outline)
        # 8x8x512
        
        # outline = Conv2D(512, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        # outline = InstanceNormalization()(outline)
        # outline = LeakyReLU(alpha=0.2)(outline)
        # 4x4x512

        # outline = Conv2D(512, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        # outline = InstanceNormalization()(outline)
        # outline = LeakyReLU(alpha=0.2)(outline)
        # 2x2x512

        # outline = Conv2D(512, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        # outline = InstanceNormalization()(outline)
        # outline = LeakyReLU(alpha=0.2)(outline)
        # 1x1x512

        # Bottleneck
        # x = Reshape((1, 1, z_dim))(noise_input)
        # x = Concatenate(axis=-1)([x, outline])
        # 1x1x612

        # Bottleneck
        x = Dense(8 * 8 * 512, use_bias=False)(noise_input)
        x = Reshape((8, 8, 512))(x)
        x = Concatenate(axis=-1)([x, outline])
        # 8x8x1024

        # x = Conv2DTranspose(612, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        # x = InstanceNormalization()(x)
        # x = ReLU()(x)
        # 2x2x612

        # x = Conv2DTranspose(612, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        # x = InstanceNormalization()(x)
        # x = ReLU()(x)
        # 4x4x612

        # x = Conv2DTranspose(612, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        # x = InstanceNormalization()(x)
        # x = ReLU()(x)
        # 8x8x612

        x = Conv2DTranspose(512, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = ReLU()(x)
        # 16x16x256

        x = Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = ReLU()(x)
        # 32x32x128

        x = Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = ReLU()(x)
        # 64x64x64

        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = Activation("tanh")(x)
        # 128x128x3

        model = tf.keras.Model([noise_input, outline_input], x, name="generator")
        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        # 128x128x3
        image_input = Input(shape=img_shape)

        # 128x128x1
        outline_shape = (self.params.img_shape[0], self.params.img_shape[1], 1)
        outline_input = Input(shape=outline_shape)

        model_input = Concatenate(axis=-1)([image_input, outline_input])
        # 128x128x4

        x = Conv2D(64, kernel_size=5, strides=2, padding="same", use_bias=False)(model_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(0.2)(x)
        # 64x64x64

        x = Conv2D(128, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalizationV1()(x)
        # x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(0.2)(x)
        # 32x32x128

        x = Conv2D(256, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalizationV1()(x)
        # x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(0.2)(x)
        # 16x16x256

        x = Conv2D(512, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalizationV1()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(0.2)(x)
        # 8x8x512

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model([image_input, outline_input], x, name="discriminator")
        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator):
        # Can't really do this easily because they both take multiple inputs
        return None


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.zoom_factor = 0.9
        self.images = get_pokemon_data256(self.params.img_shape)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # For each image, generate an outline
        outlines = np.array([self.create_outline(image, threshold2=1000) for image in self.images])
        return (self.images, outlines)

    def get_random_labels(self, n: int):
        random_indexes = np.random.choice(len(self.images), size=n, replace=False)
        random_images = self.images[random_indexes]

        # For each image, generate an outline
        outlines = np.array([self.create_outline(image) for image in random_images])
        return np.asarray(outlines)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainBCESimilarity(model, self.params, mparams, self.get_random_labels)
        # return TrainWassersteinGP(model, self.params, mparams, self.get_random_labels)

    def custom_agumentation(
        self, image: tf.Tensor, outline: Optional[tf.Tensor] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Optional[tf.Tensor]]]:
        assert outline is not None

        # Create the outline
        # Generate a shape 2 tensor with two random ints
        seed = tf.random.uniform(shape=(2,), minval=0, maxval=10, dtype=tf.int32)

        image = tf.image.stateless_random_flip_left_right(image, seed=seed)
        outline = tf.image.stateless_random_flip_left_right(outline, seed=seed)

        # Get another random int from numpy
        seednp = np.random.randint(0, 10000)
        image = tf.keras.layers.RandomRotation(0.05, seed=seednp)(image)
        outline = tf.keras.layers.RandomRotation(0.05, seed=seednp)(outline)

        # 10% zoom
        (x, y, channels) = self.params.img_shape
        image = tf.image.stateless_random_crop(image, size=[int(x * self.zoom_factor), int(y * self.zoom_factor), channels], seed=seed)
        image = tf.image.resize(image, [x, y])

        outline = tf.image.stateless_random_crop(outline, size=[int(x * self.zoom_factor), int(y * self.zoom_factor), 1], seed=seed)
        outline = tf.image.resize(outline, [x, y])

        return image, outline

    def create_outline(self, image: np.ndarray, threshold2=None) -> np.ndarray:
        """
        image: numpy array that has been normalized to -1,1
        return: numpy array of an outline, noramlized to -1,1, with the same height and width
        """
        # Will result in outlines of varying detail
        image = unnormalize_image(image)
        thresholds = [10, 100, 400, 800, 1000]
        if threshold2 is None:
            threshold2 = np.random.choice(thresholds)

        edges: np.ndarray = cv2.bitwise_not(cv2.Canny(image, threshold1=0, threshold2=threshold2))
        edges = np.expand_dims(edges, axis=-1)

        assert edges.shape == (image.shape[0], image.shape[1], 1)
        return normalize_image(edges)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.00008,
            batch_size=64,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=5,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=20,
            gradient_penalty_factor=10,
        )


        return schedule

    def get_params(self) -> HyperParams:
        name = "pokemon_conditional_outline_noise"

        exp_dir = "EXP_DIR"
        if exp_dir in os.environ:
            base_dir = os.environ["EXP_DIR"]
        else:
            base_dir = "/mnt/e/experiments"

        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            weight_path=f"{base_dir}/{name}/weights",
            checkpoint_path=f"{base_dir}/{name}/checkpoints",
            prediction_path=f"{base_dir}/{name}/predictions",
            iteration_checkpoints_path=f"{base_dir}/{name}/iteration_checkpoints",
            loss_path=f"{base_dir}/{name}/loss",
            accuracy_path=f"{base_dir}/{name}/accuracy",
            iteration_path=f"{base_dir}/{name}/iteration",
            similarity_threshold=0.0,
            generator_clip_gradients_norm=1,
            similarity_penalty=1,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
