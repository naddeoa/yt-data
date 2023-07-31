import thumbs.config_logging  # must be first
import random
import cv2
import pandas as pd
from itertools import zip_longest
import tensorflow as tf
import os
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np

from thumbs.diff_augmentation import DiffAugmentLayer
from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data256, normalize_image, unnormalize_image
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import GanModel, BuiltModel

from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization, GELU
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU
from keras.layers import (
    Activation,
    Add,
    StringLookup,
    Input,
    BatchNormalization,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    LayerNormalization,
    DepthwiseConv2D,
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

from thumbs.train import Train, TrainBCE, TrainWassersteinGP, TrainBCEPatch, TrainHinge


ngf = 32
gen_highest_f = 16
ngl = 0
ngb = 4

ndf = 32
disc_highest_f = 16
ndl = 0
ndb = 4


def block(input_x, dim, layer_scale_init_value=1e-6):
    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size=7, padding="same", depth_multiplier=1)(input_x)

    # Layer Normalization (channels_last)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Pointwise Convolution 1 (1x1 Convolution)
    x = Conv2D(4 * dim, kernel_size=1, strides=1)(x)
    x = Activation("gelu")(x)

    # Pointwise Convolution 2 (1x1 Convolution)
    x = Conv2D(dim, kernel_size=1, strides=1)(x)

    # Layer Scaling
    if layer_scale_init_value > 0:
        gamma = tf.Variable(initial_value=layer_scale_init_value * tf.ones((dim,)), trainable=True, dtype=tf.float32)
        x = gamma * x

    # Residual Connection
    x = Add()([input_x, x])

    return x


class PokemonModel(GanModel):
    def build_generator(self, z_dim):
        z = Input(shape=(z_dim,), name="z")
        x = z

        for f in np.linspace(gen_highest_f, 1, ngb):
            if f == gen_highest_f:
                x = Dense((ngf * int(f)) * 8 * 8)(x)
                x = Reshape((8, 8, ngf * int(f)))(x)
            else:
                x = Conv2DTranspose(int(f) * ngf, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
                x = InstanceNormalization()(x)
                x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
        x = Activation("tanh", name="tanh")(x)

        model = Model(z, x, name="generator")
        return model

    def build_discriminator(self, img_shape):
        dims = [96, 192, 384, 768]
        img_input = Input(shape=img_shape, name="img_input")
        x = DiffAugmentLayer()(img_input)

        x = Conv2D(dims[0], kernel_size=4, strides=4, padding="valid")(x)
        x = LayerNormalization(epsilon=1e-6)(x)

        x = block(x, dims[0], layer_scale_init_value=1e-6)

        x = LayerNormalization(epsilon=1e-6)(x)
        x = Conv2D(dims[1], kernel_size=2, strides=2, padding="valid")(x)

        x = block(x, dims[1], layer_scale_init_value=1e-6)

        x = LayerNormalization(epsilon=1e-6)(x)
        x = Conv2D(dims[2], kernel_size=2, strides=2, padding="valid")(x)

        x = block(x, dims[2], layer_scale_init_value=1e-6)

        x = LayerNormalization(epsilon=1e-6)(x)
        x = Conv2D(dims[3], kernel_size=2, strides=2, padding="valid")(x)
        x = LayerNormalization(epsilon=1e-6)(x)

        x = Flatten()(x)
        x = Dense(1)(x)

        # Define the model
        model = Model(img_input, x, name="discriminator")
        return model


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.augment_zooms = False
        self.augment_rotations = False
        # The paper says that flips seemed to be ok
        self.augment_flips = False
        self.data = get_pokemon_data256(self.params.img_shape)[:8]

    def get_data(self) -> np.ndarray:
        return self.data

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 6000000] = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=8,
            adam_b1=0.5,
            iterations=6000000,
            sample_interval=50,
            discriminator_turns=2,
            generator_turns=1,
            checkpoint_interval=1000,
            gradient_penalty_factor=10,
            dis_weight_decay= 0.05,
        )

        # schedule[6001, 1000000] = MutableHyperParams(
        #     gen_learning_rate=0.00001,
        #     dis_learning_rate=0.00002,
        #     batch_size=8,
        #     adam_b1=0.5,
        #     iterations=1000000,
        #     sample_interval=50,
        #     discriminator_turns=2,
        #     generator_turns=1,
        #     checkpoint_interval=1000,
        #     gradient_penalty_factor=10,
        # )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=4,
            name="pokemon_convnext",
            img_shape=(128, 128, 3),
            similarity_threshold=0.0,
            similarity_penalty=0,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
