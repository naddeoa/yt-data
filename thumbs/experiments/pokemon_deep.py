import thumbs.config_logging  # must be first
import random
import cv2
import pandas as pd
import tensorflow as tf
import os
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np

from thumbs.diff_augmentation import DiffAugmentLayer
from thumbs.experiment import GanExperiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data256, normalize_image, unnormalize_image
from thumbs.params import HyperParams, GanHyperParams
from thumbs.model.model import GanModel, BuiltGANModel

from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from tensorflow.keras.models import Model
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

from thumbs.train import Train, TrainBCE, TrainWassersteinGP, TrainBCEPatch


ngf = 128
ngl = 1

ndf = 128
ndl = 1


class PokemonModel(GanModel):
    def gen_block(self, input_x, f: int = 0, layers: int = ngl):
        x = Conv2DTranspose(ngf * f, kernel_size=4, strides=2, padding="same", use_bias=False, name=f"tconv_{f}")(input_x)
        x = BatchNormalization(name=f"batch_norm_{f}")(x)
        x = LeakyReLU(name=f"leaky_relu_{f}")(x)

        for i in range(1, layers + 1):
            x = Conv2DTranspose(ngf * f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"tconv_{f}_{i}")(x)
            x = BatchNormalization(name=f"instance_norm_{f}_{i}")(x)
            x = LeakyReLU(name=f"leaky_relu_{f}_{i}")(x)

        return x

    def build_generator(self, z_dim):
        z = Input(shape=(z_dim,), name="z")

        x = Dense(ngf * 4 * 8 * 8)(z)
        x = Reshape((8, 8, ngf * 4))(x)

        x = self.gen_block(x, f=4)
        x = self.gen_block(x, f=2)
        x = self.gen_block(x, f=1)

        x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same")(x)
        x = Activation("tanh")(x)

        model = Model(z, x, name="generator")

        return model

    def disc_block(self, input_x, f: int = 0, layers: int = ndl, normalize_first: bool = True, normalize_last: bool = True):
        x = SpectralNormalization(Conv2D(ndf * f, kernel_size=4, strides=2, padding="same", use_bias=False, name=f"conv_{f}"))(input_x)
        if normalize_first:
            x = InstanceNormalization(name=f"batch_norm_{f}")(x)
        x = LeakyReLU(name=f"leaky_relu_{f}")(x)

        for i in range(1, layers + 1):
            x = SpectralNormalization(Conv2D(ndf * f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"conv_{f}_{i}"))(x)
            if i == layers and normalize_last:
                x = InstanceNormalization(name=f"instance_norm_{f}_{i}")(x)
            x = LeakyReLU(name=f"leaky_relu_{f}_{i}")(x)

        return x

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")
        x = DiffAugmentLayer()(img_input)

        x = self.disc_block(x, f=1, normalize_first=False)
        x = self.disc_block(x, f=2)
        x = self.disc_block(x, f=4)
        x = self.disc_block(x, f=8, normalize_first=True, normalize_last=False)

        x = Flatten()(x)
        x = SpectralNormalization(Dense(1))(x)

        return model

    def build_gan(self, generator, discriminator) -> None:
        return None


class PokemonExperiment(GanExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.augment_zooms = False
        self.augment_rotations = False
        # The paper says that flips seemed to be ok
        self.augment_flips = True

    def get_data(self) -> np.ndarray:
        return get_pokemon_data256(self.params.img_shape)

    def get_train(self, model: BuiltGANModel, mparams: GanHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = GanHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=5,
            discriminator_turns=5,
            generator_turns=1,
            checkpoint_interval=200,
            gradient_penalty_factor=0,
            g_clipnorm=0.5,
            # d_clipnorm=1.0,
            # l1_loss_factor=200,
            # l2_loss_factor=100,
            # discriminator_ones_zeroes_shape=(batch_size, 14, 14, 1),  # patch gan discriminator
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            name="pokemon_deep_1L_clipped-0.5_gp-0",
            similarity_threshold=0.0,
            similarity_penalty=0,
        )

    def get_model(self, mparams: GanHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
