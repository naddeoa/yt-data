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
from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data256, normalize_image, unnormalize_image
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import GanModel, BuiltModel

from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
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


ngf = 128
gen_highest_f = 16
ngl = 2
ngb = 4

ndf = 128
disc_highest_f = 16
ndl = 2
ndb = 4


class PokemonModel(GanModel):
    def gen_block(self, input_x, f: int = 0, layers: int = ngl):
        x = Conv2DTranspose(ngf * f, kernel_size=4, strides=2, padding="same", use_bias=False, name=f"tconv_{f}")(input_x)
        x = InstanceNormalization(name=f"norm_{f}")(x)
        x = LeakyReLU(name=f"leaky_relu_{f}")(x)

        first = x

        for i in range(1, layers + 1):
            x = Conv2DTranspose(ngf * f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"tconv_{f}_{i}")(x)
            x = InstanceNormalization(name=f"norm_{f}_{i}")(x)
            x = LeakyReLU(name=f"leaky_relu_{f}_{i}")(x)

        if layers > 0:
            x = Add(name=f"combine_{f}")([first, x])  # TODO I can combine these in other ways to experiment
            x = LeakyReLU(name=f"leaky_relu_add_{f}")(x)

        return x

    def build_generator(self, z_dim):
        nblocks = ngb
        z = Input(shape=(z_dim,), name="z")

        x = Dense((ngf * gen_highest_f) * 4 * 4)(z)
        x = LeakyReLU()(x)
        x = Reshape((4, 4, ngf * gen_highest_f))(x)

        for i, features in enumerate(np.linspace(disc_highest_f, 1, nblocks)):
            x = self.gen_block(x, f=int(features))

        x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same")(x)  # 128,128,3
        x = Activation("tanh")(x)

        model = Model(z, x, name="generator")
        return model

    def disc_block(self, input_x, f: int = 0, layers: int = ndl, normalize_first: bool = True, normalize_last: bool = True):
        x = SpectralNormalization(Conv2D(ndf * f, kernel_size=4, strides=2, padding="same", use_bias=False, name=f"conv_{f}"))(input_x)
        if normalize_first:
            x = InstanceNormalization(name=f"instance_norm_{f}")(x)
        x = LeakyReLU(name=f"leaky_relu_{f}")(x)

        first = x

        for i in range(1, layers + 1):
            x = SpectralNormalization(Conv2D(ndf * f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"conv_{f}_{i}"))(x)
            if i == layers and not normalize_last:
                pass
            else:
                x = InstanceNormalization(name=f"instance_norm_{f}_{i}")(x)
            x = LeakyReLU(name=f"leaky_relu_{f}_{i}")(x)

        if layers > 0:
            x = Add(name=f"combine_{f}")([first, x])  # TODO I can combine these in other ways to experiment
            x = LeakyReLU(name=f"leaky_relu_add_{f}")(x)

        return x

    def build_discriminator(self, img_shape):
        nblocks = ndb
        img_input = Input(shape=img_shape, name="img_input")
        x = DiffAugmentLayer()(img_input)

        for i, features in enumerate(np.linspace(1, disc_highest_f, nblocks)):
            x = self.disc_block(x, f=int(features), normalize_first=i != 0 and i != nblocks - 1, normalize_last=i != nblocks - 1)

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
        self.augment_flips = True
        self.data = get_pokemon_data256(self.params.img_shape)

    def get_data(self) -> np.ndarray:
        return self.data

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100] = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=16,
            adam_b1=0.5,
            iterations=100,
            sample_interval=5,
            discriminator_turns=2,
            generator_turns=1,
            checkpoint_interval=200,
            gradient_penalty_factor=10,
        )

        schedule[101, 100000] = MutableHyperParams(
            gen_learning_rate=0.00001,
            dis_learning_rate=0.00002,
            batch_size=16,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=5,
            discriminator_turns=2,
            generator_turns=1,
            checkpoint_interval=200,
            gradient_penalty_factor=10,
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,
            name="pokemon_wgan_deep_fat",
            img_shape=(128, 128, 3),
            similarity_threshold=0.0,
            similarity_penalty=0,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
