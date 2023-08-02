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
from thumbs.params import HyperParams, MutableHyperParams, Sampler
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

ngf = 48
gen_highest_f = 16
ngl = 0
ngb = 4

ndf = 48
disc_highest_f = 16
ndl = 0
ndb = 4


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
        img_input = Input(shape=img_shape, name="img_input")
        x = DiffAugmentLayer()(img_input)

        for f in np.linspace(1, disc_highest_f, ndb):
            x = SpectralNormalization(Conv2D(int(f) * ndf, kernel_size=3, strides=2, padding="same", use_bias=False))(x)
            x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1)(x)

        model = Model(img_input, x, name="discriminator")
        return model


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.augment_zooms = False
        self.augment_rotations = False
        # The paper says that flips seemed to be ok
        self.augment_flips = False
        self.data = get_pokemon_data256(self.params.img_shape)[:16]

    def get_data(self) -> np.ndarray:
        return self.data

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 15300] = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=16,
            adam_b1=0.5,
            iterations=15300,
            sample_interval=100,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=1000,
            gradient_penalty_factor=10,
        )

        schedule[15301, 100000] = MutableHyperParams(
            gen_learning_rate=0.00001,
            dis_learning_rate=0.00002,
            batch_size=16,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=100,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=1000,
            gradient_penalty_factor=10,
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=20, #gen_highest_f * ngf,
            name="pkmn_overfit_48f-16x_20dim-normal_16batch-16total",
            img_shape=(128, 128, 3),
            sampler=Sampler.NORMAL
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
