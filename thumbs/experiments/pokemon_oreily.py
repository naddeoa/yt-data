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
from keras.layers import (
    Activation,
    Add,
    StringLookup,
    Conv2DTranspose,
    Conv2D,
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
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1

from thumbs.train import Train, TrainBCE, TrainWassersteinGP, TrainBCEPatch, TrainHinge

tf.keras.layers.Dropout  # TODO is this different than keras.layers.Dropout? Is it still broken?

ngf = 64
gen_highest_f = 8
ngl = 0
ngb = 5

ndf = 64
disc_highest_f = 8
ndl = 0
ndb = 5


class PokemonModel(GanModel):
    def build_generator(self, z_dim):
        z = Input(shape=(z_dim,), name="z")
        x = Reshape((1, 1, z_dim))(z)

        for f in np.linspace(gen_highest_f, 1, ngb):
            if f == gen_highest_f:
                x = Conv2DTranspose(int(f) * ngf, kernel_size=4, strides=1, padding="valid", use_bias=False)(x)
                x = BatchNormalization(momentum=0.9)(x)
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = Conv2DTranspose(int(f) * ngf, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
                x = BatchNormalization(momentum=0.9)(x)
                x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh")(x)

        model = Model(z, x, name="generator")
        return model

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")
        x = DiffAugmentLayer()(img_input)

        for i, f in enumerate(np.linspace(1, disc_highest_f, ndb)):
            x = SpectralNormalization(Conv2D(int(f) * ndf, kernel_size=4, strides=2, padding="same"))(x)
            if i != 0:
                x = BatchNormalizationV1(momentum=0.9)(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = SpectralNormalization(Conv2D(1, kernel_size=4, strides=1, padding="valid"))(x)
        x = Flatten()(x)

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
        schedule[0, 1000000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=8,
            adam_b1=0.5,
            iterations=1000000,
            sample_interval=5,
            discriminator_turns=3,
            generator_turns=1,
            checkpoint_interval=50,
            gradient_penalty_factor=10.0,
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=64,  # gen_highest_f * ngf,
            name="pkmn_oreily_64f-8x_64dim-normal_8batch-all",
            img_shape=(128, 128, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
