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
from thumbs.params import HyperParams, GanHyperParams
from thumbs.model.model import GanModel, BuiltGANModel

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
    Conv2D,
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

from thumbs.train import Train, TrainBCE, TrainWassersteinGP, TrainBCEPatch


# ngf = 64
# ndf = 128


class PokemonModel(GanModel):
    ngf = 64
    ndf = 128

    def build_generator(self, z_dim):
        model = Sequential(name="generator")

        model.add(Dense(512 * 8 * 8, input_dim=z_dim))
        model.add(Reshape((8, 8, 512)))
        # # 8x8x512

        model.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        # 16x16x512

        model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        # 32x32x256

        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        # 64x64x128

        model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same"))
        model.add(Activation("tanh"))
        model.add(DiffAugmentLayer())
        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        model = Sequential(name="discriminator")

        model.add(DiffAugmentLayer(input_shape=img_shape))
        model.add(Conv2D(self.ndf, kernel_size=4, strides=2, padding="same", use_bias=False))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(self.ndf * 2, kernel_size=4, strides=2, padding="same", use_bias=False))
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(self.ndf * 4, kernel_size=4, strides=2, padding="same", use_bias=False))
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(self.ndf * 8, kernel_size=4, strides=2, padding="same", use_bias=False))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1))

        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator) -> None:
        return None


class PokemonExperiment(Experiment):
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
        schedule[0, 5000] = GanHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=5000,
            sample_interval=20,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=200,
            gradient_penalty_factor=10,
            # l1_loss_factor=200,
            # l2_loss_factor=100,
            # discriminator_ones_zeroes_shape=(batch_size, 14, 14, 1),  # patch gan discriminator
        )

        schedule[5001, 100000] = GanHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0001,
            batch_size=64,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=20,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=200,
            gradient_penalty_factor=10,
            # l1_loss_factor=200,
            # l2_loss_factor=100,
            # discriminator_ones_zeroes_shape=(batch_size, 14, 14, 1),  # patch gan discriminator
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            name="pokemon_diff_augment",
            similarity_threshold=0.0,
            # generator_clip_gradients_norm=1,
            similarity_penalty=0,
        )

    def get_model(self, mparams: GanHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
