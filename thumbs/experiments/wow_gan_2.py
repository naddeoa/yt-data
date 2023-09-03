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
from thumbs.experiment import GanExperiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_and_pokedexno, normalize_image, unnormalize_image, get_wow_icons_64
from thumbs.params import HyperParams, GanHyperParams, Sampler
from thumbs.model.model import GanModel, BuiltGANModel

from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from tensorflow.keras.models import Model
from keras import Sequential
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
from tensorflow.keras.layers import Layer

tf.keras.layers.Dropout  # TODO is this different than keras.layers.Dropout? Is it still broken?


kern_size = 3

ngf = 64
gen_highest_f = 8
gen_blocks = [6, 12, 12]

ndf = 64
disc_highest_f = 8
disc_blocks = [12, 12, 6]


class SubtractOneLayer(Layer):
    def call(self, inputs):
        return inputs - 1


class MyModel(GanModel):
    def _generator_block(
        self,
        f: int,
        x,
        strides: int,
        kernel_size: int,
    ):
        input_x = x

        seq = Sequential()
        seq.add(
            Conv2DTranspose(
                f * ngf,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid" if kernel_size > kern_size else "same",
                use_bias=False,
            )
        )
        seq.add(InstanceNormalization())
        seq.add(LeakyReLU(alpha=0.2))

        if strides == 1 and kernel_size == kern_size:
            seq.add(
                Conv2DTranspose(
                    f * ngf,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=False,
                )
            )
            seq.add(InstanceNormalization())

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add()([x, input_x])
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def _generator_core(self, z):
        x = z
        x = Reshape((1, 1, -1))(x)

        for i, f in enumerate(np.linspace(gen_highest_f, 1, len(gen_blocks), dtype=int)):
            if f == gen_highest_f:
                x = self._generator_block(f, x, strides=1, kernel_size=8)
            else:
                x = self._generator_block(f, x, strides=2, kernel_size=kern_size)

            for i in range(gen_blocks[i]):
                x = self._generator_block(f, x, strides=1, kernel_size=kern_size)

        x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh")(x)
        return x

    def build_generator(self, z_dim):
        z = Input(shape=(z_dim,), name="z")
        x = self._generator_core(z)
        model = Model(z, x, name="generator")
        return model

    def _discriminator_block(
        self,
        f: int,
        x,
        i: Optional[str],
        strides: int,
        kernel_size: int = kern_size,
        normalize=True,
    ):
        input_x = x
        seq = Sequential()
        seq.add(
            SpectralNormalization(
                Conv2D(
                    f * ndf,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                )
            )
        )

        if normalize:
            seq.add(InstanceNormalization())
        seq.add(LeakyReLU(alpha=0.2))

        if strides == 1:
            seq.add(
                SpectralNormalization(
                    Conv2D(
                        f * ndf,
                        kernel_size=kernel_size,
                        strides=1,
                        padding="same",
                    )
                )
            )
            seq.add(InstanceNormalization())

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add()([x, input_x])
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")

        x = img_input
        x = DiffAugmentLayer()(x)

        for i, f in enumerate(np.linspace(1, disc_highest_f, len(disc_blocks), dtype=int)):
            x = self._discriminator_block(f, x, i="", strides=2, normalize=False)

            for j in range(disc_blocks[i]):
                x = self._discriminator_block(f, x, i=j, strides=1)

        x = SpectralNormalization(Conv2D(1, kernel_size=8, strides=1, padding="valid"))(x)
        x = Flatten()(x)

        model = Model(img_input, x, name="discriminator")
        return model


class MyExperiment(GanExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.data = get_wow_icons_64()

    def augment_data(self) -> bool:
        return False

    def get_data(self) -> tf.data.Dataset:
        return self.data

    def get_train(self, model: BuiltGANModel, mparams: GanHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = GanHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=1,
            generator_turns=1,
            discriminator_turns=3,
            g_clipnorm=0.001,
            d_clipnorm=0.001,
            gradient_penalty_factor=10.0,
            gen_weight_decay=0,  # Add back in after a while
            dis_weight_decay=0,
            notes="""
I give up on pokemon. Nothing is any better than anything else I've tried. I think I just need to do something that actually has data.
""",
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=200,  # gen_highest_f * ngf,
            name="wow_resnet_2",
            img_shape=(64, 64, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: GanHyperParams) -> GanModel:
        return MyModel(self.params, mparams)


if __name__ == "__main__":
    MyExperiment().start()
