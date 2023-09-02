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
from thumbs.data import get_pokemon_and_pokedexno, normalize_image, unnormalize_image, get_pokemon_data256
from thumbs.params import HyperParams, MutableHyperParams, Sampler
from thumbs.model.model import GanModel, BuiltModel

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


# ngf = 64
# gen_highest_f = 8
# ngl = 4
# ngb = 3

# ndf = 64
# disc_highest_f = 8
# ndl = 4
# ndb = 3


# class SubtractOneLayer(Layer):
#     def call(self, inputs):
#         return inputs - 1


# class MyModel(GanModel):
#     def _generator_upsample_block(
#         self,
#         f: int,
#         x,
#         strides: int,
#         kernel_size: int = 3,
#     ):
#         seq = Sequential()
#         seq.add(
#             Conv2DTranspose(
#                 f * ngf,
#                 kernel_size=kernel_size,
#                 strides=strides,
#                 padding="valid" if strides == 1 else "same",
#                 use_bias=False,
#             )
#         )
#         seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))
#         return seq(x)

#     def _generator_resnet_block(
#         self,
#         f: int,
#         x,
#     ):
#         input_x = x

#         seq = Sequential()
#         seq.add(
#             Conv2DTranspose(
#                 f * ngf,
#                 kernel_size=3,
#                 strides=1,
#                 padding="same",
#                 use_bias=False,
#             )
#         )
#         seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))
#         seq.add(
#             Conv2DTranspose(
#                 f * ngf,
#                 kernel_size=3,
#                 strides=1,
#                 padding="same",
#                 use_bias=False,
#             )
#         )
#         seq.add(InstanceNormalization())

#         x = seq(x)
#         x = Add()([x, input_x])
#         x = LeakyReLU(alpha=0.2)(x)
#         return x

#     def _generator_resnet_bottleneck_block(
#         self,
#         f: int,
#         x,
#     ):
#         shortcut = x

#         seq = Sequential()

#         # first 1x1
#         seq.add(
#             Conv2DTranspose(
#                 f * ngf,
#                 kernel_size=1,
#                 strides=1,
#                 padding="same",
#                 use_bias=False,
#             )
#         )
#         seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))

#         # second 3x3
#         seq.add(
#             Conv2DTranspose(
#                 f * ngf,
#                 kernel_size=3,
#                 strides=1,
#                 padding="same",
#                 use_bias=False,
#             )
#         )
#         seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))

#         # the  4x 1x1 conv
#         filters = f * ngf * 4
#         seq.add(
#             Conv2DTranspose(
#                 filters,
#                 kernel_size=1,
#                 strides=1,
#                 padding="same",
#                 use_bias=False,
#             )
#         )
#         seq.add(InstanceNormalization())

#         x = seq(x)
#         _, H, W, C = shortcut.shape
#         shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, filters - C]])
#         x = Add()([x, shortcut])
#         x = LeakyReLU(alpha=0.2)(x)
#         return x

#     def build_generator(self, z_dim):
#         z = Input(shape=(z_dim,), name="z")

#         x = Reshape((1, 1, -1))(z)

#         for f in [int(f) for f in np.linspace(gen_highest_f, 1, ngb)]:
#             if f == gen_highest_f:
#                 x = self._generator_upsample_block(f, x, strides=1, kernel_size=8)
#             else:
#                 x = self._generator_upsample_block(f, x, strides=2)

#             for _ in range(ngl):
#                 x = self._generator_resnet_block(f, x)

#         x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh")(x)

#         model = Model(z, x, name="generator")
#         return model

#     def _discriminator_downsample_block(
#         self,
#         f: int,
#         x,
#         normalize=True,
#     ):
#         seq = Sequential()
#         seq.add(
#             SpectralNormalization(
#                 Conv2D(
#                     f * ndf,
#                     kernel_size=3,
#                     strides=2,
#                     padding="same",
#                 )
#             )
#         )

#         if normalize:
#             seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))
#         return seq(x)

#     def _discriminator_bottleneck_block(
#         self,
#         f: int,
#         x,
#     ):
#         shortcut = x
#         seq = Sequential()
#         seq.add(
#             SpectralNormalization(
#                 Conv2D(
#                     f * ndf,
#                     kernel_size=1,
#                     strides=1,
#                     padding="same",
#                 )
#             )
#         )

#         seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))

#         seq.add(
#             SpectralNormalization(
#                 Conv2D(
#                     f * ndf,
#                     kernel_size=3,
#                     strides=1,
#                     padding="same",
#                 )
#             )
#         )
#         seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))

#         filters = f * ndf * 4
#         seq.add(
#             SpectralNormalization(
#                 Conv2D(
#                     filters,
#                     kernel_size=1,
#                     strides=1,
#                     padding="same",
#                 )
#             )
#         )
#         seq.add(InstanceNormalization())

#         x = seq(x)
#         _, H, W, C = shortcut.shape
#         shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, filters - C]])
#         x = Add()([x, shortcut])
#         x = LeakyReLU(alpha=0.2)(x)
#         return x

#     def _discriminator_resnet_block(
#         self,
#         f: int,
#         x,
#     ):
#         input_x = x
#         seq = Sequential()
#         seq.add(
#             SpectralNormalization(
#                 Conv2D(
#                     f * ndf,
#                     kernel_size=3,
#                     strides=1,
#                     padding="same",
#                 )
#             )
#         )

#         seq.add(InstanceNormalization())
#         seq.add(LeakyReLU(alpha=0.2))

#         seq.add(
#             SpectralNormalization(
#                 Conv2D(
#                     f * ndf,
#                     kernel_size=3,
#                     strides=1,
#                     padding="same",
#                 )
#             )
#         )
#         seq.add(InstanceNormalization())

#         x = seq(x)
#         x = Add()([x, input_x])
#         x = LeakyReLU(alpha=0.2)(x)
#         return x

#     def build_discriminator(self, img_shape):
#         img_input = Input(shape=img_shape, name="img_input")

#         x = img_input
#         x = DiffAugmentLayer()(x)

#         for i, f in enumerate([int(f) for f in np.linspace(1, disc_highest_f, ndb)]):
#             x = self._discriminator_downsample_block(int(f), x, normalize=i != 0)

#             for _ in range(ndl):
#                 x = self._discriminator_resnet_block(int(f), x)

#         x = SpectralNormalization(Conv2D(1, kernel_size=8, strides=1, padding="valid"))(x)
#         x = Flatten()(x)

#         model = Model(img_input, x, name="discriminator")
#         return model


kern_size = 3

ngf = 64
gen_highest_f = 8
ngl = 2
ngb = 3

ndf = 64
disc_highest_f = 8
ndl = 2
ndb = 3


class SubtractOneLayer(Layer):
    def call(self, inputs):
        return inputs - 1


class MyModel(GanModel):
    def _generator_block(
        self,
        f: int,
        x,
        i: Optional[int],
        strides: int,
        kernel_size: int,
    ):
        input_x = x

        seq = Sequential(name=f"gen_block{f}_{i}")
        seq.add(
            Conv2DTranspose(
                f * ngf,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid" if kernel_size > kern_size else "same",
                use_bias=False,
                name=f"tconf{f}_{i}",
            )
        )
        seq.add(InstanceNormalization(name=f"norm{f}_{i}"))
        seq.add(LeakyReLU(alpha=0.2, name=f"relu{f}_{i}"))

        if strides == 1 and kernel_size == kern_size:
            seq.add(
                Conv2DTranspose(
                    f * ngf,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    name=f"tconf{f}_{i}2",
                )
            )
            seq.add(InstanceNormalization(name=f"norm{f}_{i}2"))

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add(name=f"add{f}_{i}")([x, input_x])
            x = LeakyReLU(alpha=0.2, name=f"relu{f}_{i}")(x)
        return x

    def _generator_core(self, z):
        x = z
        x = Reshape((1, 1, -1))(x)

        for f in [int(f) for f in np.linspace(gen_highest_f, 1, ngb)]:
            if f == gen_highest_f:
                x = self._generator_block(f, x, i="", strides=1, kernel_size=8)
            else:
                x = self._generator_block(f, x, i="", strides=2, kernel_size=kern_size)

            # This was fixed in this model. It didn't have hidden layers on the generator's first layer
            for i in range(ngl):
                x = self._generator_block(f, x, i, strides=1, kernel_size=kern_size)

        x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh", name=f"tconf_final")(x)
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
        seq = Sequential(name=f"disc_block{f}_{i}")
        seq.add(
            SpectralNormalization(
                Conv2D(
                    f * ndf,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    name=f"conv_{f}_{i}",
                )
            )
        )

        if normalize:
            seq.add(InstanceNormalization(name=f"norm{f}_{i}"))
        seq.add(LeakyReLU(alpha=0.2, name=f"relu{f}_{i}"))

        if strides == 1:
            seq.add(
                SpectralNormalization(
                    Conv2D(
                        f * ndf,
                        kernel_size=kernel_size,
                        strides=1,
                        padding="same",
                        name=f"conv_{f}_{i}2",
                    )
                )
            )
            seq.add(InstanceNormalization(name=f"norm{f}_{i}2"))

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add(name=f"add{f}_{i}")([x, input_x])
            x = LeakyReLU(alpha=0.2, name=f"relu{f}_{i}2")(x)
        return x

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")

        x = img_input
        x = DiffAugmentLayer()(x)

        for i, f in enumerate([int(f) for f in np.linspace(1, disc_highest_f, ndb)]):
            x = self._discriminator_block(f, x, i="", strides=2, normalize=False)

            for j in range(ndl):
                x = self._discriminator_block(f, x, i=j, strides=1)

        x = SpectralNormalization(Conv2D(1, kernel_size=8, strides=1, padding="valid"))(x)
        x = Flatten()(x)

        model = Model(img_input, x, name="discriminator")
        return model


class MyExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.augment_rotations = False
        self.augment_zooms = False
        self.augment_flips = True
        self.data = get_pokemon_data256(self.get_params().img_shape)

    def augment_data(self) -> bool:
        return True 

    def get_data(self) -> np.ndarray:
        return self.data

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 900] = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=900,
            sample_interval=10,
            generator_turns=1,
            discriminator_turns=3,
            g_clipnorm=0.001,
            d_clipnorm=0.001,
            gradient_penalty_factor=10.0,
            # gen_weight_decay=0,
            # dis_weight_decay=0,
            notes="""
Had a bug in my original resnet impl. Seeing if the fixed one still has mode collapse for pkmn.
""",
        )

        schedule[901, 100000] = MutableHyperParams(
            gen_learning_rate=0.00001,
            dis_learning_rate=0.00003,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=5,
            generator_turns=1,
            discriminator_turns=5,
            # g_clipnorm=0.001,
            # d_clipnorm=0.001,
            gradient_penalty_factor=10.0,
            # gen_weight_decay=0,
            # dis_weight_decay=0,
            notes="""
Had a bug in my original resnet impl. Seeing if the fixed one still has mode collapse for pkmn.
""",
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,  # gen_highest_f * ngf,
            name="pkmn_resnet",
            img_shape=(64, 64, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return MyModel(self.params, mparams)


if __name__ == "__main__":
    MyExperiment().start()
