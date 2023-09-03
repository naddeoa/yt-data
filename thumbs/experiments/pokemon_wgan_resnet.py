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
from thumbs.params import HyperParams, GanHyperParams
from thumbs.model.model import GanModel, BuiltGANModel

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


ngf = 32
gen_highest_f = 16
ngl = 0
ngb = 4

ndf = 32
disc_highest_f = 16
ndl = 0
ndb = 4


class PokemonModel(GanModel):
    def dis_block(
        self,
        input_x,
        f: int,
        res_blocks: int = ndl,
        upsample_f: Optional[int] = None,
    ):
        return self.gen_block(input_x, f, res_blocks, upsample_f, True, Conv=Conv2D, spectral_norm=True)

    def gen_block(
        self,
        input_x,
        f: int,
        res_blocks: int = ngl,
        downsample_f: Optional[int] = None,
        downsample_norm_relu=True,
        Conv=Conv2DTranspose,
        spectral_norm: bool = False,
    ):
        x = input_x

        for i in range(1, res_blocks + 1):
            orig_input = x
            if spectral_norm:
                x = SpectralNormalization(Conv(f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"tconv_{f}_{i}"))(x)
            else:
                x = Conv(f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"tconv_{f}_{i}")(x)

            x = InstanceNormalization(name=f"norm_{f}_{i}")(x)
            x = LeakyReLU(alpha=0.2, name=f"relu_{f}_{i}")(x)

            if spectral_norm:
                x = SpectralNormalization(Conv(f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"tconv_{f}_{i}_2"))(x)
            else:
                x = Conv(f, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"tconv_{f}_{i}_2")(x)

            x = InstanceNormalization(name=f"norm_{f}_{i}_2")(x)
            x = Add(name=f"combine_{f}_{i}")([orig_input, x])  # TODO I can combine these in other ways to experiment
            x = LeakyReLU(alpha=0.2, name=f"relu_{f}_{i}_2")(x)

        if downsample_f is not None:
            if spectral_norm:
                x = SpectralNormalization(
                    Conv(downsample_f, kernel_size=4, strides=2, padding="same", use_bias=False, name=f"tconv_{f}_down")
                )(x)
            else:
                x = Conv(downsample_f, kernel_size=4, strides=2, padding="same", use_bias=False, name=f"tconv_{f}_down")(x)

            if downsample_norm_relu:
                x = InstanceNormalization(name=f"norm_{f}")(x)
                x = LeakyReLU(alpha=0.2, name=f"relu_{f}")(x)

        return x

    def build_generator(self, z_dim):
        z = Input(shape=(z_dim,), name="z")

        x = Dense((ngf * gen_highest_f) * 8 * 8)(z)
        x = Reshape((8, 8, ngf * gen_highest_f))(x)

        ls = np.linspace(gen_highest_f, 1, ngb)
        for features, next_f in zip_longest(ls, ls[1:]):
            if next_f is None:  # then its the last one
                x = self.gen_block(x, f=int(features) * ngf, downsample_f=3, downsample_norm_relu=False)
            else:
                x = self.gen_block(x, f=int(features) * ngf, downsample_f=int(next_f) * ngf)

        x = Activation("tanh", name="tanh")(x)

        model = Model(z, x, name="generator")
        return model

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")
        x = DiffAugmentLayer()(img_input)
        # x = img_input

        ls = [0, *np.linspace(1, disc_highest_f, ndb)]
        for features, next_f in zip_longest(ls, ls[1:]):
            if next_f is None:  # then its the last one
                x = self.dis_block(x, f=int(features) * ndf, upsample_f=None)
            else:
                # for the first one it has to be just three channels
                f = 3 if features == 0 else int(features) * ndf
                x = self.dis_block(x, f=f, upsample_f=int(next_f) * ndf)

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

    def get_train(self, model: BuiltGANModel, mparams: GanHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 6000] = GanHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=8,
            adam_b1=0.5,
            iterations=6000,
            sample_interval=50,
            discriminator_turns=2,
            generator_turns=1,
            checkpoint_interval=1000,
            gradient_penalty_factor=10,
        )

        schedule[6001, 1000000] = GanHyperParams(
            gen_learning_rate=0.00001,
            dis_learning_rate=0.00002,
            batch_size=8,
            adam_b1=0.5,
            iterations=1000000,
            sample_interval=50,
            discriminator_turns=2,
            generator_turns=1,
            checkpoint_interval=1000,
            gradient_penalty_factor=10,
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=4,
            name="pokemon_wgan_resnet_32f-16x_4dim_diff_8batch_0res",
            img_shape=(128, 128, 3),
            similarity_threshold=0.0,
            similarity_penalty=0,
        )

    def get_model(self, mparams: GanHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
