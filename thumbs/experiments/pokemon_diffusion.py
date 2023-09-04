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
from thumbs.experiment import DiffusionExperiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_and_pokedexno, normalize_image, unnormalize_image, get_wow_icons_64, get_pokemon_data256
from thumbs.params import DiffusionHyperParams, HyperParams, Sampler, MutableHyperParams
from thumbs.model.model import GanModel, BuiltDiffusionModel, FrameworkModel, DiffusionModel

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

from thumbs.train import Train, TrainDiffusion
from tensorflow.keras.layers import Layer

tf.keras.layers.Dropout  # TODO is this different than keras.layers.Dropout? Is it still broken?


kern_size = 3

ngf = 64
gen_highest_f = 8
gen_blocks = [2, 2, 2]

ndf = 64
disc_highest_f = 8
disc_blocks = [2, 2, 2]


class MyModel(DiffusionModel):
    def __init__(self, params: HyperParams, mparams: DiffusionHyperParams) -> None:
        super().__init__(params, mparams)
        self.embed_dim = 30

    def concat_embedding(self, x, embedding):
        _, H, W, C = x.shape
        _x = Dense(H * W, use_bias=False)(embedding)
        _x = Reshape((H, W, 1))(_x)
        return Concatenate()([x, _x])

    def get_model(self) -> Model:
        img_input = Input(shape=self.params.img_shape, name="image")
        t_input = Input(shape=(1,), name="t")
        e_embedding = Embedding(self.mparams.T, self.embed_dim, name="e_embedding")(t_input)

        x = Conv2D(64, kernel_size=3, strides=2, padding="same", use_bias=False)(img_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.concat_embedding(x, e_embedding)
        down1 = x  # 32x32x(64 + 1)

        x = Conv2D(128, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.concat_embedding(x, e_embedding)
        down2 = x  # 16x16x(128 + 1)

        x = Conv2D(256, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.concat_embedding(x, e_embedding)
        down3 = x  # 8x8x(256 + 1)

        x = Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Concatenate()([x, down2])  # 16x16x512
        x = self.concat_embedding(x, e_embedding)  # 16x16x(512 + 1)

        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Concatenate()([x, down1])  # 32x32x256
        x = self.concat_embedding(x, e_embedding)  # 32x32x(256 + 1)

        x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.concat_embedding(x, e_embedding)  # 64x64x(128 + 1)

        output = Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", use_bias=False, activation="tanh")(x)
        return Model([img_input, t_input], output, name="diffusion_model")


class MyExperiment(DiffusionExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.data = get_pokemon_data256((64,64,3))

    def augment_data(self) -> bool:
        return False

    def get_data(self) -> np.ndarray:
        return self.data

    def get_train(self, model: BuiltDiffusionModel, mparams: DiffusionHyperParams) -> Train:
        return TrainDiffusion(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 10000] = DiffusionHyperParams(
            learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=10000,
            sample_interval=20,

            T=300,
            beta=0.008,

            notes="""
First take at diffusion. Lets see.
""",
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,  # gen_highest_f * ngf,
            name="pkmn_diffusion",
            img_shape=(64, 64, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: DiffusionHyperParams) -> FrameworkModel:
        return MyModel(self.params, mparams)


if __name__ == "__main__":
    MyExperiment().start()
