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
up_blocks = [2, 2, 2]

ndf = 64
disc_highest_f = 8
down_blocks = [2, 2, 2]


class MyModel(DiffusionModel):
    def __init__(self, params: HyperParams, mparams: DiffusionHyperParams) -> None:
        super().__init__(params, mparams)
        self.embed_dim = int(mparams.T / 100)

    def down_resnet(
        self,
        f: int,
        x,
        strides: int,
        kernel_size: int = kern_size,
        normalize=True,
    ):
        input_x = x
        seq = Sequential()
        seq.add(
            # SpectralNormalization(
            Conv2D(
                f * ndf,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
            )
            # )
        )

        if normalize:
            seq.add(InstanceNormalization())
        seq.add(LeakyReLU(alpha=0.2))

        if strides == 1:
            seq.add(
                # SpectralNormalization(
                Conv2D(
                    f * ndf,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                )
                # )
            )
            seq.add(InstanceNormalization())

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add()([x, input_x])
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def up_resnet(
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
                use_bias=True,
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
                    use_bias=True,
                )
            )
            seq.add(InstanceNormalization())

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add()([x, input_x])
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def concat_embedding(self, x, embedding, name: str):
        _, H, W, C = x.shape

        s = Sequential([Dense(H * W, use_bias=True), Reshape((H, W, 1))], name=name)
        _x = s(embedding)
        return Concatenate(name=f'embed_{name}')([x, _x])

    def get_model(self) -> Model:
        img_input = Input(shape=self.params.img_shape, name="image")

        t_input = Input(shape=(1,), name="t")
        e_embedding = Embedding(self.mparams.T, self.embed_dim, name="e_embedding")(t_input)

        x = img_input

        seed = Sequential(
            [
                Conv2D(3, kernel_size=3, strides=1, padding="same", use_bias=True),
                InstanceNormalization(),
                LeakyReLU(alpha=0.2),
            ],
            name="seed",
        )
        x = seed(x)

        downs = []
        for i, f in enumerate(np.linspace(1, disc_highest_f, len(down_blocks), dtype=int)):
            x = self.down_resnet(f, x, strides=2, normalize=False)
            for _ in range(down_blocks[i]):
                x = self.down_resnet(f, x, strides=1)

            x = self.concat_embedding(x, e_embedding, name=f"embed_down{i}")
            downs.append(x)

        downs.reverse()
        for i, f in enumerate(np.linspace(gen_highest_f, 1, len(up_blocks), dtype=int)):
            x = Concatenate(name=f"resnet_concat_{i}")([x, downs[i]])
            x = self.up_resnet(f, x, strides=2, kernel_size=kern_size)
            for _ in range(up_blocks[i]):
                x = self.up_resnet(f, x, strides=1, kernel_size=kern_size)

            x = self.concat_embedding(x, e_embedding, name=f"embed_up{i}")

        output = Conv2D(3, kernel_size=3, strides=1, padding="same", use_bias=True, activation="tanh")(x)
        return Model([img_input, t_input], output, name="diffusion_model")


class MyExperiment(DiffusionExperiment):
    def __init__(self) -> None:
        super().__init__()
        # self.data = get_pokemon_data256((64,64,3))
        self.data = get_wow_icons_64()

    def augment_data(self) -> bool:
        return False

    def get_data(self) -> Union[np.ndarray, tf.data.Dataset]:
        return self.data

    def get_train(self, model: BuiltDiffusionModel, mparams: DiffusionHyperParams) -> Train:
        return TrainDiffusion(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = DiffusionHyperParams(
            learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=1,
            T=1000,
            beta=0.001,
            beta_schedule_type="linear"
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
