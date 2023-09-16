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
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras import Sequential
from keras.layers import (
    Activation,
    Add,
    StringLookup,
    Conv2DTranspose,
    Conv2D,
    Input,
    LayerNormalization,
    BatchNormalization,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    ReLU,
    LeakyReLU,
    GroupNormalization,
    LayerNormalization,
    MultiHeadAttention,
    Embedding,
    Multiply,
    Concatenate,
    # BatchNormalizationV2,
)
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from thumbs.self_attention import SelfAttention

from thumbs.train import Train, TrainDiffusion
from tensorflow.keras.layers import Layer

tf.keras.layers.Dropout  # TODO is this different than keras.layers.Dropout? Is it still broken?


kern_size = 3

ngf = 64
gen_highest_f = 4
up_blocks = [1, 1, 1]

nbn = 4

ndf = 64
disc_highest_f = 4
down_blocks = [1, 1, 1]


class MyModel(DiffusionModel):
    def __init__(self, params: HyperParams, mparams: DiffusionHyperParams) -> None:
        super().__init__(params, mparams)
        self.embed_dim = 256

    def down_resnet(
        self,
        f: int,
        x,
        strides: int,
        kernel_size: int = kern_size,
        normalize=True,
    ):
        channels = f * ndf
        input_x = x
        seq = Sequential()
        seq.add(
            Conv2D(
                channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
            )
        )

        if normalize:
            seq.add(GroupNormalization(1))
        # seq.add(ReLU())
        seq.add(tf.keras.layers.Activation("gelu"))

        if strides == 1:
            seq.add(
                Conv2D(
                    channels,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                )
            )
            seq.add(GroupNormalization(1))

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add()([x, input_x])
            # x = ReLU()(x)
            x = tf.keras.layers.Activation("gelu")(x)

        return x

    def up_resnet(
        self,
        f: int,
        x,
        strides: int,
        kernel_size: int,
    ):
        channels = f * ngf
        input_x = x

        seq = Sequential()
        seq.add(
            Conv2DTranspose(
                channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid" if kernel_size > kern_size else "same",
                use_bias=True,
            )
        )
        seq.add(GroupNormalization(1))
        # seq.add(ReLU())
        seq.add(tf.keras.layers.Activation("gelu"))

        if strides == 1 and kernel_size == kern_size:
            seq.add(
                Conv2DTranspose(
                    channels,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=True,
                )
            )
            seq.add(GroupNormalization(1))

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add()([x, input_x])
            # x = ReLU()(x)
            x = tf.keras.layers.Activation("gelu")(x)
        return x

    def concat_embedding(self, x, embedding, name: str):
        _, H, W, C = x.shape

        s = Sequential([Dense(H * W, use_bias=True), Reshape((H, W, 1))], name=name)
        _x = s(embedding)
        return Concatenate(name=f"embed_{name}")([x, _x])

    def pos_encoding(self, t, channels=256):
        inv_freq = 1.0 / (10000 ** (tf.range(0, channels, 2, dtype=tf.float32) / tf.cast(channels, tf.float32)))
        t_repeated = tf.repeat(t, repeats=[channels // 2], axis=-1)
        pos_enc_a = tf.math.sin(tf.multiply(t_repeated, inv_freq))
        pos_enc_b = tf.math.cos(tf.multiply(t_repeated, inv_freq))
        pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc

    def positional_encoding_layer(self, x, t, name: str):
        _, H, W, C = x.shape

        s = Sequential(
            [
                tf.keras.layers.Activation("swish", input_shape=(self.embed_dim,)),
                Dense(C),
            ],
            name=name,
        )  # Going to be  (batch_size, 256)

        _x = s(t)
        # Need to turn it into a (batch_size, H, W, 256) tensor so it can be added
        _x = Reshape((1, 1, C))(_x)
        _x = tf.tile(_x, [1, H, W, 1])
        return Add(name=f"embed_{name}")([x, _x])

    def get_model(self) -> Model:
        img_input = Input(shape=self.params.img_shape, name="image")

        t_input = Input(shape=(1,), name="t")
        t_pos = self.pos_encoding(t_input)

        x = img_input

        seed = Sequential(
            [
                Conv2D(img_input.shape[1], kernel_size=3, strides=1, padding="same", use_bias=True),
                GroupNormalization(1),
                tf.keras.layers.Activation("gelu"),
            ],
            name="initial",
        )
        x = seed(x)

        # Down stack
        downs = [x]
        for i, f in enumerate(np.linspace(1, disc_highest_f, len(down_blocks), dtype=int)):
            x = self.down_resnet(f, x, strides=2, normalize=False)

            for _ in range(down_blocks[i]):
                x = self.down_resnet(f, x, strides=1)

            x = SelfAttention(f * ndf)(x)
            x = self.positional_encoding_layer(x, t_pos, name=f"pos_down{i}")

            if i < len(down_blocks) - 1:
                downs.append(x)

        assert len(downs) == 3
        downs.reverse()

        # bottleneck convolutions
        for i in range(nbn):
            x = self.down_resnet(f, x, strides=1)
            x = SelfAttention(f * ndf)(x)

        # Up stack
        for i, f in enumerate(np.linspace(gen_highest_f, 1, len(up_blocks), dtype=int)):
            x = self.up_resnet(f, x, strides=2, kernel_size=kern_size)
            x = Concatenate(name=f"resnet_concat_{i}")([x, downs[i]])  # Unet concat with the down satck variant

            for _ in range(up_blocks[i]):
                x = self.up_resnet(f, x, strides=1, kernel_size=kern_size)

            if i < len(up_blocks) - 1:
                # Skipping this last one saves on a massive amount of memory
                x = SelfAttention(f * ngf)(x)

            x = self.positional_encoding_layer(x, t_pos, name=f"pos_up{i}")  # add positional information back in

        output = Conv2D(3, kernel_size=3, strides=1, padding="same")(x)
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
        schedule[0, 400] = DiffusionHyperParams(
            learning_rate=0.0002,
            batch_size=64,
            iterations=400,
            sample_interval=10,
            model_save_interval=1,
            checkpoint_interval=10,
            T=1000,
            beta_start=0.0001,
            beta_end=0.04,
            beta_schedule_type="easein",
            loss_fn=MeanAbsoluteError(),
        )

        schedule[401, 800] = DiffusionHyperParams(
            learning_rate=0.00002,
            batch_size=64,
            iterations=800,
            sample_interval=10,
            model_save_interval=1,
            checkpoint_interval=10,
            T=1000,
            beta_start=0.0001,
            beta_end=0.04,
            beta_schedule_type="easein",
            loss_fn=MeanAbsoluteError(),
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,  # gen_highest_f * ngf,
            name="wow_diffusion_attention",
            img_shape=(64, 64, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: DiffusionHyperParams) -> FrameworkModel:
        return MyModel(self.params, mparams)


if __name__ == "__main__":
    MyExperiment().start()