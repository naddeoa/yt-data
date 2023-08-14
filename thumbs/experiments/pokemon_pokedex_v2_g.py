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
from thumbs.data import get_pokemon_and_pokedexno, normalize_image, unnormalize_image
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
from tensorflow.keras.layers import Layer

tf.keras.layers.Dropout  # TODO is this different than keras.layers.Dropout? Is it still broken?


ngf = 64
gen_highest_f = 16
ngl = 0
ngb = 4

ndf = 64
disc_highest_f = 16
ndl = 0
ndb = 4


class SubtractOneLayer(Layer):
    def call(self, inputs):
        return inputs - 1


class PokemonModel(GanModel):
    def __init__(self, params: HyperParams, mparams: MutableHyperParams, highest_pokedex_no: int) -> None:
        super().__init__(params, mparams)
        self.highest_pokedex_no = highest_pokedex_no
        self.embedding_dim = 100

    def _generator_core(self, z_dim, z, embedding):
        e = Flatten()(embedding)
        # x = Concatenate()([z, e])
        # x = Reshape((1, 1, z_dim + self.embedding_dim))(e)
        x = Reshape((1, 1, self.embedding_dim))(e)

        for f in np.linspace(gen_highest_f, 1, ngb):
            if f == gen_highest_f:
                x = Conv2DTranspose(int(f) * ngf, kernel_size=8, strides=1, padding="valid", use_bias=False, name=f"tconf{f}")(x)
                x = InstanceNormalization(name=f"norm{f}")(x)
                x = LeakyReLU(alpha=0.2, name=f"relu{f}")(x)
            else:
                x = Conv2DTranspose(int(f) * ngf, kernel_size=4, strides=2, padding="same", use_bias=False, name=f"tconf{f}")(x)
                x = InstanceNormalization(name=f"norm{f}")(x)
                x = LeakyReLU(alpha=0.2, name=f"relu{f}")(x)

                x = Conv2DTranspose(int(f) * ngf, kernel_size=4, strides=1, padding="same", use_bias=False, name=f"tconf{f}_1")(x)
                x = InstanceNormalization(name=f"norm{f}_1")(x)
                x = LeakyReLU(alpha=0.2, name=f"relu{f}_1")(x)


        x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh", name=f"tconf_final")(x)
        return x

    def build_generator_embedding_input(self, z_dim):
        z = Input(shape=(z_dim,), name="z")
        e = Input(shape=(self.embedding_dim,), name="pokedex_number_embedding")

        x = self._generator_core(z_dim, z, e)
        model = Model([z, e], x, name="generator")
        return model

    def build_generator(self, z_dim):
        z = Input(shape=(z_dim,), name="z")
        pokedex_number = Input(shape=(1,), name="pokedex_number")

        e = SubtractOneLayer()(pokedex_number)  # EZ way to make the embedding index 0-based
        e = Embedding(self.highest_pokedex_no, self.embedding_dim)(e)

        x = self._generator_core(z_dim, z, e)

        model = Model([z, pokedex_number], x, name="generator")
        return model

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")
        pokedex_number = Input(shape=(1,), name="pokedex_number")

        e = pokedex_number = SubtractOneLayer()(pokedex_number)  # EZ way to make the embedding index 0-based
        e = Embedding(self.highest_pokedex_no, self.embedding_dim)(e)
        e = Dense(img_shape[0] * img_shape[1])(e)
        e = Reshape((img_shape[0], img_shape[1], 1))(e)

        x = img_input
        # x = DiffAugmentLayer()(x)
        x = Concatenate()([x, e])

        for i, f in enumerate(np.linspace(1, disc_highest_f, ndb)):
            x = SpectralNormalization(Conv2D(int(f) * ndf, kernel_size=4, strides=2, padding="same", name=f'conv_{f}'))(x)
            if i != 0:
                x = InstanceNormalization(name=f"norm{f}")(x)
            x = LeakyReLU(alpha=0.2, name=f'rely_{f}')(x)

            x = SpectralNormalization(Conv2D(int(f) * ndf, kernel_size=4, strides=1, padding="same", name=f'conv_{f}_1'))(x)
            x = InstanceNormalization(name=f"norm{f}_1")(x)
            x = LeakyReLU(alpha=0.2, name=f'rely_{f}_1')(x)

        x = SpectralNormalization(Conv2D(1, kernel_size=8, strides=1, padding="valid"))(x)
        x = Flatten()(x)

        model = Model([img_input, pokedex_number], x, name="discriminator")
        return model


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.augment_zooms = False
        self.augment_rotations = False
        self.augment_flips = False  # Everything came out looking symmetrical
        self.data = get_pokemon_and_pokedexno(self.params.img_shape)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data

    def get_random_labels(self, n: int):
        pokedex_numbers = self.data[1]
        # Just assume n is 36 since thats the default during sampling
        random_pokemon = np.random.choice(pokedex_numbers, size=(n - 24, 1))
        hardcoded_pokemon = np.array(
            [
                [101],
                [4],
                [6],
                [67],
                [362],
                [23],
                [3],
                [24],
                [94],
                [91],
                [171],
                [719],
                [502],
                [711],
                [126],
                [610],
                [265],
                [422],
                [132],
                [72],
                [452],
                [312],
                [438],
                [550],
            ]
        )
        all = np.concatenate([hardcoded_pokemon, random_pokemon])
        return (all,)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams, label_getter=self.get_random_labels)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 40000] = MutableHyperParams(
            gen_learning_rate=0.00004,
            dis_learning_rate=0.00005,
            batch_size=128,
            adam_b1=0.5,
            iterations=40000,
            sample_interval=20,
            generator_turns=1,
            discriminator_turns=2,
            gradient_penalty_factor=10.0,
            gen_weight_decay=0,
            dis_weight_decay=0,
        )


        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=1,  # gen_highest_f * ngf,
            name="pkmn-pokedex_no-noise_100embded_8x_128-batch_2:1_8deep_00005lr",
            img_shape=(128, 128, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        print(f"Using {max(self.data[1])} as the embedding input size")
        return PokemonModel(self.params, mparams, max(self.data[1]))


if __name__ == "__main__":
    PokemonExperiment().start()
