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
from thumbs.data import get_pokemon_and_pokedexno, normalize_image, unnormalize_image, get_pokemon_and_types
from thumbs.params import HyperParams, GanHyperParams, Sampler
from thumbs.model.model import GanModel, BuiltGANModel

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


kern_size = 3

ngf = 64
gen_highest_f = 8
ngl = 7
ngb = 4

ndf = 64
disc_highest_f = 8
ndl = 7
ndb = 4


class SubtractOneLayer(Layer):
    def call(self, inputs):
        return inputs - 1


class PokemonModel(GanModel):
    def __init__(self, params: HyperParams, mparams: GanHyperParams, types: List[str]) -> None:
        super().__init__(params, mparams)
        self.types = types
        self.embedding_dim = 5

    def _generator_core(self, z, type1_embedding, type2_embedding):
        type1_embedding = Flatten()(type1_embedding)
        type2_embedding = Flatten()(type2_embedding)

        x = Concatenate(name="g_concat")([z, type1_embedding, type2_embedding])
        x = Reshape((1, 1, -1))(x)

        for f in [int(f) for f in np.linspace(gen_highest_f, 1, ngb)]:
            if f == gen_highest_f:
                x = Conv2DTranspose(f * ngf, kernel_size=8, strides=1, padding="valid", use_bias=False, name=f"tconf{f}")(x)
            else:
                x = Conv2DTranspose(f * ngf, kernel_size=kern_size, strides=2, padding="same", use_bias=False, name=f"tconf{f}")(x)

            x = InstanceNormalization(name=f"norm{f}")(x)
            x = LeakyReLU(alpha=0.2, name=f"relu{f}")(x)

            # This was fixed in this model. It didn't have hidden layers on the generator's first layer
            for i in range(ngl):
                x = Conv2DTranspose(f * ngf, kernel_size=kern_size, strides=1, padding="same", use_bias=False, name=f"tconf{f}_{i}")(x)
                x = InstanceNormalization(name=f"norm{f}_{i}")(x)
                x = LeakyReLU(alpha=0.2, name=f"relu{f}_{i}")(x)

        x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh", name=f"tconf_final")(x)
        return x

    def build_generator_embedding_input(self, z_dim):
        # TODO not sure I care about interpolating between the type embeddings here
        pass
        # z = Input(shape=(z_dim,), name="z")
        # e = Input(shape=(self.embedding_dim,), name="pokedex_number_embedding")

        # x = self._generator_core(z_dim, z, e)
        # model = Model([z, e], x, name="generator")
        # return model

    def build_generator(self, z_dim):
        z = Input(shape=(z_dim,), name="z")
        type1 = Input(shape=(), name="type1", dtype=tf.string, ragged=True)
        type2 = Input(shape=(), name="type2", dtype=tf.string, ragged=True)

        sl = StringLookup(output_mode="int", name="type_lookup", vocabulary=self.types)

        e = Embedding(len(self.types) + 1, self.embedding_dim)

        type1_embedding = e(sl(type1))
        type2_embedding = e(sl(type2))

        x = self._generator_core(z, type1_embedding, type2_embedding)

        model = Model([z, type1, type2], x, name="generator")
        return model

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")

        type1 = Input(shape=(), name="type1", dtype=tf.string, ragged=True)
        type2 = Input(shape=(), name="type2", dtype=tf.string, ragged=True)

        e = Embedding(len(self.types) + 1, self.embedding_dim)
        sl = StringLookup(output_mode="int", name="type_lookup", vocabulary=self.types)

        type1_embedding = e(sl(type1))
        type2_embedding = e(sl(type2))

        e = Concatenate(name="d_concat1")([type1_embedding, type2_embedding])
        e = Dense(img_shape[0] * img_shape[1])(e)
        e = Reshape((img_shape[0], img_shape[1], 1))(e)

        x = img_input
        x = DiffAugmentLayer()(x)
        x = Concatenate(name="d_concat2")([x, e])

        for i, f in enumerate([int(f) for f in np.linspace(1, disc_highest_f, ndb)]):
            x = SpectralNormalization(Conv2D(f * ndf, kernel_size=kern_size, strides=2, padding="same", name=f"conv_{f}"))(x)
            if i != 0:
                x = InstanceNormalization(name=f"norm{f}")(x)
            x = LeakyReLU(alpha=0.2, name=f"rely_{f}")(x)

            for i in range(ndl):
                x = SpectralNormalization(Conv2D(f * ndf, kernel_size=kern_size, strides=1, padding="same", name=f"conv_{f}_{i}"))(x)
                x = InstanceNormalization(name=f"norm{f}_{i}")(x)
                x = LeakyReLU(alpha=0.2, name=f"rely_{f}_{i}")(x)

        x = SpectralNormalization(Conv2D(1, kernel_size=8, strides=1, padding="valid"))(x)
        x = Flatten()(x)

        model = Model([img_input, type1, type2], x, name="discriminator")
        return model


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.augment_zooms = False
        self.augment_rotations = False
        self.augment_flips = False  # Everything came out looking symmetrical
        self.data, self.types = get_pokemon_and_types(self.params.img_shape)
        self.string_lookup = StringLookup(output_mode="int", name="type_lookup", vocabulary=self.types)
        # self.int_types = self.string_lookup(self.types).numpy()
        self.nptypes = np.array(self.types)

    def augment_data(self) -> bool:
        return False

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # t0 = self.data[0]
        # t1 = np.array([self.string_lookup(t) for t in self.data[1]])
        # t2 = np.array([self.string_lookup(t) for t in self.data[2]])
        # return (t0, t1, t2)

        return self.data

    def get_random_labels(self, n: int):
        hardcoded_types = np.array(
            [
                ["fire", "[UNK]"],
                ["water", "[UNK]"],
                ["grass", "[UNK]"],
                ["electric", "[UNK]"],
                ["normal", "[UNK]"],
                ["fighting", "[UNK]"],
                ["flying", "[UNK]"],
                ["poison", "[UNK]"],
                ["ground", "[UNK]"],
                ["rock", "[UNK]"],
                ["bug", "[UNK]"],
                ["ghost", "[UNK]"],
                ["steel", "[UNK]"],
            ]
        )

        if n <= len(hardcoded_types):
            return (hardcoded_types[:n, 0], hardcoded_types[:n, 1])

        remaining = n - len(hardcoded_types)
        random_types = self.nptypes[np.random.randint(0, len(self.types), size=(remaining, 2))]

        all = np.concatenate([hardcoded_types, random_types])
        return (all[:, 0], all[:, 1])

    def get_train(self, model: BuiltGANModel, mparams: GanHyperParams) -> Train:
        # return TrainBCEPatch(model, self.params, mparams)
        # return TrainBCE(model, self.params, mparams)
        # return TrainHinge(model, self.params, mparams)
        return TrainWassersteinGP(model, self.params, mparams, label_getter=self.get_random_labels)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = GanHyperParams(
            gen_learning_rate=0.00001,
            dis_learning_rate=0.00002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=20,
            generator_turns=1,
            discriminator_turns=2,
            gradient_penalty_factor=10.0,
            gen_weight_decay=0,
            dis_weight_decay=0,
            notes="""
The overfit plan wasn't really goign to yield new pokemon via interpolation but I learned a lot. Switching over to do an embedding conditioning on the types. I might have to refactor two have two embedding layers, one for the primary and one for the secondary, if this looks weird. I'm using a single layer to lookup embeddings for both now.

Going batck to 128 batch sizes too. Larger ones weren't any different.
""",
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=50,  # gen_highest_f * ngf,
            name="pkmn_cond_embedding_50dim_5embded_64x8_128-batch_d2-g1_2-hidden_00002lr_3kern",
            img_shape=(128, 128, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: GanHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams, self.types)


if __name__ == "__main__":
    PokemonExperiment().start()
