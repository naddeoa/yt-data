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
from thumbs.data import get_pokemon_and_pokedexno, normalize_image, unnormalize_image, get_pokemon_and_types
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
    Dropout,
    Input,
    BatchNormalization,
    Dense,
    GaussianNoise,
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
                name=f"tconv{f}_{i}",
            )
        )
        seq.add(InstanceNormalization(name=f"norm{f}_{i}"))
        seq.add(LeakyReLU(alpha=0.2, name=f"relu{f}_{i}"))

        if strides == 1:
            seq.add(
                Conv2DTranspose(
                    f * ngf,
                    kernel_size=kern_size,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    name=f"tconv2{f}_{i}",
                )
            )
            seq.add(InstanceNormalization(name=f"norm2{f}_{i}"))

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add(name=f"add{f}_{i}")([x, input_x])
            x = LeakyReLU(alpha=0.2, name=f"relu2{f}_{i}")(x)

        return x

    def _generator_core(self, z, type1_embedding, type2_embedding):
        type1_embedding = Flatten()(type1_embedding)
        type2_embedding = Flatten()(type2_embedding)

        x = Concatenate(name="g_concat")([z, type1_embedding, type2_embedding])
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
        type1 = Input(shape=(), name="type1", dtype=tf.string, ragged=True)
        type2 = Input(shape=(), name="type2", dtype=tf.string, ragged=True)

        sl = StringLookup(output_mode="int", name="type_lookup", vocabulary=self.types)

        type1_embedding = Embedding(len(self.types) + 1, self.embedding_dim * 2)(sl(type1))
        type2_embedding = Embedding(len(self.types) + 1, self.embedding_dim)(sl(type2))

        x = self._generator_core(z, type1_embedding, type2_embedding)
        x = DiffAugmentLayer()(x)

        model = Model([z, type1, type2], x, name="generator")
        total = sum([1 for layer in model.layers if isinstance(layer, Conv2DTranspose)])
        print(f'>> Total generator convolutions: {total}')
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
                        kernel_size=kern_size,
                        strides=1,
                        padding="same",
                        name=f"conv2_{f}_{i}",
                    )
                )
            )
            seq.add(InstanceNormalization(name=f"norm2{f}_{i}"))

        x = seq(x)
        if input_x.shape == x.shape:
            x = Add(name=f"add{f}_{i}")([x, input_x])
            x = LeakyReLU(alpha=0.2, name=f"relu2{f}_{i}")(x)

        return x

    def build_discriminator(self, img_shape):
        img_input = Input(shape=img_shape, name="img_input")

        type1 = Input(shape=(), name="type1", dtype=tf.string, ragged=True)
        type2 = Input(shape=(), name="type2", dtype=tf.string, ragged=True)

        sl = StringLookup(output_mode="int", name="type_lookup", vocabulary=self.types)

        type1_embedding = Embedding(len(self.types) + 1, self.embedding_dim * 2)(sl(type1))
        type2_embedding = Embedding(len(self.types) + 1, self.embedding_dim)(sl(type2))

        types = Concatenate(name="d_concat1")([type1_embedding, type2_embedding])
        types = Dense(img_shape[0] * img_shape[1])(types)
        types = Reshape((img_shape[0], img_shape[1], 1))(types)

        x = img_input
        x = DiffAugmentLayer()(x)
        x = Concatenate(name="d_concat2")([x, types])

        for i, f in enumerate([int(f) for f in np.linspace(1, disc_highest_f, ndb)]):
            x = self._discriminator_block(f, x, i="", strides=2, normalize=False)

            for j in range(ndl):
                x = self._discriminator_block(f, x, i=j, strides=1)

        x = SpectralNormalization(Conv2D(1, kernel_size=8, strides=1, padding="valid"))(x)
        x = Flatten()(x)

        model = Model([img_input, type1, type2], x, name="discriminator")
        total = sum([1 for layer in model.layers if isinstance(layer, Conv2D)])
        print(f'>> Total discriminator convolutions: {total}')
        return model


class PokemonExperiment(GanExperiment):
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
            batch_size=64,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=5,
            generator_turns=1,
            g_clipnorm=.001,
            d_clipnorm=.001,
            discriminator_turns=2,
            gradient_penalty_factor=10.0,
            gen_weight_decay=0,
            dis_weight_decay=0,
            notes="""
The last one didn't really change in a few thousand epochs. Making this one a lot deeper with a proper resnet block.
""",
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,  # gen_highest_f * ngf,
            name="pkmn_cond_embedding3_100dim_5embed-sep_4x10_64-batch_7L_00002lr_3kern",
            img_shape=(128, 128, 3),
            sampler=Sampler.NORMAL,
        )

    def get_model(self, mparams: GanHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams, self.types)


if __name__ == "__main__":
    PokemonExperiment().start()

