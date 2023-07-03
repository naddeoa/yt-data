import thumbs.config_logging  # must be first
import pandas as pd
import tensorflow as tf
import os
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_and_types
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import Model, BuiltModel

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
    Flatten,
    Reshape,
    ReLU,
    LeakyReLU,
    LayerNormalization,
    Embedding, 
    Multiply,
    Concatenate
    # BatchNormalizationV2,
)

# from keras.layers.normalization.batch_normalization_v1 import (
#     BatchNormalization,
# )
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from thumbs.train import Train, TrainMSE, TrainBCE, TrainBCESimilarity, TrainWassersteinGP


infinity = float("inf")

ngf = 128
ndf = 128

class TileLayer(tf.keras.layers.Layer):
  def __init__(self, tiles):
    super(TileLayer, self).__init__()
    self.tiles = tiles

  def build(self, input_shape):
    pass

  def call(self, inputs):
    return tf.tile(inputs, self.tiles)



class PokemonModel(Model):
    def __init__(self, params: HyperParams, mparams: MutableHyperParams, vocab: List[str]) -> None:
        super().__init__(params, mparams)
        # self.vocab = vocab
        self.num_classes = len(vocab) + 1 # 18 pokemon types plus an OOV token

    def build_generator(self, z_dim):
        noise_input = Input(shape=(z_dim,))
        # types_input = Input(shape=(None,), dtype=tf.string, ragged=True)
        types_input = Input(shape=(self.num_classes,))
        # lookup = StringLookup(output_mode='multi_hot', name='string_lookup_gen', vocabulary=self.vocab)

        # Preprocessing 
        # types = lookup(types_input)
        x = Concatenate(axis=-1)([noise_input, types_input]) # Might need axis=-1, but it looks like tf makes it work out

        # x = Reshape((1, 1, z_dim + self.num_classes))(model_input)
        x = Dense(8*8*ngf*4)(x)
        x = Reshape((8, 8, ngf*4))(x)
        # x = Conv2DTranspose(ngf*4, kernel_size=6, strides=6, padding='valid')(x)

        x = Conv2DTranspose(ngf*3, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(ngf*2, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(ngf, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding="same")(x)
        x = Activation("tanh")(x)

        model = tf.keras.Model([noise_input, types_input], x, name="generator")
        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        image_input = Input(shape=img_shape)

        types_input = Input(shape=(self.num_classes,))
        types_input_x = Reshape((1,1,self.num_classes))(types_input)
        types_input_x = TileLayer(( 1, 128, 128, 1 ))(types_input_x)

        model_input = Concatenate(axis=-1)([image_input , types_input_x])
        # TODO stop using bias
        # , use_bias=False
        x = Conv2D(ndf, kernel_size=5, strides=2, padding="same")(model_input)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(ndf*2, kernel_size=5, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(ndf*4, kernel_size=5, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(ndf*8, kernel_size=5, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1)(x)

        model = tf.keras.Model([image_input, types_input], x, name="discriminator")
        model.summary(line_length=200)
        return model


    def build_gan(self, generator, discriminator):
        # Can't really do this easily because they both take multiple inputs
        return None


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        data, types = get_pokemon_and_types(self.params.img_shape)
        self.vocab = types
        self.lookup = StringLookup(output_mode='multi_hot', name='string_lookup_gen', vocabulary=self.vocab)

        self.images = np.array([x[0] for x in data])
        self.labels = np.array([self.lookup(item[1]).numpy() for item in data])

        self.zoom_factor = .9


        # Can lookup the type info given a pokedex number
        df = pd.read_csv('/home/anthony/workspace/yt-data/data/pokemon/stats.csv')
        df = df.drop_duplicates(subset=['#'])
        self.df = df
        self.id_to_types = {}
        for index, row in df.iterrows():
            type1 = row['Type 1'].lower()
            type2 = row['Type 2']
            both_types = [type1]
            if type2 == type2:
                both_types.append(type2.lower())

            self.id_to_types[row['#']] = both_types

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.images, self.labels

    def get_random_labels(self, n: int):
        # get n random samples from self.labels
        return self.labels[np.random.randint(0, len(self.labels), n)]

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams, self.get_random_labels)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=32,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=5,
            discriminator_turns=10,
            generator_turns=1,
            checkpoint_interval=100,
            gradient_penalty_factor=20
        )

        return schedule

    def get_params(self) -> HyperParams:
        name = "pokemon_conditional"

        exp_dir = 'EXP_DIR'
        if exp_dir in os.environ:
            base_dir = os.environ['EXP_DIR']
        else:
            base_dir = '/mnt/e/experiments'

        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            weight_path=f"{base_dir}/{name}/weights",
            checkpoint_path=f"{base_dir}/{name}/checkpoints",
            prediction_path=f"{base_dir}/{name}/predictions",
            iteration_checkpoints_path=f"{base_dir}/{name}/iteration_checkpoints",
            loss_path=f"{base_dir}/{name}/loss",
            accuracy_path=f"{base_dir}/{name}/accuracy",
            iteration_path=f"{base_dir}/{name}/iteration",
            similarity_threshold=0.0,
            similarity_penalty=20,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return PokemonModel(self.params, mparams, self.vocab)


if __name__ == "__main__":
    PokemonExperiment().start()

