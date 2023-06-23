import thumbs.config_logging  # must be first
import os
from typing import List, Tuple, Iterator
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import Model, BuiltModel

from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    LeakyReLU,
    LayerNormalization,
    # BatchNormalizationV2,
)

# from keras.layers.normalization.batch_normalization_v1 import (
#     BatchNormalization,
# )
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from thumbs.train import Train, TrainMSE, TrainBCE, TrainBCESimilarity, TrainWassersteinGP


infinity = float("inf")


class PokemonModel(Model):
    def build_generator(self, z_dim):
        model = Sequential(name="generator_2")

        model.add(Dense(2048* 4 * 4, input_dim=z_dim))
        model.add(Reshape((4, 4, 2048)))

        model.add(Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding="same"))
        model.add(Activation("tanh"))

        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        model = Sequential(name="discriminator")

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
        # model.add(LayerNormalization())
        # model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        # model.add(LayerNormalization())
        # model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        # model.add(LayerNormalization())
        # model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        # model.add(LayerNormalization())
        # model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(1024, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1))

        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator):
        model = Sequential([generator, discriminator])
        return model


class PokemonExperiment(Experiment):
    def get_data(self) -> np.ndarray:
        return get_pokemon_data(self.params.img_shape)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 4700] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=4700,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=200,
        )

        schedule[4701, 100000] = MutableHyperParams(
            gen_learning_rate=0.00002,
            dis_learning_rate=0.00002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=200,
        )

        return schedule

    def get_params(self) -> HyperParams:
        name = "pokemon_wgan_3"

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
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()

