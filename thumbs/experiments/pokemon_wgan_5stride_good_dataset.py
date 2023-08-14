import thumbs.config_logging  # must be first
import tensorflow as tf
import os
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data256
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import GanModel, BuiltModel

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

from thumbs.train import Train, TrainWassersteinGP


infinity = float("inf")


class PokemonModel(GanModel):
    def build_generator(self, z_dim):
        model = Sequential(name="generator")

        model.add(Dense(1024 * 8 * 8, input_dim=z_dim))
        model.add(Reshape((8, 8, 1024)))

        model.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding="same"))
        model.add(Activation("tanh"))

        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        model = Sequential(name="discriminator")
        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
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
        return get_pokemon_data256(self.params.img_shape)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 5600] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=32,
            adam_b1=0.5,
            iterations=5600,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=400,
        )

        schedule[5601, 6400] = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0002,
            batch_size=32,
            adam_b1=0.5,
            iterations=6400,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=400,
        )

        schedule[6401, 20500] = MutableHyperParams(
            gen_learning_rate=0.00002,
            dis_learning_rate=0.0002,
            batch_size=32,
            adam_b1=0.5,
            iterations=20500,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=400,
        )

        # Freezing disc to let the generator catch up
        schedule[20501, 100000] = MutableHyperParams(
            gen_learning_rate=0.000006,
            dis_learning_rate=0.00002,
            batch_size=32,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=400,
        )

        return schedule

    def custom_augmentation(
        self, image: tf.Tensor, labels: Optional[tf.Tensor] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Optional[tf.Tensor]]]:
        """
        No zoom for this dataset since the pokemon are much closer to the edge of the frame
        """
        if not self.augment_data():
            return image

        image = tf.image.random_flip_left_right(image)
        image = tf.keras.layers.experimental.preprocessing.RandomRotation(0.05)(image)

        return image

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            name="pokemon_wgan_5stride_good_dataset_repro",
            similarity_threshold=0.0,
            similarity_penalty=20,
        )

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
