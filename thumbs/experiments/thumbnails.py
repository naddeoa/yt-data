import thumbs.config_logging  # must be first
from typing import List, Tuple, Iterator
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.data import get_yt_data
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import Model, BuiltModel

from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU
from keras.layers import (
    Activation,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    LeakyReLU,
    LayerNormalization,
    BatchNormalization,
)

# from keras.layers.normalization.batch_normalization_v1 import (
#     BatchNormalization,
# )
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from thumbs.train import Train, TrainMSE, TrainBCE, TrainBCESimilarity


infinity = float("inf")


class ThumbnailModel(Model):
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

        # model.add(Conv2DTranspose(512, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Conv2DTranspose(256, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding="same"))
        model.add(Activation("tanh"))

        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        model = Sequential(name="discriminator")

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
        model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalizationV1())
        # model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator):
        model = Sequential([generator, discriminator])
        return model


class ThumbnailExperiment(Experiment):
    def get_data(self) -> np.ndarray:
        return get_yt_data(self.params.img_shape)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainBCE(model, self.params, mparams)

    def augment_data(self) -> bool:
        return False

    def get_mutable_params(self) -> RangeDict:
        """
        TODO
        - Inspect the intermediate layer outputs to see what they learned
        - Try an architecture that gets bigger than 128x128 but then transposed conv back down to that, auto encoder style.
        """
        schedule = RangeDict()
        schedule[0, infinity] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=200000,
            sample_interval=10,
            checkpoint_interval=20,
            discriminator_turns=1,
            generator_turns=1,
        )

        return schedule

    def get_params(self) -> HyperParams:
        name = "yt_thumbnails_shallow_small_batch"
        return HyperParams(
            latent_dim=32,
            img_shape=(128, 128, 3),
            weight_path=f"/mnt/e/experiments/{name}/weights",
            prediction_path=f"/mnt/e/experiments/{name}/predictions",
            iteration_path=f"/mnt/e/experiments/{name}/iteration",
            checkpoint_path=f"/mnt/e/experiments/{name}/checkpoints",
            similarity_threshold=0.0,
            similarity_penalty=20,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return ThumbnailModel(self.params, mparams)


if __name__ == "__main__":
    ThumbnailExperiment().start()

