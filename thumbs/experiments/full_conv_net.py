import thumbs.config_logging  # must be first
from thumbs.train import Train, TrainBCE
from typing import List
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
    BatchNormalization,
    Dense,
    Flatten,
    Reshape,
    LeakyReLU,
)
from keras.layers.convolutional import Conv2D, Conv2DTranspose


infinity = float("inf")


class FullCovModel(Model):
    def build_generator(self, z_dim):
        model = Sequential(name="generator_2")

        model.add(Reshape((1, 1, z_dim), input_shape=(z_dim,)))

        model.add(Conv2DTranspose(2048, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(2048, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(1024, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(1024, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(32, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding="same"))

        model.add(Activation("tanh"))

        model.summary()
        return model

    def build_discriminator(self, img_shape):
        model = Sequential(name="discriminator")

        model.add(
            Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
        )
        # model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(1024, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.01))

        # Output layer with sigmoid activation
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()
        return model

    def build_gan(self, generator, discriminator):
        model = Sequential([generator, discriminator])
        return model


class FullConvTrainingExperiment(Experiment):
    def get_data(self) -> np.ndarray:
        return get_yt_data(self.params.img_shape, min_views=500_000)

    def get_train(self, model: BuiltModel) -> Train:
        return TrainBCE(model, self.params)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        phase1 = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.00001,
            adam_b1=0.9,
            iterations=2000,
            sample_interval=100,
        )
        schedule[0, 2000] = phase1

        phase2 = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.0001,
            adam_b1=0.9,
            iterations=5300,
            sample_interval=100,
        )
        schedule[2001, 7300] = phase2

        phase3 = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.00005,
            adam_b1=0.9,
            iterations=2000,
            sample_interval=100,
        )
        schedule[7301, 9300] = phase3

        phase4 = MutableHyperParams(
            gen_learning_rate=0.0001,
            dis_learning_rate=0.000075,
            adam_b1=0.9,
            iterations=200000,
            sample_interval=100,
        )
        schedule[9301, infinity] = phase4

        # Can start to distinguish the beginning of objects around ~17k iterations
        # Text and teeth at 36400, mostly around 80-90% accuracy
        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=150,
            batch_size=128,
            img_shape=(128, 128, 3),
            weight_path="./experiments/full_conv/weights",
            prediction_path="./experiments/full_conv/predictions",
            iteration_path="./experiments/full_conv/iteration",
            similarity_threshold=0.0,
            similarity_penalty=10,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return FullCovModel(self.params, mparams)


if __name__ == "__main__":
    FullConvTrainingExperiment().start()
