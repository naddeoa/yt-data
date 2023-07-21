import thumbs.config_logging  # must be first
from keras.datasets import mnist
import os
from typing import List, Tuple, Iterator
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data
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

from thumbs.train import Train, TrainBCE, TrainWassersteinGP


infinity = float("inf")


class MnistModel(GanModel):
    def build_generator(self, z_dim):

        model = Sequential()

        # Reshape input into 7x7x256 tensor via a fully connected layer
        model.add(Dense(256 * 7 * 7, input_dim=z_dim))
        model.add(Reshape((7, 7, 256)))

        # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))

        # Batch normalization
        model.add(BatchNormalization())

        # Leaky ReLU activation
        model.add(LeakyReLU(alpha=0.01))

        # Transposed convolution layer, from 14x14x128 to 14x14x64 tensor
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

        # Batch normalization
        model.add(BatchNormalization())

        # Leaky ReLU activation
        model.add(LeakyReLU(alpha=0.01))

        # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

        # Output layer with tanh activation
        model.add(Activation('tanh'))

        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        model = Sequential()

        # Convolutional layer, from 28x28x1 into 14x14x32 tensor
        model.add(
            Conv2D(32,
                   kernel_size=3,
                   strides=2,
                   input_shape=img_shape,
                   padding='same'))

        # Leaky ReLU activation
        model.add(LeakyReLU(alpha=0.01))

        # Convolutional layer, from 14x14x32 into 7x7x64 tensor
        model.add(
            Conv2D(64,
                   kernel_size=3,
                   strides=2,
                   input_shape=img_shape,
                   padding='same'))

        # Batch normalization
        #model.add(BatchNormalizationV1())

        # Leaky ReLU activation
        model.add(LeakyReLU(alpha=0.01))

        # Convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
        model.add(
            Conv2D(128,
                   kernel_size=3,
                   strides=2,
                   input_shape=img_shape,
                   padding='same'))

        # Batch normalization
        #model.add(BatchNormalizationV1())

        # Leaky ReLU activation
        model.add(LeakyReLU(alpha=0.01))

        # Output layer with sigmoid activation
        model.add(Flatten())
        model.add(Dense(1))

        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator):
        model = Sequential([generator, discriminator])
        return model


class MnistExperiment(Experiment):
    def get_data(self) -> np.ndarray:
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale [0, 255] grayscale pixel values to [-1, 1]
        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)
        return X_train
        # return get_pokemon_data(self.params.img_shape)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = MutableHyperParams(
            gen_learning_rate=0.002,
            dis_learning_rate=0.002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=2,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=200,
        )

        return schedule

    def get_params(self) -> HyperParams:
        name = "mnist_wgan"

        exp_dir = 'EXP_DIR'
        if exp_dir in os.environ:
            base_dir = os.environ['EXP_DIR']
        else:
            base_dir = '/mnt/e/experiments'

        return HyperParams(
            latent_dim=100,
            img_shape=(28, 28, 1),
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

    def get_model(self, mparams: MutableHyperParams) -> GanModel:
        return MnistModel(self.params, mparams)


if __name__ == "__main__":
    MnistExperiment().start()

