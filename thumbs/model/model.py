from abc import ABC, abstractmethod
import tensorflow as tf
from thumbs.params import HyperParams, MutableHyperParams
from keras.optimizers import Adam
from dataclasses import dataclass
from typing import Optional


@dataclass
class BuiltModel:
    gan: tf.keras.Model
    discriminator: tf.keras.Model
    generator: tf.keras.Model
    generator_optimizer: tf.keras.optimizers.Optimizer


class Model(ABC):
    def __init__(self, params: HyperParams, mparams: MutableHyperParams, loss="binary_crossentropy") -> None:
        self.params = params
        self.mparams = mparams
        self.loss = loss

    @abstractmethod
    def build_discriminator(self, img_shape):
        raise NotImplementedError()

    @abstractmethod
    def build_generator(self, z_dim):
        raise NotImplementedError()

    @abstractmethod
    def build_gan(self, generator, discriminator):
        raise NotImplementedError()

    def build(self) -> BuiltModel:
        discriminator = self.build_discriminator(self.params.img_shape)
        discriminator.compile(
            loss=self.loss,
            optimizer=Adam(learning_rate=self.mparams.dis_learning_rate, beta_1=self.mparams.adam_b1),
            metrics=["accuracy"],
        )

        generator = self.build_generator(self.params.latent_dim)

        discriminator.trainable = False
        gan = self.build_gan(generator, discriminator)
        generator_optimizer = Adam(learning_rate=self.mparams.gen_learning_rate, beta_1=self.mparams.adam_b1)
        gan.compile(loss=self.loss, optimizer=generator_optimizer)

        return BuiltModel(gan, discriminator, generator, generator_optimizer)
