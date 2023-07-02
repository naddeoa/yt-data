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
    discriminator_optimizer: tf.keras.Model
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
    def build_gan(self, generator, discriminator) -> Optional[tf.keras.Model]:
        raise NotImplementedError()

    def build(self) -> BuiltModel:
        discriminator = self.build_discriminator(self.params.img_shape)
        discriminator_optimizer = Adam(learning_rate=self.mparams.dis_learning_rate, beta_1=self.mparams.adam_b1)
        discriminator.compile(
            loss=self.loss,
            optimizer=discriminator_optimizer,
            metrics=["accuracy"],
        )

        generator = self.build_generator(self.params.latent_dim)

        # TODO figure out a way of managing this flag. For the wgan that doesn't use compile it ends up
        # tanking the discriminator in the custom training loop. Maybe its enough to set it back to True
        # after compiling the gan
        discriminator.trainable = False
        gan = self.build_gan(generator, discriminator)
        generator_optimizer = Adam(learning_rate=self.mparams.gen_learning_rate, beta_1=self.mparams.adam_b1)
        if gan is not None:
            gan.compile(loss=self.loss, optimizer=generator_optimizer)
        discriminator.trainable = True

        return BuiltModel(
            gan=gan,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            generator_optimizer=generator_optimizer,
        )
