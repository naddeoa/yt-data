from abc import ABC, abstractmethod
import tensorflow as tf
from thumbs.params import HyperParams, MutableHyperParams
from keras.optimizers import Adam
from keras.optimizers.adamw import AdamW
from dataclasses import dataclass
from typing import Optional
import tensorflow as tf


@dataclass
class BuiltModel:
    discriminator: tf.keras.Model
    discriminator_optimizer: tf.keras.Model
    generator: tf.keras.Model
    generator_optimizer: tf.keras.optimizers.Optimizer


class GanModel(ABC):
    def __init__(self, params: HyperParams, mparams: MutableHyperParams) -> None:
        self.params = params
        self.mparams = mparams

    @abstractmethod
    def build_discriminator(self, img_shape):
        raise NotImplementedError()

    @abstractmethod
    def build_generator(self, z_dim):
        raise NotImplementedError()

    def build(self) -> BuiltModel:
        discriminator = self.build_discriminator(self.params.img_shape)
        discriminator_optimizer = AdamW(
            learning_rate=self.mparams.dis_learning_rate, beta_1=self.mparams.adam_b1, global_clipnorm=self.mparams.d_clipnorm
        )

        generator = self.build_generator(self.params.latent_dim)
        generator_optimizer = AdamW(
            learning_rate=self.mparams.gen_learning_rate, beta_1=self.mparams.adam_b1, global_clipnorm=self.mparams.g_clipnorm
        )

        discriminator.summary(line_length=200)
        tf.keras.utils.plot_model(discriminator, to_file=self.params.dis_diagram_path, show_shapes=True, dpi=64)
        generator.summary(line_length=200)
        tf.keras.utils.plot_model(generator, to_file=self.params.gen_diagram_path, show_shapes=True, dpi=64)

        return BuiltModel(
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            generator_optimizer=generator_optimizer,
        )
