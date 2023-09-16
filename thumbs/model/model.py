from abc import ABC, abstractmethod
import tensorflow as tf
from thumbs.params import HyperParams, GanHyperParams, GanHyperParams, DiffusionHyperParams, MutableHyperParams
from keras.optimizers import Adam, AdamW
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Generic, cast
import tensorflow as tf


@dataclass
class BuiltGANModel:
    discriminator: tf.keras.Model
    discriminator_optimizer: tf.keras.Model
    generator: tf.keras.Model
    generator_optimizer: tf.keras.optimizers.Optimizer


T = TypeVar("T")
MParams = TypeVar("MParams", bound=MutableHyperParams)


class FrameworkModel(ABC, Generic[T, MParams]):
    def __init__(self, params: HyperParams, mparams: MParams) -> None:
        self.params = params
        self.mparams = mparams

    @abstractmethod
    def build(self) -> T:
        raise NotImplementedError()


@dataclass
class BuiltDiffusionModel:
    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer


class DiffusionModel(FrameworkModel[BuiltDiffusionModel, DiffusionHyperParams]):
    def __init__(self, params: HyperParams, mparams: DiffusionHyperParams) -> None:
        self.params = params
        self.mparams = mparams

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()

    def build(self) -> BuiltDiffusionModel:
        model = self.get_model()
        model.summary(line_length=200)

        # Create a high level diagram so you get the gist
        tf.keras.utils.plot_model(model, to_file=self.params.model_diagram_path, show_shapes=True, dpi=64, expand_nested=False)

        # And a verbose one for debugging
        file_name_verbose = self.params.model_diagram_path.replace(".jpg", "_verbose.jpg")
        tf.keras.utils.plot_model(model, to_file=file_name_verbose, show_shapes=True, dpi=64, expand_nested=True, show_trainable=True)

        return BuiltDiffusionModel(
            model=model,
            optimizer=Adam(
                weight_decay=self.mparams.weight_decay,
                learning_rate=self.mparams.learning_rate,
                beta_1=self.mparams.adam_b1,
                beta_2=self.mparams.adam_b2,
                global_clipnorm=self.mparams.clipnorm,
            ),
        )


class GanModel(FrameworkModel[BuiltGANModel, GanHyperParams]):
    def __init__(self, params: HyperParams, mparams: GanHyperParams) -> None:
        self.params = params
        self.mparams = mparams

    @abstractmethod
    def build_discriminator(self, img_shape):
        raise NotImplementedError()

    @abstractmethod
    def build_generator(self, z_dim):
        raise NotImplementedError()

    def build(self) -> BuiltGANModel:
        discriminator = self.build_discriminator(self.params.img_shape)
        discriminator_optimizer = Adam(
            weight_decay=self.mparams.dis_weight_decay,
            learning_rate=self.mparams.dis_learning_rate,
            beta_1=self.mparams.adam_b1,
            beta_2=self.mparams.adam_b2,
            global_clipnorm=self.mparams.d_clipnorm,
        )

        generator = self.build_generator(self.params.latent_dim)
        generator_optimizer = Adam(
            weight_decay=self.mparams.gen_weight_decay,
            learning_rate=self.mparams.gen_learning_rate,
            beta_1=self.mparams.adam_b1,
            beta_2=self.mparams.adam_b2,
            global_clipnorm=self.mparams.g_clipnorm,
        )

        discriminator.summary(line_length=200)
        tf.keras.utils.plot_model(
            discriminator, to_file=self.params.dis_diagram_path, show_shapes=True, dpi=64, expand_nested=True, show_trainable=True
        )
        generator.summary(line_length=200)
        tf.keras.utils.plot_model(
            generator, to_file=self.params.gen_diagram_path, show_shapes=True, dpi=64, expand_nested=True, show_trainable=True
        )

        return BuiltGANModel(
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            generator_optimizer=generator_optimizer,
        )
