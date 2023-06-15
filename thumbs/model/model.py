from abc import ABC, abstractmethod
from thumbs.params import HyperParams, MutableHyperParams
from keras.optimizers import Adam


class Model(ABC):
    def __init__(self, params: HyperParams, mparams: MutableHyperParams) -> None:
        self.params = params
        self.mparams = mparams

    @abstractmethod
    def build_discriminator(self, img_shape):
        raise NotImplementedError()

    @abstractmethod
    def build_generator(self, z_dim):
        raise NotImplementedError()

    @abstractmethod
    def build_gan(self, generator, discriminator):
        raise NotImplementedError()

    def build(self):
        # Build and compile the Discriminator
        discriminator = self.build_discriminator(self.params.img_shape)
        discriminator.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.mparams.discriminator_learning_rate),
            metrics=["accuracy"],
        )

        # Build the Generator
        generator = self.build_generator(self.params.latent_dim)

        # Keep Discriminator’s parameters constant for Generator training
        discriminator.trainable = False

        # Build and compile GAN model with fixed Discriminator to train the Generator
        gan = self.build_gan(generator, discriminator)
        generator_optimizer = Adam(learning_rate=self.mparams.generator_learning_rate)

        return gan, discriminator, generator, generator_optimizer
