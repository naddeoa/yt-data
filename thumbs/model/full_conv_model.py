from thumbs.model.model import Model
from keras.models import Sequential
from keras.layers import (
    Dense,
    Reshape,
    Conv2DTranspose,
    Flatten,
    LeakyReLU
)
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Flatten,
    Reshape,
    LeakyReLU,
)
from keras.layers.convolutional import Conv2D, Conv2DTranspose


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

    def build_discriminator(self,img_shape):
        model = Sequential(name="discriminator")

        model.add(
            Conv2D(
                32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"
            )
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
