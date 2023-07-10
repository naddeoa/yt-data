import thumbs.config_logging  # must be first
import random
import cv2
import pandas as pd
import tensorflow as tf
import os
from typing import List, Tuple, Iterator, Optional, Union
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.loss import Loss
from thumbs.data import get_pokemon_data256, normalize_image, unnormalize_image
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import Model, BuiltModel

from tensorflow_addons.layers import InstanceNormalization
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU
from keras.layers import (
    Activation,
    StringLookup,
    Input,
    BatchNormalization,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    ReLU,
    LeakyReLU,
    LayerNormalization,
    Embedding,
    Multiply,
    Concatenate,
    # BatchNormalizationV2,
)

# from keras.layers.normalization.batch_normalization_v1 import (
#     BatchNormalization,
# )
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from thumbs.train import Train, TrainBCE, TrainWassersteinGP, TrainBCEPatch


infinity = float("inf")

ngf = 128
ndf = 128


class PokemonSkipModel(Model):
    def __init__(self, params: HyperParams, mparams: MutableHyperParams) -> None:
        super().__init__(params, mparams, tf.keras.losses.BinaryCrossentropy(from_logits=True))

    def build_generator(self, z_dim):
        #  TODO maybe add this back in somehow. Lets see how it goes without it
        noise_input = Input(shape=(z_dim,), name="noise input")

        # black/white only, same dimensions as image
        outline_input = Input(shape=self.params.img_shape, name="outline input")

        # Encoder
        # Layer 1: input shape is (batch_size, 128, 128, 1)
        down1 = Conv2D(64, 4, strides=2, padding="same", use_bias=False)(outline_input)  # output shape: (batch_size, 64, 64, 64)
        down1 = LeakyReLU()(down1)

        # Layer 2: input shape is (batch_size, 64, 64, 64)
        down2 = Conv2D(128, 4, strides=2, padding="same", use_bias=False)(down1)  # output shape: (batch_size, 32, 32, 128)
        down2 = BatchNormalization()(down2)
        down2 = LeakyReLU()(down2)

        # Layer 3: input shape is (batch_size, 32, 32, 128)
        down3 = Conv2D(256, 4, strides=2, padding="same", use_bias=False)(down2)  # output shape: (batch_size, 16, 16, 256)
        down3 = BatchNormalization()(down3)
        down3 = LeakyReLU()(down3)

        # Layer 4: input shape is (batch_size, 16, 16, 256)
        down4 = Conv2D(512, 4, strides=2, padding="same", use_bias=False)(down3)  # output shape: (batch_size, 8, 8, 512)
        down4 = BatchNormalization()(down4)
        down4 = LeakyReLU()(down4)

        # Decoder with skip connections
        # Layer 1: input shape is (batch_size, 8, 8, 512)
        up1 = Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False)(down4)  # output shape: (batch_size, 16, 16, 256)
        up1 = BatchNormalization()(up1)
        up1 = Concatenate()([up1, down3])  # output shape: (batch_size, 16, 16, 512)
        up1 = LeakyReLU()(up1)

        # Layer 2: input shape is (batch_size, 16, 16, 512)
        up2 = Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(up1)  # output shape: (batch_size, 32, 32, 128)
        up2 = BatchNormalization()(up2)
        up2 = Concatenate()([up2, down2])  # output shape: (batch_size, 32, 32, 256)
        up2 = LeakyReLU()(up2)

        # Layer 3: input shape is (batch_size, 32, 32, 256)
        up3 = Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(up2)  # output shape: (batch_size, 64, 64, 64)
        up3 = BatchNormalization()(up3)
        up3 = Concatenate()([up3, down1])  # output shape: (batch_size, 64, 64, 128)
        up3 = LeakyReLU()(up3)

        # Final layer: input shape is (batch_size, 64, 64, 128)
        last = Conv2DTranspose(3, 4, strides=2, padding="same", use_bias=False, activation="tanh")(
            up3
        )  # output shape: (batch_size, 128, 128, 3)

        model = tf.keras.Model([noise_input, outline_input], last, name="generator")

        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        # 128x128x3
        image_input = Input(shape=img_shape)

        # 128x128x3
        outline_input = Input(shape=self.params.img_shape)

        model_input = Concatenate(axis=-1)([image_input, outline_input])
        # 128x128x6

        down1 = Conv2D(64, 4, strides=2, padding="same")(model_input)  # (batch_size, 64, 64, 64)
        down1 = LeakyReLU()(down1)

        down2 = Conv2D(128, 4, strides=2, padding="same")(down1)  # (batch_size, 32, 32, 128)
        down2 = BatchNormalizationV1()(down2)
        down2 = LeakyReLU()(down2)

        down3 = Conv2D(256, 4, strides=2, padding="same")(down2)  # (batch_size, 16, 16, 256)
        down3 = BatchNormalizationV1()(down3)
        down3 = LeakyReLU()(down3)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 18, 18, 256)
        conv = Conv2D(512, 4, strides=1, use_bias=False)(zero_pad1)  # (batch_size, 15, 15, 512)
        norm1 = BatchNormalizationV1()(conv)
        leaky_relu = LeakyReLU()(norm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 17, 17, 512)
        last = Conv2D(1, 4, strides=1)(zero_pad2)  # (batch_size, 14, 14, 1)

        model = tf.keras.Model([image_input, outline_input], last, name="discriminator")
        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator):
        # Can't really do this easily because they both take multiple inputs
        return None


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.zoom_factor = 0.98
        self.images = get_pokemon_data256(self.params.img_shape)

    def get_data(self) -> np.ndarray:
        return self.images

    def get_random_labels(self, n: int):
        random_indexes = np.random.choice(len(self.images), size=n, replace=False)
        random_images = self.images[random_indexes]

        # For each image, generate an outline
        outlines = np.array([self.create_outline(image) for image in random_images])
        return np.asarray(outlines)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainBCEPatch(model, self.params, mparams, self.get_random_labels)

    def custom_agumentation(
        self, image: tf.Tensor, outline: Optional[tf.Tensor] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Optional[tf.Tensor]]]:
        assert outline is None
        output = super().custom_agumentation(image, None)
        assert isinstance(output, tuple)
        image, outline = output

        # This seems to be a reasonable range for the outline. Higher and some images were totally blank
        upper = tf.random.uniform(shape=[], minval=100, maxval=700, dtype=tf.int32)
        outline = tf.py_function(self.create_outline_tensor, [image, upper], tf.float32)
        return image, outline

    def create_outline_tensor(self, image: tf.Tensor, upper=255, lower=None) -> tf.Tensor:
        outline = self.create_outline(image.numpy(), upper=upper.numpy(), lower=lower)
        tensor: tf.Tensor  = tf.convert_to_tensor(outline, dtype=tf.float32)
        return tensor

    def create_outline(self, image: np.ndarray, upper=255, lower=None) -> np.ndarray:
        """
        image: numpy array that has been normalized to -1,1
        return: numpy array of an outline, noramlized to -1,1, with the same height and width
        """
        # Will result in outlines of varying detail
        image = unnormalize_image(image)
        # thresholds = [10, 100, 400, 800, 1000]
        if lower is None:
            lower = upper // 3

        edges: np.ndarray = cv2.bitwise_not(cv2.Canny(image, threshold1=lower, threshold2=upper))
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        assert edges.shape == image.shape
        return normalize_image(edges)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
            checkpoint_interval=200,
            gradient_penalty_factor=10,
            l1_loss_factor=100,
            discriminator_ones_zeroes_shape=(128, 14, 14, 1),  # patch gan discriminator
        )

        return schedule

    def get_params(self) -> HyperParams:
        name = "pokemon_conditional_outline_noise"

        exp_dir = "EXP_DIR"
        if exp_dir in os.environ:
            base_dir = os.environ["EXP_DIR"]
        else:
            base_dir = "/mnt/e/experiments"

        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            weight_path=f"{base_dir}/{name}/weights",
            checkpoint_path=f"{base_dir}/{name}/checkpoints",
            prediction_path=f"{base_dir}/{name}/predictions",
            iteration_checkpoints_path=f"{base_dir}/{name}/iteration_checkpoints",
            loss_path=f"{base_dir}/{name}/loss",
            accuracy_path=f"{base_dir}/{name}/accuracy",
            iteration_path=f"{base_dir}/{name}/iteration",
            similarity_threshold=0.0,
            # generator_clip_gradients_norm=1,
            similarity_penalty=0,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return PokemonSkipModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
