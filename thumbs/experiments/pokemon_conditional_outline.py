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
    Concatenate
    # BatchNormalizationV2,
)

# from keras.layers.normalization.batch_normalization_v1 import (
#     BatchNormalization,
# )
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from thumbs.train import Train, TrainMSE, TrainBCE, TrainBCESimilarity, TrainWassersteinGP


infinity = float("inf")

ngf = 128
ndf = 128

class TileLayer(tf.keras.layers.Layer):
  def __init__(self, tiles):
    super(TileLayer, self).__init__()
    self.tiles = tiles

  def build(self, input_shape):
    pass

  def call(self, inputs):
    return tf.tile(inputs, self.tiles)



class PokemonModel(Model):
    def build_generator(self, z_dim):
        noise_input = Input(shape=(z_dim,), name="noise input")
        
        # black/white only, same dimensions as image
        outline_shape = (self.params.img_shape[0], self.params.img_shape[1], 1)
        outline_input = Input(shape=outline_shape, name="outline input")

        # Got a whole nother model in here just to downsample the outline enough to concat it with the noise
        outline = Conv2D(8, kernel_size=5, strides=2, padding="same", use_bias=False)(outline_input)
        outline = LeakyReLU()(outline)

        outline  = Conv2D(16, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        outline = BatchNormalization()(outline )
        outline = LeakyReLU()(outline )

        outline  = Conv2D(32, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)
        outline = BatchNormalization()(outline )
        outline = LeakyReLU()(outline )

        outline  = Conv2D(64, kernel_size=5, strides=2, padding="same", use_bias=False)(outline)

        x = Dense(8*8*ngf*4)(noise_input)
        x = Reshape((8, 8, ngf*4))(x)

        x = Concatenate(axis=-1)([x, outline])
        # x = Conv2DTranspose(ngf*4, kernel_size=6, strides=6, padding='valid')(x)

        x = Conv2DTranspose(ngf*3, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(ngf*2, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(ngf, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
        x = Activation("tanh")(x)

        model = tf.keras.Model([noise_input, outline_input], x, name="generator")
        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        image_input = Input(shape=img_shape)

        outline_shape = (self.params.img_shape[0], self.params.img_shape[1], 1)
        outline_input = Input(shape=outline_shape)
        outline = Dropout(.2)(outline_input)

        model_input = Concatenate(axis=-1)([image_input , outline])
        x = Conv2D(ndf, kernel_size=5, strides=2, padding="same", use_bias=False)(model_input)
        x = LeakyReLU(alpha=0.2)(x)
        # outline = Dropout(.4)(outline_input)

        x = Conv2D(ndf*2, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        # outline = Dropout(.4)(outline_input)

        x = Conv2D(ndf*4, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        # outline = Dropout(.4)(outline_input)

        x = Conv2D(ndf*8, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1)(x)

        model = tf.keras.Model([image_input, outline_input], x, name="discriminator")
        model.summary(line_length=200)
        return model


    def build_gan(self, generator, discriminator):
        # Can't really do this easily because they both take multiple inputs
        return None


class PokemonExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.zoom_factor = .9
        self.images = get_pokemon_data256(self.params.img_shape)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # For each image, generate an outline
        outlines = np.array([self.create_outline(image, threshold2=1000) for image in self.images])
        return (self.images, outlines)

    def get_random_labels(self, n: int):
        random_indexes = np.random.choice(len(self.images), size=n, replace=False)
        random_images = self.images[random_indexes]

        # For each image, generate an outline
        outlines = np.array([self.create_outline(image) for image in random_images])
        return np.asarray(outlines)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainWassersteinGP(model, self.params, mparams, self.get_random_labels)


    def custom_agumentation(self, image: tf.Tensor, outline: Optional[tf.Tensor] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, Optional[tf.Tensor]]]:
        assert outline is not None

        # Create the outline
        # Generate a shape 2 tensor with two random ints
        seed = tf.random.uniform(shape=(2,), minval=0, maxval=10, dtype=tf.int32)

        image = tf.image.stateless_random_flip_left_right(image, seed=seed)
        outline = tf.image.stateless_random_flip_left_right(outline, seed=seed)

        # Get another random int from numpy
        seednp = np.random.randint(0, 10000)
        image = tf.keras.layers.RandomRotation(0.05, seed=seednp)(image)
        outline = tf.keras.layers.RandomRotation(0.05, seed=seednp)(outline)

        # 10% zoom
        (x, y, channels) = self.params.img_shape
        image = tf.image.stateless_random_crop(image, size=[int(x * self.zoom_factor ), int(y * self.zoom_factor), channels], seed=seed)
        image = tf.image.resize(image, [x, y])

        outline = tf.image.stateless_random_crop(outline, size=[int(x * self.zoom_factor ), int(y * self.zoom_factor), 1], seed=seed)
        outline = tf.image.resize(outline, [x, y])

        return image, outline


    def create_outline(self, image: np.ndarray, threshold2=None) -> np.ndarray:
        """
            image: numpy array that has been normalized to -1,1
            return: numpy array of an outline, noramlized to -1,1, with the same height and width
        """
        # Will result in outlines of varying detail
        image = unnormalize_image(image)
        thresholds = [10, 100, 400, 800, 1000]
        if threshold2 is None:
            threshold2 = np.random.choice(thresholds)

        edges: np.ndarray = cv2.bitwise_not(cv2.Canny(image, threshold1=0, threshold2=threshold2))
        edges = np.expand_dims(edges, axis=-1)

        assert edges.shape == (image.shape[0], image.shape[1], 1)
        return normalize_image(edges)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=32,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=5,
            discriminator_turns=5,
            generator_turns=1,
            checkpoint_interval=100,
            gradient_penalty_factor=10
        )

        return schedule

    def get_params(self) -> HyperParams:
        name = "pokemon_conditional_outline_noise"

        exp_dir = 'EXP_DIR'
        if exp_dir in os.environ:
            base_dir = os.environ['EXP_DIR']
        else:
            base_dir = '/mnt/e/experiments'

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
            generator_clip_gradients_norm=1,
            similarity_penalty=20,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()

