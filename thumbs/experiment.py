import tensorflow as tf
from thumbs.viz import show_samples
from thumbs.params import HyperParams, MutableHyperParams
import os
from rangedict import RangeDict
import numpy as np
from thumbs.train import Train, load_iterations
from thumbs.model.model import Model, BuiltModel
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator, Union


class Experiment(ABC):
    def __init__(self) -> None:
        self.params = self.get_params()

    @abstractmethod
    def get_train(self, model: BuiltModel) -> Train:
        raise NotImplementedError()

    @abstractmethod
    def get_params(self) -> HyperParams:
        raise NotImplementedError()

    @abstractmethod
    def get_mutable_params(self) -> RangeDict:
        raise NotImplementedError()

    @abstractmethod
    def get_model(self, mparams: MutableHyperParams) -> Model:
        raise NotImplementedError()

    @abstractmethod
    def get_data(self) -> np.ndarray:
        raise NotImplementedError()

    def augment_data(self) -> bool:
        return True

    def get_samples(self):
        schedule = self.get_mutable_params()
        mparams: MutableHyperParams = schedule[0]
        model = self.get_model(mparams).build()

        show_samples(
            model.generator,
            self.get_params().latent_dim,
            file_name="",
            dir="",
            rows=6,
            cols=6,
        )

    def _custom_agumentation(self, image: tf.Tensor) -> tf.Tensor:
        """
        flip, rotate, zoom randomly
        """
        if not self.augment_data():
            return image

        image = tf.image.random_flip_left_right(image)
        image = tf.keras.layers.experimental.preprocessing.RandomRotation(0.05)(image)

        # 10% zoom
        (x, y, channels) = self.params.img_shape
        image = tf.image.random_crop(image, size=[int(x * 0.95), int(y * 0.95), channels])
        image = tf.image.resize(image, [x, y])
        return image

    def prepare_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return (
            dataset.shuffle(buffer_size=1000)
            .map(self._custom_agumentation)
            .batch(self.params.batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

    def start(self) -> None:
        schedule = self.get_mutable_params()
        i = load_iterations(self.params.iteration_path) or 0
        i += 1
        dataset = tf.data.Dataset.from_tensor_slices(self.get_data())

        while True:
            mparams: MutableHyperParams = schedule[i]
            if i >= mparams.iterations:
                print(f"Checkpointed at iteration {i} but only training for {mparams.iterations} iterations")
                os._exit(0)

            print("------------------------------------------------------------")
            print(f"Training with params {mparams}, starting from iteration {i} to {mparams.iterations}")
            print("------------------------------------------------------------")

            model = self.get_model(mparams)
            train = self.get_train(model.build())

            for j in train.train(self.prepare_data(dataset), mparams, start_iter=i):
                i = j

            i += 1
