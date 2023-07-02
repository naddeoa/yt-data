import tensorflow as tf
from thumbs.viz import show_samples
from thumbs.params import HyperParams, MutableHyperParams
import os
from rangedict import RangeDict
import numpy as np
from thumbs.train import Train, load_iterations
from thumbs.model.model import Model, BuiltModel
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator, Union, Optional


class Experiment(ABC):
    def __init__(self) -> None:
        self.params = self.get_params()
        self.zoom_factor = 0.95

    @abstractmethod
    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
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
    def get_data(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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

    def custom_agumentation(self, image: tf.Tensor, labels: Optional[tf.Tensor] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, Optional[tf.Tensor]]]:
        """
        flip, rotate, zoom randomly
        """
        if not self.augment_data():
            return image, labels

        image = tf.image.random_flip_left_right(image)
        image = tf.keras.layers.experimental.preprocessing.RandomRotation(0.05)(image)

        # 10% zoom
        (x, y, channels) = self.params.img_shape
        image = tf.image.random_crop(image, size=[int(x * self.zoom_factor ), int(y * self.zoom_factor ), channels])
        image = tf.image.resize(image, [x, y])
        return image, labels

    def prepare_data(self, dataset: tf.data.Dataset, mparams: MutableHyperParams) -> tf.data.Dataset:
        return (
            dataset.shuffle(buffer_size=1000)
            .map(self.custom_agumentation)
            .batch(mparams.batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

    def start(self) -> None:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        schedule = self.get_mutable_params()
        loaded_i = load_iterations(self.params.iteration_path)
        if loaded_i is not None:
            i = loaded_i + 1 # we already did this iteration and it was saved so add one
        else:
            i = 0
        dataset = tf.data.Dataset.from_tensor_slices(self.get_data())

        while True:
            try:
                mparams: MutableHyperParams = schedule[i+1]
                print(mparams)
            except KeyError:
                print(f"Checkpointed at iteration {i} but only training for {mparams.iterations} iterations")
                os._exit(0)

            if i >= mparams.iterations:
                continue

            model = self.get_model(mparams)
            train = self.get_train(model.build(), mparams)

            for j in train.train(self.prepare_data(dataset, mparams), start_iter=i):
                i = j

