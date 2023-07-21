import tensorflow as tf
from thumbs.viz import show_samples
from thumbs.params import HyperParams, MutableHyperParams
import os
from rangedict import RangeDict
import numpy as np
from thumbs.train import Train, load_iterations
from thumbs.model.model import GanModel, BuiltModel
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator, Union, Optional
from scipy.ndimage import rotate
from PIL import Image


class Experiment(ABC):
    def __init__(self) -> None:
        self.params = self.get_params()
        self.zoom_factor = 0.95
        self.augment_flips = True
        self.augment_rotations = True
        self.augment_zooms = True

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
    def get_model(self, mparams: MutableHyperParams) -> GanModel:
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

    def rotate_tensor(self, image_tf: tf.Tensor) -> tf.Tensor:
        # Assuming you have an image file called 'image.jpg'
        # Load the image using PIL (Python Imaging Library)
        image = image_tf.numpy()

        # Rotate the image array by 20 degrees counterclockwise
        rotated_array = rotate(image, angle=20, reshape=False, mode="")

        # Convert the rotated array back to a PIL image
        rotated_tensor: tf.Tensor = tf.convert_to_tensor(rotated_array)

        return rotated_tensor

    def custom_agumentation(self, image: tf.Tensor, labels: Optional[tf.Tensor] = None) -> Union[tf.Tensor, tuple]:
        """
        flip, rotate, zoom randomly
        """
        if not self.augment_data():
            return image if labels is None else (image, labels)

        if self.augment_flips:
            image = tf.image.random_flip_left_right(image)
        # Fill with 1 which is white in an image normalized to -1,1. Default is to reflect part of the image
        # to fill the space in the rotation but that would introduce parts of the pokemon that that shouldn't be there.

        if self.augment_rotations:
            image = tf.keras.layers.RandomRotation(0.05, fill_mode="constant", fill_value=1)(image)

        # 10% zoom
        if self.augment_zooms:
            (x, y, channels) = self.params.img_shape
            image = tf.image.random_crop(image, size=[int(x * self.zoom_factor), int(y * self.zoom_factor), channels])
            image = tf.image.resize(image, [x, y])

        return image if labels is None else (image, labels)

    def prepare_data(self, dataset: tf.data.Dataset, mparams: MutableHyperParams) -> tf.data.Dataset:
        return (
            dataset.shuffle(buffer_size=1000)
            .map(self.custom_agumentation)
            .batch(mparams.batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

    def start(self) -> None:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
        schedule = self.get_mutable_params()
        loaded_i = load_iterations(self.params.iteration_path)
        if loaded_i is not None:
            i = loaded_i + 1  # we already did this iteration and it was saved so add one
        else:
            i = 0
        dataset = tf.data.Dataset.from_tensor_slices(self.get_data())

        while True:
            try:
                mparams: MutableHyperParams = schedule[i + 1]
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
