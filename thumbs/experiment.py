import tensorflow as tf
import textwrap
from thumbs.viz import show_samples
from thumbs.params import HyperParams, MutableHyperParams, GanHyperParams, DiffusionHyperParams
import os
from rangedict import RangeDict
import numpy as np
from thumbs.train import Train, load_iterations
from thumbs.model.model import GanModel, BuiltGANModel, FrameworkModel, BuiltDiffusionModel
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator, Union, Optional, TypeVar, Generic
from scipy.ndimage import rotate
from PIL import Image

Params = TypeVar("Params", bound=MutableHyperParams)

BuiltModel = TypeVar("BuiltModel", bound=Union[BuiltGANModel, BuiltDiffusionModel])

class Experiment(ABC, Generic[Params, BuiltModel]):
    def __init__(self) -> None:
        self.params = self.get_params()
        self.zoom_factor = 0.95
        self.augment_flips = True
        self.augment_rotations = True
        self.augment_zooms = True

    @abstractmethod
    def get_train(self, model: BuiltModel, mparams: Params) -> Train:
        raise NotImplementedError()

    @abstractmethod
    def get_params(self) -> HyperParams:
        raise NotImplementedError()

    @abstractmethod
    def get_mutable_params(self) -> RangeDict:
        raise NotImplementedError()

    @abstractmethod
    def get_model(self, mparams: Params) -> FrameworkModel:
        raise NotImplementedError()

    @abstractmethod
    def get_data(self) -> Union[tf.data.Dataset, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def augment_data(self) -> bool:
        return True

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

    def prepare_data(self, dataset: tf.data.Dataset, mparams: Params) -> tf.data.Dataset:
        d = dataset.shuffle(buffer_size=1000)

        if self.augment_data():
            d = d.map(self.custom_agumentation)

        return d.batch(mparams.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def write_params(self) -> None:
        params = f"""
params:
{textwrap.indent(self.params.get_yaml(), '    ')}
"""

        schedule = self.get_mutable_params()
        for rnge, mparams in schedule.items():
            params += f"""
{rnge}:
{textwrap.indent(mparams.get_yaml(), '    ')}
"""

        with open(self.params.params_path, "w") as f:
            f.write(params)

    def start(self) -> None:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
        self.write_params()
        schedule = self.get_mutable_params()
        loaded_i = load_iterations(self.params.iteration_path)
        if loaded_i is not None:
            i = loaded_i + 1  # we already did this iteration and it was saved so add one
        else:
            i = 0

        dataset = self.get_data()
        if not isinstance(dataset, tf.data.Dataset):
            dataset = tf.data.Dataset.from_tensor_slices(dataset)

        while True:
            try:
                print(f"Looking up hyper params for iteration {i+1}")
                mparams: Params = schedule[i + 1]
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


class GanExperiment(Experiment[GanHyperParams, BuiltGANModel]):
    pass


class DiffusionExperiment(Experiment[DiffusionHyperParams, BuiltDiffusionModel]):
    pass
