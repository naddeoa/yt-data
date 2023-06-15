from thumbs.params import HyperParams, MutableHyperParams
from rangedict import RangeDict
import numpy as np
from thumbs.train import Train, load_iterations
from thumbs.model.model import Model
from abc import ABC, abstractmethod
from typing import List


class Experiment(ABC):
    def __init__(self) -> None:
        self.params = self.get_params()
        self.data = self.get_data()

    def _init_train(self, model: Model) -> Train:
        (
            gan,
            discriminator,
            generator,
            generator_optimizer,
        ) = model.build()
        return Train(
            gan,
            generator,
            discriminator,
            generator_optimizer,
            self.params,
        )

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

    def start(self) -> None:
        schedule = self.get_mutable_params()
        i = load_iterations(self.params.iteration_path) or 0

        while True:
            mparams: MutableHyperParams = schedule[i]
            print("------------------------------------------------------------")
            print(f"Training with params {mparams}, starting from iteration {i} to {mparams.iterations}")
            print("------------------------------------------------------------")

            model = self.get_model(mparams)
            train = self._init_train(model)

            for j in train.train(self.data, mparams.iterations, mparams.sample_interval, start_iter=i):
                i = j
