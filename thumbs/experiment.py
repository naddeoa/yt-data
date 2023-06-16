from thumbs.params import HyperParams, MutableHyperParams
from rangedict import RangeDict
import numpy as np
from thumbs.train import Train, load_iterations
from thumbs.model.model import Model, BuiltModel
from abc import ABC, abstractmethod
from typing import List


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

    def start(self) -> None:
        schedule = self.get_mutable_params()
        i = load_iterations(self.params.iteration_path) or 0
        data = self.get_data()

        while True:
            mparams: MutableHyperParams = schedule[i]
            if i > mparams.iterations:
                raise Exception(f"Checkpointed at iteration {i} but only training for {mparams.iterations} iterations")

            print("------------------------------------------------------------")
            print(f"Training with params {mparams}, starting from iteration {i} to {mparams.iterations}")
            print("------------------------------------------------------------")

            model = self.get_model(mparams)
            train = self.get_train(model.build())

            for j in train.train(data, mparams.iterations, mparams.sample_interval, start_iter=i):
                i = j
