import thumbs.config_logging  # must be first
from typing import List
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.data import get_data
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.full_conv_model import FullCovModel
from thumbs.model.model import Model

infinity = float('inf')

class FullConvTrainingSchedule(Experiment):
    def get_data(self) -> np.ndarray:
        return get_data(self.params.img_shape, min_views=500_000)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        phase1 = MutableHyperParams(
            generator_learning_rate=0.0001,
            discriminator_learning_rate=0.00001,
            iterations=2000,
            sample_interval=100,
        )
        schedule[0,2000] = phase1

        phase2 = MutableHyperParams(
            generator_learning_rate=0.0001,
            discriminator_learning_rate=0.0001,
            iterations=5300,
            sample_interval=100,
        )
        schedule[2001,7300] = phase2

        phase3 = MutableHyperParams(
            generator_learning_rate=0.0001,
            discriminator_learning_rate=0.00005,
            iterations=2000,
            sample_interval=100,
        )
        schedule[7301,9300] = phase3

        phase4 = MutableHyperParams(
            generator_learning_rate=0.0001,
            discriminator_learning_rate=0.000075,
            iterations=200000,  # +1300 + 13500
            sample_interval=100,
        )
        # Can start to distinguish the beginning of objects around 7k iterations of phase4
        schedule[9301,infinity] = phase4

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams(
            latent_dim=150,
            batch_size=128,
            img_shape=(128, 128, 3),
            weight_path="./experiments/full_conv/weights",
            prediction_path="./experiments/full_conv/predictions",
            iteration_path = "./experiments/full_conv/iteration",
            similarity_threshold=0.0,
            similarity_penalty=10,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return FullCovModel(self.params, mparams)


if __name__ == "__main__":
    FullConvTrainingSchedule().start()
