from dataclasses import dataclass
from typing import Tuple


@dataclass
class HyperParams:
    latent_dim: int # = 150
    batch_size: int # = 128
    img_shape: Tuple[int, int, int] # = (128, 128, 3)
    weight_path: str # = "./experiments/model_name/weights"
    prediction_path: str # = "./experiments/model_name/predictions"
    iteration_path: str # = "./experiments/model_name/iteration"
    similarity_threshold: float # = 0.0
    similarity_penalty: float # = 10.0


@dataclass
class MutableHyperParams:
    iterations: int # = 200_000
    sample_interval: int # = 100
    gen_learning_rate: float # = 0.0001
    dis_learning_rate: float # = 0.00001
