from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class HyperParams:
    latent_dim: int  # = 150
    img_shape: Tuple[int, int, int]  # = (128, 128, 3)
    weight_path: str  # = "./experiments/model_name/weights"
    prediction_path: str  # = "./experiments/model_name/predictions"
    iteration_path: str  # = "./experiments/model_name/iteration"
    iteration_checkpoints_path: str  # = "./experiments/model_name/iteration"
    loss_path: str  # = "./experiments/model_name/loss"
    accuracy_path: str  # = "./experiments/model_name/accuracy"
    similarity_threshold: float  # = 0.0
    similarity_penalty: float  # = 10.0
    checkpoint_path: str  # = "./experiments/model_name/checkpoint"
    generator_clip_gradients_norm: Optional[float] = None  # = None


@dataclass
class MutableHyperParams:
    iterations: int  # = 200_000
    batch_size: int  # = 128
    sample_interval: int  # = 100
    gen_learning_rate: float  # = 0.0001
    dis_learning_rate: float  # = 0.00001
    adam_b1: float  # = 0.5
    generator_turns: int = 1
    discriminator_turns: int = 1
    checkpoint_interval: int = 200
    discriminator_training: bool = True
    generator_training: bool = True
    gradient_penalty_factor: float = 10
    discriminator_ones_zeroes_shape: Optional[tuple] = None 

    def __post_init__(self):
        if self.discriminator_ones_zeroes_shape is None:
            self.discriminator_ones_zeroes_shape = (self.batch_size, 1)
