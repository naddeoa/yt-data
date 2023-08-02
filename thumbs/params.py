from dataclasses import dataclass
import yaml
import os
import numpy as np
from typing import Tuple, Optional
from enum import Enum


# Create an enum
class Sampler(Enum):
    UNIFORM = 0
    NORMAL = 1
    UNIFORM_NORMAL = 2
    BERNOULLI = 3
    CENSORED_NORMAL = 4


@dataclass
class HyperParams:
    latent_dim: int  # = 150
    name: str  # = "pokemon_deep_1L_clipped-0.5_gp-0"

    img_shape: Tuple[int, int, int]  # = (128, 128, 3)
    generator_clip_gradients_norm: Optional[float] = None  # = None
    base_dir: str = os.environ["EXP_DIR"] if "EXP_DIR" in os.environ else "/mnt/e/experiments"
    sampler: Sampler = Sampler.NORMAL

    similarity_threshold: float = 0  # = 0.0
    similarity_penalty: float = 10  # = 10.0

    def latent_sample(self, batch_size: int) -> np.ndarray:
        if self.sampler == Sampler.UNIFORM:
            return np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        elif self.sampler == Sampler.NORMAL:
            return np.random.normal(0, 1, (batch_size, self.latent_dim))
        elif self.sampler == Sampler.UNIFORM_NORMAL:
            latent = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            latent += np.random.normal(0, 0.1, (batch_size, self.latent_dim))
            return latent
        elif self.sampler == Sampler.BERNOULLI:
            return np.random.binomial(1, 0.5, (batch_size, self.latent_dim))
        elif self.sampler == Sampler.CENSORED_NORMAL:
            mean = 1
            latent = np.random.normal(0, mean, (batch_size, self.latent_dim))
            latent[latent > mean] = mean
            latent[latent < -mean] = -mean
            return latent
        else:
            raise ValueError("Invalid sampler")

    @property
    def gen_weight_path(self):
        return f"{self.base_dir}/{self.name}/weights_gen"

    @property
    def dis_weight_path(self):
        return f"{self.base_dir}/{self.name}/weights_dis"

    @property
    def prediction_path(self):
        return f"{self.base_dir}/{self.name}/predictions"

    @property
    def iteration_path(self):
        return f"{self.base_dir}/{self.name}/iteration"

    @property
    def iteration_checkpoints_path(self):
        return f"{self.base_dir}/{self.name}/iteration_checkpoints"

    @property
    def loss_path(self):
        return f"{self.base_dir}/{self.name}/loss"

    @property
    def params_path(self):
        return f"{self.base_dir}/{self.name}/params.yaml"

    @property
    def accuracy_path(self):
        return f"{self.base_dir}/{self.name}/accuracy"

    @property
    def checkpoint_path(self):
        return f"{self.base_dir}/{self.name}/checkpoints"

    @property
    def gen_diagram_path(self):
        return f"{self.base_dir}/{self.name}/generator.jpg"

    @property
    def dis_diagram_path(self):
        return f"{self.base_dir}/{self.name}/discriminator.jpg"

    def __post_init__(self):
        os.makedirs(f"{self.base_dir}/{self.name}", exist_ok=True)

    def get_yaml(self) -> str:
        return str(yaml.dump(self.__dict__, indent=4))


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
    l1_loss_factor: float = 0
    l2_loss_factor: float = 0
    discriminator_ones_zeroes_shape: tuple = ()
    g_clipnorm: Optional[float] = None
    d_clipnorm: Optional[float] = None
    dis_weight_decay: float = 0.004
    gen_weight_decay: float = 0.004

    def __post_init__(self):
        if self.discriminator_ones_zeroes_shape == ():
            self.discriminator_ones_zeroes_shape = (self.batch_size, 1)

    def get_yaml(self) -> str:
        return str(yaml.dump(self.__dict__, indent=4))
