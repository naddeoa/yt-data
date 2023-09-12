from dataclasses import dataclass
from keras.losses import MeanSquaredError, MeanAbsoluteError
import tensorflow as tf
import yaml
import os
import numpy as np
from typing import Tuple, Optional, List, Callable, Dict
from enum import Enum


# Create an enum
class Sampler(Enum):
    UNIFORM = 0
    NORMAL = 1
    UNIFORM_NORMAL = 2
    BERNOULLI = 3
    CENSORED_NORMAL = 4


# Create an enum
class TurnMode(Enum):
    NEW_SAMMPLES = 0
    SAME_SAMPLES = 1


@dataclass
class HyperParams:
    latent_dim: int  # = 150
    name: str  # = "pokemon_deep_1L_clipped-0.5_gp-0"

    img_shape: Tuple[int, int, int]  # = (128, 128, 3)
    generator_clip_gradients_norm: Optional[float] = None  # = None
    base_dir: str = os.environ["EXP_DIR"] if "EXP_DIR" in os.environ else "/mnt/e/experiments"
    sampler: Sampler = Sampler.NORMAL  # TODO move this to GanParams

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
    def weight_path(self):
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
    def model_diagram_path(self):
        return f"{self.base_dir}/{self.name}/model.jpg"

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
    model_save_interval: int = -1
    learning_rate: float = -1
    checkpoint_interval: int = -1
    clipnorm: Optional[float] = None
    weight_decay: float = 0.004
    adam_b1: float = 0.9
    adam_b2: float = 0.999  # = 0.5
    notes: Optional[str] = None
    l1_loss_factor: Optional[float] = None
    l2_loss_factor: Optional[float] = None

    def get_yaml(self) -> str:
        return str(yaml.dump(self.__dict__, indent=4))

    def __post_init__(self):
        if self.checkpoint_interval == -1:
            self.checkpoint_interval = self.sample_interval * 10
        
        if self.model_save_interval == -1:
            self.model_save_interval = self.sample_interval


import numpy as np


def cos_linspace(start, stop, num_points):
    # Generate a linear space between 0 and pi
    linear_space = np.linspace(np.pi, 0, num_points)

    # Get the cosine of that linear space
    cosine_space = np.cos(linear_space)

    # Scale and translate to the desired range
    scaled_cosine_space = start + (stop - start) * (cosine_space + 1) / 2

    return scaled_cosine_space


def in_out_easing(start, end, steps):
    # Generate the range
    t = tf.linspace(start, end, steps)

    # Normalize t to the range [0, 1]
    t_norm = (t - start) / (end - start)

    # Apply the smoothstep function for in-out easing
    t_smooth = 3 * tf.math.pow(t_norm, 2) - 2 * tf.math.pow(t_norm, 3)

    # Scale it back to the [start, end] range
    t_smooth = start + (end - start) * t_smooth

    return t_smooth

def ease_in(start, end, steps):
    # Generate the time range
    t = tf.linspace(start, end, steps)
    
    # Normalize t to [0, 1]
    t_norm = (t - start) / (end - start)
    
    # Ease-in curve, cubic
    t_ease_in = tf.math.pow(t_norm, 3)
    
    # Scale it back to [start, end]
    t_ease_in = start + (end - start) * t_ease_in
    
    return t_ease_in

@dataclass
class DiffusionHyperParams(MutableHyperParams):
    T: int = 1000
    beta_start: float = 0.0001  # beta value at the first timestep
    beta_end: float = 0.2  # beta value at the last timestep
    beta: tf.Tensor = tf.constant(-1)
    beta_schedule_type: str = "linear"  # "linear" or "cos", "easein"
    loss_fn: Callable = MeanSquaredError()

    def __post_init__(self):
        if self.beta == -1:
            if self.beta_schedule_type == "linear":
                # self.beta = tf.linspace(self.beta_start, self.beta_end, self.T)
                self.beta = tf.convert_to_tensor(np.linspace(self.beta_start, self.beta_end, self.T, dtype=np.float32))
            elif self.beta_schedule_type == "easein":
                beta_schedule = ease_in(self.beta_start, self.beta_end, self.T)
                self.beta = tf.convert_to_tensor(beta_schedule, dtype=tf.float32)
            else:
                raise ValueError("Invalid beta schedule type")


@dataclass
class GanHyperParams(MutableHyperParams):
    gen_learning_rate: float = 0.0001
    dis_learning_rate: float = 0.0002

    generator_turns: int = 1
    discriminator_turns: int = 1

    g_clipnorm: Optional[float] = None  # For gans
    d_clipnorm: Optional[float] = None  # For gans

    dis_weight_decay: float = 0.004  # For gans
    gen_weight_decay: float = 0.004  # For gans

    generator_turns_mode: TurnMode = TurnMode.SAME_SAMPLES
    discriminator_turns_mode: TurnMode = TurnMode.SAME_SAMPLES

    discriminator_training: bool = True
    generator_training: bool = True
    gradient_penalty_factor: float = 10
    discriminator_ones_zeroes_shape: tuple = ()

    learning_rate: float = -1  # Not used

    def __post_init__(self):
        if self.discriminator_ones_zeroes_shape == ():
            self.discriminator_ones_zeroes_shape = (self.batch_size, 1)
