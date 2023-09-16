import thumbs.config_logging  # must be first
import tensorflow as tf
from typing import List, Tuple, Iterator, Optional, TypedDict, Union
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import DiffusionExperiment
from thumbs.data import (
    get_pokemon_and_pokedexno,
    normalize_image,
    unnormalize_image,
    get_wow_icons_64,
    get_pokemon_data256,
    get_wow_icons_128,
    get_wow_icons_256,
)
from thumbs.params import DiffusionHyperParams, HyperParams, Sampler, MutableHyperParams
from thumbs.model.model import GanModel, BuiltDiffusionModel, FrameworkModel, DiffusionModel

from tensorflow.keras.models import Model
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras import Sequential
from keras.layers import (
    Activation,
    Add,
    LayerNormalization,
    StringLookup,
    Conv2DTranspose,
    Conv2D,
    Input,
    LayerNormalization,
    BatchNormalization,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    ReLU,
    LeakyReLU,
    GroupNormalization,
    LayerNormalization,
    MultiHeadAttention,
    Embedding,
    Multiply,
    Concatenate,
)
from thumbs.self_attention import SelfAttention
from thumbs.train import Train, TrainDiffusion

tf.keras.layers.Dropout  # TODO is this different than keras.layers.Dropout? Is it still broken?


class ModelParams(TypedDict):
    kern_size: int
    nuf: int
    up_highest_f: int
    up_blocks: List[int]
    ns: int
    nbn: int
    ndf: int
    down_highest_f: int
    down_blocks: List[int]


class UpResNetLayer(tf.keras.layers.Layer):
    def __init__(self, f, model_params, strides=1, **kwargs):
        super(UpResNetLayer, self).__init__(**kwargs)
        self.f = f
        self.strides = strides
        self.model_params = model_params
        self.kernel_size = self.model_params["kern_size"]
        self.channels = f * self.model_params["nuf"]

    def build(self, input_shape):
        self.seq = Sequential()
        self.seq.add(Conv2DTranspose(self.channels, kernel_size=self.kernel_size, strides=self.strides, padding="same"))
        self.seq.add(GroupNormalization(1))
        self.seq.add(Activation("gelu"))

        if self.strides == 1:
            self.seq.add(Conv2DTranspose(self.channels, kernel_size=self.kernel_size, strides=1, padding="same"))
            self.seq.add(GroupNormalization(1))

    def call(self, inputs):
        x = self.seq(inputs)
        if inputs.shape == x.shape:
            x = Add()([x, inputs])
            x = Activation("gelu")(x)
        return x


class DownResNetLayer(tf.keras.layers.Layer):
    def __init__(self, f, model_params, strides=1, normalize=True, **kwargs):
        super(DownResNetLayer, self).__init__(**kwargs)
        self.f = f
        self.strides = strides
        self.normalize = normalize
        self.model_params = model_params
        self.kernel_size = self.model_params["kern_size"]
        self.channels = f * self.model_params["ndf"]

    def build(self, input_shape):
        self.seq = Sequential()
        self.seq.add(Conv2D(self.channels, kernel_size=self.kernel_size, strides=self.strides, padding="same"))

        if self.normalize:
            self.seq.add(GroupNormalization(1))
        self.seq.add(Activation("gelu"))

        if self.strides == 1:
            self.seq.add(Conv2D(self.channels, kernel_size=self.kernel_size, strides=1, padding="same"))
            self.seq.add(GroupNormalization(1))

    def call(self, inputs):
        x = self.seq(inputs)
        if inputs.shape == x.shape:
            x = Add()([x, inputs])
            x = Activation("gelu")(x)
        return x


class MyModel(DiffusionModel):
    def __init__(self, params: HyperParams[ModelParams], mparams: DiffusionHyperParams) -> None:
        super().__init__(params, mparams)
        self.embed_dim = 256
        self.model_params = params.model_params

    def concat_embedding(self, x, embedding, name: str):
        _, H, W, C = x.shape

        s = Sequential([Dense(H * W, use_bias=True), Reshape((H, W, 1))], name=name)
        _x = s(embedding)
        return Concatenate(name=f"embed_{name}")([x, _x])

    def pos_encoding(self, t, channels=256):
        inv_freq = 1.0 / (10000 ** (tf.range(0, channels, 2, dtype=tf.float32) / tf.cast(channels, tf.float32)))
        t_repeated = tf.repeat(t, repeats=[channels // 2], axis=-1)
        pos_enc_a = tf.math.sin(tf.multiply(t_repeated, inv_freq))
        pos_enc_b = tf.math.cos(tf.multiply(t_repeated, inv_freq))
        pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc

    def positional_encoding_layer(self, x, t, name: str):
        _, H, W, C = x.shape

        s = Sequential(
            [
                tf.keras.layers.Activation("swish", input_shape=(self.embed_dim,)),
                Dense(C),
            ],
            name=name,
        )

        _x = s(t)
        # Need to turn it into a (batch_size, H, W, 256) tensor so it can be added
        _x = Reshape((1, 1, C))(_x)
        _x = tf.tile(_x, [1, H, W, 1])
        return Add(name=f"embed_{name}")([x, _x])

    def get_model(self) -> Model:
        img_input = Input(shape=self.params.img_shape, name="image")

        t_input = Input(shape=(1,), name="t")
        t_pos = self.pos_encoding(t_input)

        x = img_input

        down_highest_f = self.model_params["down_highest_f"]
        down_blocks = self.model_params["down_blocks"]
        nbn = self.model_params["nbn"]
        up_highest_f = self.model_params["up_highest_f"]
        up_blocks = self.model_params["up_blocks"]
        ndf = self.model_params["ndf"]
        nuf = self.model_params["nuf"]

        # Seed block
        for i in range(self.model_params["ns"]):
            x = DownResNetLayer(1, self.model_params, strides=1, name=f"seed_{i}")(x)

        # Down stack
        downs = [x]
        for i, f in enumerate(np.linspace(1, down_highest_f, len(down_blocks), dtype=int)):
            x = DownResNetLayer(f, self.model_params, strides=2, normalize=False, name=f"down_resnet_f{f}")(x)

            for j in range(down_blocks[i]):
                x = DownResNetLayer(f, self.model_params, name=f"down_resnet_f{f}-{j}")(x)

            if i >= len(down_blocks) - 2:
                x = SelfAttention(f * ndf)(x)

            x = self.positional_encoding_layer(x, t_pos, name=f"pos_down{i}")

            if i < len(down_blocks) - 1:
                downs.append(x)

        downs.reverse()

        # Bottleneck
        for i in range(nbn):
            # x = self.down_resnet(f, x, strides=1)
            x = DownResNetLayer(f, self.model_params, name=f"bottleneck_{i}")(x)
            x = SelfAttention(f * ndf)(x)

        # Up stack
        for i, f in enumerate(np.linspace(up_highest_f, 1, len(up_blocks), dtype=int)):
            # x = self.up_resnet(f, x, strides=2)
            x = UpResNetLayer(f, self.model_params, strides=2, name=f"up_resnet_f{f}")(x)
            x = Concatenate(name=f"unet_concat_{i}")([x, downs[i]])  # Unet concat with the down satck variant

            for j in range(up_blocks[i]):
                # x = self.up_resnet(f, x, strides=1)
                x = UpResNetLayer(f, self.model_params, name=f"up_resnet_f{f}-{j}")(x)

            if i < len(up_blocks) - 2:
                # Skipping this last one saves on a massive amount of memory
                x = SelfAttention(f * nuf)(x)

            x = self.positional_encoding_layer(x, t_pos, name=f"pos_up{i}")  # add positional information back in

        output = Conv2D(3, kernel_size=3, strides=1, padding="same")(x)
        return Model([img_input, t_input], output, name="diffusion_model")


class MyExperiment(DiffusionExperiment):
    def __init__(self) -> None:
        super().__init__()
        # self.data = get_pokemon_data256((64,64,3))
        self.data = get_wow_icons_128()
        # self.data = get_wow_icons_256()

    def augment_data(self) -> bool:
        return False

    def get_data(self) -> Union[np.ndarray, tf.data.Dataset]:
        return self.data

    def get_train(self, model: BuiltDiffusionModel, mparams: DiffusionHyperParams) -> Train:
        return TrainDiffusion(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = DiffusionHyperParams(
            learning_rate=0.0002,
            batch_size=4,
            iterations=100000,
            sample_interval=5,
            model_save_interval=1,
            checkpoint_interval=10,
            T=1000,
            beta_start=0.0001,
            beta_end=0.04,
            beta_schedule_type="easein",
            loss_fn=MeanAbsoluteError(),
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams[ModelParams](
            latent_dim=100,
            name="wow_diffusion_mae_128_attn",
            img_shape=(128, 128, 3),
            sampler=Sampler.NORMAL,
            model_params=ModelParams(
                kern_size=3,
                # Up stack
                nuf=32,
                up_highest_f=8,
                up_blocks=[1, 1, 1, 1],
                # Seed blocks
                ns=2,
                # Bottleneck blocks
                nbn=4,
                # Down stack
                ndf=32,
                down_highest_f=8,
                down_blocks=[1, 1, 1, 1],
            ),
        )

    def get_model(self, mparams: DiffusionHyperParams) -> FrameworkModel:
        return MyModel(self.params, mparams)


if __name__ == "__main__":
    MyExperiment().start()
