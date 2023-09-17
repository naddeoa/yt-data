import thumbs.config_logging  # must be first
import tensorflow as tf
from typing import List, Tuple, Iterator, Optional, TypedDict, Union
from rangedict import RangeDict
import numpy as np
from thumbs.diff_augmentation import DiffAugmentLayer

from thumbs.experiment import GanExperiment
from thumbs.data import (
    get_pokemon_and_pokedexno,
    normalize_image,
    unnormalize_image,
    get_wow_icons_64,
    get_pokemon_data256,
    get_wow_icons_128,
    get_wow_icons_256,
)
from thumbs.params import GanHyperParams, DiffusionHyperParams, HyperParams, Sampler, MutableHyperParams
from thumbs.model.model import GanModel, BuiltDiffusionModel, FrameworkModel, DiffusionModel, BuiltGANModel

from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
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
from thumbs.train import Train, TrainDiffusion, TrainWassersteinGP, TrainWassersteinDiffusion

tf.keras.layers.Dropout  # TODO is this different than keras.layers.Dropout? Is it still broken?


class ModelParams(TypedDict):
    kern_size: int
    gen_nuf: int
    gen_up_highest_f: int
    gen_up_blocks: List[int]
    gen_ns: int
    gen_nbn: int
    gen_ndf: int
    gen_down_highest_f: int
    gen_down_blocks: List[int]

    disc_features: int
    disc_down_highest_f: int
    disc_down_blocks: List[int]


class UpResNetLayer(tf.keras.layers.Layer):
    def __init__(self, f, model_params, strides=1, **kwargs):
        super(UpResNetLayer, self).__init__(**kwargs)
        self.f = f
        self.strides = strides
        self.model_params = model_params
        self.kernel_size = self.model_params["kern_size"]
        self.channels = f * self.model_params["gen_nuf"]

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
    def __init__(self, f, model_params, strides=1, normalize=True, use_spectral_norm=False, **kwargs):
        super(DownResNetLayer, self).__init__(**kwargs)
        self.f = f
        self.use_spectral_norm = use_spectral_norm
        self.strides = strides
        self.normalize = normalize
        self.model_params = model_params
        self.kernel_size = self.model_params["kern_size"]
        self.channels = f * self.model_params["gen_ndf"]

    def build(self, input_shape):
        self.seq = Sequential()
        if self.use_spectral_norm:
            self.seq.add(SpectralNormalization(Conv2D(self.channels, kernel_size=self.kernel_size, strides=self.strides, padding="same")))
        else:
            self.seq.add(Conv2D(self.channels, kernel_size=self.kernel_size, strides=self.strides, padding="same"))

        if self.normalize:
            self.seq.add(GroupNormalization(1))
        self.seq.add(Activation("gelu"))

        if self.strides == 1:
            if self.use_spectral_norm:
                self.seq.add(SpectralNormalization(Conv2D(self.channels, kernel_size=self.kernel_size, strides=1, padding="same")))
            else:
                self.seq.add(Conv2D(self.channels, kernel_size=self.kernel_size, strides=1, padding="same"))

            self.seq.add(GroupNormalization(1))

    def call(self, inputs):
        x = self.seq(inputs)
        if inputs.shape == x.shape:
            x = Add()([x, inputs])
            x = Activation("gelu")(x)
        return x


class MyModel(GanModel):
    def __init__(self, params: HyperParams[ModelParams], mparams: GanHyperParams) -> None:
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

    def build_generator(self, z_dim) -> Model:
        # TODO noise?
        z_input = Input(shape=(z_dim,), name="z")
        noisy_image_input = Input(shape=self.params.img_shape, name="noisy_image")

        t_input = Input(shape=(1,), name="t")
        t_pos = self.pos_encoding(t_input)

        x = noisy_image_input

        gen_down_highest_f = self.model_params["gen_down_highest_f"]
        gen_down_blocks = self.model_params["gen_down_blocks"]
        gen_nbn = self.model_params["gen_nbn"]
        gen_up_highest_f = self.model_params["gen_up_highest_f"]
        gen_up_blocks = self.model_params["gen_up_blocks"]
        gen_ndf = self.model_params["gen_ndf"]
        gen_nuf = self.model_params["gen_nuf"]

        # Seed block
        # for i in range(self.model_params["gen_ns"]):
        # x = DownResNetLayer(1, self.model_params, strides=1, name=f"seed_{i}")(x)
        seed = Sequential(
            [
                Conv2D(noisy_image_input.shape[1], kernel_size=3, strides=1, padding="same", use_bias=True),
                GroupNormalization(1),
                tf.keras.layers.Activation("gelu"),
            ],
            name="initial",
        )
        x = seed(x)

        # Down stack
        downs = [x]
        for i, f in enumerate(np.linspace(1, gen_down_highest_f, len(gen_down_blocks), dtype=int)):
            x = DownResNetLayer(f, self.model_params, strides=2, normalize=False, name=f"down_resnet_f{f}")(x)

            for j in range(gen_down_blocks[i]):
                x = DownResNetLayer(f, self.model_params, name=f"down_resnet_f{f}-{j}")(x)

            # x = SelfAttention(f * gen_ndf)(x)
            x = self.positional_encoding_layer(x, t_pos, name=f"pos_down{i}")

            if i < len(gen_down_blocks) - 1:
                downs.append(x)

        downs.reverse()

        # Convert zdim into 2 channel
        z = Dense(8 * 8 * 2)(z_input)
        z = Reshape((8, 8, 2))(z)

        # Bottleneck
        for i in range(gen_nbn):
            x = DownResNetLayer(f, self.model_params, name=f"bottleneck_{i}")(x)
            # x = SelfAttention(f * gen_ndf)(x)
            # Unclear what the right way to incorporate the latent noise vector is. Doing it here is effiecient, and
            # doing it repeatedly might make sure it pays attention more. Having the noise should allow it to predict
            # more types of noise with diff augment. I guess its ignored at worst.
            x = Concatenate(name=f"bottleneck_noise_concat_{i}")([x, z])  # TODO is this the best way?

        # Up stack
        for i, f in enumerate(np.linspace(gen_up_highest_f, 1, len(gen_up_blocks), dtype=int)):
            x = UpResNetLayer(f, self.model_params, strides=2, name=f"up_resnet_f{f}")(x)
            x = Concatenate(name=f"unet_concat_{i}")([x, downs[i]])  # Unet concat with the down satck variant

            for j in range(gen_up_blocks[i]):
                x = UpResNetLayer(f, self.model_params, name=f"up_resnet_f{f}-{j}")(x)

            if i < len(gen_up_blocks) - 1:
                x = SelfAttention(f * gen_nuf)(x)

            x = self.positional_encoding_layer(x, t_pos, name=f"pos_up{i}")  # add positional information back in

        output = Conv2D(3, kernel_size=3, strides=1, padding="same")(x)
        return Model([z_input, noisy_image_input, t_input], output, name="generator")

    def build_discriminator(self, img_shape):
        noisy_image_input = Input(shape=self.params.img_shape, name="noisy_image")
        noise_input = Input(shape=self.params.img_shape, name="noise")

        t_input = Input(shape=(1,), name="t")
        t_pos = self.pos_encoding(t_input)

        x = Concatenate()([noisy_image_input, noise_input])

        disc_down_highest_f = self.model_params["disc_down_highest_f"]
        disc_down_blocks = self.model_params["disc_down_blocks"]
        disc_features = self.model_params["disc_features"]

        for i, f in enumerate(np.linspace(1, disc_down_highest_f, len(disc_down_blocks), dtype=int)):
            x = DownResNetLayer(f, self.model_params, strides=2, normalize=False, name=f"down_resnet_f{f}", use_spectral_norm=True)(x)

            for j in range(disc_down_blocks[i]):
                x = DownResNetLayer(f, self.model_params, name=f"down_resnet_f{f}-{j}", use_spectral_norm=True)(x)

            if i < len(disc_down_blocks) - 1:
                x = SelfAttention(f * disc_features)(x)

            x = self.positional_encoding_layer(x, t_pos, name=f"pos_down{i}")

        x = SpectralNormalization(Conv2D(1, kernel_size=8, strides=1, padding="valid"))(x)
        x = Flatten()(x)
        model = tf.keras.Model([noisy_image_input, noise_input, t_input], x, name="discriminator")
        return model


class MyExperiment(GanExperiment):
    def __init__(self) -> None:
        super().__init__()
        # self.data = get_pokemon_data256((64,64,3))
        self.data = get_wow_icons_64()
        # self.data = get_wow_icons_128()
        # self.data = get_wow_icons_256()

    def augment_data(self) -> bool:
        return False

    def get_data(self) -> Union[np.ndarray, tf.data.Dataset]:
        return self.data

    def get_train(self, model: BuiltGANModel, mparams: GanHyperParams) -> Train:
        diffusion_mparams = DiffusionHyperParams(
            learning_rate=0.0002,
            batch_size=4,
            iterations=100000,
            sample_interval=1,
            model_save_interval=1,
            checkpoint_interval=10,
            # Only these matter
            T=1000,
            beta_start=0.0001,
            beta_end=.0085,
            beta_schedule_type="linear",
            loss_fn=MeanAbsoluteError(),
        )

        return TrainWassersteinDiffusion(model, self.params, mparams, diffusion_mparams, diff_augment=True)

    def get_mutable_params(self) -> RangeDict:
        schedule = RangeDict()
        schedule[0, 100000] = GanHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0003,
            batch_size=64,
            adam_b1=0.5,
            iterations=100000,
            sample_interval=1,
            model_save_interval=1,
            generator_turns=1,
            discriminator_turns=2,
            g_clipnorm=0.01,
            d_clipnorm=0.01,
            gradient_penalty_factor=10.0,
            l1_loss_factor=100.0,
            # gen_weight_decay=0,
            # dis_weight_decay=0,
        )

        return schedule

    def get_params(self) -> HyperParams:
        return HyperParams[ModelParams](
            latent_dim=100,
            name="wow_diffusion_gan_64_attn_linear_diffaugment",
            img_shape=(64, 64, 3),
            sampler=Sampler.NORMAL,
            model_params=ModelParams(
                kern_size=3,
                #
                # Generator
                #
                # Up stack
                gen_nuf=64,
                gen_up_highest_f=4,
                gen_up_blocks=[1, 1, 1],
                # Seed blocks
                gen_ns=1,
                # Bottleneck blocks
                gen_nbn=4,
                # Down stack
                gen_ndf=64,
                gen_down_highest_f=4,
                gen_down_blocks=[1, 1, 1],
                #
                # Discriminator
                #
                disc_features=64,
                disc_down_highest_f=8,
                disc_down_blocks=[2, 2, 2],
            ),
        )

    def get_model(self, mparams: GanHyperParams) -> FrameworkModel:
        return MyModel(self.params, mparams)


if __name__ == "__main__":
    MyExperiment().start()
