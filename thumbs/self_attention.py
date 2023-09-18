from typing import Any, Optional
from rangedict import RangeDict
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    def build(self, input_shape) -> None:
        channels = input_shape[-1]
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=channels)
        self.ln = tf.keras.layers.LayerNormalization(axis=-1)
        self.ff_self = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(axis=-1),
                tf.keras.layers.Dense(channels, activation=None),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.Dense(channels, activation=None),
            ]
        )

    def call(self, x):
        # print(f'orig x shape: {x.shape}')
        orig = x
        size = x.shape[1]
        channels = x.shape[-1]
        x = tf.reshape(x, (-1, size * size, channels))
        x_ln = self.ln(x)
        attention_value = self.mha(x_ln, x_ln, x_ln)  # MultiHeadAttention in TF doesn't return attention scores by default
        attention_value += x
        attention_value = self.ff_self(attention_value) + attention_value
        x = tf.reshape(attention_value, (-1, size, size, channels))

        assert orig.shape == x.shape
        return x


attn_patch_sizes = RangeDict()
attn_patch_sizes[0, 63] = 2
attn_patch_sizes[64, 127] = 4
attn_patch_sizes[128, 255] = 8


class PatchBasedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, patch_size: Optional[int] = None, convert_to_original_shape=True):
        super(PatchBasedSelfAttention, self).__init__()
        self.patch_size = patch_size

        self.convert_to_original_shape = convert_to_original_shape

    def build(self, input_shape) -> None:
        if self.patch_size is None:
            self.patch_size = attn_patch_sizes[input_shape[1]]

        channels = input_shape[-1]
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=channels)
        self.ln = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x):
        batch_size = tf.shape(x)[0]

        # Extract patches
        patches = tf.image.extract_patches(
            x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Reshape for MHA
        patches_channels = patches.shape[-1]
        patches_size = patches.shape[1]
        patches_reshaped = tf.reshape(patches, [batch_size, patches_size * patches_size, patches_channels])

        # Self Attention magic
        x_ln = self.ln(patches_reshaped)
        attention_value = self.mha(x_ln, x_ln, x_ln)
        attention_value += patches_reshaped
        attention_value = tf.reshape(attention_value, tf.shape(patches))

        if self.convert_to_original_shape:
            resized = tf.nn.depth_to_space(attention_value, block_size=self.patch_size)
            assert x.shape == resized.shape
            return resized

        return attention_value
