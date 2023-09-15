import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
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
        x = tf.reshape(x, (-1, size * size, self.channels))
        x_ln = self.ln(x)
        attention_value = self.mha(x_ln, x_ln, x_ln)  # MultiHeadAttention in TF doesn't return attention scores by default
        attention_value += x
        attention_value = self.ff_self(attention_value) + attention_value
        x = tf.reshape(attention_value, (-1, size, size, self.channels))

        assert orig.shape == x.shape
        return x
