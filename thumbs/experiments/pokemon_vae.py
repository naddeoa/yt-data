import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from thumbs.data import get_pokemon_data256

# Hyper params
latent_dim = 100
image_shape = (128, 128, 3)
input_img = Input(shape=image_shape)

# Encoder
x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)
x = Flatten()(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_var) * epsilon


z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])


# Decoder
decoder_input = Input(K.int_shape(z)[1:])
x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = Reshape(shape_before_flattening[1:])(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(3, 3, padding='same', activation='tanh')(x) # was sigmoid originally, lets see
decoder = Model(decoder_input, x)
z_decoded = decoder(z)

class CustomVariationalLayer(tf.keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = mse(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([input_img, z_decoded])

vae = Model(input_img, y)
vae.compile(optimizer='adam', loss=None)

x_train = get_pokemon_data256(image_shape)

# Assuming your images are stored in a numpy array x_train
vae.fit(x=x_train,
        y=None,
        shuffle=True,
        epochs=10,
        batch_size=32,
        validation_split=0.2)
