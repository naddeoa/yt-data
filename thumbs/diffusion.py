from thumbs.params import DiffusionHyperParams, HyperParams
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import cast

from thumbs.viz import visualize_thumbnails


class Diffusion:
    def __init__(self, mparams: DiffusionHyperParams, params: HyperParams, model) -> None:
        self.mparams = mparams
        self.params = params
        self.model = model
        self.alpha = 1.0 - mparams.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)

    def show_samples(self, dataset: tf.data.Dataset, file_name=None, rows=6, cols=6):
        n_imgs = 4
        random_batch = next(iter(dataset))
        random_img = random_batch[:n_imgs]

        min_t = 500
        t = tf.constant(
            [
                [np.random.randint(min_t, self.mparams.T - 1)],
                [np.random.randint(min_t, self.mparams.T - 1)],
                [np.random.randint(min_t, self.mparams.T - 1)],
                [np.random.randint(min_t, self.mparams.T - 1)],
            ],
            dtype=tf.int32,
        )

        noisy, real_noise = self.add_noise(random_img, t)
        # Predict the noise
        predicted_noise = self.model([noisy, t], training=False)

        dir = self.params.prediction_path
        imgs_per_row = 4

        labels = []
        for i in range(n_imgs):
            labels += [
                "Original",
                "Reconstructed",
                f"Noisy (t={t[i].numpy()[0]})",
                "Predicted Noise",
            ]

        denoised_img = self.remove_noise(noisy, predicted_noise, t)
        images = [random_img.numpy(), denoised_img.numpy(), noisy.numpy(), predicted_noise.numpy()]
        images = [img for sublist in zip(*images) for img in sublist]
        visualize_thumbnails(images, rows=n_imgs, cols=imgs_per_row, dir=dir, file_name=file_name, label_list=labels)

    @tf.function
    def sample(
        self,
        n,
        clip=False,
    ) -> tf.Tensor:
        x = tf.random.normal((n, *self.params.img_shape))
        start = self.mparams.T - 1
        for i in tf.range(start, 0, -1):
            t = tf.ones(n, dtype=tf.int32) * i
            predicted_noise = self.model([x, t], training=False)

            alpha = tf.gather(self.alpha, t)
            alpha = tf.reshape(alpha, [-1, 1, 1, 1])

            alpha_hat = tf.gather(self.alpha_hat, t)
            alpha_hat = tf.reshape(alpha_hat, [-1, 1, 1, 1])

            beta = tf.gather(self.mparams.beta, t)
            beta = tf.reshape(beta, [-1, 1, 1, 1])

            if i > 1:
                noise = tf.random.normal(tf.shape(x))
            else:
                noise = tf.zeros(tf.shape(x))

            x = (1 / tf.sqrt(alpha)) * (x - ((1 - alpha) / tf.sqrt(1 - alpha_hat)) * predicted_noise) + tf.sqrt(beta) * noise

            if clip:
                x = tf.clip_by_value(x, -1, 1)

            # if i % 100 == 0:
            # tf.print(i)

        x = tf.clip_by_value(x, -1, 1)
        x = (x + 1) / 2
        x = tf.cast(x * 255, tf.uint8)
        return cast(tf.Tensor, x)

    def sample_2(
        self,
        n,
        steps_size=20,
    ) -> np.ndarray:
        x = tf.random.normal((n, *self.params.img_shape))
        results = tf.constant([])
        for i in tqdm(list(reversed(range(1, self.mparams.T, steps_size)))):
            t = tf.ones(n, dtype=tf.int32) * i
            predicted_noise = self.model.predict([x, t], verbose=False)  # Assuming model accepts x and t
            x = self.remove_noise(x, predicted_noise, t)
            results = x
            if i > 1:
                x, _ = self.add_noise(x, t)

        results = tf.clip_by_value(x, -1, 1)
        results = (x + 1) / 2
        results = tf.cast(x * 255, tf.uint8)
        return cast(np.ndarray, results.numpy())

    # @tf.function
    def add_noise(self, x, t):
        sqrt_alpha_hat = tf.math.sqrt(tf.gather(self.alpha_hat, t))
        sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [-1, 1, 1, 1])

        sqrt_one_minus_alpha_hat = tf.math.sqrt(1 - tf.gather(self.alpha_hat, t))
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [-1, 1, 1, 1])

        noise = tf.random.normal(tf.shape(x))
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def remove_noise(self, x_noisy, noise, t, device="/cpu:0"):
        with tf.device(device):
            # Grab the same sqrt_alpha_hat and sqrt_one_minus_alpha_hat values
            sqrt_alpha_hat = tf.math.sqrt(tf.gather(self.alpha_hat, t))
            sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [-1, 1, 1, 1])

            sqrt_one_minus_alpha_hat = tf.math.sqrt(1 - tf.gather(self.alpha_hat, t))
            sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [-1, 1, 1, 1])

            # Reverse the noise addition to recover the original x
            x_original = (x_noisy - sqrt_one_minus_alpha_hat * noise) / sqrt_alpha_hat

        return x_original
