import tensorflow as tf
from tensorflow import reduce_mean
from thumbs.params import HyperParams


class Loss:
    def __init__(self, params: HyperParams) -> None:
        self.params = params

    def similarity_penalty_loss(self, similarity_score):
        return tf.maximum(
            0.0,
            self.params.similarity_penalty * (self.params.similarity_threshold - similarity_score),
        )

    def wasserstein_loss(self, y_true, y_pred):
        return reduce_mean(y_true * y_pred)

    # @tf.function
    def worst_cosine_similarity(self, images):
        similarities = [
            tf.reduce_mean(tf.keras.losses.cosine_similarity(images[i], images[j]))
            for i in range(len(images))
            for j in range(i + 1, len(images))
        ]
        return tf.reduce_min(similarities)

    def compute_similarity_score(self, images1, images2):
        similarity = tf.keras.losses.cosine_similarity(images1, images2)
        return tf.reduce_mean(similarity)

    def bce_similarity_loss(self, disc_output, generated_images):
        loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(disc_output), disc_output)
        worst_similarity = self.worst_cosine_similarity(generated_images[:8])
        additional_loss = self.similarity_penalty_loss(worst_similarity)
        return loss + additional_loss
