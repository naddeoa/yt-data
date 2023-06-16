import tensorflow as tf
import numpy as np
from thumbs.model.model import BuiltModel
from thumbs.util import get_current_time
from thumbs.viz import show_accuracy_plot, show_loss_plot, show_samples
from thumbs.params import HyperParams
from thumbs.loss import Loss
from typing import List, Optional
from abc import ABC, abstractmethod


from tqdm import tqdm


def trunc(loss):
    """Truncate loss values for display in tqdm"""
    return f"{loss:.4f}"


def load_iterations(iteration_path: str) -> Optional[int]:
    try:
        with open(iteration_path, "r") as f:
            i = int(f.read())
            print(f"Loaded previous iteration count: {i}")
            return i

    except Exception as e:
        print("No save file for iteration count")
        return None


def save_iterations(iteration_path: str, iterations: int):
    with open(iteration_path, "w") as f:
        f.write(str(iterations))


def load_weights(gan, weight_path: str):
    try:
        gan.load_weights(weight_path)
        print("Loaded previous weights")
    except Exception as e:
        print(e)


class Train(ABC):
    def __init__(self, built_model: BuiltModel, params: HyperParams) -> None:
        self.gan = built_model.gan
        self.generator = built_model.generator
        self.discriminator = built_model.discriminator
        self.generator_optimizer = built_model.generator_optimizer
        self.params = params
        self.loss = Loss(params)

        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.iteration_checkpoints: List[int] = []

    @abstractmethod
    def train_discriminator(self, gen_imgs, real_imgs):
        raise NotImplementedError()

    @abstractmethod
    def train_generator(self, z):
        raise NotImplementedError()

    def train(self, dataset, iterations, sample_interval, start_iter=0):
        load_weights(self.gan, self.params.weight_path)

        # Labels for fake images: all zeros
        initial_sample = False

        progress = tqdm(range(start_iter, iterations), total=iterations, initial=start_iter)
        for iteration in progress:
            # Get a random batch of real images
            idx = np.random.randint(0, dataset.shape[0], self.params.batch_size)
            imgs = dataset[idx]

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (self.params.batch_size, self.params.latent_dim))
            gen_imgs = self.generator.predict(z, verbose=0)

            # -------------------------
            #  Train the Discriminator
            # -------------------------
            (
                d_loss,
                d_loss_fake,
                d_loss_real,
                d_acc,
                d_fake_acc,
                d_real_acc,
            ) = self.train_discriminator(gen_imgs, imgs)

            # ---------------------
            #  Train the Generator
            # ---------------------
            z = np.random.normal(0, 1, (self.params.batch_size, self.params.latent_dim))
            gen_imgs = self.generator.predict(z, verbose=0)
            g_loss = self.train_generator(z)

            progress.set_postfix(
                {
                    "d_loss": trunc(d_loss),
                    "g_loss": trunc(g_loss),
                    "d_acc": trunc(d_acc),
                    "d_fake_acc": trunc(d_fake_acc),
                    "d_real_acc": trunc(d_real_acc),
                }
            )

            self.save_sample(
                iteration=iteration,
                sample_interval=sample_interval,
                initial_sample=initial_sample,
                dataset=dataset,
                d_loss=d_loss,
                g_loss=g_loss,
                d_acc=d_acc,
                d_fake_acc=d_fake_acc,
                d_real_acc=d_real_acc,
            )
            initial_sample = True

            yield iteration

    def save_sample(
        self,
        iteration,
        sample_interval,
        d_loss,
        g_loss,
        d_acc,
        d_fake_acc,
        d_real_acc,
        initial_sample,
        dataset,
    ):
        if (iteration) % sample_interval == 0 or not initial_sample:
            # Save losses and accuracies so they can be plotted after training
            self.losses.append((d_loss, g_loss))
            self.accuracies.append(100.0 * d_acc)
            self.iteration_checkpoints.append(iteration)

            num_images_seen = float(iteration * self.params.batch_size)
            epochs = num_images_seen / dataset.shape[0]

            # Output training progress
            print(
                "%d [D loss: %f, acc.: %.2f%%, fake: %f, real: %f] [G loss: %f] [Epochs: %f, Images seen: %d]"
                % (
                    iteration + 1,
                    d_loss,
                    100.0 * d_acc,
                    d_fake_acc,
                    d_real_acc,
                    g_loss,
                    epochs,
                    num_images_seen,
                )
            )

            # Output a sample of generated image
            self.gan.save_weights(self.params.weight_path)
            save_iterations(self.params.iteration_path, iteration + 1)
            file_name = get_current_time()
            show_samples(
                self.generator,
                self.params.latent_dim,
                file_name=file_name,
                dir=self.params.prediction_path,
            )
            show_loss_plot(
                self.losses,
                self.iteration_checkpoints,
                dir=self.params.prediction_path,
                file_name=file_name,
            )
            show_accuracy_plot(
                self.accuracies,
                self.iteration_checkpoints,
                dir=self.params.prediction_path,
                file_name=file_name,
            )


class TrainBCE(Train):
    def __init__(self, built_model: BuiltModel, params: HyperParams) -> None:
        super().__init__(built_model, params)
        self.loss = Loss(params)

    def train_discriminator(self, gen_imgs, real_imgs):
        real = np.ones((self.params.batch_size, 1))
        fake = np.zeros((self.params.batch_size, 1))
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(real_imgs, real)
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_real_acc, d_fake_acc)
        return d_loss, d_loss_fake, d_loss_real, d_acc, d_fake_acc, d_real_acc

    def train_generator(self, z):
        real = np.ones((self.params.batch_size, 1))
        g_loss = self.gan.train_on_batch(z, real)
        return g_loss


class TrainBCESimilarity(Train):
    """
    BCE with an additional loss based on cosine similarity
    """

    def __init__(self, built_model: BuiltModel, params: HyperParams) -> None:
        super().__init__(built_model, params)
        self.loss = Loss(params)

    def train_discriminator(self, gen_imgs, real_imgs):
        real = np.ones((self.params.batch_size, 1))
        fake = np.zeros((self.params.batch_size, 1))
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(real_imgs, real)
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_real_acc, d_fake_acc)
        return d_loss, d_loss_fake, d_loss_real, d_acc, d_fake_acc, d_real_acc

    @tf.function
    def train_generator(self, z):
        """
        Need a custom train loop for the generator because I want to factor in generators predictions
        """
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(z, training=True)

            # Get the discriminator's predictions on the fake images
            fake_preds = self.discriminator(generated_images, training=False)

            # Calculate the loss using the generator's output (generated_images)
            # and the discriminator's predictions (fake_preds)
            loss = self.loss.bce_similarity_loss(fake_preds, generated_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.generator.trainable_weights)

        # Update the weights of the generator
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return loss


class TrainMSE(Train):
    def __init__(self, built_model: BuiltModel, params: HyperParams) -> None:
        super().__init__(built_model, params)
        self.loss = Loss(params)

    def train_discriminator(self, gen_imgs, real_imgs):
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(real_imgs, np.ones((self.params.batch_size, 1)))
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(gen_imgs, -np.ones((self.params.batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_real_acc, d_fake_acc)

        return d_loss, d_loss_fake, d_loss_real, d_acc, d_fake_acc, d_real_acc

    def train_generator(self, z):
        real = np.ones((self.params.batch_size, 1))
        g_loss = self.gan.train_on_batch(z, real)
        return g_loss

    @tf.function
    def _train_generator_similarity(self, z):
        """
        Need a custom train loop for the generator because I want to factor in generators predictions
        """
        misleading_labels = np.zeros((self.params.batch_size, 1))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(z, training=True)

            # Get the discriminator's predictions on the fake images
            predictions = self.discriminator(generated_images, training=False)

            g_loss = tf.keras.losses.MSE(misleading_labels, predictions)
            worst_similarity = self.loss.worst_cosine_similarity(generated_images[:8])
            additional_loss = self.loss.similarity_penalty_loss(worst_similarity)
            g_loss += additional_loss

        grads = tape.gradient(g_loss, self.generator.trainable_weights)

        # Update the weights of the generator
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss
