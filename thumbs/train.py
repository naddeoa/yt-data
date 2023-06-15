import tensorflow as tf
import numpy as np
from thumbs.util import get_current_time
from thumbs.viz import show_accuracy_plot, show_loss_plot, show_samples
from thumbs.params import HyperParams
from thumbs.loss import Loss
from typing import List, Optional


from tqdm import tqdm


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


class Train:
    def __init__(
        self, gan, generator, discriminator, generator_optimizer, params: HyperParams
    ) -> None:
        self.gan = gan
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.params = params
        self.loss = Loss(params)

        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.iteration_checkpoints: List[int] = []

    @tf.function
    def _train_generator(self, z):
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
            loss = self.loss.custom_generator_loss(fake_preds, generated_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.generator.trainable_weights)

        # Update the weights of the generator
        self.generator_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )
        return loss

    def train(self, dataset, iterations, sample_interval, start_iter=0):
        load_weights(self.gan, self.params.weight_path)

        if start_iter > iterations:
            raise Exception(
                f"Checkpointed at iteration {start_iter} but only training for {iterations} iterations"
            )

        # Labels for real images: all ones
        real = np.ones((self.params.batch_size, 1))

        # Labels for fake images: all zeros
        fake = np.zeros((self.params.batch_size, 1))
        initial_sample = False

        for iteration in tqdm(range(start_iter, iterations), total=iterations, initial=start_iter):
            # -------------------------
            #  Train the Discriminator
            # -------------------------

            # Get a random batch of real images
            idx = np.random.randint(0, dataset.shape[0], self.params.batch_size)
            imgs = dataset[idx]

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (self.params.batch_size, self.params.latent_dim))
            gen_imgs = self.generator.predict(z, verbose=0)

            # Train Discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train the Generator
            # ---------------------

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (self.params.batch_size, self.params.latent_dim))
            gen_imgs = self.generator.predict(z, verbose=0)

            # Train Generator
            # g_loss = gan.train_on_batch(z, real)
            g_loss = self._train_generator(z)

            if (iteration + 1) % sample_interval == 0 or not initial_sample:
                initial_sample = True
                # Save losses and accuracies so they can be plotted after training
                self.losses.append((d_loss, g_loss))
                self.accuracies.append(100.0 * accuracy)
                self.iteration_checkpoints.append(iteration + 1)

                # Output training progress
                print(
                    "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                    % (iteration + 1, d_loss, 100.0 * accuracy, g_loss)
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

            yield iteration
