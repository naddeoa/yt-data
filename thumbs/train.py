import tensorflow as tf
import numpy as np
from thumbs.model.model import BuiltModel
from thumbs.util import get_current_time
from thumbs.viz import show_accuracy_plot, show_loss_plot, show_samples
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.loss import Loss
from typing import Iterable, List, Optional, Tuple
from abc import ABC, abstractmethod
import pathlib
import os


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
    pathlib.Path(os.path.dirname(iteration_path)).mkdir(parents=True, exist_ok=True)
    with open(iteration_path, "w") as f:
        f.write(str(iterations))


def save_model(gan: tf.keras.Model, weight_path: str, iterations: int):
    pathlib.Path(weight_path).mkdir(parents=True, exist_ok=True)
    gan.save_weights(weight_path)


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

        self.losses: List[Tuple[float, float]] = []
        self.accuracies: List[float] = []
        self.accuracies_rf: List[Tuple[float, float]] = []
        self.iteration_checkpoints: List[int] = []

    @abstractmethod
    def train_discriminator(self, gen_imgs, real_imgs):
        raise NotImplementedError()

    @abstractmethod
    def train_generator(self, z):
        raise NotImplementedError()

    def train(self, dataset: tf.data.Dataset, mparams: MutableHyperParams, start_iter=0):
        load_weights(self.gan, self.params.weight_path)

        # Labels for fake images: all zeros
        initial_sample = False

        accuracies_rf = []
        loss_dg: List[Tuple[float, float]] = []
        progress = tqdm(
            range(start_iter, mparams.iterations + 1), total=mparams.iterations, initial=start_iter, position=0, leave=True, desc="epoch"
        )
        for iteration in progress:
            # batches = dataset.shuffle(buffer_size=1024).batch(self.params.batch_size, drop_remainder=True)
            for imgs in tqdm(dataset, position=1, leave=False, desc="batch"):
                # -------------------------
                #  Train the Discriminator
                # -------------------------
                # Generate a batch of fake images
                for _ in range(mparams.discriminator_turns):
                    z = np.random.normal(0, 1, (self.params.batch_size, self.params.latent_dim))
                    gen_imgs = self.generator.predict(z, verbose=0)
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
                # gen_imgs = self.generator.predict(z, verbose=0)
                for _ in range(mparams.generator_turns):
                    z = np.random.normal(0, 1, (self.params.batch_size, self.params.latent_dim))
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

                # accuracies.append(d_acc)
                accuracies_rf.append((d_real_acc, d_fake_acc))
                loss_dg.append((d_loss, g_loss))

            updated = self.save_sample(
                iteration=iteration,
                initial_sample=initial_sample,
                loss_dg=loss_dg,
                accuracies_rf=accuracies_rf,
                mparams=mparams,
            )
            if updated:
                initial_sample = True

            yield iteration

    def save_sample(
        self,
        iteration: int,
        loss_dg: List[Tuple[float, float]],
        accuracies_rf: List[Tuple[float, float]],
        initial_sample: bool,
        mparams: MutableHyperParams,
    ) -> bool:
        sample_interval = mparams.sample_interval
        checkpoint_path = self.params.checkpoint_path
        checkpoint_interval = mparams.checkpoint_interval

        if checkpoint_path is not None and checkpoint_interval is not None and (iteration) % checkpoint_interval == 0:
            save_iterations(f"{checkpoint_path}/{iteration}/iteration", iteration)
            self.gan.save_weights(f"{checkpoint_path}/{iteration}/weights")

        if (iteration) % sample_interval == 0 or not initial_sample:
            # Save losses and accuracies so they can be plotted after training
            self.losses.append(np.mean(loss_dg, axis=0))
            self.accuracies_rf.append(100.0 * np.mean(accuracies_rf, axis=0))
            self.iteration_checkpoints.append(iteration)

            file_name = str(iteration)
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
                self.accuracies_rf,
                self.iteration_checkpoints,
                dir=self.params.prediction_path,
                file_name=file_name,
            )
            self.gan.save_weights(self.params.weight_path)
            save_iterations(self.params.iteration_path, iteration)

            return True
        else:
            return False


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
