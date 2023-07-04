import tensorflow as tf
import json
from tensorflow.keras import backend as K
import numpy as np
from thumbs.model.model import BuiltModel
from thumbs.util import get_current_time, is_colab
from thumbs.viz import show_accuracy_plot, show_loss_plot, show_samples
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.loss import Loss
from typing import Iterable, List, Optional, Tuple, Any, Optional, Callable
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


def save_as_json(l: Any, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(l, f)


def load_from_json(path: str) -> Optional[Any]:
    # load something that was serialized with save_as_json
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
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


LabelGetter = Optional[Callable[[int], np.ndarray]]
class Train(ABC):
    # label_getter is a function that takes a number and returns an array of np.ndarray
    def __init__(self, built_model: BuiltModel, params: HyperParams, mparams: MutableHyperParams, label_getter: LabelGetter  = None) -> None:
        self.gan = built_model.gan
        self.mparams = mparams
        self.label_getter = label_getter
        self.generator = built_model.generator
        self.discriminator = built_model.discriminator
        self.discriminator_optimizer  = built_model.discriminator_optimizer
        self.generator_optimizer = built_model.generator_optimizer
        self.params = params
        self.loss = Loss(params)

        self.losses: List[Tuple[float, float]] = load_from_json(self.params.loss_path) or []
        self.accuracies: List[float] = load_from_json(self.params.accuracy_path) or []
        self.iteration_checkpoints: List[int] = load_from_json(self.params.iteration_checkpoints_path) or []
        self.accuracies_rf: List[Tuple[float, float]] = []

        print("------------------------------------------------------------")
        print("PARAMS")
        print(params)

        print()
        print("MUTABLE PARAMS")
        print(mparams)

        print()
        print(f"Restored losses: {len(self.losses)}")
        print(f"Restored iterations: {len(self.iteration_checkpoints)}")
        print("------------------------------------------------------------")


    @abstractmethod
    def train_discriminator(self, gen_imgs, real_imgs, labels=None):
        raise NotImplementedError()

    @abstractmethod
    def train_generator(self, z, labels=None):
        raise NotImplementedError()

    def train(self, dataset: tf.data.Dataset, start_iter=0):
        if self.gan is not None:
            load_weights(self.gan, self.params.weight_path)
        else:
            load_weights(self.generator, f'{self.params.weight_path}_gen')
            load_weights(self.discriminator, f'{self.params.weight_path}_dis')


        accuracies_rf = []
        loss_dg: List[Tuple[float, float]] = []

        if not self.mparams.discriminator_training:
            self.discriminator.trainable = False
            self.discriminator.compile()

        if not self.mparams.generator_training:
            self.generator.trainable = False
            self.generator.compile()

        progress = tqdm(
            range(start_iter, self.mparams.iterations + 1), total=self.mparams.iterations, initial=start_iter, position=0, leave=True, desc="epoch"
        )
        for iteration in progress:
            # Google colab can't handle two progress bars, so overwrite the epoch one each time.
            for item in tqdm(dataset, position=1 if not is_colab() else 0, leave=False if not is_colab() else True, desc="batch"):
                if isinstance(item, tuple):
                    imgs, labels = item
                else:
                    imgs = item
                    labels = None

                # -------------------------
                #  Train the Discriminator
                # -------------------------
                # Generate a batch of fake images
                for d_turn in range(self.mparams.discriminator_turns):
                    cur_imgs = imgs
                    cur_labels = labels

                    z = np.random.normal(0, 1, (self.mparams.batch_size, self.params.latent_dim))
                    gen_input = z if cur_labels is None else [z, cur_labels]
                    gen_imgs = self.generator.predict(gen_input , verbose=0)

                    if d_turn > 0:
                        # Get a new random shuffle from the dataset for multiple turns
                        next_imgs = dataset.__iter__().__next__()
                        if isinstance(next_imgs, tuple):
                            cur_imgs, cur_labels = next_imgs 
                        else:
                            cur_imgs = next_imgs
                            cur_labels = None

                    (
                        d_loss,
                        d_loss_fake,
                        d_loss_real,
                        d_acc,
                        d_fake_acc,
                        d_real_acc,
                    ) = self.train_discriminator(gen_imgs, cur_imgs, cur_labels)

                # ---------------------
                #  Train the Generator
                # ---------------------
                # gen_imgs = self.generator.predict(z, verbose=0)
                for _ in range(self.mparams.generator_turns):
                    z = np.random.normal(0, 1, (self.mparams.batch_size, self.params.latent_dim))
                    # TODO make the generator get new labels on each iteration too, but I may never train the generator > 1 anyway
                    g_loss = self.train_generator(z, labels)

                postfix = {
                    "d_loss": trunc(d_loss),
                    "g_loss": trunc(g_loss),
                    "d_acc": trunc(d_acc),
                    "d_fake_acc": trunc(d_fake_acc),
                    "d_real_acc": trunc(d_real_acc),
                }

                progress.set_postfix(postfix)

                # accuracies.append(d_acc)
                accuracies_rf.append((d_real_acc, d_fake_acc))
                loss_dg.append((d_loss, g_loss))

            updated = self.save_sample(
                iteration=iteration,
                loss_dg=loss_dg,
                accuracies_rf=accuracies_rf,
            )
            if is_colab():
                print(postfix)
            if updated:
                accuracies_rf = []
                loss_dg = []

            yield iteration

    def save_sample(
        self,
        iteration: int,
        loss_dg: List[Tuple[float, float]],
        accuracies_rf: List[Tuple[float, float]],
    ) -> bool:
        sample_interval = self.mparams.sample_interval
        checkpoint_path = self.params.checkpoint_path
        checkpoint_interval = self.mparams.checkpoint_interval

        if checkpoint_path is not None and checkpoint_interval is not None and (iteration) % checkpoint_interval == 0:
            save_iterations(f"{checkpoint_path}/{iteration}/iteration", iteration)
            if self.gan is not None:
                self.gan.save_weights(f"{checkpoint_path}/{iteration}/weights")
            else:
                self.generator.save_weights(f"{checkpoint_path}/{iteration}/weights_gen")
                self.discriminator.save_weights(f"{checkpoint_path}/{iteration}/weights_dis")

            loss_file_name = self.params.loss_path.split("/")[-1]
            save_as_json(self.losses, f"{checkpoint_path}/{iteration}/{loss_file_name}")
            iter_checkpoint_file_name = self.params.iteration_checkpoints_path.split("/")[-1]
            save_as_json(self.iteration_checkpoints, f"{checkpoint_path}/{iteration}/{iter_checkpoint_file_name}")


        if (iteration) % sample_interval == 0:
            # Save losses and accuracies so they can be plotted after training
            if self.gan is not None:
                self.gan.save_weights(self.params.weight_path)
            else:
                self.generator.save_weights(f'{self.params.weight_path}_gen')
                self.discriminator.save_weights(f'{self.params.weight_path}_dis')
            save_iterations(self.params.iteration_path, iteration)

            self.losses.append(np.mean(loss_dg, axis=0).tolist())
            self.accuracies_rf.append(np.mean(accuracies_rf, axis=0).tolist())
            self.iteration_checkpoints.append(iteration)

            # save_as_json(self.accuracies, self.params.accuracy_path)
            save_as_json(self.losses, self.params.loss_path)
            save_as_json(self.iteration_checkpoints, self.params.iteration_checkpoints_path)

            file_name = str(iteration)
            show_samples(
                self.generator,
                self.params.latent_dim,
                file_name=file_name,
                dir=self.params.prediction_path,
                label_getter=self.label_getter,
            )
            show_loss_plot(
                self.losses,
                self.iteration_checkpoints,
                dir=self.params.prediction_path,
                file_name=file_name,
            )

            # most recent data only
            show_loss_plot(
                self.losses[-50:],
                self.iteration_checkpoints[-50:],
                dir=self.params.prediction_path,
                file_name='zoom',
                save_as_latest=False
            )

            return True
        else:
            return False


class TrainWassersteinGP(Train):
    def __init__(self, built_model: BuiltModel, params: HyperParams, mparams: MutableHyperParams, label_getter: LabelGetter = None) -> None:
        super().__init__(built_model, params, mparams, label_getter )
        self.loss = Loss(params)

    def gradient_penalty(self, real_images, fake_images, labels=None):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([self.mparams.batch_size, 1, 1, 1], 0.0, 1.0)
        # alpha = tf.random.uniform(shape=[self.mparams.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)

        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            disc_input = interpolated if labels is None else [interpolated, labels]
            pred = self.discriminator(disc_input, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss


    @tf.function
    def train_discriminator(self, gen_imgs, real_imgs, labels):
        # Get the latent vector
        with tf.GradientTape() as tape:
            # Get the logits for the fake images
            disc_gen_inputs = gen_imgs if labels is None else [gen_imgs, labels]
            fake_logits = self.discriminator(disc_gen_inputs, training=True)
            # Get the logits for the real images
            disc_real_inputs = real_imgs if labels is None else [real_imgs, labels]
            real_logits = self.discriminator(disc_real_inputs, training=True)
            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(real_imgs, gen_imgs, labels)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + (gp * self.mparams.gradient_penalty_factor)

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.discriminator_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )
        return d_loss, 0, 0, 0, 0, 0


    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

    @tf.function
    def train_generator(self, z, labels):
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generator_input = z if labels is None else [z, labels]
            generated_images = self.generator(generator_input , training=True)
            # Get the discriminator logits for fake images
            discrim_input = generated_images if labels is None else [generated_images, labels]
            gen_img_logits = self.discriminator(discrim_input, training=True)
            # Calculate the generator loss
            g_loss = self.generator_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return g_loss









class TrainBCE(Train):
    def __init__(self, built_model: BuiltModel, params: HyperParams, mparams: MutableHyperParams) -> None:
        super().__init__(built_model, params, mparams)
        self.loss = Loss(params)

    def train_discriminator(self, gen_imgs, real_imgs):
        real = np.ones((self.mparams.batch_size, 1))
        fake = np.zeros((self.mparams.batch_size, 1))
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(real_imgs, real)
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_real_acc, d_fake_acc)
        return d_loss, d_loss_fake, d_loss_real, d_acc, d_fake_acc, d_real_acc

    def train_generator(self, z):
        real = np.ones((self.mparams.batch_size, 1))
        g_loss = self.gan.train_on_batch(z, real)
        return g_loss


class TrainBCESimilarity(Train):
    """
    BCE with an additional loss based on cosine similarity
    """

    def __init__(self, built_model: BuiltModel, params: HyperParams, mparams: MutableHyperParams) -> None:
        super().__init__(built_model, params, mparams)
        self.loss = Loss(params)

    def train_discriminator(self, gen_imgs, real_imgs):
        real = np.ones((self.mparams.batch_size, 1))
        fake = np.zeros((self.mparams.batch_size, 1))
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
            fake_preds = self.discriminator(generated_images, training=True)

            # Calculate the loss using the generator's output (generated_images)
            # and the discriminator's predictions (fake_preds)
            loss = self.loss.bce_similarity_loss(fake_preds, generated_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.generator.trainable_weights)

        if self.params.generator_clip_gradients_norm is not None:
            grads = [tf.clip_by_norm(grad, self.params.generator_clip_gradients_norm) for grad in grads]

        # Update the weights of the generator
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return loss


class TrainMSE(Train):
    def __init__(self, built_model: BuiltModel, params: HyperParams, mparams: MutableHyperParams) -> None:
        super().__init__(built_model, params, mparams)
        self.loss = Loss(params)

    def train_discriminator(self, gen_imgs, real_imgs):
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(real_imgs, np.ones((self.mparams.batch_size, 1)))
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(gen_imgs, -np.ones((self.mparams.batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_real_acc, d_fake_acc)

        return d_loss, d_loss_fake, d_loss_real, d_acc, d_fake_acc, d_real_acc

    def train_generator(self, z):
        real = np.ones((self.mparams.batch_size, 1))
        g_loss = self.gan.train_on_batch(z, real)
        return g_loss

    @tf.function
    def _train_generator_similarity(self, z):
        """
        Need a custom train loop for the generator because I want to factor in generators predictions
        """
        misleading_labels = np.zeros((self.mparams.batch_size, 1))
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
