import tensorflow as tf
import json
import numpy as np
from thumbs.model.model import BuiltModel
from thumbs.util import is_colab
from thumbs.viz import show_loss_plot, show_samples
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.loss import Loss
from typing import List, Optional, Tuple, Any, Optional, Callable, Dict, Union
from abc import ABC, abstractmethod
import pathlib
import os
from tqdm import tqdm


def trunc(loss: Union[float, tf.Tensor, np.float_]) -> str:
    """Truncate loss values for display in tqdm"""

    _loss: Union[float, np.float_, np.ndarray]
    if isinstance(loss, tf.Tensor):
        _loss = loss.numpy()
    else:
        _loss = loss

    return f"{_loss:.4f}"


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
    with open(path, "w") as f:
        json.dump(l, f)


def load_from_json(path: str) -> Optional[Any]:
    # load something that was serialized with save_as_json
    try:
        with open(path, "r") as f:
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


def to_postfix(d_loss: tf.Tensor, g_loss: tf.Tensor, d_other: Dict[str, tf.Tensor], g_other: Dict[str, tf.Tensor]) -> Dict[str, str]:
    postfix = {k: trunc(v) for k, v in {**d_other, **g_other}.items()}
    postfix["d_loss"] = trunc(d_loss)
    postfix["g_loss"] = trunc(g_loss)
    return postfix


class InputMapper(ABC):
    @abstractmethod
    def get_real_images(self, data_batch: Union[tuple, tf.Tensor]) -> tf.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_generator_input(self, data_batch: Union[tuple, tf.Tensor], noise: np.ndarray) -> Union[tuple, tf.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def get_discriminator_input_real(self, data_batch: Union[tuple, tf.Tensor]) -> Union[tuple, tf.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def get_discriminator_input_fake(self, data_batch: Union[tuple, tf.Tensor], generated_imgs: np.ndarray) -> Union[tuple, tf.Tensor]:
        raise NotImplementedError()


class DefaultInputMapper(InputMapper):
    def get_real_images(self, data_batch: Union[tuple, tf.Tensor]) -> tf.Tensor:
        if not isinstance(data_batch, tuple):
            return data_batch

        t: tf.Tensor = data_batch[0]
        return t

    def get_generator_input(self, data_batch: Union[tuple, tf.Tensor], noise: np.ndarray) -> Union[tuple, tf.Tensor]:
        if not isinstance(data_batch, tuple) or len(data_batch) == 1:
            return (noise,)

        return (noise,) + data_batch[1:]

    def get_discriminator_input_real(self, data_batch: Union[tuple, tf.Tensor]) -> Union[tuple, tf.Tensor]:
        return data_batch

    def get_discriminator_input_fake(self, data_batch: Union[tuple, tf.Tensor], generated_imgs: np.ndarray) -> Union[tuple, tf.Tensor]:
        if not isinstance(data_batch, tuple) or len(data_batch) == 1:
            return (generated_imgs,)

        return (generated_imgs,) + data_batch[1:]


class Train(ABC):
    # label_getter is a function that takes a number and returns an array of np.ndarray
    def __init__(
        self,
        built_model: BuiltModel,
        params: HyperParams,
        mparams: MutableHyperParams,
        label_getter: LabelGetter = None,
        input_mapper: InputMapper = DefaultInputMapper(),
    ) -> None:
        self.gan = built_model.gan
        self.mparams = mparams
        self.generator = built_model.generator
        self.label_getter = label_getter
        self.discriminator = built_model.discriminator
        self.discriminator_optimizer = built_model.discriminator_optimizer
        self.generator_optimizer = built_model.generator_optimizer
        self.input_mapper = input_mapper
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

    # TODO make these return a (tensor, dict) where the dict has all of the components of the loss
    @abstractmethod
    def train_discriminator(self, gen_imgs, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def train_generator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        raise NotImplementedError()

    def train(self, dataset: tf.data.Dataset, start_iter=0):
        if self.gan is not None:
            load_weights(self.gan, self.params.weight_path)
        else:
            load_weights(self.generator, f"{self.params.weight_path}_gen")
            load_weights(self.discriminator, f"{self.params.weight_path}_dis")

        loss_dg: List[Tuple[float, float]] = []

        if not self.mparams.discriminator_training:
            self.discriminator.trainable = False
            self.discriminator.compile()

        if not self.mparams.generator_training:
            self.generator.trainable = False
            self.generator.compile()

        progress = tqdm(
            range(start_iter, self.mparams.iterations + 1),
            total=self.mparams.iterations,
            initial=start_iter,
            position=0,
            leave=True,
            desc="epoch",
        )
        for iteration in progress:
            # Google colab can't handle two progress bars, so overwrite the epoch one each time.
            for item in tqdm(dataset, position=1 if not is_colab() else 0, leave=False if not is_colab() else True, desc="batch"):
                # -------------------------
                #  Train the Discriminator
                # -------------------------
                for d_turn in range(self.mparams.discriminator_turns):
                    cur_item: tuple = item

                    z = np.random.normal(0, 1, (self.mparams.batch_size, self.params.latent_dim))
                    generator_input = self.input_mapper.get_generator_input(cur_item, z)
                    gen_imgs = self.generator.predict(generator_input, verbose=0)

                    if d_turn > 0:
                        # Get a new random shuffle from the dataset for multiple turns
                        cur_item = dataset.__iter__().__next__()

                    d_loss, d_other = self.train_discriminator(gen_imgs, cur_item)

                # ---------------------
                #  Train the Generator
                # ---------------------
                for g_turn in range(self.mparams.generator_turns):
                    cur_item = item
                    z = np.random.normal(0, 1, (self.mparams.batch_size, self.params.latent_dim))

                    if g_turn > 0:
                        # Get a new random shuffle from the dataset for multiple turns
                        cur_item = dataset.__iter__().__next__()

                    g_loss, g_other = self.train_generator(z, cur_item)

                postfix = to_postfix(d_loss, g_loss, d_other, g_other)
                progress.set_postfix(postfix)
                loss_dg.append((float(d_loss), float(g_loss)))

            updated = self.save_sample(
                iteration=iteration,
                loss_dg=loss_dg,
            )
            if is_colab():
                print(postfix)
            if updated:
                loss_dg = []

            yield iteration

    def save_sample(
        self,
        iteration: int,
        loss_dg: List[Tuple[float, float]],
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
                self.generator.save_weights(f"{self.params.weight_path}_gen")
                self.discriminator.save_weights(f"{self.params.weight_path}_dis")
            save_iterations(self.params.iteration_path, iteration)

            self.losses.append(np.mean(loss_dg, axis=0).tolist())
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
                self.losses[-50:], self.iteration_checkpoints[-50:], dir=self.params.prediction_path, file_name="zoom", save_as_latest=False
            )

            return True
        else:
            return False


class TrainWassersteinGP(Train):
    def gradient_penalty(self, fake_images, data: tuple):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([self.mparams.batch_size, 1, 1, 1], 0.0, 1.0)
        # alpha = tf.random.uniform(shape=[self.mparams.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)

        real_images = self.input_mapper.get_real_images(data)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            disc_input = self.input_mapper.get_discriminator_input_fake(data, interpolated)
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
        return fake_loss - real_loss, {"d_loss_real": real_loss, "d_loss_fake": fake_loss}

    @tf.function
    def train_discriminator(self, gen_imgs, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        # def train_discriminator(self, gen_imgs, real_imgs, labels) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        # Get the latent vector
        with tf.GradientTape() as tape:
            # Get the logits for the fake images
            real_input = self.input_mapper.get_discriminator_input_real(data)
            real_logits = self.discriminator(real_input, training=True)

            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            fake_logits = self.discriminator(fake_input, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost, other = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(gen_imgs, data)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + (gp * self.mparams.gradient_penalty_factor)

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.discriminator_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
        return d_loss, other

    def generator_loss(self, disc_output, gen_output, target):
        loss = -tf.reduce_mean(disc_output)  # normal loss for wgan
        l1_loss = self.mparams.l1_loss_factor * tf.reduce_mean(tf.abs(target - gen_output))
        l2_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.square(target - gen_output))
        total_gen_loss = loss + l1_loss + l2_loss

        losses = {"g_mean_loss": loss}
        if self.mparams.l1_loss_factor > 0:
            losses["g_l1_loss"] = l1_loss
        if self.mparams.l2_loss_factor > 0:
            losses["g_l2_loss"] = l2_loss

        return total_gen_loss, losses

    @tf.function
    def train_generator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generator_input = self.input_mapper.get_generator_input(data, z)
            generated_images = self.generator(generator_input, training=True)
            # Get the discriminator logits for fake images
            discrim_input = self.input_mapper.get_discriminator_input_fake(data, generated_images)
            disc_output = self.discriminator(discrim_input, training=True)
            # Calculate the generator loss
            real_images = self.input_mapper.get_real_images(data)
            g_loss, other = self.generator_loss(disc_output, generated_images, real_images)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        if self.params.generator_clip_gradients_norm is not None:
            gen_gradient = [tf.clip_by_norm(grad, self.params.generator_clip_gradients_norm) for grad in gen_gradient]
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return g_loss, other


class TrainBCE(Train):
    """
    BCE with an additional loss based on cosine similarity
    """
    def train_discriminator(self, gen_imgs, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        real = np.ones(self.mparams.discriminator_ones_zeroes_shape)
        fake = np.zeros((self.mparams.discriminator_ones_zeroes_shape))

        real_input = self.input_mapper.get_discriminator_input_real(data)
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(real_input, real)

        fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(fake_input, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_real_acc, d_fake_acc)
        losses = {
            "d_loss_fake": d_loss_fake,
            "d_loss_real": d_loss_real,
            "d_acc": d_acc,
            "d_fake_acc": d_fake_acc,
            "d_real_acc": d_real_acc,
        }
        return d_loss, losses

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.abs(target - gen_output))
        l2_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.square(target - gen_output))

        losses = {"g_bce_loss": gan_loss}
        if self.mparams.l1_loss_factor > 0:
            losses["g_l1_loss"] = l1_loss
        if self.mparams.l2_loss_factor > 0:
            losses["g_l2_loss"] = l2_loss

        total_gen_loss = gan_loss + l1_loss + l2_loss
        return total_gen_loss, losses

    @tf.function
    def train_generator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Need a custom train loop for the generator because I want to factor in generators predictions
        """
        with tf.GradientTape() as tape:
            generator_input = self.input_mapper.get_generator_input(data, z)
            generated_images = self.generator(generator_input, training=True)

            # Get the discriminator's predictions on the fake images
            fake_input = self.input_mapper.get_discriminator_input_fake(data, generated_images)
            fake_preds = self.discriminator(fake_input, training=True)

            # Calculate the loss using the generator's output (generated_images)
            # and the discriminator's predictions (fake_preds)
            real_images = self.input_mapper.get_real_images(data)
            loss, other = self.generator_loss(fake_preds, generated_images, real_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.generator.trainable_weights)

        if self.params.generator_clip_gradients_norm is not None:
            grads = [tf.clip_by_norm(grad, self.params.generator_clip_gradients_norm) for grad in grads]

        # Update the weights of the generator
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return loss, other


class TrainBCEPatch(Train):
    """
    BCE based on patch output logits from the discriminator, not a single sigmoid output.
    """
    def train_discriminator(self, gen_imgs, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        real = np.ones(self.mparams.discriminator_ones_zeroes_shape)
        fake = np.zeros((self.mparams.discriminator_ones_zeroes_shape))

        real_input = self.input_mapper.get_discriminator_input_real(data)
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(real_input, real)

        fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(fake_input, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_real_acc, d_fake_acc)

        losses = {
            "d_loss_fake": d_loss_fake,
            "d_loss_real": d_loss_real,
            "d_acc": d_acc,
            "d_fake_acc": d_fake_acc,
            "d_real_acc": d_real_acc,
        }
        return d_loss, losses

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = self.mparams.l1_loss_factor * tf.reduce_mean(tf.abs(target - gen_output))
        l2_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.square(target - gen_output))
        total_gen_loss = gan_loss + l1_loss + l2_loss

        losses = {"g_bce_loss": gan_loss}
        if self.mparams.l1_loss_factor > 0:
            losses["g_l1_loss"] = l1_loss
        if self.mparams.l2_loss_factor > 0:
            losses["g_l2_loss"] = l2_loss
        return total_gen_loss, losses

    @tf.function
    def train_generator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Need a custom train loop for the generator because I want to factor in generators predictions
        """
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generator_input = self.input_mapper.get_generator_input(data, z)
            generated_images = self.generator(generator_input, training=True)

            # Get the discriminator's predictions on the fake images
            fake_input = self.input_mapper.get_discriminator_input_fake(data, generated_images)
            fake_preds = self.discriminator(fake_input, training=True)

            # Calculate the loss using the generator's output (generated_images)
            # and the discriminator's predictions (fake_preds)
            real_images = self.input_mapper.get_real_images(data)
            loss, other = self.generator_loss(fake_preds, generated_images, real_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.generator.trainable_weights)

        if self.params.generator_clip_gradients_norm is not None:
            grads = [tf.clip_by_norm(grad, self.params.generator_clip_gradients_norm) for grad in grads]

        # Update the weights of the generator
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return loss, other
