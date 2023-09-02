import tensorflow as tf
import json
import numpy as np
from thumbs.model.model import BuiltModel
from thumbs.util import is_colab
from thumbs.viz import show_loss_plot, visualize_thumbnails
from thumbs.params import HyperParams, MutableHyperParams, TurnMode
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


def save_model(model: tf.keras.Model, weight_path: str, iterations: int):
    pathlib.Path(weight_path).mkdir(parents=True, exist_ok=True)
    model.save_weights(weight_path)


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
    losses: Dict[str, List[float]]

    def __init__(
        self,
        built_model: BuiltModel,
        params: HyperParams,
        mparams: MutableHyperParams,
        label_getter: LabelGetter = None,
        input_mapper: InputMapper = DefaultInputMapper(),
    ) -> None:
        self.mparams = mparams
        self.generator = built_model.generator
        self.label_getter = label_getter
        self.discriminator = built_model.discriminator
        self.discriminator_optimizer = built_model.discriminator_optimizer
        self.generator_optimizer = built_model.generator_optimizer
        self.input_mapper = input_mapper
        self.params = params
        self.loss = Loss(params)

        losses = load_from_json(self.params.loss_path) or {}
        if isinstance(losses, list):
            # The older format was a List[Tuple[flaot, float]] so those have to be converted to the dict format.
            # The floats were the discriminator and generator loss, respectively.
            print(f"Converting {len(losses)} losses to dict format")
            self.losses = {
                "Discriminator Loss": [l[0] for l in losses],
                "Generator Loss": [l[1] for l in losses],
            }
        elif isinstance(losses, dict):
            self.losses = losses
        else:
            raise Exception(f"Unknown loss format: {losses}")

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
        retored_losses_count: int
        if len(self.losses) == 0:
            retored_losses_count = 0
        else:
            retored_losses_count = len(self.losses[list(self.losses.keys())[0]])

        print(f"Restored losses: {retored_losses_count}")
        print(f"Restored iterations: {len(self.iteration_checkpoints)}")
        print("------------------------------------------------------------")

    @abstractmethod
    def get_loss_plot(self, losses: Dict[str, Union[float, tf.Tensor]]) -> Dict[str, float]:
        raise NotImplementedError()

    @abstractmethod
    def train_body(self, data: tuple, dataset: tf.data.Dataset) -> Dict[str, Union[float, tf.Tensor]]:
        """
        Returns a dict of loss and loss components. For a gan, copmponents might
        include the real/fake loss and the discriminator/generator loss. Other modles might just have one.
        """
        raise NotImplementedError()

    def apply_discriminator_gradients(self, gradient) -> None:
        if self.mparams.discriminator_training:
            self.discriminator_optimizer.apply_gradients(zip(gradient, self.discriminator.trainable_variables))

    def apply_generator_gradients(self, gradient) -> None:
        if self.mparams.generator_training:
            self.generator_optimizer.apply_gradients(zip(gradient, self.generator.trainable_variables))

    def train(self, dataset: tf.data.Dataset, start_iter=0):
        load_weights(self.generator, self.params.gen_weight_path)
        load_weights(self.discriminator, self.params.dis_weight_path)

        loss: Dict[str, List[float]] = {}

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
                losses = self.train_body(item, dataset)

                progress.set_postfix(losses)
                for k, v in self.get_loss_plot(losses).items():
                    if k not in loss:
                        loss[k] = []
                    loss[k].append(float(v))

            updated = self.save_sample(
                iteration=iteration,
                loss=loss,
            )

            if is_colab():
                print(losses)

            if updated:
                loss = {}

            yield iteration

    def save_sample(
        self,
        iteration: int,
        loss: Dict[str, List[float]],
    ) -> bool:
        sample_interval = self.mparams.sample_interval
        checkpoint_path = self.params.checkpoint_path
        checkpoint_interval = self.mparams.checkpoint_interval

        if checkpoint_path is not None and checkpoint_interval is not None and (iteration) % checkpoint_interval == 0:
            save_iterations(f"{checkpoint_path}/{iteration}/iteration", iteration)
            self.generator.save_weights(f"{checkpoint_path}/{iteration}/weights_gen")
            self.discriminator.save_weights(f"{checkpoint_path}/{iteration}/weights_dis")

            loss_file_name = self.params.loss_path.split("/")[-1]
            save_as_json(self.losses, f"{checkpoint_path}/{iteration}/{loss_file_name}")
            iter_checkpoint_file_name = self.params.iteration_checkpoints_path.split("/")[-1]
            save_as_json(self.iteration_checkpoints, f"{checkpoint_path}/{iteration}/{iter_checkpoint_file_name}")

        if (iteration) % sample_interval == 0:
            # Save losses and accuracies so they can be plotted after training
            self.generator.save_weights(self.params.gen_weight_path)
            self.discriminator.save_weights(self.params.dis_weight_path)
            save_iterations(self.params.iteration_path, iteration)

            self.iteration_checkpoints.append(iteration)

            # The content of loss should be turned into a Dict[str, float] by taking the mean of the values, and then
            # Each one shoudl be appended to the values stored in self.losses
            for k, v in loss.items():
                if k not in self.losses:
                    self.losses[k] = []
                self.losses[k].append(np.mean(v, axis=0).tolist())

            # save_as_json(self.accuracies, self.params.accuracy_path)
            save_as_json(self.losses, self.params.loss_path)
            save_as_json(self.iteration_checkpoints, self.params.iteration_checkpoints_path)

            file_name = str(iteration)
            self.show_samples(file_name=file_name)
            show_loss_plot(
                self.losses,
                self.iteration_checkpoints,
                dir=self.params.prediction_path,
                file_name=file_name,
            )

            # Show the plot again but only use the last 50 items for each series
            most_recent = {k: v[-50:] for k, v in self.losses.items()}

            show_loss_plot(
                most_recent, self.iteration_checkpoints[-50:], dir=self.params.prediction_path, file_name="zoom", save_as_latest=False
            )

            return True
        else:
            return False

    def show_samples(self, file_name=None, rows=6, cols=6):
        noise = self.params.latent_sample(rows * cols)
        if self.label_getter is not None:
            labels = self.label_getter(rows * cols)
            generated_thumbnails = self.generator.predict((noise, *labels), verbose=0)
        else:
            generated_thumbnails = self.generator.predict(noise, verbose=0)

        dir = self.params.prediction_path
        visualize_thumbnails(generated_thumbnails, rows, cols, dir, file_name)


class TrainGAN(Train):
    @abstractmethod
    def train_discriminator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def train_generator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        raise NotImplementedError()

    def train_body(self, item: tuple, dataset: tf.data.Dataset) -> Dict[str, Union[float, tf.Tensor]]:
        # -------------------------
        #  Train the Discriminator
        # -------------------------
        for d_turn in range(self.mparams.discriminator_turns):
            cur_item: tuple = item

            if d_turn > 0 and self.mparams.discriminator_turns_mode == TurnMode.NEW_SAMMPLES:
                # Get a new random shuffle from the dataset for multiple turns
                cur_item = dataset.__iter__().__next__()

            z = self.params.latent_sample(self.mparams.batch_size)
            d_loss, d_other = self.train_discriminator(z, cur_item)

        # ---------------------
        #  Train the Generator
        # ---------------------
        for g_turn in range(self.mparams.generator_turns):
            cur_item = item
            z = self.params.latent_sample(self.mparams.batch_size)

            if g_turn > 0 and self.mparams.generator_turns_mode == TurnMode.NEW_SAMMPLES:
                # Get a new random shuffle from the dataset for multiple turns
                cur_item = dataset.__iter__().__next__()

            g_loss, g_other = self.train_generator(z, cur_item)

        # postfix = to_postfix(d_loss, g_loss, d_other, g_other)
        # progress.set_postfix(postfix)
        # loss.append((float(d_loss), float(g_loss)))

        d_other_float = {k: float(v) for k, v in d_other.items()}
        g_other_float = {k: float(v) for k, v in g_other.items()}
        return {
            "d_loss": float(d_loss),
            "g_loss": float(g_loss),
            **d_other_float ,
            **g_other_float,
        }

    def get_loss_plot(self, losses: Dict[str, Union[float, tf.Tensor]]) -> Dict[str, float]:
        return {
            "Discriminator Loss": float(losses["d_loss"]),
            "Generator Loss": float(losses["g_loss"]),
        }


class TrainWassersteinGP(TrainGAN):
    def gradient_penalty(self, fake_images, data: tuple):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        alpha = tf.random.normal([self.mparams.batch_size, 1, 1, 1], 0.0, 1.0)
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
    def train_discriminator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        generator_input = self.input_mapper.get_generator_input(data, z)
        real_input = self.input_mapper.get_discriminator_input_real(data)
        with tf.GradientTape() as tape:
            gen_imgs = self.generator(generator_input, training=True)
            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            # Get the logits for the fake images
            real_logits = self.discriminator(real_input, training=True)
            fake_logits = self.discriminator(fake_input, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost, other = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            if self.mparams.gradient_penalty_factor > 0:
                gp = self.gradient_penalty(gen_imgs, data) * self.mparams.gradient_penalty_factor
            else:
                gp = 0

            d_loss = d_cost + gp

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.apply_discriminator_gradients(d_gradient)
        return d_loss, other

    def generator_loss(self, disc_output, gen_output, target):
        loss = -tf.reduce_mean(disc_output)  # normal loss for wgan
        losses = {"g_mean_loss": loss}
        total_gen_loss = loss

        if self.mparams.l1_loss_factor is not None:
            l1_loss = self.mparams.l1_loss_factor * tf.reduce_mean(tf.abs(target - gen_output))
            losses["g_l1_loss"] = l1_loss
            total_gen_loss += l1_loss
        if self.mparams.l2_loss_factor is not None:
            l2_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.square(target - gen_output))
            losses["g_l2_loss"] = l2_loss
            total_gen_loss += l2_loss

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

        # Update the weights of the generator using the generator optimizer
        self.apply_generator_gradients(gen_gradient)
        return g_loss, other


class TrainBCE(TrainGAN):
    """
    BCE with an additional loss based on cosine similarity
    """

    def train_discriminator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        generator_input = self.input_mapper.get_generator_input(data, z)
        real = np.ones(self.mparams.discriminator_ones_zeroes_shape)
        fake = np.zeros((self.mparams.discriminator_ones_zeroes_shape))

        with tf.GradientTape() as tape:
            gen_imgs = self.generator(generator_input, training=True)
            real_input = self.input_mapper.get_discriminator_input_real(data)
            real_output = self.discriminator(real_input, training=True)
            d_loss_real = tf.keras.losses.BinaryCrossentropy()(real, real_output)

            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            fake_output = self.discriminator(fake_input, training=True)
            d_loss_fake = tf.keras.losses.BinaryCrossentropy()(fake, fake_output)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Get the gradients
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

        # Apply the gradients
        self.apply_discriminator_gradients(grads)

        d_real_acc = tf.reduce_mean(tf.cast(tf.equal(real, tf.round(real_output)), tf.float32))
        d_fake_acc = tf.reduce_mean(tf.cast(tf.equal(fake, tf.round(fake_output)), tf.float32))
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
        losses = {"g_bce_loss": gan_loss}
        total_gen_loss = gan_loss

        if self.mparams.l1_loss_factor is not None:
            l1_loss = self.mparams.l1_loss_factor * tf.reduce_mean(tf.abs(target - gen_output))
            losses["g_l1_loss"] = l1_loss
            total_gen_loss += l1_loss
        if self.mparams.l2_loss_factor is not None:
            l2_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.square(target - gen_output))
            losses["g_l2_loss"] = l2_loss
            total_gen_loss += l2_loss

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

        # Update the weights of the generator
        self.apply_generator_gradients(grads)
        return loss, other


class TrainHinge(TrainGAN):
    @tf.function
    def train_discriminator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        generator_input = self.input_mapper.get_generator_input(data, z)
        with tf.GradientTape() as disc_tape:
            gen_imgs = self.generator(generator_input, training=True)
            real_input = self.input_mapper.get_discriminator_input_real(data)
            real_output = self.discriminator(real_input, training=True)

            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            fake_output = self.discriminator(fake_input, training=True)

            d_real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
            d_fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
            d_loss = d_real_loss + d_fake_loss

        # Calculate the gradients
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        # Apply the gradients
        self.apply_discriminator_gradients(gradients_of_discriminator)

        losses = {
            "d_loss_fake": d_fake_loss,
            "d_loss_real": d_real_loss,
        }

        return d_loss, losses

    def generator_loss(self, disc_generated_output, gen_output, target):
        gen_loss = -tf.reduce_mean(disc_generated_output)  # We're just changing this to hinge loss, bro
        losses = {"g_loss": gen_loss}
        total_gen_loss = gen_loss

        if self.mparams.l1_loss_factor is not None:
            l1_loss = self.mparams.l1_loss_factor * tf.reduce_mean(tf.abs(target - gen_output))
            losses["g_l1_loss"] = l1_loss
            total_gen_loss += l1_loss
        if self.mparams.l2_loss_factor is not None:
            l2_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.square(target - gen_output))
            losses["g_l2_loss"] = l2_loss
            total_gen_loss += l2_loss

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

        # Update the weights of the generator
        self.apply_generator_gradients(grads)
        return loss, other


class TrainBCEPatch(TrainGAN):
    """
    BCE based on patch output logits from the discriminator, not a single sigmoid output.
    """

    @tf.function
    def train_discriminator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        generator_input = self.input_mapper.get_generator_input(data, z)
        real = np.ones(self.mparams.discriminator_ones_zeroes_shape)
        fake = np.zeros((self.mparams.discriminator_ones_zeroes_shape))

        with tf.GradientTape() as disc_tape:
            gen_imgs = self.generator(generator_input, training=True)
            real_input = self.input_mapper.get_discriminator_input_real(data)
            real_output = self.discriminator(real_input, training=True)
            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real, real_output, from_logits=True))

            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            fake_output = self.discriminator(fake_input, training=True)
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake, fake_output, from_logits=True))

            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.apply_discriminator_gradients(gradients_of_discriminator)

        d_real_acc = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, 0.0), tf.float32))
        d_fake_acc = tf.reduce_mean(tf.cast(tf.math.less(fake_output, 0.0), tf.float32))
        d_acc = 0.5 * (d_real_acc + d_fake_acc)

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
        losses = {"g_bce_loss": gan_loss}
        total_gen_loss = gan_loss

        if self.mparams.l1_loss_factor is not None:
            l1_loss = self.mparams.l1_loss_factor * tf.reduce_mean(tf.abs(target - gen_output))
            losses["g_l1_loss"] = l1_loss
            total_gen_loss += l1_loss
        if self.mparams.l2_loss_factor is not None:
            l2_loss = self.mparams.l2_loss_factor * tf.reduce_mean(tf.square(target - gen_output))
            losses["g_l2_loss"] = l2_loss
            total_gen_loss += l2_loss

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

        # Update the weights of the generator
        self.apply_generator_gradients(grads)
        return loss, other
