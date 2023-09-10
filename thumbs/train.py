import tensorflow as tf
import json
import numpy as np
from thumbs.model.model import BuiltGANModel, BuiltDiffusionModel
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from thumbs.util import is_colab
from thumbs.viz import show_loss_plot, visualize_thumbnails
from thumbs.params import HyperParams, GanHyperParams, TurnMode, MutableHyperParams, DiffusionHyperParams
from thumbs.loss import Loss
from typing import List, Optional, Tuple, Any, Optional, Callable, Dict, Union, TypeVar, Generic
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


MParams = TypeVar("MParams", bound=MutableHyperParams)
BuiltModel = TypeVar("BuiltModel", bound=Union[BuiltGANModel, BuiltDiffusionModel])


class Train(ABC, Generic[MParams, BuiltModel]):
    # label_getter is a function that takes a number and returns an array of np.ndarray
    losses: Dict[str, List[float]]

    def __init__(
        self,
        built_model: BuiltModel,
        params: HyperParams,
        mparams: MParams,
        label_getter: LabelGetter = None,
        input_mapper: InputMapper = DefaultInputMapper(),
    ) -> None:
        self.mparams = mparams
        self.built_model = built_model
        self.label_getter = label_getter
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
    def save_weights_checkpoint(self, checkpoint_path: str, iteration: int):
        raise NotImplementedError()

    @abstractmethod
    def save_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def load_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def train_body(self, data: tuple, dataset: tf.data.Dataset) -> Dict[str, Union[float, tf.Tensor]]:
        """
        Returns a dict of loss and loss components. For a gan, copmponents might
        include the real/fake loss and the discriminator/generator loss. Other modles might just have one.
        """
        raise NotImplementedError()

    # TODO can't make this a tf.function until i refactor prepare_data to return tensors
    # @tf.function
    def train(self, dataset: tf.data.Dataset, start_iter=0):
        self.load_weights()
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
                dataset=dataset,
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
        dataset: tf.data.Dataset,
    ) -> bool:
        sample_interval = self.mparams.sample_interval
        checkpoint_path = self.params.checkpoint_path
        checkpoint_interval = self.mparams.checkpoint_interval

        if checkpoint_path is not None and checkpoint_interval is not None and (iteration) % checkpoint_interval == 0:
            save_iterations(f"{checkpoint_path}/{iteration}/iteration", iteration)
            self.save_weights_checkpoint(checkpoint_path, iteration)

            loss_file_name = self.params.loss_path.split("/")[-1]
            save_as_json(self.losses, f"{checkpoint_path}/{iteration}/{loss_file_name}")
            iter_checkpoint_file_name = self.params.iteration_checkpoints_path.split("/")[-1]
            save_as_json(self.iteration_checkpoints, f"{checkpoint_path}/{iteration}/{iter_checkpoint_file_name}")

        if (iteration) % sample_interval == 0:
            # Save losses and accuracies so they can be plotted after training
            self.save_weights()
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
            self.show_samples(file_name=file_name, dataset=dataset)
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

    @abstractmethod
    def show_samples(self, dataset: tf.data.Dataset, file_name=None, rows=6, cols=6):
        raise NotImplementedError()


class TrainDiffusion(Train[DiffusionHyperParams, BuiltDiffusionModel]):
    def __init__(
        self,
        built_model: BuiltDiffusionModel,
        params: HyperParams,
        mparams: DiffusionHyperParams,
        label_getter: LabelGetter = None,
        input_mapper: InputMapper = DefaultInputMapper(),
    ) -> None:
        super().__init__(built_model, params, mparams, label_getter, input_mapper)

        # Fancy math stuff to make sure we don't have to loop to add noise, because its super slow
        # Calculate alpha values from beta
        self.alpha = 1 - mparams.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)

    # I made this up and it does'nt work well. I remove all the noise, then add back the noise for t-foo, and repeat.
    def my_sample_new_images(self, n, step_size=10, noise=None):
        if noise == None:
            noise = tf.random.normal(shape=(n, *self.params.img_shape), mean=0.0, stddev=1.0)
            # Rescale the noise to -1 to 1 range
            noise = 2 * (noise - tf.reduce_min(noise)) / (tf.reduce_max(noise) - tf.reduce_min(noise)) - 1
        else:
            # reshape the noise to have n batch size
            noise = tf.reshape(noise, (n, *noise.shape))

        x = noise
        saved = []
        last_i = -1
        for i, cur_t in enumerate(tqdm(range(self.mparams.T - 1, 0, -step_size))):
            t = tf.constant(cur_t, shape=(n, 1), dtype=tf.int32)
            if i > 0:
                # already noisy at the start
                x, _ = self.forward_diffusion_sample(x, t)

            predicted_noise = self.built_model.model.predict([x, t], verbose=False)
            x = self.reverse_diffusion_sample(x, predicted_noise, t)
            saved.append((x, cur_t))
            last_i = i

        if last_i != 0:
            # Once more for t=0
            t = tf.constant(0, shape=(n, 1), dtype=tf.int32)
            predicted_noise = self.built_model.model.predict([x, t], verbose=False)
            x = self.reverse_diffusion_sample(x, predicted_noise, t)
            saved.append((x, 0))

        return x, saved


    def gpt_sample_images(self, n, step_size=10, stop_at=0, noise=None):
        # Initialize your random noise z. This will be transformed into the image.
        # z = tf.random.normal(shape=(n, *self.params.img_shape), mean=0.0, stddev=1.0)
        samples = []
        if noise == None:
            x = tf.random.normal(shape=(n, *self.params.img_shape), mean=0.0, stddev=1.0)
            # Rescale the noise to -1 to 1 range
            # TODO unclear if I should even do sacling to -1,1. Looks like the noise might constantly make this thing 
            # go outside those bound even during training.
            # z = 2 * (z - tf.reduce_min(z)) / (tf.reduce_max(z) - tf.reduce_min(z)) - 1
        else:
            # reshape the noise to have n batch size
            x = tf.reshape(noise, (n, *noise.shape))
        
        # Iterating backward through the timesteps
        for t in tqdm(list(reversed(range(stop_at, self.mparams.T, step_size)))):
            t_tensor = tf.constant(t, dtype=tf.int32, shape=(n, 1))
            
            # Get predicted noise from the model
            predicted_noise = self.built_model.model([x, t_tensor], training=False)
            
            # Compute the alphas and betas for the current step
            alpha_t = tf.gather(self.alpha, t)
            alpha_t = tf.reshape(alpha_t, [-1, 1, 1, 1])

            beta_t = tf.gather(self.mparams.beta, t)
            beta_t = tf.reshape(beta_t, [-1, 1, 1, 1])

            # Actual reverse diffusion step
            x = (x - tf.sqrt(beta_t) * predicted_noise) / tf.sqrt(alpha_t)
            samples.append(x.numpy())

        # At this point, z should approximate the original image
        generated_images = x

        return generated_images, samples

    # Got this online but it doesn't look like it works for my code
    # @tf.function
    def sample(self, n, ):
        x = tf.random.normal((n, *self.params.img_shape))
        samples = []
        for i in tqdm(list(reversed(range(1, self.mparams.T)))):
            t = tf.ones(n, dtype=tf.int32) * i
            predicted_noise = self.built_model.model([x, t])  # Assuming model accepts x and t

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
            samples.append(x.numpy())

        x = tf.clip_by_value(x, -1, 1)
        x = (x + 1) / 2
        x = tf.cast(x * 255, tf.uint8)
        return x, samples

    def get_loss_plot(self, losses: Dict[str, Union[float, tf.Tensor]]) -> Dict[str, float]:
        if isinstance(losses["loss"], float):
            loss = losses["loss"]
        else:
            loss = float(losses["loss"].numpy())

        return {
            "Loss": loss,
        }

    def show_samples(self, dataset: tf.data.Dataset, file_name=None, rows=6, cols=6):
        n_imgs = 4
        random_batch = next(iter(dataset))
        random_img = random_batch[:n_imgs]

        # Noise up the img, all the way to T
        # t = tf.constant(self.mparams.T - 1, dtype=tf.int32, shape=(n_imgs, 1, 1, 1))

        t = tf.constant(
            [
                [np.random.randint(0, self.mparams.T - 1)],
                [np.random.randint(0, self.mparams.T - 1)],
                [np.random.randint(0, self.mparams.T - 1)],
                [np.random.randint(0, self.mparams.T - 1)],
            ],
            dtype=tf.int32,
        )

        noisy, real_noise = self.forward_diffusion_sample(random_img, t)
        # Predict the noise
        predicted_noise = self.built_model.model([noisy, t], training=False)

        dir = self.params.prediction_path
        # Print all the shapes

        imgs_per_row = 4

        labels = []
        for i in range(n_imgs):
            labels += [
                "Original",
                "Reconstructed",
                f"Noisy (t={t[i].numpy()[0]})",
                "Predicted Noise",
            ]

        denoised_img = self.reverse_diffusion_sample(noisy, predicted_noise, t)
        images = [random_img, denoised_img, noisy, predicted_noise]
        images = [img.numpy() for img in images]
        images = [img for sublist in zip(*images) for img in sublist]
        visualize_thumbnails(images, rows=n_imgs, cols=imgs_per_row, dir=dir, file_name=file_name, label_list=labels)

    def load_weights(self):
        load_weights(self.built_model.model, self.params.weight_path)

    def save_weights(self):
        self.built_model.model.save_weights(self.params.weight_path)

    def save_weights_checkpoint(self, checkpoint_path: str, iteration: int):
        self.built_model.model.save_weights(f"{checkpoint_path}/{iteration}/weights")

    # @tf.function
    def forward_diffusion_sample(self, x, t, device="/cpu:0"):
        with tf.device(device):
            sqrt_alpha_hat = tf.sqrt(tf.gather(self.alpha_hat, t))
            sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [-1, 1, 1, 1])

            sqrt_one_minus_alpha_hat = tf.sqrt(1 - tf.gather(self.alpha_hat, t))
            sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [-1, 1, 1, 1])

            noise = tf.random.normal(tf.shape(x))
            return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise


        #     x = tf.cast(x_0, tf.float32)

        #     sqrt_alpha_hat = tf.math.sqrt(tf.gather(self.alpha_hat, t))
        #     sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [-1, 1, 1, 1])

        #     sqrt_one_minus_alpha_hat = tf.math.sqrt(1 - tf.gather(self.alpha_hat, t))
        #     sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [-1, 1, 1, 1])

        #     # Generate random noise
        #     noise = tf.random.normal(shape=x.shape, mean=0.0, stddev=1.0)

        #     # Calculate the noised-up image using the pre-computed scaling terms
        #     x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

        # return x_noisy, noise

    def reverse_diffusion_sample(self, x_noisy, noise, t, device="/cpu:0"):
        with tf.device(device):
            # Grab the same sqrt_alpha_hat and sqrt_one_minus_alpha_hat values
            sqrt_alpha_hat = tf.math.sqrt(tf.gather(self.alpha_hat, t))
            sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [-1, 1, 1, 1])

            sqrt_one_minus_alpha_hat = tf.math.sqrt(1 - tf.gather(self.alpha_hat, t))
            sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [-1, 1, 1, 1])

            # Reverse the noise addition to recover the original x
            x_original = (x_noisy - sqrt_one_minus_alpha_hat * noise) / sqrt_alpha_hat

        return x_original

    def train_body(self, data, dataset: tf.data.Dataset) -> Dict[str, Union[float, tf.Tensor]]:
        # Apparently we don't want to ever sample 0
        t = tf.random.uniform(shape=(self.mparams.batch_size,), minval=1, maxval=self.mparams.T, dtype=tf.int32)

        with tf.GradientTape() as tape:
            # Generate noisy image and real noise for this timestep
            # Assuming you have a function forward_diffusion_sample to do this
            noisy_item, real_noise = self.forward_diffusion_sample(data, t, device="/gpu:0")

            # Forward pass: Get model's prediction of the noise added at this timestep
            predicted_noise = self.built_model.model([noisy_item, t], training=True)

            # Compute loss between the real noise and the predicted noise
            loss = self.mparams.loss_fn(real_noise, predicted_noise)

        # Backprop and update weights
        grads = tape.gradient(loss, self.built_model.model.trainable_variables)
        self.built_model.optimizer.apply_gradients(zip(grads, self.built_model.model.trainable_variables))

        return {"loss": loss}


class TrainGAN(Train[GanHyperParams, BuiltGANModel]):
    def save_weights(self):
        self.built_model.generator.save_weights(self.params.gen_weight_path)
        self.built_model.discriminator.save_weights(self.params.dis_weight_path)

    def save_weights_checkpoint(self, checkpoint_path: str, iteration: int):
        self.built_model.generator.save_weights(f"{checkpoint_path}/{iteration}/weights_gen")
        self.built_model.discriminator.save_weights(f"{checkpoint_path}/{iteration}/weights_dis")

    def load_weights(self):
        load_weights(self.built_model.generator, self.params.gen_weight_path)
        load_weights(self.built_model.discriminator, self.params.dis_weight_path)

    def apply_discriminator_gradients(self, gradient) -> None:
        if self.mparams.discriminator_training:
            self.built_model.discriminator_optimizer.apply_gradients(zip(gradient, self.built_model.discriminator.trainable_variables))

    def apply_generator_gradients(self, gradient) -> None:
        if self.mparams.generator_training:
            self.built_model.generator_optimizer.apply_gradients(zip(gradient, self.built_model.generator.trainable_variables))

    def show_samples(self, dataset: tf.data.Dataset, file_name=None, rows=6, cols=6):
        noise = self.params.latent_sample(rows * cols)
        if self.label_getter is not None:
            labels = self.label_getter(rows * cols)
            generated_thumbnails = self.built_model.generator.predict((noise, *labels), verbose=0)
        else:
            generated_thumbnails = self.built_model.generator.predict(noise, verbose=0)

        dir = self.params.prediction_path
        visualize_thumbnails(generated_thumbnails, rows, cols, dir, file_name)

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
            **d_other_float,
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
            pred = self.built_model.discriminator(disc_input, training=True)

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
            gen_imgs = self.built_model.generator(generator_input, training=True)
            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            # Get the logits for the fake images
            real_logits = self.built_model.discriminator(real_input, training=True)
            fake_logits = self.built_model.discriminator(fake_input, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost, other = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            if self.mparams.gradient_penalty_factor > 0:
                gp = self.gradient_penalty(gen_imgs, data) * self.mparams.gradient_penalty_factor
            else:
                gp = 0

            d_loss = d_cost + gp

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.built_model.discriminator.trainable_variables)
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
            generated_images = self.built_model.generator(generator_input, training=True)
            # Get the discriminator logits for fake images
            discrim_input = self.input_mapper.get_discriminator_input_fake(data, generated_images)
            disc_output = self.built_model.discriminator(discrim_input, training=True)
            # Calculate the generator loss
            real_images = self.input_mapper.get_real_images(data)
            g_loss, other = self.generator_loss(disc_output, generated_images, real_images)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.built_model.generator.trainable_variables)

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
            gen_imgs = self.built_model.generator(generator_input, training=True)
            real_input = self.input_mapper.get_discriminator_input_real(data)
            real_output = self.built_model.discriminator(real_input, training=True)
            d_loss_real = tf.keras.losses.BinaryCrossentropy()(real, real_output)

            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            fake_output = self.built_model.discriminator(fake_input, training=True)
            d_loss_fake = tf.keras.losses.BinaryCrossentropy()(fake, fake_output)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Get the gradients
        grads = tape.gradient(d_loss, self.built_model.discriminator.trainable_weights)

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
            generated_images = self.built_model.generator(generator_input, training=True)

            # Get the discriminator's predictions on the fake images
            fake_input = self.input_mapper.get_discriminator_input_fake(data, generated_images)
            fake_preds = self.built_model.discriminator(fake_input, training=True)

            # Calculate the loss using the generator's output (generated_images)
            # and the discriminator's predictions (fake_preds)
            real_images = self.input_mapper.get_real_images(data)
            loss, other = self.generator_loss(fake_preds, generated_images, real_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.built_model.generator.trainable_weights)

        # Update the weights of the generator
        self.apply_generator_gradients(grads)
        return loss, other


class TrainHinge(TrainGAN):
    @tf.function
    def train_discriminator(self, z, data: tuple) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        generator_input = self.input_mapper.get_generator_input(data, z)
        with tf.GradientTape() as disc_tape:
            gen_imgs = self.built_model.generator(generator_input, training=True)
            real_input = self.input_mapper.get_discriminator_input_real(data)
            real_output = self.built_model.discriminator(real_input, training=True)

            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            fake_output = self.built_model.discriminator(fake_input, training=True)

            d_real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
            d_fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
            d_loss = d_real_loss + d_fake_loss

        # Calculate the gradients
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.built_model.discriminator.trainable_variables)

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
            generated_images = self.built_model.generator(generator_input, training=True)

            # Get the discriminator's predictions on the fake images
            fake_input = self.input_mapper.get_discriminator_input_fake(data, generated_images)
            fake_preds = self.built_model.discriminator(fake_input, training=True)

            # Calculate the loss using the generator's output (generated_images)
            # and the discriminator's predictions (fake_preds)
            real_images = self.input_mapper.get_real_images(data)
            loss, other = self.generator_loss(fake_preds, generated_images, real_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.built_model.generator.trainable_weights)

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
            gen_imgs = self.built_model.generator(generator_input, training=True)
            real_input = self.input_mapper.get_discriminator_input_real(data)
            real_output = self.built_model.discriminator(real_input, training=True)
            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real, real_output, from_logits=True))

            fake_input = self.input_mapper.get_discriminator_input_fake(data, gen_imgs)
            fake_output = self.built_model.discriminator(fake_input, training=True)
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake, fake_output, from_logits=True))

            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.built_model.discriminator.trainable_variables)
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
            generated_images = self.built_model.generator(generator_input, training=True)

            # Get the discriminator's predictions on the fake images
            fake_input = self.input_mapper.get_discriminator_input_fake(data, generated_images)
            fake_preds = self.built_model.discriminator(fake_input, training=True)

            # Calculate the loss using the generator's output (generated_images)
            # and the discriminator's predictions (fake_preds)
            real_images = self.input_mapper.get_real_images(data)
            loss, other = self.generator_loss(fake_preds, generated_images, real_images)

        # Calculate the gradients of the loss with respect to the generator's weights
        grads = tape.gradient(loss, self.built_model.generator.trainable_weights)

        # Update the weights of the generator
        self.apply_generator_gradients(grads)
        return loss, other
