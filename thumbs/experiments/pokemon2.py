import thumbs.config_logging  # must be first
from typing import List, Tuple, Iterator
from rangedict import RangeDict
import numpy as np

from thumbs.experiment import Experiment
from thumbs.data import get_pokemon_data
from thumbs.params import HyperParams, MutableHyperParams
from thumbs.model.model import Model, BuiltModel

from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    GaussianNoise,
    Dropout,
    Flatten,
    Reshape,
    LeakyReLU,
    LayerNormalization,
    # BatchNormalizationV2,
)

# from keras.layers.normalization.batch_normalization_v1 import (
#     BatchNormalization,
# )
from tensorflow.compat.v1.keras.layers import BatchNormalization as BatchNormalizationV1
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from thumbs.train import Train, TrainMSE, TrainBCE, TrainBCESimilarity


infinity = float("inf")


class PokemonModel(Model):
    def build_generator(self, z_dim):
        model = Sequential(name="generator_2")

        model.add(Dense(2048* 4 * 4, input_dim=z_dim))
        model.add(Reshape((4, 4, 2048)))

        model.add(Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding="same"))
        model.add(Activation("tanh"))

        model.summary(line_length=200)
        return model

    def build_discriminator(self, img_shape):
        model = Sequential(name="discriminator")

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
        model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
        model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalizationV1())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary(line_length=200)
        return model

    def build_gan(self, generator, discriminator):
        model = Sequential([generator, discriminator])
        return model


class PokemonExperiment(Experiment):
    def get_data(self) -> np.ndarray:
        return get_pokemon_data(self.params.img_shape)

    def get_train(self, model: BuiltModel, mparams: MutableHyperParams) -> Train:
        return TrainBCE(model, self.params, mparams)

    def get_mutable_params(self) -> RangeDict:
        """
        - The additional transpose only add ~3mil parameters
        - FAIL Just adding more layers to the generator seemed to overpower the discriminator. Even at
            equal learning  rates the accuracy was often under 50% after 400 epochs.
        - NEXT to add some complexity to the discriminator, revering  the generator to the original
        - Going with near equal complexity. 8mil vs 6mil
        - Disc caught up and was ~100% accurate by epoch 126. Bumped the LR to 10x less than gen, but might
            have been too late to save the training since I caught it around epoch 200.
        - GOOD the images do look a little better actually. I'm worried that the discriminator is too good
            but I guess I need to keep training to see. Determined this at 500 epochs.
        - Ok i'm going to start again without the gausian noise in the disc without the need to have different schedules.
        - Also switching the batch norm to come after the transpose, before the relu because this thing https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        - Adding batch norm to the disc always fucks it all up. no idea why. I must be missing something.
        - GOOD increasing batch size to 128 dramatically sped up training (10 minutes for 500 epochs instead of 100).
        - I wonder if I should add more data (via data aug) per epoch now because the dataset is pretty small. Its only 6 iterations for the entire dataset.
        - After 500 epochs it looks like it might be memorizing the entire dataset. I might have to make the model simpler
        - the discriminator got way better and gen got way worse around 600 epochs for some reason
        - Around 1100 the generator loss plumets and the images become horrible. It recovered from the 600 mark though so lets see.
        - It seems doomed, going to restart with more training turns for the generator
        - BAD noticed that the quality of the 64,64 images were pretty bad. My expectations were based on the raw data but the downscaled
            images looked horrible so I can't expect much from a model of that size. Bumping back to 128,128 which is close to the original sizes.
            Also added data augmentation with zoom, flip, and rotate.
            I also updated the sample code to convert the images back to 0,255 which I found to look more like the originals compared with -1,1, no surprise.

        - NEXT use the original arch but remove the injected noise
        - GOOD by far the best traiing cycle was with the BatchNormalizationV1. Its the only way I could stabalize the disc. No idea whats wrong with the default batch norm.

        TODO
        - When calculating D loss for real images, instead of 1.0 for labels, feed it with 0.9.
        - You should use batch normalization, but not on the first conv layer in D, and not on logits in G.
        """
        schedule = RangeDict()
        schedule[0, 1730] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=2000,
            sample_interval=10,
            discriminator_turns=1,
            generator_turns=1,
        )

        # Even at 2x gen steps the discriminator is >90% accuracy. Images do seem to  be getting better though.
        schedule[1731, 3000] = MutableHyperParams(
            gen_learning_rate=0.0002,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=200000,
            sample_interval=10,
            discriminator_turns=1,
            checkpoint_interval=200,
            generator_turns=2,
        )

        schedule[3001, infinity] = MutableHyperParams(
            gen_learning_rate=0.00005,
            dis_learning_rate=0.0002,
            batch_size=128,
            adam_b1=0.5,
            iterations=200000,
            sample_interval=10,
            discriminator_turns=1,
            checkpoint_interval=200,
            generator_turns=4,
        )

        return schedule

    def get_params(self) -> HyperParams:
        name = "pokemon_batch_norm_v1"
        return HyperParams(
            latent_dim=100,
            img_shape=(128, 128, 3),
            weight_path=f"./experiments/{name}/weights",
            checkpoint_path=f"./experiments/{name}/checkpoints",
            prediction_path=f"./experiments/{name}/predictions",
            iteration_path=f"./experiments/{name}/iteration",
            similarity_threshold=0.0,
            similarity_penalty=20,
        )

    def get_model(self, mparams: MutableHyperParams) -> Model:
        return PokemonModel(self.params, mparams)


if __name__ == "__main__":
    PokemonExperiment().start()
