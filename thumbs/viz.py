import tensorflow as tf
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import PIL.Image
from sklearn.decomposition import PCA
import numpy as np
from functools import partial
from PIL import Image
import numpy as np
from thumbs.util import is_notebook, get_current_time
from thumbs.util import normalize_image, unnormalize_image
from datetime import datetime
from typing import List, Tuple, Iterator, Optional, Union, Dict


def show_accuracy_plot(accuracies_rf, iteration_checkpoints, dir: str, file_name: str) -> None:
    accuracies_np = np.asarray(accuracies_rf)
    accuracies_real = accuracies_np.T[0]
    accuracies_fake = accuracies_np.T[1]

    # Plot Discriminator accuracy
    plt.figure(figsize=(10, 2))
    plt.plot(iteration_checkpoints, accuracies_fake, label="Fake accuracy")
    plt.plot(iteration_checkpoints, accuracies_real, label="Real accuracy")

    plt.xticks(iteration_checkpoints, rotation=90)
    plt.yticks(range(0, 100, 5))

    plt.title("Discriminator Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.annotate(text=f"{round(accuracies_fake[-1])}", xy=(iteration_checkpoints[-1], accuracies_fake[-1]))
    plt.annotate(text=f"{round(accuracies_real[-1])}", xy=(iteration_checkpoints[-1], accuracies_real[-1]))

    # Ensure predictions exists
    if not os.path.exists(f"{dir}"):
        os.mkdir(f"{dir}")
    plt.savefig(f"{dir}/_latest-acc.jpg")
    plt.savefig(f"{dir}/acc-{file_name}.jpg")
    plt.close()


def show_loss_plot_gan(
    losses: List[Tuple[float, float]], iteration_checkpoints: List[int], dir: str, file_name: str, save_as_latest: bool = True
) -> None:
    losses_np = np.asarray(losses)
    disc_loss = losses_np.T[0]
    gen_loss = losses_np.T[1]

    # Plot training losses for Discriminator and Generator
    plt.figure(figsize=(10, 2))
    plt.plot(iteration_checkpoints, disc_loss, label="Discriminator loss")
    plt.plot(iteration_checkpoints, gen_loss, label="Generator loss")

    plt.xticks(iteration_checkpoints, rotation=90)

    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    # Show label on the last data point
    plt.annotate(text=f"{disc_loss[-1]}", xy=(iteration_checkpoints[-1], disc_loss[-1]))
    plt.annotate(text=f"{gen_loss[-1]}", xy=(iteration_checkpoints[-1], gen_loss[-1]))

    # Show label on the first data point
    plt.annotate(text=f"{disc_loss[0]}", xy=(iteration_checkpoints[0], disc_loss[0]))
    plt.annotate(text=f"{gen_loss[0]}", xy=(iteration_checkpoints[0], gen_loss[0]))

    # Ensure predictions exists
    if not os.path.exists(f"{dir}"):
        os.mkdir(f"{dir}")

    if save_as_latest:
        plt.savefig(f"{dir}/_latest-loss.jpg")
    plt.savefig(f"{dir}/loss-{file_name}.jpg")
    plt.close()


def show_loss_plot(losses: Dict[str, List[float]], iteration_checkpoints, dir: str, file_name: str, save_as_latest: bool = True) -> None:
    """
    Does the same thing as show_loss_plot_gan but it doesn't assume two series of losses.
    """
    plt.figure(figsize=(10, 2))
    for label, loss in losses.items():
        plt.plot(iteration_checkpoints, loss, label=label)

    plt.xticks(iteration_checkpoints, rotation=90)

    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    # Show label on the last data point
    for label, loss in losses.items():
        plt.annotate(text=f"{loss[-1]}", xy=(iteration_checkpoints[-1], loss[-1]))

    # Show label on the first data point
    for label, loss in losses.items():
        plt.annotate(text=f"{loss[0]}", xy=(iteration_checkpoints[0], loss[0]))

    # Ensure predictions exists
    if not os.path.exists(f"{dir}"):
        os.mkdir(f"{dir}")

    if save_as_latest:
        plt.savefig(f"{dir}/_latest-loss.jpg")
    plt.savefig(f"{dir}/loss-{file_name}.jpg")
    plt.close()


def visualize_image_distribution(images):
    # Flatten all pixel values into a single list
    all_pixels = np.array(images).flatten()

    # Plot a histogram for the pixel values
    plt.hist(all_pixels, bins=256, color="gray", alpha=0.7)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()


def visualize_image_scatter(images):
    # Flatten each image into a 1D array
    flattened_images = np.array([img.flatten() for img in images])

    # Apply PCA with 2 components
    pca = PCA(2)
    reduced_data = pca.fit_transform(flattened_images)

    # Plot the reduced data as a scatter plot
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title("PCA Scatter Plot")
    plt.show()
    plt.close()


# def compare_scatters(dataset1, dataset2, dir, name1='Dataset 1', name2='Dataset 2'):
#     global total_epochs_so_far
#     plt.cla()
#     plt.clf()
#     # Flatten each image into a 1D array
#     flattened_images1 = np.array([img.flatten() for img in dataset1])
#     flattened_images2 = np.array([img.flatten() for img in dataset2])

#     # Apply PCA with 2 components
#     pca = PCA(2)

#     # Fit PCA on the combined datasets and transform each separately
#     pca.fit(np.concatenate((flattened_images1, flattened_images2)))
#     reduced_data1 = pca.transform(flattened_images1)
#     reduced_data2 = pca.transform(flattened_images2)

#     # Plot the reduced data as a scatter plot
#     plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], label=name1, alpha=0.5)
#     plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], label=name2, alpha=0.5)
#     plt.title('PCA Scatter Plot')
#     plt.legend()
#     if is_notebook():
#         plt.show()
#         plt.close()
#     else:
#         # Ensure predictions exists
#         if not os.path.exists(dir):
#             os.mkdir(dir)
#         plt.savefig(f'{dir}/_latest-plot.jpg')
#         plt.savefig(f'{dir}/plot-{total_epochs_so_far}.jpg')
#         plt.close()


def visualize_preprocessed_image(image, size=None):
    image = unnormalize_image(image)

    if size is not None:
        plt.figure(figsize=size)  # Set the figure size to be 10 inches wide and 5 inches tall
    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()


def visualize_thumbnails(image_list, rows, cols, dir=None, file_name=None, label_list: Optional[List[str]] = None):
    plt.cla()
    plt.clf()
    # Create a grid of subplots to display the images
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))

    # Make a copy of image_list
    image_list = list(image_list)
    for row in range(rows):
        for col in range(cols):
            image = unnormalize_image(image_list.pop())
            if label_list:
                label = label_list.pop()

            if rows == 1:
                axs[col].imshow(image)
                axs[col].axis("off")
                if label_list is not None:
                    axs[col].set_title(label)
            else:
                axs[row, col].imshow(image)
                axs[row, col].axis("off")
                if label_list is not None:
                    axs[row, col].set_title(label)

    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()

    # Show the plot
    # Ensure predictions exists
    if is_notebook():
        plt.show()

    if dir is not None and file_name is not None:
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(f"{dir}/_latest.jpg", bbox_inches="tight")
        plt.savefig(f"{dir}/thumbnail-{file_name}.jpg", bbox_inches="tight")
    plt.close()


def visualize_grid(
    images,
    rows=None,
    cols=None,
    normalized=True,
    dir=None,
    file_name=None,
    title_index=False,
):
    n_images = len(images)

    if normalized:
        images = unnormalize_image(images)

    # Determine number of rows and columns
    if rows is None and cols is None:
        rows = int(np.sqrt(n_images))
        cols = rows
    elif rows is not None:
        cols = int(np.ceil(n_images / rows))
    elif cols is not None:
        rows = int(np.ceil(n_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))

    # Make sure axes is an array (useful for edge cases with 0 row or 1 column)
    axes = np.array(axes).reshape(-2)

    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis("off")
        if title_index:
            ax.set_title(f"Index: {idx}")

    # Turn off any extra subplots
    for ax in axes[n_images:]:
        ax.axis("off")

    if is_notebook():
        plt.show()
    if dir is not None and file_name is not None:
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(f"{dir}/_latest_grid.jpg", bbox_inches="tight")
        plt.savefig(f"{dir}/grid-{file_name}.jpg", bbox_inches="tight")


# deprecated
def show_samples(generator, latent_dim, file_name, dir: str, rows=6, cols=6, label_getter=None):
    # noise = np.random.uniform(-1, 1, size=(rows * cols, latent_dim))
    noise = np.random.normal(0, 1, (rows * cols, latent_dim))
    if label_getter is not None:
        labels = label_getter(rows * cols)
        generated_thumbnails = generator.predict((noise, *labels), verbose=0)
    else:
        generated_thumbnails = generator.predict(noise, verbose=0)
    visualize_thumbnails(generated_thumbnails, rows, cols, dir, file_name)
