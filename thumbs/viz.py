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
from datetime import datetime


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


def show_loss_plot(losses, iteration_checkpoints, dir: str, file_name: str) -> None:
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

    plt.annotate(text=f"{disc_loss[-1]}", xy=(iteration_checkpoints[-1], disc_loss[-1]))
    plt.annotate(text=f"{gen_loss[-1]}", xy=(iteration_checkpoints[-1], gen_loss[-1]))

    # Ensure predictions exists
    if not os.path.exists(f"{dir}"):
        os.mkdir(f"{dir}")
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


def process_prediction_image(image):
    # # Scale the image from [-1, 1] to [0, 1]
    # image = (image + 1) / 2
    # # Clip values to [0, 1] in case of any numerical instability
    # image = tf.clip_by_value(image, 0, 1)
    # # Convert the image tensor to a NumPy array
    # return image.numpy()

    # This scales it back up to the range [0, 255]. It looks a little different
    # rendered from here than it does from -1,1
    image = (image + 1) * 127.5
    return image.astype(np.uint8)


def visualize_preprocessed_image(image, size=None):
    image = process_prediction_image(image)

    if size is not None:
        plt.figure(figsize=size)  # Set the figure size to be 10 inches wide and 5 inches tall
    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()


def visualize_thumbnails(image_list, rows, cols, dir, file_name):
    plt.cla()
    plt.clf()
    # Create a grid of subplots to display the images
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))

    # Make a copy of image_list
    image_list = list(image_list)
    for row in range(rows):
        for col in range(cols):
            image = process_prediction_image(image_list.pop())
            if rows == 1:
                axs[col].imshow(image)
                axs[col].axis("off")
            else:
                axs[row, col].imshow(image)
                axs[row, col].axis("off")

    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()

    # Show the plot
    # Ensure predictions exists
    if is_notebook():
        plt.show()

    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.savefig(f"{dir}/_latest.jpg", bbox_inches="tight")
    plt.savefig(f"{dir}/thumbnail-{file_name}.jpg", bbox_inches="tight")
    plt.close()


def show_samples(generator, latent_dim, file_name, dir: str, rows=6, cols=6, dataset=None):
    # noise = np.random.uniform(-1, 1, size=(rows * cols, latent_dim))
    noise = np.random.normal(0, 1, (rows * cols, latent_dim))
    generated_thumbnails = generator.predict(noise, verbose=0)
    visualize_thumbnails(generated_thumbnails, rows, cols, dir, file_name)
