import json
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import PIL
from typing import List, Tuple, Optional, Dict, Any
import os
from thumbs.util import is_notebook
from thumbs.viz import visualize_preprocessed_image, visualize_image_scatter, visualize_thumbnails
import numpy as np
from PIL import Image
from thumbs.util import normalize_image, unnormalize_image


#
# Resize, normalize, etc
#
def load_and_preprocess_image(img_path, size: Tuple[int, int, int], file_type: str = "jpg"):
    # Open the image file
    x, y, _ = size
    img = Image.open(img_path)

    # Resize the image
    img = img.resize((x, y))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # If it's a grayscale image, convert it to RGB
    if len(img_array.shape) == 2:
        img_array = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)

    return normalize_image(img_array)


def get_cat_data(
    size: Tuple[int, int, int] = (128, 128, 3),
) -> np.ndarray:
    data_dir = "/mnt/e/data/afhq/train/cat"

    print(f"Images in {data_dir}")
    file_list = os.listdir(data_dir)
    print(file_list[:10])
    print(f"Found {len(file_list)} total files")
    jpg_file_list = [file for file in file_list if file.endswith(".jpg") or file.endswith(".jpeg")]

    print(f"Found {len(jpg_file_list)} jpgs")

    jpgs = [load_and_preprocess_image(f"{data_dir}/{img_path}", size) for img_path in tqdm(jpg_file_list)]

    images = np.array(jpgs)
    print(f"Shape of images: {images.shape}")

    if is_notebook():
        # Make sure the preprocessing worked
        for image in images[:2]:
            visualize_preprocessed_image(image)
        visualize_image_scatter(images)

    return images


def get_pokemon_data(
    size: Tuple[int, int, int] = (128, 128, 3),
) -> np.ndarray:
    data_dir = "/home/anthony/workspace/yt-data/pokemon"
    print(f"Images in {data_dir}")
    file_list = os.listdir(data_dir)
    print(file_list[:10])
    print(f"Found {len(file_list)} total files")
    jpg_file_list = [file for file in file_list if file.endswith(".jpg") or file.endswith(".jpeg")]

    print(f"Found {len(jpg_file_list)} jpgs")

    jpgs = [load_and_preprocess_image(f"{data_dir}/{img_path}", size) for img_path in tqdm(jpg_file_list)]

    images = np.array(jpgs)
    print(f"Shape of images: {images.shape}")

    if is_notebook():
        # Make sure the preprocessing worked
        for image in images[:2]:
            visualize_preprocessed_image(image)
        visualize_image_scatter(images)

    return images


def get_yt_data(
    size: Tuple[int, int, int] = (64, 64, 3),
    n: Optional[int] = None,
    min_views: int = 1_000_000,
) -> np.ndarray:
    extract_data_dir = "/home/anthony/workspace/yt-data/data/"
    print(f"Thumbnails in {extract_data_dir}")

    file_list = os.listdir(extract_data_dir)
    print(file_list[:10])
    print(f"Found {len(file_list)} total files")
    jpg_file_list = [file for file in file_list if file.endswith(".jpg")]
    print(f"Found {len(jpg_file_list)} jpgs")

    #
    # Filter out worse data
    #
    thumbnail_data: List[Tuple[str, str]] = []  # tuples of (path, id)

    for file_name in jpg_file_list:
        file_path = os.path.join(extract_data_dir, file_name)

        # Filter out the dataset by viewCount > 50,000
        json_file_path = file_path.replace(".jpg", ".json")
        with open(json_file_path, "r") as json_file:
            json_data = json.load(json_file)
            if json_data["viewCount"] < min_views:
                continue

            if "shorts" in json_data["title"].lower() or "shorts" in json_data["description"].lower():
                continue

        thumbnail_data.append((file_path, file_name.replace(".jpg", "")))

    print(thumbnail_data[:10])
    print(f"Found {len(thumbnail_data)} thumbnails")

    if is_notebook():
        PIL.Image.open(thumbnail_data[0][0])

    if n is not None:
        data_subset = thumbnail_data[:n]
    else:
        data_subset = thumbnail_data

    images = [load_and_preprocess_image(t[0], size) for t in tqdm(data_subset)]

    if is_notebook():
        # Make sure the preprocessing worked
        for image in images[:2]:
            visualize_preprocessed_image(image)

        visualize_image_scatter(images)

    #
    # Get rid of youtube shorts
    #
    def is_above_line(point, line_point1, line_point2):
        (x, y) = point
        (x1, y1) = line_point1
        (x2, y2) = line_point2

        # Calculate the gradient (slope) of the line
        m = (y2 - y1) / (x2 - x1)

        # Calculate the y-intercept (c) of the line
        c = y1 - m * x1

        # Calculate the y-coordinate of the point on the line with the same x-coordinate
        line_y = m * x + c
        return y > line_y

    lineToCutAbove = {
        (64, 64, 3): ([-75.0, -10.0], [100.0, 90.0]),
        (128, 128, 3): ([-150.0, 0.0], [100.0, 100.0]),
    }

    if size not in lineToCutAbove:
        raise Exception(f"No line to cut above for size {size}")

    print(f"looking up size {size}")
    p1, p2 = lineToCutAbove[size]

    def scatter(images):
        # Flatten each image into a 1D array
        flattened_images = np.array([img.flatten() for img in images])

        # Apply PCA with 2 components
        pca = PCA(2)
        reduced_data = pca.fit_transform(flattened_images)

        # Plot the reduced data as a scatter plot
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.title("PCA Scatter Plot")
        # plt.plot([-75, 100], [-10, 90], 'k-')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-")
        plt.show()

    def get_rid_of_shorts(dataset):
        pca = PCA(2)

        flattened_images = np.array([img.flatten() for img in dataset])
        pca.fit(flattened_images)

        pca_data = pca.transform(flattened_images)

        # # zip the reduced data with the original data
        # List of (pca, index, original)
        all_data = list(zip(pca_data, list(range(len(dataset))), dataset))

        no_shorts_thumbnail_data = [x for x in all_data if not is_above_line(x[0], p1, p2)]

        if is_notebook():
            scatter([x[0] for x in no_shorts_thumbnail_data])

        indexes_to_keep = [x[1] for x in no_shorts_thumbnail_data]
        filtered_dataset = [dataset[i] for i in indexes_to_keep]

        print(f"Left with {len(filtered_dataset )} thumbnails")
        if is_notebook():
            visualize_preprocessed_image(filtered_dataset[0])
            visualize_preprocessed_image(filtered_dataset[1])
        return filtered_dataset

    if is_notebook():
        scatter(images)

    no_shorts_dataset = get_rid_of_shorts(images)
    return np.asarray(no_shorts_dataset)


def get_wow_icons_64(size: Tuple[int, int, int] = (64, 64, 3)) -> tf.data.Dataset:
    def load_image(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size[:2])
        img = (img - 127.5) / 127.5  # Normalize to [-1,1]
        return img

    dataset = tf.data.Dataset.list_files("/mnt/e/data/wow-icons/*.PNG").map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # num_items = 0
    # for _ in dataset:
    #     num_items += 1

    # print(f"Found {num_items} wow icons.")

    return dataset


def get_pokemon_data256(
    size: Tuple[int, int, int] = (128, 128, 3),
) -> np.ndarray:
    data_dir = "/home/anthony/workspace/yt-data/data/pokemon"
    print(f"Images in {data_dir}")
    file_list = os.listdir(data_dir)
    print(file_list[:10])
    print(f"Found {len(file_list)} total files")
    jpg_file_list = [file for file in file_list if file.endswith(".jpg") or file.endswith(".jpeg")]

    print(f"Found {len(jpg_file_list)} jpgs")

    jpgs = [load_and_preprocess_image(f"{data_dir}/{img_path}", size) for img_path in tqdm(jpg_file_list)]

    images = np.array(jpgs)

    if is_notebook():
        # Get 36 random images
        # get vector of 36 random ints between 0 and len(images)
        ids = np.random.choice(len(images), 36)
        random_images = images[ids]
        visualize_thumbnails(random_images, rows=6, cols=6, dir="/tmp", file_name="preview.jpg")
        visualize_image_scatter(images)

    return images


def get_pokemon_and_types(
    size: Tuple[int, int, int] = (128, 128, 3), drop_megas=True  # need to manually label
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], List[str]]:
    """
    Gets the dataset of pokemon, which is a list of [image as ndarray, types as List[str]]
    Also gets all of the possible pokemon types as the second return value
    """
    data_dir = "/home/anthony/workspace/yt-data/data/pokemon"
    print(f"Images in {data_dir}")
    file_list = os.listdir(data_dir)
    print(file_list[:10])
    print(f"Found {len(file_list)} total files")
    jpg_file_list = [file for file in file_list if file.endswith(".jpg") or file.endswith(".jpeg")]
    print(f"Found {len(jpg_file_list)} jpgs")

    # Get type/stat info
    df = pd.read_csv("/home/anthony/workspace/yt-data/data/pokemon/stats.csv")
    df = df.drop_duplicates(subset=["#"])
    types: List[str] = list(set(df["Type 1"].unique().tolist() + df["Type 2"].unique().tolist()))
    types = [type.lower() for type in types if type == type]  # remove nan
    types.sort()

    # Can lookup the type info given a pokedex number
    pokemon_types: Dict[str, Tuple[str, str]] = {}
    for index, row in df.iterrows():
        type1 = row["Type 1"].lower()
        type2 = row["Type 2"]
        pokemon_types[row["#"]] = (type1, type2.lower() if isinstance(type2, str) else "[UNK]")

    def get_pokedex_num(name):
        # each name starts with some number of digits.  Find them all
        # and return them as a string
        numbers = []
        for c in name:
            if c.isdigit():
                numbers.append(c)
            else:
                break
        return int("".join(numbers))

    images = np.asarray(
        [
            load_and_preprocess_image(f"{data_dir}/{img_path}", size)
            for img_path in tqdm(jpg_file_list)
            # if "-" not in img_path  # Drop mega evolutions and visual variants for now
        ]
    )

    type1 = np.asarray(
        [
            pokemon_types[get_pokedex_num(img_path)][0]
            for img_path in tqdm(jpg_file_list)
            # if "-" not in img_path  # Drop mega evolutions and visual variants for now
        ]
    )

    type2 = np.asarray(
        [
            pokemon_types[get_pokedex_num(img_path)][1]
            for img_path in tqdm(jpg_file_list)
            # if "-" not in img_path  # Drop mega evolutions and visual variants for now
        ]
    )

    if is_notebook():
        # Get 36 random images
        # get vector of 36 random ints between 0 and len(images)
        images = np.asarray(images, dtype=object)
        just_images = images
        ids = np.random.choice(len(images), 36)
        random_images = just_images[ids]
        visualize_thumbnails(random_images, rows=6, cols=6, dir="/tmp", file_name="preview.jpg")
        visualize_image_scatter(just_images)

    return (images, type1, type2), types


def get_pokemon_and_pokedexno(
    size: Tuple[int, int, int] = (128, 128, 3), drop_megas=False  # need to manually label
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets the dataset of pokemon, which is a list of [image as ndarray, types as List[str]]
    Also gets all of the possible pokemon types as the second return value
    """
    data_dir = "/home/anthony/workspace/yt-data/data/pokemon"
    print(f"Images in {data_dir}")
    file_list = os.listdir(data_dir)
    print(file_list[:10])
    print(f"Found {len(file_list)} total files")
    jpg_file_list = [file for file in file_list if file.endswith(".jpg") or file.endswith(".jpeg")]
    print(f"Found {len(jpg_file_list)} jpgs")

    # Get type/stat info

    # Can lookup the type info given a pokedex number
    def get_pokedex_num(name):
        # each name starts with some number of digits.  Find them all
        # and return them as a string
        numbers = []
        for c in name:
            if c.isdigit():
                numbers.append(c)
            else:
                break
        return int("".join(numbers))

    images = np.asarray(
        [
            load_and_preprocess_image(f"{data_dir}/{img_path}", size)
            for img_path in tqdm(jpg_file_list)
            # if "-" not in img_path  # Drop mega evolutions and visual variants for now
        ]
    )

    pokedex_numbers = np.asarray(
        [
            get_pokedex_num(img_path)
            for img_path in tqdm(jpg_file_list)
            # if "-" not in img_path  # Drop mega evolutions and visual variants for now
        ]
    )

    if is_notebook():
        # Get 36 random images
        # get vector of 36 random ints between 0 and len(images)
        print(f"shapes {images.shape} {pokedex_numbers.shape}")
        ids = np.random.choice(len(images), 36)
        random_images = images[ids]
        visualize_thumbnails(random_images, rows=6, cols=6)
        visualize_image_scatter(images)

    return images, pokedex_numbers
