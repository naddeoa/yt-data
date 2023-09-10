import numpy as np
from datetime import datetime


def is_notebook() -> bool:
    try:
        # Check if the 'get_ipython' function exists
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Running in a Jupyter notebook
        else:
            return False  # Running
    except Exception:
        return False


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except ModuleNotFoundError:
        return False


def create_batches(lst, batch_size, length):
    """Yield successive n-sized batches from lst."""
    for i in range(0, length, batch_size):
        yield lst[i : i + batch_size]


def get_current_time() -> str:
    now = datetime.now()
    # return now.strftime("%Y-%m-%d_%I-%M_%p")
    return now.strftime("%Y-%m-%d %H-%M")


def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """
    Normalize from [0, 255] to [-1, 1]
    """
    return (img_array.astype("float32") - 127.5) / 127.5

def unnormalize_image(image):
    # This scales it back up to the range [0, 255]. It looks a little different
    # rendered from here than it does from -1,1
    image = np.clip(image, -1, 1)
    image = (image + 1) * 127.5
    return image.astype(np.uint8)