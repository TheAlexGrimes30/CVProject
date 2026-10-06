import cv2
import numpy as np


def gaussian_kernel(
    size: int,
    sigma: float,
) -> np.ndarray:

    radius = size // 2
    coordinates = np.arange(-radius, radius + 1,)
    x, y = np.meshgrid(coordinates, coordinates)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)

    return kernel

def adaptive_gaussian_native(
    image: np.ndarray,
    block_size: int = 11,
    sigma: float = 2.0,
    c: float = 2.0,
) -> np.ndarray:

    radius = block_size // 2
    kernel = gaussian_kernel(block_size, sigma)

    padded = np.pad(
        image.astype(np.float32),
        (
            (radius, radius),
            (radius, radius),
        ),
        mode="reflect",
    )

    result = np.zeros_like(image, dtype=np.uint8)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            region = padded[y:y + block_size, x:x + block_size]
            local_threshold = (np.sum(region * kernel) - c)
            result[y, x] = 255 if image[y, x] > local_threshold else 0

    return result

def adaptive_gaussian_opencv(
    image: np.ndarray,
    block_size: int = 11,
    c: float = 2.0,
) -> np.ndarray:

    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )