import cv2
import numpy as np


def median_filter_native(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:

    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be odd and >= 3")

    radius = kernel_size // 2

    padded = np.pad(
        image,
        (
            (radius, radius),
            (radius, radius)
        ),
        mode="edge",
    )

    result = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            region = padded[
                y:y + kernel_size,
                x:x + kernel_size
            ]

            result[y, x] = np.median(region)

    return result


def median_filter_opencv(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:

    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be odd and >= 3")

    return cv2.medianBlur(image, kernel_size)