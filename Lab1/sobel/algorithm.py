import cv2
import numpy as np

SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

SOBEL_Y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float32,
)


def convolution(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:

    kernel_height, kernel_width = kernel.shape

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(
        image.astype(np.float32),
        (
            (pad_y, pad_y),
            (pad_x, pad_x),
        ),
        mode="edge",
    )

    result = np.zeros_like(image, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            region = padded[
                y:y + kernel_height,
                x:x + kernel_width
            ]

            result[y, x] = np.sum(region * kernel)

    return result


def sobel_native(
    image: np.ndarray,
) -> np.ndarray:

    gradient_x = convolution(image, SOBEL_X)
    gradient_y = convolution(image, SOBEL_Y)

    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    gradient = np.clip(
        gradient,
        0,
        255,
    )

    return gradient.astype(np.uint8)


def sobel_opencv(
    image: np.ndarray,
) -> np.ndarray:

    gradient_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    )

    gradient_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    )

    gradient = cv2.magnitude(gradient_x, gradient_y)

    gradient = np.clip(
        gradient,
        0,
        255
    )

    return gradient.astype(np.uint8)
