import cv2
import numpy as np

KERNEL = np.ones((3, 3), dtype=np.uint8)

def binarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    return (image >= threshold).astype(np.uint8)

def dilation_native(
    binary: np.ndarray,
    kernel: np.ndarray = KERNEL,
) -> np.ndarray:

    kernel_height, kernel_width = kernel.shape
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(
        binary,
        (
            (pad_y, pad_y),
            (pad_x, pad_x),
        ),
        mode="constant",
        constant_values=0,
    )

    result = np.zeros_like(binary)

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            region = padded[
                y:y + kernel_height,
                x:x + kernel_width
            ]

            result[y, x] = 1 if np.any(region[kernel == 1] == 1) else 0

    return result

def dilation_opencv(
    binary: np.ndarray,
    kernel: np.ndarray = KERNEL,
) -> np.ndarray:

    return cv2.dilate(
        binary,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )
