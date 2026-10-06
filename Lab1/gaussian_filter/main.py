from pathlib import Path

import numpy as np

from Lab1.gaussian_filter.algorithm import adaptive_gaussian_opencv, adaptive_gaussian_native
from Lab1.gaussian_filter.utils import load_gray, save_gray, print_benchmark
from Lab1.sobel.utils import benchmark

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "image.jpg"

BLOCK_SIZE = 11
SIGMA = 2.0
C = 2.0

REPEATS = 5


def main() -> None:

    gray = load_gray(str(IMAGE_PATH))

    native_result, native_time, native_std = benchmark(
        adaptive_gaussian_native,
        gray,
        BLOCK_SIZE,
        SIGMA,
        C,
        repeats=REPEATS,
    )

    opencv_result, opencv_time, opencv_std = benchmark(
        adaptive_gaussian_opencv,
        gray,
        BLOCK_SIZE,
        C,
        repeats=REPEATS,
    )

    save_gray(native_result,"results/native.png")
    save_gray(opencv_result,"results/opencv.png",)
    difference = np.abs(native_result.astype(np.int16) - opencv_result.astype(np.int16))
    save_gray(difference,"results/difference.png",)

    print_benchmark(
        native_time,
        native_std,
        opencv_time,
        opencv_std,
    )

    print("Mean difference:", np.mean(difference),)


if __name__ == "__main__":
    main()