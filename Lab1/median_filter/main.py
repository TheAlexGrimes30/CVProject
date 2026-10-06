from pathlib import Path

import numpy as np

from Lab1.median_filter.algorithm import median_filter_native, median_filter_opencv
from Lab1.median_filter.utils import load_gray, benchmark, save_gray, print_benchmark

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "image.jpg"

KERNEL_SIZE = 3
REPEATS = 5


def main() -> None:

    image = load_gray(str(IMAGE_PATH))

    native_result, native_time, native_std = benchmark(
        median_filter_native,
        image,
        KERNEL_SIZE,
        repeats=REPEATS,
    )

    opencv_result, opencv_time, opencv_std = benchmark(
        median_filter_opencv,
        image,
        KERNEL_SIZE,
        repeats=REPEATS,
    )

    save_gray(native_result, "results/native.png",)
    save_gray(opencv_result,"results/opencv.png",)

    difference = np.abs(native_result.astype(np.int16) - opencv_result.astype(np.int16))

    save_gray(difference,"results/difference.png",)

    print_benchmark(
        native_time,
        native_std,
        opencv_time,
        opencv_std,
    )

    print("Mean difference:", np.mean(difference))


if __name__ == "__main__":
    main()