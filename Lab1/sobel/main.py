from pathlib import Path

import numpy as np

from Lab1.sobel.algorithm import sobel_opencv, sobel_native
from Lab1.sobel.utils import print_benchmark, save_gray, benchmark, load_gray

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "image.jpg"
REPEATS = 5


def main() -> None:

    gray = load_gray(str(IMAGE_PATH))

    native_result, native_time, native_std = benchmark(
        sobel_native,
        gray,
        repeats=REPEATS,
    )

    opencv_result, opencv_time, opencv_std = benchmark(
        sobel_opencv,
        gray,
        repeats=REPEATS
    )

    save_gray(native_result,"results/native.png",)
    save_gray(opencv_result,"results/opencv.png",)

    difference = np.abs(native_result.astype(np.int16) - opencv_result.astype(np.int16))

    save_gray(difference, "results/difference.png",)

    print_benchmark(
        native_time,
        native_std,
        opencv_time,
        opencv_std,
    )

    print("Mean difference:", np.mean(difference))


if __name__ == "__main__":
    main()