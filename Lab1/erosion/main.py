from pathlib import Path

import numpy as np

from Lab1.erosion.algorithm import erosion_opencv, erosion_native, binarize
from Lab1.erosion.utils import print_benchmark, save_gray, benchmark, load_gray

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "image.jpg"

THRESHOLD = 128
REPEATS = 5


def main() -> None:

    gray = load_gray(str(IMAGE_PATH))
    binary = binarize(gray, THRESHOLD)

    native_result, native_time, native_std = benchmark(
        erosion_native,
        binary,
        repeats=REPEATS,
    )

    opencv_result, opencv_time, opencv_std = benchmark(
        erosion_opencv,
        binary,
        repeats=REPEATS,
    )

    save_gray(binary * 255,"results/binary.png",)
    save_gray(native_result * 255,"results/native.png",)
    save_gray(opencv_result * 255,"results/opencv.png",)

    difference = (
        native_result
        != opencv_result
    ).astype(np.uint8) * 255

    save_gray(difference, "results/difference.png")

    print_benchmark(
        native_time,
        native_std,
        opencv_time,
        opencv_std,
    )

    print(
        "Results equal:",
        np.array_equal(native_result, opencv_result)
    )


if __name__ == "__main__":
    main()