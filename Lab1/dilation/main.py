from pathlib import Path

import numpy as np

from Lab1.dilation.algorithm import binarize, dilation_opencv, dilation_native
from Lab1.dilation.utils import load_gray, print_benchmark, save_gray, benchmark

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "image.jpg"

THRESHOLD = 128
REPEATS = 5

def main() -> None:
    gray = load_gray(str(IMAGE_PATH))
    binary = binarize(gray, THRESHOLD)

    native_result, native_time, native_std = benchmark(
        dilation_native,
        binary,
        repeats=REPEATS,
    )

    opencv_result, opencv_time, opencv_std = benchmark(
        dilation_opencv,
        binary,
        repeats=REPEATS,
    )

    save_gray(binary * 255,"results/binary.png")
    save_gray(native_result * 255,"results/native.png",)
    save_gray(opencv_result * 255,"results/opencv.png",)

    difference = (native_result != opencv_result).astype(np.uint8) * 255

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