import time
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)

def benchmark(
    func,
    *args,
    repeats: int = 5,
    **kwargs,
):
    times = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return result, float(np.mean(times)), float(np.std(times))

def print_benchmark(
    native_time: float,
    native_std: float,
    opencv_time: float,
    opencv_std: float,
) -> None:

    speedup = native_time / opencv_time if opencv_time > 0 else float("inf")

    print(
        f"Native: "
        f"{native_time:.6f} ± "
        f"{native_std:.6f} s"
    )

    print(
        f"OpenCV: "
        f"{opencv_time:.6f} ± "
        f"{opencv_std:.6f} s"
    )

    print(
        f"OpenCV speedup: "
        f"{speedup:.2f}x"
    )

def save_gray(image: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)
