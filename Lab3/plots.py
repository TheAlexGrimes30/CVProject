from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


def make_comparison_figure(
    reference: Image.Image,
    degraded: Image.Image,
    mask: np.ndarray,
    output: Image.Image,
    destination: Path,
    title: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), constrained_layout=True)
    r = np.asarray(reference.convert("RGB"), dtype=np.float32)
    o = np.asarray(output.convert("RGB"), dtype=np.float32)
    diff = np.abs(r - o).mean(axis=2)
    views = [reference, degraded, mask * 255, output, diff]
    captions = ["HR (ground truth)", "Degraded (LQ->HR)", "Binary mask", "SR-inpainting", "Abs. error (HR vs result)"]
    for ax, view, caption in zip(axes, views, captions):
        if isinstance(view, Image.Image):
            ax.imshow(view)
        else:
            ax.imshow(view, cmap="gray", vmin=0, vmax=255)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
