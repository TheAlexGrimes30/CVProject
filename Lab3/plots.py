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


def plot_metric_summary(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("psnr_output", "PSNR (dB) ↑", "psnr_by_mask.png"),
        ("lpips_output", "LPIPS ↓", "lpips_by_mask.png"),
        ("clip_iqa_output", "CLIP-IQA ↑", "clip_iqa_by_mask.png"),
        ("psnr_gain", "PSNR gain (dB)", "psnr_gain_by_mask.png"),
    )
    for column, ylabel, filename in definitions:
        valid = df.replace([float("inf"), float("-inf")], np.nan)
        pivot = valid.pivot_table(index="degradation", columns="mask", values=column, aggfunc="mean")
        if pivot.empty:
            continue
        ax = pivot.plot.bar(figsize=(10, 5), rot=15, width=0.78)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Degradation")
        ax.legend(title="Mask")
        ax.grid(axis="y", alpha=0.3)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
