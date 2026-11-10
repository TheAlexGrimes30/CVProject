import numpy as np
from PIL import Image


def create_mask(mask_type: str, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if mask_type == "zeros":
        return np.zeros((h, w), dtype=np.uint8)
    if mask_type == "ones":
        return np.ones((h, w), dtype=np.uint8)
    if mask_type == "center":
        out = np.zeros((h, w), dtype=np.uint8)
        x0, x1 = w // 4, w - w // 4
        y0, y1 = h // 4, h - h // 4
        out[y0:y1, x0:x1] = 1
        return out
    if mask_type == "checker":
        yy, xx = np.indices((h, w))
        return (((xx // max(1, w // 8)) + (yy // max(1, h // 8))) % 2).astype(np.uint8)
    raise ValueError(f"Unknown mask_type={mask_type!r}. Use zeros, ones, center or checker.")

def validate_mask(mask: np.ndarray, size: tuple[int, int]) -> None:
    w, h = size
    if mask.shape != (h, w):
        raise ValueError(f"Mask must have shape {(h, w)}; got {mask.shape}")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask must be binary (0/1), not 0/255 or floating values")

def mask_to_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

def hard_composite(base: Image.Image, restored: Image.Image, mask: np.ndarray) -> Image.Image:
    validate_mask(mask, base.size)
    if base.size != restored.size:
        raise ValueError("Base and restoration sizes must match")
    b = np.asarray(base.convert("RGB"), dtype=np.uint8)
    r = np.asarray(restored.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(np.where(mask[..., None].astype(bool), r, b).astype(np.uint8), mode="RGB")


def max_outside_difference(base: Image.Image, output: Image.Image, mask: np.ndarray) -> int:
    outside = mask == 0
    if not outside.any():
        return 0
    b = np.asarray(base.convert("RGB"), dtype=np.int16)
    o = np.asarray(output.convert("RGB"), dtype=np.int16)
    return int(np.abs(b[outside] - o[outside]).max())
