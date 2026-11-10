from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)\
        if folder.exists() else []

def prepare_hr_image(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if min(w, h) < size:
        ratio = size / min(w, h)
        image = image.resize((round(w * ratio), round(h * ratio)), Image.Resampling.BICUBIC)
        w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return image.crop((left, top, left + size, top + size))

def load_external_lq_image(folder: Path, name: str, target_size: tuple[int, int]) -> Image.Image | None:
    if not folder.is_dir():
        return None

    stem = Path(name).stem
    paths = sorted(p for p in folder.iterdir() if p.is_file() and
                   p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        return None
    with Image.open(paths[0]) as img:
        return img.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)

def pil_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)
