from io import BytesIO

import numpy as np
from PIL import ImageFilter, Image


def degrade(image: Image.Image, name: str, *, scale: int = 4, seed: int = 42) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if w < scale or h < scale:
        raise ValueError("Image is too small for the chosen scale")

    def down_up(x: Image.Image) -> Image.Image:
        low = x.resize((w // scale, h // scale), Image.Resampling.BICUBIC)
        return low.resize((w, h), Image.Resampling.BICUBIC)

    if name == "bicubic_x4":
        return down_up(image)
    if name == "blur_noise_x4":
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1.6))
        recon = down_up(blurred)
        rng = np.random.default_rng(seed)
        array = np.asarray(recon, dtype=np.float32)
        array += rng.normal(loc=0, scale=7.0, size=array.shape)
        return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    if name == "complex_x4":
        blurred = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        recon = down_up(blurred)
        buff = BytesIO()
        recon.save(buff, format="JPEG", quality=27, subsampling=2)
        buff.seek(0)
        with Image.open(buff) as jpeg:
            array = np.asarray(jpeg.convert("RGB"), dtype=np.float32).copy()
        rng = np.random.default_rng(seed)
        array += rng.normal(loc=0, scale=4.0, size=array.shape)
        return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    raise ValueError(f"Unknown synthetic degradation {name!r}")
