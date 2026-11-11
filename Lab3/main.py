import gc
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from Lab3.degrade import degrade
from Lab3.image_loader import list_images, prepare_hr_image
from Lab3.image_masks import create_mask, mask_to_pil, max_outside_difference
from Lab3.metrics import psnr
from Lab3.models import MaskDfModel
from Lab3.plots import make_comparison_figure

ROOT = Path(__file__).resolve().parent

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = True

@dataclass
class ExperimentConfig:
    hr_dir: Path = ROOT / "images"
    output_dir: Path = ROOT / "results"
    image_size: int = 256
    max_images: int = 1

    degradations: tuple[str, ...] = ("bicubic_x4", "blur_noise_x4", "complex_x4")
    masks: tuple[str, ...] = ("zeros", "ones", "center", "checker")
    scale: int = 4
    seed: int = 42

    model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    device: str = "auto"
    use_cpu_offload_on_cuda: bool = True
    num_inference_steps: int = 4
    strength: float = 0.75
    guidance_scale: float = 1.0
    prompt: str = (
        "realistic high resolution photograph of a historic Spanish town, "
        "stone bridge over a deep gorge, natural rock textures, sharp details"
    )

    negative_prompt: str = "blurry, distortion, painting, artifacts, low quality, unrealistic"
    evaluate_perceptual: bool = True


CONFIG = ExperimentConfig()


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_inference(cfg: ExperimentConfig) -> list[dict]:
    images = list_images(cfg.hr_dir)
    if not images:
        raise FileNotFoundError(f"Добавь хотя бы одно HR-изображение в: {cfg.hr_dir}")

    images = images[:cfg.max_images]
    cases: list[dict] = []
    model = None

    try:
        for image_idx, image_path in enumerate(images, start=1):
            with Image.open(image_path) as img:
                hr = prepare_hr_image(img, cfg.image_size)

            image_name = f"{image_idx:02d}_{image_path.stem}"
            hr_path = cfg.output_dir / "hr" / f"{image_name}.png"
            hr_path.parent.mkdir(parents=True, exist_ok=True)
            hr.save(hr_path)
            print(f"\n[HR] {image_path.name}: {hr.size} -> {hr_path}")

            for degradation_name in cfg.degradations:
                lq = degrade(hr, degradation_name, scale=cfg.scale, seed=cfg.seed)
                lq_path = cfg.output_dir / "lq" / image_name / f"{degradation_name}.png"
                lq_path.parent.mkdir(parents=True, exist_ok=True)
                lq.save(lq_path)

                for mask_name in cfg.masks:
                    mask = create_mask(mask_name, lq.size)
                    output_folder = cfg.output_dir / "restored" / image_name / degradation_name
                    output_folder.mkdir(parents=True, exist_ok=True)
                    mask_path = output_folder / f"mask_{mask_name}.png"
                    mask_to_pil(mask).save(mask_path)

                    print(f"[RUN] {image_name} | {degradation_name} | mask={mask_name}", flush=True)
                    if mask_name == "zeros":
                        sr = lq.copy()
                    else:
                        if model is None:
                            model = MaskDfModel(cfg)
                        sr = model.enhance(lq, mask, seed=cfg.seed)

                    sr_path = output_folder / f"sr_{mask_name}.png"
                    sr.save(sr_path)
                    figure_path = cfg.output_dir / "comparisons" / image_name / f"{degradation_name}_{mask_name}.png"
                    make_comparison_figure(
                        hr, lq, mask, sr, figure_path,
                        title=f"{image_name}: {degradation_name}, mask={mask_name}",
                    )
                    case = {
                        "image": image_name,
                        "degradation": degradation_name,
                        "mask": mask_name,
                        "mask_coverage": float(mask.mean()),
                        "hr_path": str(hr_path.relative_to(cfg.output_dir)),
                        "lq_path": str(lq_path.relative_to(cfg.output_dir)),
                        "mask_path": str(mask_path.relative_to(cfg.output_dir)),
                        "sr_path": str(sr_path.relative_to(cfg.output_dir)),
                        "comparison_path": str(figure_path.relative_to(cfg.output_dir)),
                        "outside_max_abs_diff": max_outside_difference(lq, sr, mask),
                    }
                    cases.append(case)
                    _write_json(cfg.output_dir / "cases.json", cases)
    finally:
        if model is not None:
            model.close()
            del model
        gc.collect()
    return cases


def evaluate(cfg: ExperimentConfig, cases: list[dict]) -> pd.DataFrame:

    evaluator = None
    if cfg.evaluate_perceptual:
        from metrics import PerceptualMetrics
        print("\n[METRICS] Загружаю LPIPS и CLIP-IQA на CPU...")
        evaluator = PerceptualMetrics(device="cpu")

    baseline_cache: dict[tuple[str, str], dict[str, float]] = {}
    rows: list[dict] = []

    try:
        for i, case in enumerate(cases, 1):
            with Image.open(cfg.output_dir / case["hr_path"]) as img:
                hr = img.convert("RGB")
            with Image.open(cfg.output_dir / case["lq_path"]) as img:
                lq = img.convert("RGB")
            with Image.open(cfg.output_dir / case["sr_path"]) as img:
                sr = img.convert("RGB")
            with Image.open(cfg.output_dir / case["mask_path"]) as img:
                mask = (np.asarray(img.convert("L")) > 127).astype(np.uint8)

            key = (case["image"], case["degradation"])
            if key not in baseline_cache:
                baseline_cache[key] = {
                    "psnr_input": psnr(hr, lq),
                    "lpips_input": evaluator.lpips(hr, lq) if evaluator is not None else float("nan"),
                    "clip_iqa_input": evaluator.clip_iqa(lq) if evaluator is not None else float("nan"),
                }
            baseline = baseline_cache[key]

            if mask.any():
                psnr_output = psnr(hr, sr)
                lpips_output = evaluator.lpips(hr, sr) if evaluator is not None else float("nan")
                clip_iqa_output = evaluator.clip_iqa(sr) if evaluator is not None else float("nan")
            else:
                psnr_output = baseline["psnr_input"]
                lpips_output = baseline["lpips_input"]
                clip_iqa_output = baseline["clip_iqa_input"]

            row = {
                "image": case["image"],
                "degradation": case["degradation"],
                "mask": case["mask"],
                "mask_coverage": case["mask_coverage"],
                **baseline,
                "psnr_output": psnr_output,
                "psnr_gain": psnr_output - baseline["psnr_input"],
                "psnr_mask_input": psnr(hr, lq, mask),
                "psnr_mask_output": psnr(hr, sr, mask),
                "lpips_output": lpips_output,
                "clip_iqa_output": clip_iqa_output,
                "outside_max_abs_diff": case["outside_max_abs_diff"],
                "sr_path": case["sr_path"],
                "comparison_path": case["comparison_path"],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(cfg.output_dir / "metrics.csv", index=False)
            print(
                f"[METRIC {i:02d}/{len(cases)}] {row['degradation']:14s} {row['mask']:7s} "
                f"PSNR={psnr_output:.2f} dB, Δ={row['psnr_gain']:+.2f} dB"
            )
    finally:
        del evaluator
        gc.collect()

    return pd.DataFrame(rows)

def main() -> None:
    cfg = CONFIG
    if cfg.image_size <= 0 or cfg.image_size % 8 != 0:
        raise ValueError("image_size должен быть положительным и кратным 8")
    if not (0.0 < cfg.strength <= 1.0):
        raise ValueError("strength должен находиться в диапазоне (0, 1]")
    if cfg.max_images < 1 or cfg.num_inference_steps < 1 or cfg.scale < 1:
        raise ValueError("max_images, num_inference_steps и scale должны быть >= 1")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_dir / "config.json", asdict(cfg))
    print(f"Входные изображения: {cfg.hr_dir.resolve()}")
    print(f"Результаты: {cfg.output_dir.resolve()}")

    cases = run_inference(cfg)

    if not cases:
        raise RuntimeError("Нет завершённых экспериментов")



if __name__ == "__main__":
    main()
