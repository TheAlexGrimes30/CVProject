import numpy as np
import torch
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPImageQualityAssessment


def psnr(reference: Image.Image, prediction: Image.Image, mask: np.ndarray | None = None) -> float:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float64) / 255.0
    pred = np.asarray(prediction.convert("RGB"), dtype=np.float64) / 255.0
    if ref.shape != pred.shape:
        raise ValueError("PSNR requires aligned images with the same dimensions")
    squared = np.square(ref - pred)
    if mask is not None:
        if mask.shape != ref.shape[:2]:
            raise ValueError("Incorrect mask dimensions for masked PSNR")
        selected = mask.astype(bool)
        if not selected.any():
            return float("nan")
        mse = float(np.mean(squared[selected]))
    else:
        mse = float(np.mean(squared))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)


class PerceptualMetrics:

    def __init__(self, device: str = "cpu") -> None:

        self.device = torch.device(device)
        self.lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device).eval()
        self.clip_iqa_model = CLIPImageQualityAssessment(
            model_name_or_path="clip_iqa", prompts=("quality",), data_range=1.0
        ).to(self.device).eval()

    @torch.inference_mode()
    def lpips(self, reference: Image.Image, prediction: Image.Image) -> float:
        a = to_tensor(reference).to(self.device)
        b = to_tensor(prediction).to(self.device)
        return float(self.lpips_model(a, b).detach().cpu().item())

    @torch.inference_mode()
    def clip_iqa(self, image: Image.Image) -> float:
        a = to_tensor(image).to(self.device)
        prediction = self.clip_iqa_model(a)
        if isinstance(prediction, dict):
            prediction = prediction["quality"]
        return float(prediction.reshape(-1)[0].detach().cpu().item())
    