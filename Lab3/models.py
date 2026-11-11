import gc

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline

from Lab3.image_masks import validate_mask, mask_to_pil, hard_composite


class MaskDfModel:
    def __init__(self, config):

        requested = config.device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else (
                "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
            )

        self.device_name = requested
        self.device = torch.device(requested)
        dtype = torch.float16 if requested == "cuda" else torch.float32
        print(f"Loading {config.model_id} on {self.device} ({dtype}); first use may download weights...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(config.model_id, dtype=dtype)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        if requested == "cuda" and config.use_cpu_offload_on_cuda:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=False)
        self.config = config

    @torch.inference_mode()
    def enhance(self, degraded: Image.Image, mask: np.ndarray, seed: int) -> Image.Image:
        degraded = degraded.convert("RGB")
        validate_mask(mask, degraded.size)
        if not np.any(mask):
            return degraded.copy()
        if degraded.width % 8 or degraded.height % 8:
            raise ValueError("Image sides must be divisible by 8 for Stable Diffusion VAE")

        pipe = self.pipe
        device = pipe._execution_device
        vae_dtype = pipe.vae.dtype
        image_tensor = pipe.image_processor.preprocess(
            degraded, height=degraded.height, width=degraded.width
        ).to(device=device, dtype=vae_dtype)
        clean_latents = pipe.vae.encode(image_tensor).latent_dist.mode() * pipe.vae.config.scaling_factor
        latent_h, latent_w = clean_latents.shape[-2:]
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        latent_mask = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")
        latent_mask = latent_mask.to(device=device, dtype=clean_latents.dtype)

        fixed_generator = torch.Generator(device=device).manual_seed(seed + 12345)
        fixed_noise = torch.randn(
            clean_latents.shape, generator=fixed_generator,
            device=device, dtype=clean_latents.dtype,
        )
        inference_generator = torch.Generator(device=device).manual_seed(seed)
        steps = self.config.num_inference_steps
        strength = self.config.strength

        def anchor_callback(current_pipe, step: int, timestamp, callback_kwargs: dict) -> dict:
            latents = callback_kwargs["latents"]
            active_times, _ = current_pipe.get_timesteps(steps, strength, device)
            if step + 1 < len(active_times):
                next_t = active_times[step + 1].reshape(1).to(device=device)
                original_noised = current_pipe.scheduler.add_noise(clean_latents, fixed_noise, next_t)
            else:
                original_noised = clean_latents
            callback_kwargs["latents"] = latent_mask * latents + (1.0 - latent_mask) * original_noised
            return callback_kwargs

        result = pipe(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            image=degraded,
            mask_image=mask_to_pil(mask),
            height=degraded.height,
            width=degraded.width,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=self.config.guidance_scale,
            generator=inference_generator,
            callback_on_step_end=anchor_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="pil"
        )

        enhanced = result.images[0]
        final = hard_composite(degraded, enhanced, mask)
        del clean_latents, image_tensor, fixed_noise, latent_mask, result, enhanced
        return final

    def close(self) -> None:
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
