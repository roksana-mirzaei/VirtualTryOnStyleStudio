import inspect
import os
from typing import Union
import PIL
import numpy as np
import torch
from transformers import CLIPImageProcessor
from accelerate import load_checkpoint_in_model
import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from utils import (prepare_image, compute_vae_encodings, numpy_to_pil, prepare_mask_image, resize_and_crop, resize_and_padding)
from tryon.attn_processor import SkipAttnProcessor
from tryon.utils import get_trainable_module, init_adapter


def validate_inputs(image, condition_image, mask, output_width, output_height):
    if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
        return image, condition_image, mask
    assert image.size == mask.size, "Image and mask dimensions must match."
    image = resize_and_crop(image, (output_width, output_height))
    mask = resize_and_crop(mask, (output_width, output_height))
    condition_image = resize_and_padding(condition_image, (output_width, output_height))
    return image, condition_image, mask


class Pipeline:
    def __init__(
            self,
            model_path,
            attention_weights,
            precision=torch.float32,
            device='cuda',
            compile_model=False,
            enable_safety=True,
            use_tf32_acceleration=True,
    ):
        self.device = device
        self.precision = precision
        self.enable_safety = enable_safety

        self.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=precision)
        self.unet_model = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device, dtype=precision)

        if enable_safety:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path, subfolder="safety_checker").to(device, dtype=precision)

        init_adapter(self.unet_model, cross_attention_class=SkipAttnProcessor)
        self.attention_modules = get_trainable_module(self.unet_model, "attention")
        load_checkpoint_in_model(self.attention_modules, attention_weights)

        if compile_model:
            self.unet_model = torch.compile(self.unet_model)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")

        if use_tf32_acceleration:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def apply_safety_checker(self, processed_image):
        if self.safety_checker is None:
            bad_image_flag = None
        else:
            safety_input = self.feature_extractor(processed_image, return_tensors="pt").to(self.device)
            processed_image, bad_image_flag = self.safety_checker(
                images=processed_image, clip_input=safety_input.pixel_values.to(self.precision)
            )
        return processed_image, bad_image_flag

    def prepare_extra_scheduler_args(self, random_generator, noise_eta):
        extra_args = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_args["eta"] = noise_eta
        if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_args["generator"] = random_generator
        return extra_args

    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            condition_image: Union[PIL.Image.Image, torch.Tensor],
            mask: Union[PIL.Image.Image, torch.Tensor],
            inference_steps: int,
            guidance_strength: float,
            output_height: int = 1024,
            output_width: int = 768,
            random_generator=None,
            noise_eta=1.0,
            **kwargs
    ):
        with torch.no_grad():
            concat_axis = -2

            image, condition_image, mask = validate_inputs(image, condition_image, mask, output_width, output_height)

            image = prepare_image(image).to(self.device, dtype=self.precision)
            condition_image = prepare_image(condition_image).to(self.device, dtype=self.precision)
            mask = prepare_mask_image(mask).to(self.device, dtype=self.precision)

            masked_image = image * (mask < 0.5)
            masked_latent = compute_vae_encodings(masked_image, self.vae)
            masked_person_image = numpy_to_pil(masked_image.cpu().permute(0, 2, 3, 1).float().numpy())

            conditioned_latent = compute_vae_encodings(condition_image, self.vae)
            resized_mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

            del image, mask, condition_image

            masked_latent_stack = torch.cat([masked_latent, conditioned_latent], dim=concat_axis)
            resized_mask_latent_stack = torch.cat([resized_mask_latent, torch.zeros_like(resized_mask_latent)], dim=concat_axis)
            latent_noise = randn_tensor(
                masked_latent_stack.shape,
                generator=random_generator,
                device=masked_latent_stack.device,
                dtype=self.precision,
            )
            self.scheduler.set_timesteps(inference_steps, device=self.device)
            time_steps = self.scheduler.timesteps
            latent_noise = latent_noise * self.scheduler.init_noise_sigma
            apply_guidance = (guidance_strength > 1.0)
            if apply_guidance:
                masked_latent_stack = torch.cat(
                    [
                        torch.cat([masked_latent, torch.zeros_like(conditioned_latent)], dim=concat_axis),
                        masked_latent_stack,
                    ]
                )
                resized_mask_latent_stack = torch.cat([resized_mask_latent_stack] * 2)

            extra_scheduler_args = self.prepare_extra_scheduler_args(random_generator, noise_eta)
            warmup_steps = (len(time_steps) - inference_steps * self.scheduler.order)

            with tqdm.tqdm(total=inference_steps) as progress_bar:
                for i, timestep in enumerate(time_steps):
                    model_input_latent = (torch.cat([latent_noise] * 2) if apply_guidance else latent_noise)
                    model_input_latent = self.scheduler.scale_model_input(model_input_latent, timestep)
                    inpainting_model_input = torch.cat([model_input_latent, resized_mask_latent_stack, masked_latent_stack], dim=1)
                    noise_prediction = self.unet_model(
                        inpainting_model_input,
                        timestep.to(self.device),
                        encoder_hidden_states=None,
                        return_dict=False,
                    )[0]
                    if apply_guidance:
                        unconditioned_pred, conditioned_pred = noise_prediction.chunk(2)
                        noise_prediction = unconditioned_pred + guidance_strength * (conditioned_pred - unconditioned_pred)
                    latent_noise = self.scheduler.step(
                        noise_prediction, timestep, latent_noise, **extra_scheduler_args
                    ).prev_sample
                    if i == len(time_steps) - 1 or ((i + 1) > warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            latent_noise = latent_noise.split(latent_noise.shape[concat_axis] // 2, dim=concat_axis)[0]
            latent_noise = 1 / self.vae.config.scaling_factor * latent_noise
            decoded_image = self.vae.decode(latent_noise.to(self.device, dtype=self.precision)).sample
            decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
            decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = numpy_to_pil(decoded_image)

            if self.enable_safety:
                bad_image = '/app/bad_content3.jpg'
                bad_image = PIL.Image.open(bad_image).resize(image[0].size)
                numpy_img = np.array(image)
                _, bad_img_flagged = self.apply_safety_checker(processed_image=numpy_img)
                for i, flagged in enumerate(bad_img_flagged):
                    if flagged:
                        image[i] = bad_image

            return {"image": image, "masked_person_image": masked_person_image}
