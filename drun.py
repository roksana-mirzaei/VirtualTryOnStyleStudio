from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageDraw, ImageFont
from tryon.mask import Masker
from tryon.pipeline import Pipeline
from utils import resize_and_crop, resize_and_padding
import shutil
import uvicorn
import gc
from evaluation.SSIM import SSIMCalculator
from evaluation.PSNR import PSNRCalculator

app = FastAPI()
metrics = {}

# Base directory paths
BASE_DIR = "/app"
stablediffusion_model = "booksforcharlie/stable-diffusion-inpainting"

CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "final_output")


def force_make_dirs(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    print(f"{dir} created.")


pipeline = Pipeline(
    model_path=stablediffusion_model,
    attention_weights=os.path.join(CHECKPOINTS_DIR, 'attn/tryon.safetensors'),
    device='cuda'
)

mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
masker = Masker(
    densepose_models_path=os.path.join(CHECKPOINTS_DIR, 'densepose'),
    human_parsing_path=os.path.join(CHECKPOINTS_DIR, "human_parsing"),
    device='cuda',
)


def memory_snapshot():
    snapshot = torch.cuda.memory_snapshot()
    total_size = sum(segment["total_size"] for segment in snapshot if segment["device"] == 0)
    allocated_size = sum(segment["allocated_size"] for segment in snapshot if segment["device"] == 0)
    active_size = sum(segment["active_size"] for segment in snapshot if segment["device"] == 0)
    requested_size = sum(segment["requested_size"] for segment in snapshot if segment["device"] == 0)

    return total_size, allocated_size, active_size, requested_size


@app.post("/process")
async def run_app(
        person_image: UploadFile = File(..., description="Human Image"),
        cloth_image: UploadFile = File(..., description="Cloth Image"),
        cloth_type: str = Form(..., description="top, bottom, full"),
):
    try:
        gc.collect()
        torch.cuda.empty_cache()

        total_size_before, allocated_size_before, active_size_before, requested_size_before = memory_snapshot()
        print(f"Before Processing - Total: {total_size_before / (1024 ** 2):.2f} MB, Allocated: {allocated_size_before / (1024 ** 2):.2f} MB, Active: {active_size_before / (1024 ** 2):.2f} MB, Requested: {requested_size_before / (1024 ** 2):.2f} MB")

        guidance_scale = 2.5
        seed = 42
        num_inference_steps = 50

        [force_make_dirs(dir) for dir in [UPLOAD_DIR, OUTPUT_DIR]]

        person_image_path = os.path.join(UPLOAD_DIR, person_image.filename)
        cloth_image_path = os.path.join(UPLOAD_DIR, cloth_image.filename)

        with open(person_image_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)

        with open(cloth_image_path, "wb") as buffer:
            shutil.copyfileobj(cloth_image.file, buffer)

        person_image = Image.open(person_image_path).convert("RGB")
        cloth_image = Image.open(cloth_image_path).convert("RGB")

        person_image = resize_and_crop(person_image, (768, 1024))
        cloth_image = resize_and_padding(cloth_image, (768, 1024))

        mask = masker(person_image, cloth_type)['mask']
        mask = mask_processor.blur(mask, blur_factor=9)

        generator = torch.Generator(device='cuda').manual_seed(seed) if seed != -1 else None

        # Run inference
        result_image = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            inference_steps=num_inference_steps,
            guidance_strength=guidance_scale,
            random_generator=generator
        )

        result_image = result_image['image'][0]

        result_save_path = os.path.join(OUTPUT_DIR, f"output_.png")
        result_image.save(result_save_path)

        SSIMEval = SSIMCalculator(person_image_path, result_save_path)
        ssim_score = SSIMEval()

        PSNREval = PSNRCalculator(person_image_path, result_save_path)
        psnr_score, mse_score = PSNREval()

        global metrics
        metrics = {
            "Structural Similarity Index (SSIM) ---> ": ssim_score,
            "Peak Signal-to-Noise Ratio (PSNR) ---> ": psnr_score,
            "Mean Squared Error (MSE) ---> ": mse_score
        }

        del person_image, cloth_image, mask, result_image, generator

        (total_size_after,
         allocated_size_after,
         active_size_after,
         requested_size_after) = memory_snapshot()

        print(f"After Processing - Total: {total_size_after / (1024 ** 2):.2f} MB, Allocated: {allocated_size_after / (1024 ** 2):.2f} MB, Active: {active_size_after / (1024 ** 2):.2f} MB, Requested: {requested_size_after / (1024 ** 2):.2f} MB")
        print(f"Active Memory Used: {(active_size_after - active_size_before) / (1024 ** 2):.2f} MB")

        gc.collect()
        torch.cuda.empty_cache()

        return FileResponse(result_save_path)

    except Exception as e:
        return {"error": str(e)}


@app.get("/get_eval_metrics")
async def get_metrics():
    try:
        return metrics
    except Exception as e:
        return {"error": str(e)}



@app.post("/masks_and_segmentations")
async def masks_and_segmentations(
        person_image: UploadFile = File(..., description="Human Image"),
        cloth_image: UploadFile = File(..., description="Cloth Image"),
        cloth_type: str = Form(..., description="top, bottom, full"),
):
    try:

        guidance_scale = 2.5
        seed = 42
        num_inference_steps = 50

        [force_make_dirs(dir) for dir in [UPLOAD_DIR, OUTPUT_DIR]]

        person_image_path = os.path.join(UPLOAD_DIR, person_image.filename)
        cloth_image_path = os.path.join(UPLOAD_DIR, cloth_image.filename)

        with open(person_image_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)

        with open(cloth_image_path, "wb") as buffer:
            shutil.copyfileobj(cloth_image.file, buffer)

        person_image = Image.open(person_image_path).convert("RGB")
        cloth_image = Image.open(cloth_image_path).convert("RGB")

        person_image = resize_and_crop(person_image, (768, 1024))
        cloth_image = resize_and_padding(cloth_image, (768, 1024))

        mask = masker(person_image, cloth_type)['mask']
        mask_blurred = mask_processor.blur(mask, blur_factor=9)

        densepose_segmentation = masker(person_image, cloth_type)['densepose']
        schp_lip_segmentation = masker(person_image, cloth_type)['schp_lip']
        schp_atr_segmentation = masker(person_image, cloth_type)['schp_atr']

        generator = torch.Generator(device='cuda').manual_seed(seed) if seed != -1 else None

        # Run inference
        result_image = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask_blurred,
            inference_steps=num_inference_steps,
            guidance_strength=guidance_scale,
            random_generator=generator
        )

        result_image = result_image['masked_person_image'][0]

        #just to view
        images = [
            (mask, "Mask"),
            (densepose_segmentation, "DensePose Segmentation"),
            (schp_lip_segmentation, "SCHP-LIP Segmentation"),
            (schp_atr_segmentation, "SCHP-ATR Segmentation"),
            (result_image, "Masked Person Image - Input 1"),
            (cloth_image, "Cloth Image - Input 2")
        ]
        font = ImageFont.load_default(size=30.0)

        widths, heights = zip(*(img[0].size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        concatenated_image = Image.new("RGB", (total_width, max_height + 50), "white")

        draw = ImageDraw.Draw(concatenated_image)
        x_offset = 0
        for img, label in images:
            concatenated_image.paste(img, (x_offset, 50))
            draw.text((x_offset + img.width // 2 - len(label) * 5, 10), label, fill="black", font=font)
            x_offset += img.width

        result_save_path = os.path.join(OUTPUT_DIR, f"output_labeled.png")
        concatenated_image.save(result_save_path)

        #save memory
        del person_image, concatenated_image, mask, densepose_segmentation, schp_lip_segmentation, schp_atr_segmentation, result_image, generator

        return FileResponse(result_save_path)

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
