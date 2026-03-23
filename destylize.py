#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from controlnet_aux import MidasDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)


DEFAULT_PROMPT = "raw photograph, 8k uhd, dslr, soft lighting, highly detailed nature landscape"
DEFAULT_NEG_PROMPT = (
    "painting, brushstrokes, impressionist, illustration, stylized, art, cartoon, "
    "drawing, sketch, watercolor, oil paint"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Destylize paintings using SDXL + ControlNet Depth Img2Img.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with stylized input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for destylized images.")
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None,
        help="Optional directory for saving intermediate depth maps. Defaults to sibling 'depth_maps'.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="Optional directory for saving resized/padded debug inputs. Defaults to sibling 'processed_inputs'.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Photorealistic SDXL checkpoint.",
    )
    parser.add_argument(
        "--controlnet_model",
        type=str,
        default="diffusers/controlnet-depth-sdxl-1.0",
        help="ControlNet depth checkpoint.",
    )
    parser.add_argument(
        "--depth_annotator",
        type=str,
        default="lllyasviel/Annotators",
        help="HF repo for controlnet_aux MiDaS annotator.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.9)
    parser.add_argument("--strength", type=float, default=0.4, help="Img2Img denoise strength. Lower preserves more.")
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="pad",
        choices=["pad", "crop"],
        help="How to fit source aspect ratio into target resolution.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def pad_or_crop_resize(path: Path, width: int, height: int, resize_mode: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    dst_ratio = width / height

    if resize_mode == "crop":
        if src_ratio > dst_ratio:
            new_w = int(src_h * dst_ratio)
            left = (src_w - new_w) // 2
            image = image.crop((left, 0, left + new_w, src_h))
        else:
            new_h = int(src_w / dst_ratio)
            top = (src_h - new_h) // 2
            image = image.crop((0, top, src_w, top + new_h))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    scale = min(width / src_w, height / src_h)
    rs_w = max(64, int(round((src_w * scale) / 64) * 64))
    rs_h = max(64, int(round((src_h * scale) / 64) * 64))
    rs_w = min(rs_w, width)
    rs_h = min(rs_h, height)

    resized = image.resize((rs_w, rs_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    ox = (width - rs_w) // 2
    oy = (height - rs_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def save_depth(depth_img: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    depth_img.save(target)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.depth_dir is None:
        depth_dir = output_dir.parent / "depth_maps"
    else:
        depth_dir = Path(args.depth_dir)
    if args.processed_dir is None:
        processed_dir = output_dir.parent / "processed_inputs"
    else:
        processed_dir = Path(args.processed_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=dtype)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    if device == "cuda":
        pipe.enable_attention_slicing()

    depth_estimator = MidasDetector.from_pretrained(args.depth_annotator)

    print(f"[INFO] Found {len(images)} images in {input_dir}")
    print(f"[INFO] Writing destylized outputs to {output_dir}")
    print(f"[INFO] Writing depth maps to {depth_dir}")
    print(f"[INFO] Writing processed inputs to {processed_dir}")
    print(
        "[INFO] Hyperparameters: "
        f"controlnet_conditioning_scale={args.controlnet_conditioning_scale}, "
        f"guidance_scale={args.guidance_scale}, "
        f"strength={args.strength}, "
        f"steps={args.num_inference_steps}, "
        f"resize_mode={args.resize_mode}"
    )

    for idx, img_path in enumerate(images, start=1):
        processed = pad_or_crop_resize(img_path, args.width, args.height, args.resize_mode)
        depth_map = depth_estimator(processed)

        stem = img_path.stem
        depth_out = depth_dir / f"{stem}_depth.png"
        processed_out = processed_dir / f"{stem}_processed.png"
        out_path = output_dir / f"{stem}_destylized.png"

        save_depth(depth_map, depth_out)
        processed.save(processed_out)

        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=processed,
            control_image=depth_map,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            strength=args.strength,
            generator=generator,
            width=args.width,
            height=args.height,
        ).images[0]
        result.save(out_path)
        print(f"[{idx}/{len(images)}] {img_path.name} -> {out_path.name}")

    print("[INFO] Destylization complete.")


if __name__ == "__main__":
    main()
