#!/usr/bin/env python3
"""Pre-download all HuggingFace models to a local cache.

Run this on the LOGIN NODE (which has internet) before submitting Slurm jobs.
The script warms the HF cache so compute nodes can load everything offline.

Usage:
    conda activate cos526_destylize
    python download_models.py [--cache_dir /scratch/network/$USER/hf_cache]
"""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download HF models for offline use.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join("/scratch/network", os.environ["USER"], "hf_cache"),
        help="Local directory to cache model weights.",
    )
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")

    from controlnet_aux import MidasDetector
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

    print(f"[INFO] Downloading models to cache: {args.cache_dir}")

    print("[1/3] ControlNet (SDXL Depth): diffusers/controlnet-depth-sdxl-1.0")
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0")

    print("[2/3] Base model (SDXL): stabilityai/stable-diffusion-xl-base-1.0")
    StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
    )

    print("[3/3] Depth estimator: lllyasviel/Annotators (MiDaS)")
    MidasDetector.from_pretrained("lllyasviel/Annotators")

    print(f"[INFO] All models cached in {args.cache_dir}")
    print("[INFO] You can now submit the Slurm job with offline mode.")


if __name__ == "__main__":
    main()
