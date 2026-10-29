#!/usr/bin/env python3
from __future__ import annotations

import sys
sys.path.append("vggt")

import argparse
import os
import tempfile
from typing import Iterable

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

try:
    # Run this script from a cloned VGGT repository.
    from visual_util import predictions_to_glb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "visual_util.py was not found. Run this script from the VGGT repository "
        "or add that repository to PYTHONPATH."
    ) from exc


# DEFAULT_MODEL = "facebook/VGGT-1B-Commercial"
DEFAULT_MODEL = "facebook/VGGT-1B"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT through a Gradio UI")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help="Hugging Face model id; commercial default is VGGT-1B-Commercial")
    parser.add_argument("--weights", default=None,
                        help="Optional local VGGT checkpoint; overrides --model-name")
    parser.add_argument("--image-size", type=int, default=518,
                        help="Default inference resolution; must be divisible by 14")
    parser.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop",
                        help="Crop to a square-ish input or preserve content with padding")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Inference device. Defaults to CUDA when available.")
    parser.add_argument("--server-name", default=None,
                        help="Bind address, e.g. 0.0.0.0 for a LAN/server deployment")
    parser.add_argument("--local-network", action="store_true",
                        help="Equivalent to --server-name 0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Ask Gradio to create a temporary public share URL")
    parser.add_argument("--tmp-dir", default=None,
                        help="Directory for generated GLB files")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Maximum number of uploaded images per reconstruction")
    parser.add_argument("--silent", action="store_true", help="Reduce startup logging")
    return parser.parse_args()


def choose_dtype(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def load_model(model_name: str, weights: str | None, device: str) -> VGGT:
    if weights:
        print(f"Loading local VGGT checkpoint: {weights}")
        model = VGGT()
        checkpoint = torch.load(weights, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Loading VGGT checkpoint: {model_name}")
        model = VGGT.from_pretrained(model_name)
    model.eval()
    return model.to(device)


def load_images_at_resolution(image_paths: list[str], resolution: int, mode: str) -> torch.Tensor:
    """Resize inputs while preserving the preprocessing conventions used by VGGT."""
    if resolution < 14 or resolution % 14 != 0:
        raise gr.Error("Resolution must be a positive multiple of 14, such as 224, 364, 518, or 672.")

    tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        if mode == "pad":
            scale = resolution / max(width, height)
            new_width = max(14, round(width * scale / 14) * 14)
            new_height = max(14, round(height * scale / 14) * 14)
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            tensor = TF.to_tensor(image)
            pad_h = resolution - tensor.shape[1]
            pad_w = resolution - tensor.shape[2]
            if pad_h < 0 or pad_w < 0:
                tensor = TF.resize(tensor, [resolution, resolution])
                pad_h = pad_w = 0
            tensor = F.pad(
                tensor,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                value=1.0,
            )
        else:
            new_width = resolution
            new_height = max(14, round(height * resolution / width / 14) * 14)
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            tensor = TF.to_tensor(image)
            if new_height > resolution:
                start = (new_height - resolution) // 2
                tensor = tensor[:, start : start + resolution, :]
        tensors.append(tensor)

    # Batch inference requires equal tensor sizes. Pad differing aspect ratios.
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    tensors = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]), value=1.0) for t in tensors]
    return torch.stack(tensors)


def normalize_files(files: Iterable) -> list[str]:
    paths: list[str] = []
    for item in files or []:
        path = item if isinstance(item, str) else getattr(item, "name", None)
        if path and os.path.isfile(path):
            paths.append(path)
    return sorted(paths)


def reconstruct(
    files,
    confidence_percentile: float,
    prediction_mode: str,
    show_cameras: bool,
    max_images: int,
    resolution: int,
    preprocess_mode: str,
    model: VGGT,
    device: str,
    output_dir: str,
):
    image_paths = normalize_files(files)
    if len(image_paths) < 2:
        raise gr.Error("Upload at least two images of the same scene.")
    if len(image_paths) > max_images:
        raise gr.Error(f"Please upload no more than {max_images} images.")

    print(f"Reconstructing {len(image_paths)} images on {device}")
    images = load_images_at_resolution(image_paths, resolution, preprocess_mode).to(device)
    dtype = choose_dtype(device)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

        # VGGT returns pose encoding; convert it to OpenCV-style extrinsics
        # and intrinsics for the visualization/export helper.
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )

        depth = predictions.get("depth")
        depth_conf = predictions.get("depth_conf")
        if depth is None:
            raise gr.Error("The selected checkpoint did not return depth maps.")

        world_points_from_depth = unproject_depth_map_to_point_map(
            depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )

    # visual_util.predictions_to_glb expects CPU NumPy arrays without batch dims.
    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy().squeeze(0)
        return value

    prediction_dict = {
        "images": images.detach().float().cpu().numpy().squeeze(0).transpose(0, 2, 3, 1),
        "extrinsic": to_numpy(extrinsic),
        "intrinsic": to_numpy(intrinsic),
        "world_points_from_depth": to_numpy(world_points_from_depth),
        "depth_conf": to_numpy(depth_conf) if depth_conf is not None else None,
    }

    point_map = predictions.get("world_points")
    point_conf = predictions.get("world_points_conf")
    if point_map is not None:
        prediction_dict["world_points"] = to_numpy(point_map)
    if point_conf is not None:
        prediction_dict["world_points_conf"] = to_numpy(point_conf)

    # Remove optional None values so the helper can apply its normal fallbacks.
    prediction_dict = {k: v for k, v in prediction_dict.items() if v is not None}

    scene = predictions_to_glb(
        prediction_dict,
        conf_thres=float(confidence_percentile),
        show_cam=bool(show_cameras),
        prediction_mode=prediction_mode,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vggt_reconstruction.glb")
    scene.export(output_path, file_type="glb")
    print(f"Wrote {output_path}")
    return output_path, output_path


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if args.tmp_dir:
        output_dir = args.tmp_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="vggt_gradio_")

    model = load_model(args.model_name, args.weights, device)

    with gr.Blocks(title="VGGT Image-to-3D") as app:
        gr.Markdown(
            "# VGGT Image-to-3D\n"
            "Upload multiple overlapping images of one scene. "
            "The result is a colored GLB point cloud with optional camera frustums."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.File(
                    label="Scene images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                confidence = gr.Slider(
                    minimum=0, maximum=99, value=50, step=1,
                    label="Discard lowest-confidence percentile",
                )
                mode = gr.Radio(
                    choices=["Predicted Pointmap", "Depthmap and Camera"],
                    value="Predicted Pointmap",
                    label="Geometry branch",
                )
                resolution = gr.Dropdown(
                    choices=[224, 336, 364, 518, 672],
                    value=args.image_size,
                    label="Inference resolution",
                    info="Higher values use more VRAM and may preserve more detail.",
                )
                preprocess_mode = gr.Radio(
                    choices=["crop", "pad"], value=args.preprocess_mode,
                    label="Image preprocessing",
                )
                show_cameras = gr.Checkbox(value=True, label="Show camera frustums")
                run = gr.Button("Reconstruct", variant="primary")
            with gr.Column():
                output = gr.Model3D(label="Reconstruction (GLB)")
                download = gr.File(label="Download GLB")

        run.click(
            fn=lambda files, conf, selected_mode, selected_resolution, selected_preprocess, cameras: reconstruct(
                files, conf, selected_mode, cameras,
                args.max_images, int(selected_resolution), selected_preprocess,
                model, device, output_dir,
            ),
            inputs=[image_input, confidence, mode, resolution, preprocess_mode, show_cameras],
            outputs=[output, download],
        )

    app.launch(
        server_name=args.server_name or ("0.0.0.0" if args.local_network else None),
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
