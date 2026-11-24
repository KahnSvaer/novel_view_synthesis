from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..common.colmap import IMAGE_EXTENSIONS
from ..common.loader import INPUT_DIR, load_cameras, load_ply
from .dataset import (
    FrameRecord,
    _build_frame_records,
    _estimate_near_far,
    _normalize_scene_records,
)
from .renderer import get_rays_for_image, render_rays
from .training import get_nerf_models


def _infer_encoding_bands(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Infer (Lx, Ld) positional encoding bands from a saved NeRF state_dict."""
    in_pts = state_dict["linear_layer_1.0.weight"].shape[1]
    in_dirs = state_dict["linear_layer_3.0.weight"].shape[1] - 256
    return in_pts // 6, in_dirs // 6


def load_models_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load coarse and fine NeRF models from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    coarse_sd = ckpt["coarse_model_state_dict"]
    fine_sd = ckpt["fine_model_state_dict"]

    lx, ld = _infer_encoding_bands(coarse_sd)
    coarse_model, fine_model = get_nerf_models(Lx=lx, Ld=ld)
    coarse_model.load_state_dict(coarse_sd)
    fine_model.load_state_dict(fine_sd)

    coarse_model.to(device).eval()
    fine_model.to(device).eval()
    return coarse_model, fine_model


def _rotation_log(rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rotation axis and angle (radians) from a 3x3 rotation matrix."""
    trace = torch.clamp((torch.trace(rot) - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(trace)
    if angle < 1e-6:
        return torch.tensor([1.0, 0.0, 0.0], device=rot.device, dtype=rot.dtype), torch.zeros((), device=rot.device, dtype=rot.dtype)
    skew = torch.stack(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ]
    )
    axis = skew / (2.0 * torch.sin(angle))
    return axis, angle


def _rotation_exp(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula: build a 3x3 rotation matrix from unit axis and angle."""
    if torch.abs(angle) < 1e-8:
        return torch.eye(3, device=axis.device, dtype=axis.dtype)
    kx, ky, kz = axis[0], axis[1], axis[2]
    zero = torch.zeros_like(kx)
    k_mat = torch.stack(
        [
            torch.stack([zero, -kz, ky]),
            torch.stack([kz, zero, -kx]),
            torch.stack([-ky, kx, zero]),
        ]
    )
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return eye + torch.sin(angle) * k_mat + (1.0 - torch.cos(angle)) * (k_mat @ k_mat)


def interpolate_c2w(c2w_a: torch.Tensor, c2w_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate two 4x4 camera-to-world matrices (SO(3) SLERP + linear translation)."""
    rot_a, rot_b = c2w_a[:3, :3], c2w_b[:3, :3]
    rot_rel = rot_a.T @ rot_b
    axis, angle = _rotation_log(rot_rel)
    rot_interp = rot_a @ _rotation_exp(axis, angle * alpha)

    t_interp = (1.0 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]

    c2w = torch.eye(4, device=c2w_a.device, dtype=c2w_a.dtype)
    c2w[:3, :3] = rot_interp
    c2w[:3, 3] = t_interp
    return c2w


def _scale_camera(camera: dict, scale: float) -> dict:
    """Scale camera intrinsics and image dimensions while preserving distortion params."""
    if scale == 1.0:
        return camera

    model = camera["model"]
    params = list(camera["params"])
    new_w = max(1, int(round(camera["width"] * scale)))
    new_h = max(1, int(round(camera["height"] * scale)))
    sx = new_w / camera["width"]
    sy = new_h / camera["height"]

    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        params[0] = params[0] * 0.5 * (sx + sy)
        params[1] = params[1] * sx
        params[2] = params[2] * sy
    elif model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
        params[0] = params[0] * sx
        params[1] = params[1] * sy
        params[2] = params[2] * sx
        params[3] = params[3] * sy

    return {
        "model": model,
        "width": new_w,
        "height": new_h,
        "params": params,
    }


def build_interpolated_trajectory(
    frames: list[FrameRecord],
    *,
    steps_per_pair: int = 8,
    loop: bool = False,
    render_scale: float = 1.0,
) -> list[tuple[dict, torch.Tensor]]:
    """Generate a smooth sequence of (scaled_camera, c2w) poses between input views."""
    if not frames:
        raise ValueError("Need at least one frame to build a trajectory.")
    if len(frames) == 1:
        return [(_scale_camera(frames[0].camera, render_scale), frames[0].camera_to_world)]

    pairs = list(zip(frames[:-1], frames[1:]))
    if loop:
        pairs.append((frames[-1], frames[0]))

    trajectory: list[tuple[dict, torch.Tensor]] = []
    for pair_idx, (frame_a, frame_b) in enumerate(pairs):
        cam = _scale_camera(frame_a.camera, render_scale)
        for step in range(steps_per_pair):
            alpha = step / steps_per_pair
            c2w = interpolate_c2w(frame_a.camera_to_world, frame_b.camera_to_world, alpha)
            trajectory.append((cam, c2w))
        if not loop and pair_idx == len(pairs) - 1:
            cam_last = _scale_camera(frame_b.camera, render_scale)
            trajectory.append((cam_last, frame_b.camera_to_world))

    return trajectory


@torch.no_grad()
def render_camera_frame(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    camera: dict,
    camera_to_world: torch.Tensor,
    *,
    near: float,
    far: float,
    n_coarse: int = 64,
    n_fine: int = 128,
    white_background: bool = False,
    ray_chunk: int = 4096,
    model_chunk: int = 65536,
    fp16: bool = True,
    device: torch.device,
) -> np.ndarray:
    """Render a full RGB image [H, W, 3] uint8 from a camera pose."""
    height = int(camera["height"])
    width = int(camera["width"])
    rays_o, rays_d = get_rays_for_image(camera=camera, camera_to_world=camera_to_world, device=device)

    use_amp = fp16 and device.type == "cuda"
    rgb_chunks = []
    for start in range(0, rays_o.shape[0], ray_chunk):
        end = start + ray_chunk
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = render_rays(
                coarse_model=coarse_model,
                fine_model=fine_model,
                rays_o=rays_o[start:end],
                rays_d=rays_d[start:end],
                near=near,
                far=far,
                n_coarse=n_coarse,
                n_fine=n_fine,
                perturb=False,
                white_background=white_background,
                chunk_size=model_chunk,
            )
        rgb_chunks.append(outputs["fine_rgb"].float().cpu())

    rgb = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3)
    rgb = torch.clamp(rgb * 255.0, 0.0, 255.0).to(torch.uint8).numpy()
    return rgb


def render_trajectory_gif(
    *,
    scene_name: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    steps_per_pair: int = 8,
    render_scale: float = 1.0,
    fps: int = 15,
    n_coarse: int = 64,
    n_fine: int = 128,
    near: float | None = None,
    far: float | None = None,
    normalize_scene: bool = True,
    white_background: bool = False,
    loop: bool = False,
    ray_chunk: int = 4096,
    fp16: bool = True,
    frames_dir: str | Path | None = None,
    resume: bool = True,
    device: str | torch.device | None = None,
) -> Path:
    """Render an interpolated camera trajectory from a NeRF checkpoint and save as a GIF."""
    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    cameras_payload = load_cameras(scene_name)
    points3d, _ = load_ply(scene_name)

    image_root = INPUT_DIR / scene_name
    image_names = sorted(
        path.relative_to(image_root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    frames = _build_frame_records(cameras_payload, image_names)

    if normalize_scene:
        frames, points3d = _normalize_scene_records(frames, points3d)

    if near is None or far is None:
        est_near, est_far = _estimate_near_far(frames, points3d)
        near = est_near if near is None else near
        far = est_far if far is None else far

    coarse_model, fine_model = load_models_from_checkpoint(checkpoint_path, torch_device)
    trajectory = build_interpolated_trajectory(
        frames,
        steps_per_pair=steps_per_pair,
        loop=loop,
        render_scale=render_scale,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_root = Path(frames_dir) if frames_dir is not None else out_path.with_name(f"{out_path.stem}_frames")
    frame_root.mkdir(parents=True, exist_ok=True)

    pil_frames: list[Image.Image] = []
    total = len(trajectory)
    for idx, (cam, c2w) in enumerate(trajectory):
        frame_path = frame_root / f"frame_{idx:04d}.png"
        expected_size = (int(cam["width"]), int(cam["height"]))
        if resume and frame_path.exists():
            with Image.open(frame_path) as cached_im:
                if cached_im.size == expected_size:
                    pil_frames.append(cached_im.convert("RGB"))
                    print(f"[render] frame {idx + 1}/{total} ({expected_size[0]}x{expected_size[1]}) [cached]")
                    continue

        print(f"[render] frame {idx + 1}/{total} ({expected_size[0]}x{expected_size[1]})")
        rgb = render_camera_frame(
            coarse_model=coarse_model,
            fine_model=fine_model,
            camera=cam,
            camera_to_world=c2w,
            near=near,
            far=far,
            n_coarse=n_coarse,
            n_fine=n_fine,
            white_background=white_background,
            ray_chunk=ray_chunk,
            fp16=fp16,
            device=torch_device,
        )
        frame_im = Image.fromarray(rgb)
        frame_im.save(frame_path)
        pil_frames.append(frame_im)

    duration_ms = max(1, int(round(1000.0 / max(fps, 1))))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF ({total} frames) to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an interpolated GIF from a trained NeRF.")
    parser.add_argument("--scene", type=str, default="images")
    parser.add_argument("--checkpoint", type=str, default="outputs/images_nerf/latest.pt")
    parser.add_argument("--output", type=str, default="outputs/images_nerf/render.gif")
    parser.add_argument("--steps-per-pair", type=int, default=8, help="Frames per consecutive camera pair.")
    parser.add_argument("--render-scale", type=float, default=1.0, help="Resolution scale factor (1.0 = full image size).")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--n-coarse", type=int, default=64)
    parser.add_argument("--n-fine", type=int, default=128)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--no-normalize-scene", action="store_true")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--loop", action="store_true", help="Loop back from last camera to first.")
    parser.add_argument("--ray-chunk", type=int, default=4096)
    parser.add_argument("--no-fp16", action="store_true", help="Disable CUDA FP16 autocast during inference.")
    parser.add_argument("--frames-dir", type=str, default=None, help="Directory to save incremental PNG frames.")
    parser.add_argument("--no-resume", action="store_true", help="Re-render frames even if already saved.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_trajectory_gif(
        scene_name=args.scene,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        steps_per_pair=args.steps_per_pair,
        render_scale=args.render_scale,
        fps=args.fps,
        n_coarse=args.n_coarse,
        n_fine=args.n_fine,
        near=args.near,
        far=args.far,
        normalize_scene=not args.no_normalize_scene,
        white_background=args.white_background,
        loop=args.loop,
        ray_chunk=args.ray_chunk,
        fp16=not args.no_fp16,
        frames_dir=args.frames_dir,
        resume=not args.no_resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
