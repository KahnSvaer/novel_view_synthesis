from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from ..common.colmap import IMAGE_EXTENSIONS
from ..common.loader import INPUT_DIR, load_cameras, load_ply
from .renderer import find_camera_ray


@dataclass(frozen=True)
class FrameRecord:
    image_name: str
    camera: dict
    camera_to_world: torch.Tensor


class NeRFRayDataset(Dataset):
    """Samples random pixels from images and computes world-space rays on the fly."""

    def __init__(
        self,
        image_root: str | Path,
        frames: list[FrameRecord],
        rays_per_item: int,
        near: float,
        far: float,
        *,
        deterministic: bool,
        seed: int,
    ) -> None:
        super().__init__()
        if rays_per_item <= 0:
            raise ValueError("rays_per_item must be a positive integer.")
        if near <= 0.0 or far <= near:
            raise ValueError("Require 0 < near < far.")

        self.image_root = Path(image_root)
        self.frames = frames
        self.rays_per_item = rays_per_item
        self.near = near
        self.far = far
        self.deterministic = deterministic
        self.seed = seed

        self._images: list[torch.Tensor] = []
        self._load_images()

    def _load_images(self) -> None:
        for frame in self.frames:
            image_path = self.image_root / frame.image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                width, height = img.size
                rgb = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())

            expected_w = int(frame.camera["width"])
            expected_h = int(frame.camera["height"])
            if width != expected_w or height != expected_h:
                raise ValueError(
                    f"Image/camera size mismatch for {frame.image_name}: "
                    f"image=({width}, {height}), camera=({expected_w}, {expected_h})"
                )

            self._images.append(rgb)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame = self.frames[index]
        rgb = self._images[index]
        height, width, _ = rgb.shape

        if self.deterministic:
            generator = torch.Generator()
            generator.manual_seed(self.seed + index)
            px = torch.randint(0, width, (self.rays_per_item,), generator=generator)
            py = torch.randint(0, height, (self.rays_per_item,), generator=generator)
        else:
            px = torch.randint(0, width, (self.rays_per_item,))
            py = torch.randint(0, height, (self.rays_per_item,))

        rays_o, rays_d = find_camera_ray(
            pixel_x=px.to(torch.float32),
            pixel_y=py.to(torch.float32),
            camera=frame.camera,
            camera_to_world=frame.camera_to_world,
        )
        target_rgb = rgb[py, px].to(torch.float32) / 255.0

        near = torch.full((self.rays_per_item, 1), self.near, dtype=torch.float32)
        far = torch.full((self.rays_per_item, 1), self.far, dtype=torch.float32)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target_rgb": target_rgb,
            "near": near,
            "far": far,
        }


def _collate_rays(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate ray dictionaries into a flat batch [N_rays, ...]."""
    keys = batch[0].keys()
    return {key: torch.cat([sample[key] for sample in batch], dim=0) for key in keys}


def _build_frame_records(cameras_payload: dict, image_names: list[str]) -> list[FrameRecord]:
    image_to_info = {
        image_info["name"]: image_info
        for image_info in cameras_payload["images"].values()
    }

    records: list[FrameRecord] = []
    for image_name in image_names:
        if image_name not in image_to_info:
            continue

        image_info = image_to_info[image_name]
        camera_id = str(image_info["camera_id"])
        camera = cameras_payload["cameras"].get(camera_id)
        if camera is None:
            continue

        c2w = torch.tensor(image_info["camera_to_world"], dtype=torch.float32)
        records.append(
            FrameRecord(
                image_name=image_name,
                camera=camera,
                camera_to_world=c2w,
            )
        )

    if not records:
        raise RuntimeError("No usable camera/image records found for NeRF training.")

    return records


def _normalize_scene_records(
    records: list[FrameRecord],
    points3d: np.ndarray,
) -> tuple[list[FrameRecord], np.ndarray]:
    """Center and scale camera poses and sparse points so the scene fits near [-1, 1]^3."""
    if len(points3d) > 0:
        center = np.median(points3d, axis=0)
        dists = np.linalg.norm(points3d - center, axis=-1)
        scale = float(np.percentile(dists, 95))
    else:
        cam_positions = np.stack([r.camera_to_world[:3, 3].numpy() for r in records], axis=0)
        center = cam_positions.mean(axis=0)
        scale = float(np.max(np.linalg.norm(cam_positions - center, axis=-1)))

    scale = max(scale, 1e-3)
    center_t = torch.tensor(center, dtype=torch.float32)

    normalized: list[FrameRecord] = []
    for r in records:
        c2w = r.camera_to_world.clone()
        c2w[:3, 3] = (c2w[:3, 3] - center_t) / scale
        normalized.append(
            FrameRecord(
                image_name=r.image_name,
                camera=r.camera,
                camera_to_world=c2w,
            )
        )

    norm_points = (points3d - center) / scale if len(points3d) > 0 else points3d
    return normalized, norm_points


def _estimate_near_far(
    records: list[FrameRecord],
    points3d: np.ndarray,
) -> tuple[float, float]:
    """Estimate near/far depth bounds along camera forward (+Z) axes from sparse points."""
    if len(points3d) == 0:
        return 0.1, 6.0

    z_samples = []
    for r in records:
        cam_pos = r.camera_to_world[:3, 3].numpy()
        forward = r.camera_to_world[:3, 2].numpy()
        depths = (points3d - cam_pos) @ forward
        positive = depths[depths > 1e-3]
        if len(positive) > 0:
            z_samples.append(positive)

    if not z_samples:
        return 0.1, 6.0

    all_z = np.concatenate(z_samples, axis=0)
    near = max(0.05, float(np.percentile(all_z, 1)) * 0.8)
    far = max(near + 0.5, float(np.percentile(all_z, 99)) * 1.2)
    return near, far


def create_ray_dataloaders(
    scene_name: str,
    *,
    rays_per_image: int = 1024,
    batch_size_images: int = 1,
    val_ratio: float = 0.2,
    near: float | None = None,
    far: float | None = None,
    normalize_scene: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Build train/val ray dataloaders from COLMAP camera metadata and scene images.

    Each dataset item samples `rays_per_image` random rays from one image.
    The collate function flattens multiple sampled images into one ray batch.
    """
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1.")

    cameras_payload = load_cameras(scene_name)
    points3d, _ = load_ply(scene_name)

    image_root = INPUT_DIR / scene_name
    if not image_root.exists():
        raise FileNotFoundError(f"Scene image folder not found: {image_root}")

    image_names = sorted(
        path.relative_to(image_root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    frame_records = _build_frame_records(cameras_payload, image_names)

    if normalize_scene:
        frame_records, points3d = _normalize_scene_records(frame_records, points3d)

    if near is None or far is None:
        est_near, est_far = _estimate_near_far(frame_records, points3d)
        near = est_near if near is None else near
        far = est_far if far is None else far

    rng = random.Random(seed)
    indices = list(range(len(frame_records)))
    rng.shuffle(indices)

    val_count = int(len(indices) * val_ratio)
    if val_ratio > 0.0:
        val_count = max(1, val_count)
    train_count = len(indices) - val_count
    if train_count <= 0:
        raise ValueError("Not enough frames after train/val split; lower val_ratio.")

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    train_frames = [frame_records[i] for i in train_indices]
    val_frames = [frame_records[i] for i in val_indices]

    train_ds = NeRFRayDataset(
        image_root=image_root,
        frames=train_frames,
        rays_per_item=rays_per_image,
        near=near,
        far=far,
        deterministic=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_images,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_rays,
        pin_memory=torch.cuda.is_available(),
    )

    if not val_frames:
        return train_loader, None

    val_ds = NeRFRayDataset(
        image_root=image_root,
        frames=val_frames,
        rays_per_item=rays_per_image,
        near=near,
        far=far,
        deterministic=True,
        seed=seed + 10_000,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_images,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_rays,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
