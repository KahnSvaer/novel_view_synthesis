import json
from pathlib import Path

import numpy as np
import open3d as o3d

from .colmap import run_colmap

INPUT_DIR = Path("./data/archive")
OUTPUT_DIR = Path("./data/processed")


def load_cameras(
        input_dir: str|Path, 
        output_dir: str|Path|None = None
    ) -> dict:
    """
    Load camera parameters from COLMAP output.

    Returns dictionary with camera intrinsics (per different cameras) and extrinsics (per image).
    """
    if isinstance(input_dir, str) and "/" not in input_dir:
        input_dir = INPUT_DIR / input_dir
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = OUTPUT_DIR / input_dir.name
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        run_colmap(
            input_dir, 
            output_dir
        )

    camera_json = output_dir / "cameras.json"
    with open(camera_json, "r") as f:
        cameras = json.load(f)
    return cameras


def load_ply(
        input_dir: str|Path, 
        output_dir: str|Path|None = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from COLMAP output.

    Returns:
        points: Nx3 array of 3D points.
        colors: Nx3 array of RGB colors.
    """
    if isinstance(input_dir, str) and "/" not in input_dir:
        input_dir = INPUT_DIR / input_dir
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = OUTPUT_DIR / input_dir.name
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        run_colmap(
            input_dir, 
            output_dir
        )

    ply_file = output_dir / "sparse_points.ply"

    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    return points, colors