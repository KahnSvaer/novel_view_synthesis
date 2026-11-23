import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycolmap

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def run_colmap(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    image_list: list[str | Path] | None = None,
    *,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    max_num_features: int = 8192,
    device: pycolmap.Device = pycolmap.Device.auto,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Perform complete sparse COLMAP Structure-from-Motion.

    Pipeline:
        1. Image discovery
        2. Feature extraction
        3. Feature matching
        4. Incremental SfM
        5. Bundle adjustment (performed during mapping)
        6. Reconstruction selection
        7. Sparse point cloud export
        8. Camera and reconstruction metadata export
    
    Args:
        input_dir: Path to the input directory containing images.
        output_dir: Path to the output directory for COLMAP results.
        image_list: Optional list of image filenames to process. If None, all images in input_dir are used.
        camera_mode: Camera mode for COLMAP (AUTO, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, FISHEYE, SIMPLE_PINHOLE).
        max_num_features: Maximum number of SIFT features to extract per image.
        device: Device to use for COLMAP (CPU or GPU).
        overwrite: If True, existing output directory will be overwritten.
        verbose: If True, prints detailed progress and summary information.
    """

    # Validate inputs
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if output_dir is None:
        output_dir = input_dir / "colmap_output"
    output_dir = Path(output_dir).resolve()

    if max_num_features <= 0:
        raise ValueError("max_num_features must be positive.")

    if image_list is None:

        image_paths = sorted(
            path
            for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    else:

        image_paths = []

        for image in image_list:
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = input_dir / image_path

            image_path = image_path.resolve()

            try:
                image_path.relative_to(input_dir)
            except ValueError:
                raise ValueError(
                    f"Image is outside input directory: {image}"
                )

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image_paths.append(image_path)

        image_paths = sorted(set(image_paths))

    if len(image_paths) < 2:
        raise ValueError(
            f"At least two images are required. Found {len(image_paths)}."
        )

    image_names = [
        path.relative_to(input_dir).as_posix()
        for path in image_paths
    ]

    print(f"[COLMAP] Found {len(image_names)} images.")
    print(f"[COLMAP] Input: {input_dir}")
    print(f"[COLMAP] Output: {output_dir}")

    if output_dir.exists():

        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. "
                "Set overwrite=True to replace it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    pycolmap.logging.set_log_destination(
        pycolmap.logging.INFO,
        output_dir / "colmap.log",
    )

    database_path = output_dir / "database.db"
    mapper_path = output_dir / "mapper"
    sparse_path = output_dir / "sparse"

    mapper_path.mkdir()
    sparse_path.mkdir()


    # Feature extraction
    print("\n[COLMAP] Extracting SIFT features...")
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = max_num_features

    pycolmap.extract_features(
        database_path=database_path,
        image_path=input_dir,
        image_names=image_names,
        camera_mode=camera_mode,
        extraction_options=extraction_options,
        device=device,
    )


    # Feature matching
    print("\n[COLMAP] Performing exhaustive matching...")
    pycolmap.match_exhaustive(
        database_path=database_path,
        device=device,
    )

    # Incremental SfM
    print("\n[COLMAP] Running incremental mapping...")
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.multiple_models = True
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=input_dir,
        output_path=mapper_path,
        options=pipeline_options,
    )

    if not reconstructions:
        raise RuntimeError(
            "COLMAP failed to produce a reconstruction. "
            "Check image overlap, texture and feature matching."
        )


    # Select best reconstruction
    best_reconstruction = max(
        reconstructions.values(),
        key=lambda reconstruction: (
            reconstruction.num_reg_images(),
            reconstruction.num_points3D(),
        ),
    )

    registered_images = best_reconstruction.num_reg_images()
    num_points = best_reconstruction.num_points3D()

    if registered_images < 2 or num_points == 0:
        raise RuntimeError(
            "COLMAP produced an insufficient reconstruction."
        )


    # Export
    best_model_path = sparse_path / "0"
    best_model_path.mkdir(parents=True, exist_ok=True)

    best_reconstruction.write(str(best_model_path))

    point_cloud_path = output_dir / "sparse_points.ply"
    best_reconstruction.export_PLY(str(point_cloud_path))


    # Camera intrinsics
    cameras = {}

    for camera_id, camera in best_reconstruction.cameras.items():
        cameras[str(camera_id)] = {
            "model": camera.model_name,
            "width": camera.width,
            "height": camera.height,
            "params": camera.params.tolist(),
        }


    # Camera extrinsics
    images = {}

    for image_id, image in best_reconstruction.images.items():
        if not image.has_pose:
            continue

    # COLMAP world-to-camera transformation (3x4).
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = image.cam_from_world().matrix()

    # Invert to obtain camera-to-world.
    c2w = np.linalg.inv(w2c)

    images[str(image_id)] = {
        "name": image.name,
        "camera_id": image.camera_id,
        "world_to_camera": w2c.tolist(),
        "camera_to_world": c2w.tolist(),
    }


    # Save camera metadata
    camera_data = {
        "cameras": cameras,
        "images": images,
    }

    camera_path = output_dir / "cameras.json"

    with camera_path.open("w", encoding="utf-8") as f:
        json.dump(camera_data, f, indent=4)


    # Reconstruction statistics
    mean_reprojection_error = float(
        best_reconstruction.compute_mean_reprojection_error()
    )

    registered_image_names = sorted(
        image.name
        for image in best_reconstruction.images.values()
        if image.has_pose
    )

    statistics = {
        "input_directory": str(input_dir),
        "total_images": len(image_names),
        "registered_images": registered_images,
        "unregistered_images": len(image_names) - registered_images,
        "num_points3D": num_points,
        "num_cameras": len(best_reconstruction.cameras),
        "mean_reprojection_error": mean_reprojection_error,
        "num_reconstructions": len(reconstructions),
        "registered_image_names": registered_image_names,
    }

    statistics_path = output_dir / "statistics.json"

    with statistics_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=4)


    # Summary
    if not verbose:
        return
    print("\n========== COLMAP SUMMARY ==========")

    print(f"Input images:             {len(image_names)}")
    print(f"Registered images:        {registered_images}")
    print(f"Unregistered images:      {len(image_names) - registered_images}")
    print(f"3D points:                {num_points}")
    print(f"Cameras:                  {len(best_reconstruction.cameras)}")
    print(f"Reprojection error:       {mean_reprojection_error:.4f} px")
    print(f"Reconstructions:          {len(reconstructions)}")

    print("====================================")

    print(f"\nSparse model: {best_model_path}")
    print(f"Point cloud:  {point_cloud_path}")
    print(f"Cameras:      {camera_path}")
    print(f"Statistics:   {statistics_path}")


def process_scene(folder: Path, output_dir: Path):
    """Run COLMAP for one scene in the current process."""

    run_colmap(
        input_dir=folder,
        output_dir=output_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
        max_num_features=4096,
        device=pycolmap.Device.auto,
        overwrite=False,
        verbose=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()

    # Worker mode: process exactly one scene.
    if args.worker:

        if args.input_dir is None or args.output_dir is None:
            parser.error("--worker requires --input-dir and --output-dir")

        try:
            process_scene(args.input_dir, args.output_dir)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

        sys.exit(0)

    # Parent mode: discover and process all scenes.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    archive_dir = PROJECT_ROOT / "data" / "archive"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    allowed_folders = [
        "images", "Ammonite"
    ]
    folder_paths = sorted(
        directory
        for directory in archive_dir.iterdir()
        if directory.is_dir()
        and directory.name in allowed_folders
    )

    for folder in folder_paths:
        print(folder.name)
        print(f"\n{'=' * 50}")
        print(f"Processing scene: {folder.name}")
        print(f"{'=' * 50}")

        output_dir = processed_dir / folder.name

        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--input-dir",
                str(folder),
                "--output-dir",
                str(output_dir),
            ],
            check=False,
        )

        if result.returncode == 0:
            print(f"[SUCCESS] {folder.name}")
        else:
            print(
                f"[ERROR] {folder.name} failed "
                f"(exit code: {result.returncode})"
            )

    print("\nAll scenes processed.")