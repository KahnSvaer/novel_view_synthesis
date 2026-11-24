# Novel View Synthesis & 3D Reconstruction

A PyTorch implementation of **NeRF** (and upcoming **3D Gaussian Splatting**) trained on the [3D Object Reconstruction from Images](https://www.kaggle.com/datasets/kpoviesistphane/3d-object-reconstruction-from-images) Kaggle dataset. **COLMAP** is used for camera calibration and sparse reconstruction in NeRF, and its camera poses and sparse point cloud will be reused directly to initialize **3D Gaussian Splatting**.

## Project Status
- [x] **COLMAP SfM Pipeline** (`src/common/colmap.py`, `src/common/loader.py`): SIFT feature extraction, exhaustive matching, incremental mapping, sparse point cloud (`.ply`), and camera intrinsics/extrinsics (`cameras.json`).
- [x] **NeRF** (`src/nerf/`): Coarse & fine MLPs (`Lx=10`, `Ld=4`), OpenCV/COLMAP ray generation with radial/tangential undistortion, scene normalization & automatic `near`/`far` estimation, chunked volume rendering, training (`main.py`), and an interpolated trajectory GIF visualizer (`src/nerf/visualizer.py` with SO(3) SLERP, CUDA FP16 autocast, and incremental frame caching).
- [ ] **3D Gaussian Splatting** (`src/gs/`): **Next step.**

## Quickstart (NeRF)

### Train
```bash
uv run python main.py \
  --scene images \
  --val-ratio 0.0 \
  --epochs 3000 \
  --save-every-epochs 100 \
  --checkpoint-dir outputs/images_nerf
```

### Render Trajectory GIF
```bash
uv run python -m src.nerf.visualizer \
  --scene images \
  --checkpoint outputs/images_nerf/latest.pt \
  --output results/nerf_images_dataset_kaggle/render.gif \
  --steps-per-pair 8 \
  --render-scale 0.25 \
  --ray-chunk 32768
```

## Results
- **Kaggle `images` scene (`11` views):** See [results/nerf_images_dataset_kaggle/analysis.md](results/nerf_images_dataset_kaggle/analysis.md) for training metrics, rendered output (`render.gif`), and qualitative observations.
- **Dataset source:** [3D Object Reconstruction from Images (Kaggle)](https://www.kaggle.com/datasets/kpoviesistphane/3d-object-reconstruction-from-images)
