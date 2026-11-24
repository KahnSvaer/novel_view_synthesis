from __future__ import annotations

import argparse
from functools import partial

import torch

from src.nerf.dataset import create_ray_dataloaders
from src.nerf.renderer import render_batch_fn
from src.nerf.training import get_nerf_models, train_nerf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeRF from COLMAP outputs.")
    parser.add_argument("--scene", type=str, default="images", help="Scene folder under data/archive.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-final", type=float, default=5e-5, help="Final exponential decay LR.")
    parser.add_argument("--Lx", type=int, default=10, help="Positional encoding bands for 3D points.")
    parser.add_argument("--Ld", type=int, default=4, help="Positional encoding bands for view directions.")
    parser.add_argument("--batch-images", type=int, default=1, help="Images sampled per dataloader step.")
    parser.add_argument("--rays-per-image", type=int, default=1024)
    parser.add_argument("--n-coarse", type=int, default=64)
    parser.add_argument("--n-fine", type=int, default=128)
    parser.add_argument("--near", type=float, default=None, help="Near bound (auto-estimated if omitted).")
    parser.add_argument("--far", type=float, default=None, help="Far bound (auto-estimated if omitted).")
    parser.add_argument("--no-normalize-scene", action="store_true", help="Disable scene centering/scaling.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--validate-every-epochs", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/nerf")
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-perturb", action="store_true", help="Disable stratified perturbation.")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto if omitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader = create_ray_dataloaders(
        scene_name=args.scene,
        rays_per_image=args.rays_per_image,
        batch_size_images=args.batch_images,
        val_ratio=args.val_ratio,
        near=args.near,
        far=args.far,
        normalize_scene=not args.no_normalize_scene,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    coarse_model, fine_model = get_nerf_models(Lx=args.Lx, Ld=args.Ld)

    render_fn = partial(
        render_batch_fn,
        n_coarse=args.n_coarse,
        n_fine=args.n_fine,
        perturb=not args.no_perturb,
        white_background=args.white_background,
    )

    history = train_nerf(
        coarse_model=coarse_model,
        fine_model=fine_model,
        train_loader=train_loader,
        val_loader=val_loader,
        render_batch_fn=render_fn,
        val_render_batch_fn=render_fn,
        epochs=args.epochs,
        lr=args.lr,
        lr_final=args.lr_final,
        device=args.device,
        log_every=args.log_every,
        validate_every_epochs=args.validate_every_epochs,
        max_val_batches=args.max_val_batches,
        checkpoint_dir=args.checkpoint_dir,
        save_every_epochs=args.save_every_epochs,
    )

    print("Training complete.")
    print(
        {
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history.get("val_loss", [None])[-1],
            "num_epochs": len(history["train_loss"]),
        }
    )


if __name__ == "__main__":
    main()
