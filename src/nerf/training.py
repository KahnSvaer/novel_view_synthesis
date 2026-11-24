from pathlib import Path

import torch

from .model import CoarseNeRFModel, FineNeRFModel


def get_nerf_models(Lx: int = 10, Ld: int = 4):
    """
    Get the NeRF model based on the specified type.

    Args:
        Lx: Number of positional encoding frequencies for 3D points.
        Ld: Number of positional encoding frequencies for view directions.

    Returns:
        An instance of the specified NeRF model.
    """
    assert Lx > 0, "Lx must be a positive integer."
    assert Ld > 0, "Ld must be a positive integer."
    return CoarseNeRFModel(Lx, Ld), FineNeRFModel(Lx, Ld)


def loss(coarse_predictions: torch.Tensor,
         fine_predictions: torch.Tensor,
         targets: torch.Tensor) -> torch.Tensor:
    """
    MSE loss for coarse and fine NeRF outputs.

    Args:
        coarse_predictions: Predicted RGB values from the coarse model.
        fine_predictions: Predicted values from the fine model.
        targets: Ground-truth RGB values.
    """
    return (torch.nn.functional.mse_loss(coarse_predictions, targets)
        + torch.nn.functional.mse_loss(fine_predictions, targets))


def _to_device(batch, device: torch.device):
    """Move nested tensors in a batch to the target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_to_device(v, device) for v in batch)
    return batch


def _run_validation(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    val_loader,
    val_render_batch_fn,
    device: torch.device,
    max_val_batches: int | None = None,
) -> tuple[float, int]:
    """Run one validation pass and return (mean_loss, num_batches)."""
    coarse_model.eval()
    fine_model.eval()

    val_running_loss = 0.0
    val_batch_count = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = _to_device(val_batch, device)
            coarse_rgb, fine_rgb, target_rgb = val_render_batch_fn(
                coarse_model,
                fine_model,
                val_batch,
            )
            val_batch_loss = loss(coarse_rgb, fine_rgb, target_rgb)
            val_running_loss += val_batch_loss.item()
            val_batch_count += 1

            if max_val_batches is not None and val_batch_count >= max_val_batches:
                break

    mean_val_loss = val_running_loss / max(val_batch_count, 1)
    return mean_val_loss, val_batch_count


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    epoch: int,
    global_step: int,
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict,
    best_val_loss: float | None,
) -> None:
    """Persist full training state for resume/evaluation."""
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "coarse_model_state_dict": coarse_model.state_dict(),
        "fine_model_state_dict": fine_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def train_nerf(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    train_loader,
    render_batch_fn,
    epochs: int,
    val_loader=None,
    val_render_batch_fn=None,
    lr: float = 5e-4,
    lr_final: float | None = 5e-5,
    device: str | torch.device | None = None,
    log_every: int = 50,
    validate_every_epochs: int | None = 1,
    max_val_batches: int | None = None,
    grad_clip_norm: float | None = None,
    checkpoint_dir: str | Path | None = None,
    save_every_epochs: int | None = None,
):
    """
    Train coarse and fine NeRF models.
    
    Args:
        coarse_model: Coarse NeRF model.
        fine_model: Fine NeRF model.
        train_loader: Iterable of training batches.
        render_batch_fn: Callable with signature
            `(coarse_model, fine_model, batch) -> (coarse_rgb, fine_rgb, target_rgb)`.
        epochs: Number of epochs to train.
        lr: Learning rate.
        device: Training device. If None, automatically picks CUDA if available.
        val_loader: Optional validation data loader.
        val_render_batch_fn: Optional render fn for validation. Defaults to
            `render_batch_fn`.
        log_every: Print training loss every N steps.
        validate_every_epochs: If set, run validation every N epochs (default 1;
            set to None to disable epoch-end validation).
        max_val_batches: Optional cap on number of val batches per validation
            run (useful for fast periodic validation).
        grad_clip_norm: Optional max gradient norm for clipping.
        checkpoint_dir: Optional folder where checkpoints are written.
        save_every_epochs: Optional periodic checkpoint cadence.

    Returns:
        Dict with `train_loss` and optional `val_loss` per epoch.
    """
    assert epochs > 0, "epochs must be a positive integer."
    assert lr > 0.0, "lr must be positive."
    if validate_every_epochs is not None:
        assert validate_every_epochs > 0, "validate_every_epochs must be > 0."
    if max_val_batches is not None:
        assert max_val_batches > 0, "max_val_batches must be > 0."
    if save_every_epochs is not None:
        assert save_every_epochs > 0, "save_every_epochs must be > 0."

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    coarse_model = coarse_model.to(device)
    fine_model = fine_model.to(device)

    optimizer = torch.optim.Adam(
        list(coarse_model.parameters()) + list(fine_model.parameters()),
        lr=lr,
    )

    scheduler = None
    if lr_final is not None and 0.0 < lr_final < lr:
        total_steps = max(epochs * len(train_loader), 1)
        gamma = (lr_final / lr) ** (1.0 / total_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if val_render_batch_fn is None:
        val_render_batch_fn = render_batch_fn

    checkpoint_root = None
    if checkpoint_dir is not None:
        checkpoint_root = Path(checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    global_step = 0
    best_val_loss = None
    for epoch in range(1, epochs + 1):
        coarse_model.train()
        fine_model.train()

        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            batch = _to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            coarse_rgb, fine_rgb, target_rgb = render_batch_fn(
                coarse_model,
                fine_model,
                batch,
            )
            batch_loss = loss(coarse_rgb, fine_rgb, target_rgb)
            batch_loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(coarse_model.parameters()) + list(fine_model.parameters()),
                    max_norm=grad_clip_norm,
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += batch_loss.item()
            batch_count += 1
            global_step += 1

            if log_every > 0 and global_step % log_every == 0:
                print(
                    f"[train] epoch={epoch}/{epochs} step={global_step} "
                    f"loss={batch_loss.item():.6f}"
                )

        mean_train_loss = running_loss / max(batch_count, 1)
        history["train_loss"].append(mean_train_loss)

        should_validate_epoch = (
            val_loader is not None
            and validate_every_epochs is not None
            and (epoch % validate_every_epochs == 0 or epoch == epochs)
        )
        if not should_validate_epoch:
            print(f"[epoch {epoch}] train_loss={mean_train_loss:.6f}")

            if checkpoint_root is not None:
                _save_checkpoint(
                    checkpoint_root / "latest.pt",
                    epoch=epoch,
                    global_step=global_step,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    optimizer=optimizer,
                    history=history,
                    best_val_loss=best_val_loss,
                )
                if save_every_epochs is not None and epoch % save_every_epochs == 0:
                    _save_checkpoint(
                        checkpoint_root / f"epoch_{epoch:04d}.pt",
                        epoch=epoch,
                        global_step=global_step,
                        coarse_model=coarse_model,
                        fine_model=fine_model,
                        optimizer=optimizer,
                        history=history,
                        best_val_loss=best_val_loss,
                    )
            continue

        mean_val_loss, _ = _run_validation(
            coarse_model=coarse_model,
            fine_model=fine_model,
            val_loader=val_loader,
            val_render_batch_fn=val_render_batch_fn,
            device=device,
            max_val_batches=max_val_batches,
        )
        history["val_loss"].append(mean_val_loss)

        print(
            f"[epoch {epoch}] train_loss={mean_train_loss:.6f} "
            f"val_loss={mean_val_loss:.6f}"
        )

        if best_val_loss is None or mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            if checkpoint_root is not None:
                _save_checkpoint(
                    checkpoint_root / "best.pt",
                    epoch=epoch,
                    global_step=global_step,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    optimizer=optimizer,
                    history=history,
                    best_val_loss=best_val_loss,
                )

        if checkpoint_root is not None:
            _save_checkpoint(
                checkpoint_root / "latest.pt",
                epoch=epoch,
                global_step=global_step,
                coarse_model=coarse_model,
                fine_model=fine_model,
                optimizer=optimizer,
                history=history,
                best_val_loss=best_val_loss,
            )
            if save_every_epochs is not None and epoch % save_every_epochs == 0:
                _save_checkpoint(
                    checkpoint_root / f"epoch_{epoch:04d}.pt",
                    epoch=epoch,
                    global_step=global_step,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    optimizer=optimizer,
                    history=history,
                    best_val_loss=best_val_loss,
                )

        coarse_model.train()
        fine_model.train()

    return history

