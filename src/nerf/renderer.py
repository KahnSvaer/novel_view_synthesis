import torch


def _extract_pinhole_intrinsics(camera: dict) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from COLMAP-style camera metadata."""
    model = camera["model"]
    params = camera["params"]

    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)

    if model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)

    raise ValueError(f"Unsupported camera model for NeRF pinhole rays: {model}")


def _undistort_normalized(
    u_d: torch.Tensor,
    v_d: torch.Tensor,
    camera: dict,
    num_iters: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively invert COLMAP radial/tangential lens distortion in normalized coords."""
    model = camera["model"]
    params = camera["params"]

    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return u_d, v_d

    if model == "SIMPLE_RADIAL":
        k1 = float(params[3])
        k2, p1, p2 = 0.0, 0.0, 0.0
    elif model == "RADIAL":
        k1, k2 = float(params[3]), float(params[4])
        p1, p2 = 0.0, 0.0
    elif model == "OPENCV":
        k1, k2, p1, p2 = (float(params[i]) for i in range(4, 8))
    else:
        return u_d, v_d

    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
        return u_d, v_d

    u = u_d.clone()
    v = v_d.clone()
    for _ in range(num_iters):
        r2 = u * u + v * v
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        du_tang = 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u)
        dv_tang = p1 * (r2 + 2.0 * v * v) + 2.0 * p2 * u * v
        u = (u_d - du_tang) / radial
        v = (v_d - dv_tang) / radial
    return u, v


def _ensure_2d_tensor(values, n_rows: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Broadcast scalar or 1D near/far values to [N_rays, 1]."""
    if isinstance(values, (float, int)):
        return torch.full((n_rows, 1), float(values), device=device, dtype=dtype)

    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(n_rows, 1)
    if tensor.ndim == 1:
        if tensor.shape[0] != n_rows:
            raise ValueError("1D near/far tensor must have length N_rays.")
        return tensor.unsqueeze(-1)
    if tensor.ndim == 2 and tensor.shape == (n_rows, 1):
        return tensor

    raise ValueError("near/far must be scalar, [N_rays], or [N_rays, 1].")


def get_ray_directions(
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build camera-frame ray directions for all image pixels.

    Returns:
        Tensor of shape [H, W, 3] in camera coordinates.
    """
    x = torch.arange(width, device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

    # OpenCV/COLMAP camera convention: +X right, +Y down, +Z forward.
    dirs = torch.stack(
        [
            (grid_x - cx) / fx,
            (grid_y - cy) / fy,
            torch.ones_like(grid_x),
        ],
        dim=-1,
    )
    return dirs


def find_camera_ray(
    pixel_x,
    pixel_y,
    camera: dict,
    camera_to_world: torch.Tensor,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute world-space ray origin and direction for one or more image pixels.

    Args:
        pixel_x: Pixel x coordinate(s), scalar or tensor.
        pixel_y: Pixel y coordinate(s), scalar or tensor.
        camera: Camera metadata dictionary containing model and params.
        camera_to_world: [4, 4] camera-to-world transform in COLMAP/OpenCV convention.
        normalize: If True, return unit-norm direction vectors.

    Returns:
        Tuple of:
            ray_origin: [N, 3]
            ray_direction: [N, 3]
    """
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4, 4].")

    fx, fy, cx, cy = _extract_pinhole_intrinsics(camera)
    device = camera_to_world.device
    dtype = camera_to_world.dtype

    px = torch.as_tensor(pixel_x, device=device, dtype=dtype).reshape(-1)
    py = torch.as_tensor(pixel_y, device=device, dtype=dtype).reshape(-1)
    if px.shape != py.shape:
        raise ValueError("pixel_x and pixel_y must have matching shapes.")

    u_d = (px - cx) / fx
    v_d = (py - cy) / fy
    u, v = _undistort_normalized(u_d, v_d, camera)

    dirs_cam = torch.stack([u, v, torch.ones_like(u)], dim=-1)

    rotation = camera_to_world[:3, :3]
    ray_d = dirs_cam @ rotation.T
    if normalize:
        ray_d = ray_d / torch.clamp(torch.linalg.norm(ray_d, dim=-1, keepdim=True), min=1e-8)

    ray_o = camera_to_world[:3, 3].expand_as(ray_d)
    return ray_o, ray_d


def get_rays_for_image(
    camera: dict,
    camera_to_world,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate world-space rays for every pixel in an image.

    Returns:
        rays_o: [H*W, 3]
        rays_d: [H*W, 3]
    """
    height = int(camera["height"])
    width = int(camera["width"])
    fx, fy, cx, cy = _extract_pinhole_intrinsics(camera)

    c2w = torch.as_tensor(camera_to_world, dtype=dtype)
    if c2w.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4, 4].")
    if device is not None:
        c2w = c2w.to(device)

    directions_cam = get_ray_directions(
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        device=c2w.device,
        dtype=c2w.dtype,
    )
    u, v = _undistort_normalized(directions_cam[..., 0], directions_cam[..., 1], camera)
    directions_cam = torch.stack([u, v, directions_cam[..., 2]], dim=-1)

    rotation = c2w[:3, :3]
    rays_d = directions_cam.reshape(-1, 3) @ rotation.T
    if normalize:
        rays_d = rays_d / torch.clamp(torch.linalg.norm(rays_d, dim=-1, keepdim=True), min=1e-8)

    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near,
    far,
    n_samples: int,
    *,
    perturb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified depth sampling along rays."""
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    n_rays = rays_o.shape[0]
    device = rays_o.device
    dtype = rays_o.dtype

    near_t = _ensure_2d_tensor(near, n_rays, device=device, dtype=dtype)
    far_t = _ensure_2d_tensor(far, n_rays, device=device, dtype=dtype)

    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
    z_vals = near_t * (1.0 - t_vals) + far_t * t_vals

    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        noise = torch.rand(z_vals.shape, device=device, dtype=dtype)
        z_vals = lower + (upper - lower) * noise

    points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    return z_vals, points


def _query_nerf(model: torch.nn.Module, points: torch.Tensor, directions: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Query NeRF model in chunks to avoid OOM."""
    flat_points = points.reshape(-1, 3)
    expanded_dirs = directions[:, None, :].expand_as(points).reshape(-1, 3)

    sigma_chunks = []
    rgb_chunks = []

    for start in range(0, flat_points.shape[0], chunk_size):
        end = start + chunk_size
        sigma_chunk, rgb_chunk = model(flat_points[start:end], expanded_dirs[start:end])
        sigma_chunks.append(sigma_chunk)
        rgb_chunks.append(rgb_chunk)

    sigma = torch.cat(sigma_chunks, dim=0).reshape(*points.shape[:-1], 1)
    rgb = torch.cat(rgb_chunks, dim=0).reshape(*points.shape[:-1], 3)
    return sigma, rgb


def volume_render(
    sigma: torch.Tensor,
    rgb: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    *,
    white_background: bool,
) -> dict[str, torch.Tensor]:
    """Alpha compositing for volume rendering as in NeRF."""
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = torch.full_like(deltas[:, :1], 1e10)
    deltas = torch.cat([deltas, delta_inf], dim=-1)

    ray_norm = torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    deltas = deltas * ray_norm

    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * deltas)

    trans = torch.cumprod(
        torch.cat(
            [torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10],
            dim=-1,
        ),
        dim=-1,
    )[:, :-1]
    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map)[..., None]

    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "acc_map": acc_map,
        "weights": weights,
        "alpha": alpha,
    }


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    *,
    deterministic: bool,
) -> torch.Tensor:
    """Hierarchical importance sampling from piecewise-constant PDF."""
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device, dtype=bins.dtype)
        u = u.expand(cdf.shape[0], n_samples).contiguous()
    else:
        u = torch.rand(cdf.shape[0], n_samples, device=bins.device, dtype=bins.dtype)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    cdf_g0 = torch.gather(cdf, 1, below)
    cdf_g1 = torch.gather(cdf, 1, above)

    bins_g0 = torch.gather(bins, 1, below)
    bins_g1 = torch.gather(bins, 1, above)

    denom = torch.where((cdf_g1 - cdf_g0) < 1e-8, torch.ones_like(cdf_g1), cdf_g1 - cdf_g0)
    t = (u - cdf_g0) / denom
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples


def render_rays(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near,
    far,
    *,
    n_coarse: int = 64,
    n_fine: int = 128,
    perturb: bool = True,
    white_background: bool = False,
    chunk_size: int = 65536,
) -> dict[str, torch.Tensor]:
    """Render a batch of rays using coarse and fine NeRF models."""
    z_coarse, pts_coarse = sample_stratified(
        rays_o=rays_o,
        rays_d=rays_d,
        near=near,
        far=far,
        n_samples=n_coarse,
        perturb=perturb,
    )

    sigma_coarse, rgb_coarse = _query_nerf(
        model=coarse_model,
        points=pts_coarse,
        directions=rays_d,
        chunk_size=chunk_size,
    )
    coarse_outputs = volume_render(
        sigma=sigma_coarse,
        rgb=rgb_coarse,
        z_vals=z_coarse,
        rays_d=rays_d,
        white_background=white_background,
    )

    z_mid = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])
    z_fine_samples = sample_pdf(
        bins=z_mid,
        weights=coarse_outputs["weights"][:, 1:-1].detach(),
        n_samples=n_fine,
        deterministic=not perturb,
    )

    z_all = torch.sort(torch.cat([z_coarse, z_fine_samples], dim=-1), dim=-1).values
    pts_fine = rays_o[:, None, :] + rays_d[:, None, :] * z_all[..., None]

    sigma_fine, rgb_fine = _query_nerf(
        model=fine_model,
        points=pts_fine,
        directions=rays_d,
        chunk_size=chunk_size,
    )
    fine_outputs = volume_render(
        sigma=sigma_fine,
        rgb=rgb_fine,
        z_vals=z_all,
        rays_d=rays_d,
        white_background=white_background,
    )

    return {
        "coarse_rgb": coarse_outputs["rgb_map"],
        "coarse_depth": coarse_outputs["depth_map"],
        "coarse_acc": coarse_outputs["acc_map"],
        "coarse_weights": coarse_outputs["weights"],
        "coarse_z_vals": z_coarse,
        "fine_rgb": fine_outputs["rgb_map"],
        "fine_depth": fine_outputs["depth_map"],
        "fine_acc": fine_outputs["acc_map"],
        "fine_weights": fine_outputs["weights"],
        "fine_z_vals": z_all,
    }


def render_batch_fn(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    batch: dict,
    *,
    n_coarse: int = 64,
    n_fine: int = 128,
    perturb: bool = True,
    white_background: bool = False,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training adapter matching train_nerf expectations.

    Required batch keys:
        rays_o: [N_rays, 3]
        rays_d: [N_rays, 3]
        target_rgb: [N_rays, 3]

    Optional batch keys:
        near: scalar, [N_rays], or [N_rays, 1]
        far: scalar, [N_rays], or [N_rays, 1]
    """
    required = {"rays_o", "rays_d", "target_rgb"}
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Missing required batch keys: {missing}")

    near = batch.get("near", 2.0)
    far = batch.get("far", 6.0)

    outputs = render_rays(
        coarse_model=coarse_model,
        fine_model=fine_model,
        rays_o=batch["rays_o"],
        rays_d=batch["rays_d"],
        near=near,
        far=far,
        n_coarse=n_coarse,
        n_fine=n_fine,
        perturb=perturb,
        white_background=white_background,
        chunk_size=chunk_size,
    )

    target_rgb = batch["target_rgb"]
    return outputs["coarse_rgb"], outputs["fine_rgb"], target_rgb
