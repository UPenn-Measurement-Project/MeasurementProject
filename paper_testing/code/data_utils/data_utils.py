#imports
import torch
import torch.nn.functional as F

#warping function
def warp_image(
    imgs,
    mode="affine",                # 'affine' or 'perspective'
    rot_deg=(-20, 20),
    scale=(0.9, 1.1),
    translate_frac=(-0.05, 0.05),
    shear_deg=(-8, 8),
    perspective_jitter=0.06,
    padding_mode="border",        # 'zeros' | 'border' | 'reflection'
    align_corners=True,
):
    """
    imgs: torch.Tensor of shape [B, C, H, W] in [0,1].
    Returns: warped images, same shape.
    Each image gets its own random warp.
    """
    assert imgs.dim() == 4, "Expected [B,C,H,W] tensor"
    B, C, H, W = imgs.shape
    device, dtype = imgs.device, imgs.dtype

    if mode == "affine":
        # Random parameters per image
        rot = torch.empty(B, device=device).uniform_(*rot_deg).deg2rad()
        sx  = torch.empty(B, device=device).uniform_(*scale)
        sy  = torch.empty(B, device=device).uniform_(*scale)
        tx  = torch.empty(B, device=device).uniform_(*translate_frac) * 2.0
        ty  = torch.empty(B, device=device).uniform_(*translate_frac) * 2.0
        sh  = torch.tan(torch.empty(B, device=device).uniform_(*shear_deg).deg2rad())

        cos, sin = torch.cos(rot), torch.sin(rot)

        # Build batched affine matrices [B,2,3]
        A = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        A[:,0,0] = cos * sx
        A[:,0,1] = -sin * sy + sh
        A[:,1,0] = sin * sx
        A[:,1,1] = cos * sy
        A[:,0,2] = tx
        A[:,1,2] = ty

        grid = F.affine_grid(A, size=(B, C, H, W), align_corners=align_corners)

    elif mode == "perspective":
        grids = []
        base = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], device=device, dtype=dtype)  # (4,2)
        for _ in range(B):
            noise = (torch.rand_like(base) * 2 - 1) * (2 * perspective_jitter)
            dst = base + noise
            src = base

            # Direct Linear Transform to compute H (3x3)
            Arows = []
            for i in range(4):
                x0, y0 = src[i]
                u, v = dst[i]
                Arows.append([-x0, -y0, -1,   0,   0,   0, x0*u, y0*u, u])
                Arows.append([  0,   0,   0, -x0, -y0, -1, x0*v, y0*v, v])
            A = torch.tensor(Arows, device=device, dtype=dtype)
            _, _, Vh = torch.linalg.svd(A)
            Hm = Vh[-1].view(3,3)
            Hinv = torch.linalg.inv(Hm)

            ys, xs = torch.meshgrid(
                torch.linspace(-1,1,H, device=device, dtype=dtype),
                torch.linspace(-1,1,W, device=device, dtype=dtype),
                indexing="ij"
            )
            ones = torch.ones_like(xs)
            grid_h = torch.stack([xs, ys, ones], dim=-1).view(-1,3)   # (HW,3)
            src_coords = (grid_h @ Hinv.T)
            src_coords = src_coords[:, :2] / src_coords[:, 2:].clamp(min=1e-8)
            grid = src_coords.view(1, H, W, 2)                        # (1,H,W,2)
            grids.append(grid)
        grid = torch.cat(grids, dim=0)   # [B,H,W,2]

    else:
        raise ValueError("mode must be 'affine' or 'perspective'")

    # Warp batch
    warped = F.grid_sample(imgs, grid, mode="bilinear",
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return warped
