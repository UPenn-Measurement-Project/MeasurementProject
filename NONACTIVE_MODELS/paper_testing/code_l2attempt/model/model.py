# PyTorch ≥ 1.10
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import List, Tuple

# -----------------------------
# Utils
# -----------------------------

def _init_weight(modules):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

def conv_block(in_ch, out_ch, k=3, s=1, *, dilation=1, bias=True,
               batch_norm=True, layer_norm=False, activation="ReLU"):
    # same padding for given stride/dilation
    pad = (dilation * (k - 1) + 2 - s) // 2
    seq = nn.Sequential()
    seq.add_module("conv", nn.Conv2d(in_ch, out_ch, k, s, pad, dilation, bias=bias))
    if batch_norm:
        seq.add_module("bn", nn.BatchNorm2d(out_ch))
    elif layer_norm:
        seq.add_module("ln", LayerNorm())
    if activation is not None:
        seq.add_module("act", getattr(nn, activation)(inplace=True))
    return seq

class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.shape[1:])

# -----------------------------
# Heatmap -> keypoints (soft-argmax by axes)
# -----------------------------

@torch.no_grad()
def _coord_axis_softmax(logits: torch.Tensor, other_axis: int, axis_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: (B, N, H, W). other_axis=2 uses H-aggregation to get x; other_axis=3 uses W-aggregation to get y
    returns (coord in [-1,1], axis probability)
    """
    # average along the other spatial axis → (B, N, axis)
    g_c_prob = torch.mean(logits, dim=other_axis)
    g_c_prob = F.softmax(g_c_prob, dim=2)
    coord_pts = torch.linspace(-1.0, 1.0, axis_size, device=logits.device).view(1, 1, axis_size)
    coord = torch.sum(g_c_prob * coord_pts, dim=2)  # (B, N)
    return coord, g_c_prob

def get_coord_2d(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, N, H, W) → centers mu in [-1,1]^2 as (y, x) per original code
    """
    B, N, H, W = logits.shape
    mu_y, _ = _coord_axis_softmax(logits, other_axis=3, axis_size=H)  # aggregate over W
    mu_x, _ = _coord_axis_softmax(logits, other_axis=2, axis_size=W)  # aggregate over H
    mu = torch.stack([mu_y, mu_x], dim=2)  # (B, N, 2) with [y, x]
    return mu

def get_gaussian_maps(mu: torch.Tensor, shape_hw: Tuple[int, int], inv_std: float, mode: str = "ankush") -> torch.Tensor:
    """
    mu: (B, N, 2) with (y, x) in [-1,1]
    returns heatmaps: (B, N, H, W)
    Modes:
      - 'rot'  : exp(-( (y-μy)^2 + (x-μx)^2 ) * inv_std^2)
      - 'flat' : exp(-((...)*inv_std^2 + eps)^(1/4))
      - 'ankush' (used in the reference impl): separable exp(-sqrt( eps + |(y-μy)*inv_std| )) ⊗ same for x
    """
    H, W = shape_hw
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, H, device=mu.device)
    x = torch.linspace(-1.0, 1.0, W, device=mu.device)

    if mode in ("rot", "flat"):
        mu_y = mu_y.unsqueeze(-1)  # (B,N,1)
        mu_x = mu_x.unsqueeze(-1)
        y = y.view(1, 1, H, 1)
        x = x.view(1, 1, 1, W)
        g_y = (y - mu_y) ** 2
        g_x = (x - mu_x) ** 2
        dist = (g_y + g_x) * (inv_std ** 2)
        if mode == "rot":
            g = torch.exp(-dist)
        else:
            g = torch.exp(-torch.pow(dist + 1e-5, 0.25))
    elif mode == "ankush":
        # separable “L1-ish” kernel with sqrt; matches the public code
        y = y.view(1, 1, H)
        x = x.view(1, 1, W)
        gy = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_y - y) * inv_std)))  # (B,N,H)
        gx = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_x - x) * inv_std)))  # (B,N,W)
        gy = gy.unsqueeze(3)  # (B,N,H,1)
        gx = gx.unsqueeze(2)  # (B,N,1,W)
        g = torch.matmul(gy, gx)  # outer product → (B,N,H,W)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return g

# -----------------------------
# Nets
# -----------------------------

class Encoder(nn.Module):
    """Shared feature tower"""
    def __init__(self, in_ch: int, nf: int, batch_norm: bool = True, layer_norm: bool = False):
        super().__init__()
        blocks = nn.ModuleList()
        blocks.append(conv_block(in_ch, nf, k=7, s=1, batch_norm=batch_norm, layer_norm=layer_norm))
        blocks.append(conv_block(nf, nf, k=3, s=1, batch_norm=batch_norm, layer_norm=layer_norm))
        # 3 downsampling stages (×2 filters, stride 2 then 1)
        for _ in range(3):
            filters = nf * 2
            blocks.append(conv_block(nf, filters, k=3, s=2, batch_norm=batch_norm, layer_norm=layer_norm))
            blocks.append(conv_block(filters, filters, k=3, s=1, batch_norm=batch_norm, layer_norm=layer_norm))
            nf = filters
        self.blocks = blocks

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats  # [C1, C2, C3, C4, C5] (spatial size halves thrice)

class ImageEncoder(nn.Module):
    """Φ (appearance) encoder; returns pyramid incl. input as in ref code."""
    def __init__(self, in_ch: int, nf: int):
        super().__init__()
        self.backbone = Encoder(in_ch, nf)
    def forward(self, x):
        feats = self.backbone(x)
        return [x] + feats

class PoseEncoder(nn.Module):
    """
    Ψ (geometry) encoder that outputs:
      - mu (B,N,2) from heatmap logits via axis-wise softmax
      - multi-scale gaussian maps (list) rendered at renderer scales
    """
    def __init__(self, in_ch: int, nf: int, n_maps: int, map_sizes: List[int],
                 gauss_std: float = 0.1, gauss_mode: str = "ankush"):
        super().__init__()
        self.map_sizes = map_sizes
        self.gauss_std = gauss_std
        self.gauss_mode = gauss_mode
        self.backbone = Encoder(in_ch, nf)
        self.heat_logits = conv_block(nf * 8, n_maps, k=1, s=1, batch_norm=False, activation=None)

    def forward(self, x):
        feats = self.backbone(x)
        x_last = feats[-1]               # (B, nf*8, h, w)
        logits = self.heat_logits(x_last)  # (B, N, h, w)
        mu = get_coord_2d(logits)          # (B, N, 2) in [-1,1]
        gmaps = []
        inv_std = 1.0 / self.gauss_std
        for sz in self.map_sizes:
            gmaps.append(get_gaussian_maps(mu, (sz, sz), inv_std, mode=self.gauss_mode))
        return mu, gmaps  # multi-scale, last is the smallest or largest depending on schedule

class Renderer(nn.Module):
    """
    U-Net-ish upsampling head that starts at the smallest render resolution and
    upsamples to the requested output size. Mirrors the sequence/ratios of the reference.
    """
    def __init__(self, map_size_hw: Tuple[int, int], map_filters: int, nf: int,
                 out_ch: int, final_res: int, batch_norm: bool = True):
        super().__init__()
        H, W = map_size_hw
        assert H == W
        seq = nn.Sequential()
        i = 1
        cur_filters = map_filters
        cur_res = H
        while cur_res <= final_res:
            seq.add_module(f"conv_render{i}", conv_block(cur_filters, nf, k=3, s=1, batch_norm=batch_norm))
            if cur_res == final_res:
                seq.add_module("conv_render_final", conv_block(nf, out_ch, k=3, s=1, batch_norm=False, activation=None))
                break
            # another conv + upsample
            seq.add_module(f"conv_render{i+1}", conv_block(nf, nf, k=3, s=1, batch_norm=batch_norm))
            next_res = cur_res * 2
            seq.add_module(f"upsample_render{i+1}", nn.Upsample(size=(next_res, next_res), mode="nearest"))
            cur_res = next_res
            cur_filters = nf
            if nf >= 8:
                nf //= 2
            i += 2
        self.seq = seq

    def forward(self, x):
        y = self.seq(x)
        return torch.sigmoid(y)  # images in [0,1]

class KeypointDetectionModel(nn.Module):
    """
    AssembleNet variant (L2 loss): ImageEncoder (appearance) + PoseEncoder (geometry) + Renderer
    """
    def __init__(self,
                 in_ch: int = 3,
                 nf: int = 32,
                 n_maps: int = 10,
                 gauss_std: float = 0.1,
                 renderer_stride: int = 2,
                 render_nf: int = 32,
                 out_ch: int = 3,
                 max_size: Tuple[int, int] = (128, 128),
                 min_size: Tuple[int, int] = (16, 16),
                 gauss_mode: str = "ankush"):
        super().__init__()
        assert max_size[0] == max_size[1] and min_size[0] == min_size[1]
        self._render_sizes = self._make_sizes(max_size[0], min_size[0], renderer_stride)
        self.image_encoder = ImageEncoder(in_ch, nf)
        self.pose_encoder = PoseEncoder(in_ch, nf, n_maps, map_sizes=self._render_sizes,
                                        gauss_std=gauss_std, gauss_mode=gauss_mode)
        # concat last appearance feature (nf*8) with last pose map (N)
        self.map_filters = nf * 8 + n_maps
        self.renderer = Renderer(min_size, self.map_filters, render_nf, out_ch, final_res=max_size[0])
        _init_weight(self.modules())

    @staticmethod
    def _make_sizes(max_side: int, min_side: int, stride: int) -> List[int]:
        sizes = []
        cur = max_side
        while cur >= min_side:
            sizes.append(cur)
            # renderer consumes maps from smallest to largest; the list holds all scales
            cur = max_side // stride
            max_side = cur
        return sizes

    def forward(self, img_src: torch.Tensor, img_tgt: torch.Tensor):
        """
        img_src: source/appearance image (B,3,H,W)
        img_tgt: target/geometry image  (B,3,H,W)
        Returns:
          y_pred: reconstruction of img_tgt using appearance from img_src & geometry from img_tgt
          mu: (B,N,2) keypoints in [-1,1] (y, x)
          g_small: the smallest-resolution gaussian stack used for rendering (B,N,h,w)
        """
        app_feats = self.image_encoder(img_src)          # pyramid; app_feats[-1] is the bottleneck
        mu, gmaps = self.pose_encoder(img_tgt)           # list of gaussian stacks at multiple scales
        joint = torch.cat([app_feats[-1], gmaps[-1]], dim=1)  # concat bottleneck + smallest maps
        y_pred = self.renderer(joint)
        return y_pred, mu, gmaps[-1]
