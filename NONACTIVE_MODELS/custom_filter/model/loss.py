# loss.py (torchvision-only version, no 255 preprocessing)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# conv name -> torchvision vgg16.features index (pre-ReLU)
CONV_INDEX = {
    "conv1_1": 0,  "conv1_2": 2,
    "conv2_1": 5,  "conv2_2": 7,
    "conv3_1": 10, "conv3_2": 12, "conv3_3": 14,
    "conv4_1": 17, "conv4_2": 19, "conv4_3": 21,
    "conv5_1": 24, "conv5_2": 26, "conv5_3": 28,
}

@dataclass
class PerceptualConfig:
    comp: Sequence[str] = field(default_factory=lambda: ['input','conv1_2','conv2_2','conv3_2','conv4_2','conv5_2'])
    init_ws: Sequence[float] = field(default_factory=lambda: [100.0,1.6,2.3,1.8,2.8,100.0])
    l2: bool = True
    ema_momentum: float = 0.99
    net_file: Optional[str] = None  # if None, use torchvision pretrained
    # torchvision stats (on 0–1 inputs)
    tv_mean01: Sequence[float] = (0.485, 0.456, 0.406)
    tv_std01:  Sequence[float] = (0.229, 0.224, 0.225)
    # pixel term behavior
    input_feature_mode: str = "raw"   # {"raw", "preproc"}

class EmaScalar(nn.Module):
    def __init__(self, init_value: float, momentum: float):
        super().__init__()
        self.register_buffer("value", torch.tensor(float(init_value)))
        self.momentum = float(momentum)
    def update(self, x_scalar: torch.Tensor) -> torch.Tensor:
        self.value.mul_(self.momentum).add_(x_scalar.detach() * (1.0 - self.momentum))
        return self.value

class VGG16ConvFeatures(nn.Module):
    """Returns dict of requested conv layer activations (pre-ReLU)."""
    def __init__(self, conv_names: Sequence[str], net_file: Optional[str]):
        super().__init__()
        try:
            vgg = models.vgg16(weights=None if net_file else models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            vgg = models.vgg16(pretrained=(net_file is None))
        if net_file:
            sd = torch.load(net_file, map_location="cpu")
            vgg.load_state_dict(sd)
        self.features = vgg.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.targets: List[int] = []
        self.names: List[str] = []
        for name in conv_names:
            if name not in CONV_INDEX:
                raise ValueError(f"Unknown conv layer '{name}'. Allowed: {sorted(CONV_INDEX.keys())}")
            self.targets.append(CONV_INDEX[name])
            self.names.append(name)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        cur = x
        want = set(self.targets)
        for i, layer in enumerate(self.features):
            cur = layer(cur)
            if i in want:
                nm = next(n for n, j in zip(self.names, self.targets) if j == i)
                out[nm] = cur
                if len(out) == len(self.targets):
                    break
        return out

class PerceptualLoss(nn.Module):
    """
    - Compares 'input' (raw pixels) + specified pre-ReLU conv features
    - Per-layer L1/L2 with optional mask
    - EMA normalization per layer
    - Final sum scaled by 1000.0
    - Uses torchvision normalization (on 0–1 inputs)
    """
    def __init__(self, cfg: PerceptualConfig):
        super().__init__()
        self.cfg = cfg
        if len(cfg.comp) != len(cfg.init_ws):
            raise ValueError(f"len(comp)={len(cfg.comp)} must match len(init_ws)={len(cfg.init_ws)}")
        if cfg.comp[0].lower() != "input":
            raise ValueError("First entry of comp must be 'input'")
        conv_list = [c for c in cfg.comp if c.lower() != "input"]
        self.vgg = VGG16ConvFeatures(conv_list, cfg.net_file)
        mean = torch.tensor(cfg.tv_mean01).view(1,3,1,1).float()
        std  = torch.tensor(cfg.tv_std01).view(1,3,1,1).float()
        self.register_buffer("tv_mean01", mean)
        self.register_buffer("tv_std01", std)
        self.ema = nn.ModuleDict({n: EmaScalar(w, cfg.ema_momentum) for n,w in zip(cfg.comp,cfg.init_ws)})
        self.eps = 1e-12
    def _preprocess_vgg(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.tv_mean01.to(x.device, x.dtype)) / self.tv_std01.to(x.device, x.dtype)
    def _resize_mask(self, mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        return F.interpolate(mask, size=(h,w), mode="bilinear", align_corners=False)
    def _reduce(self, diff: torch.Tensor, layer: str, mask: Optional[torch.Tensor], training: bool) -> torch.Tensor:
        if mask is not None:
            m = self._resize_mask(mask, diff.size(-2), diff.size(-1))
            diff = diff * m
        meanval = diff.mean()
        denom = self.ema[layer].update(meanval) if training else self.ema[layer].value
        return (diff / (denom + self.eps)).mean()
    def forward(self, gt: torch.Tensor, pr: torch.Tensor,
                loss_mask: Optional[torch.Tensor]=None, training: Optional[bool]=None) -> torch.Tensor:
        if training is None:
            training = self.training
        # 1) 'input' term
        if self.cfg.input_feature_mode == "preproc":
            gt_in, pr_in = self._preprocess_vgg(gt), self._preprocess_vgg(pr)
        else:
            gt_in, pr_in = gt, pr
        diff_in = (gt_in - pr_in).pow(2) if self.cfg.l2 else (gt_in - pr_in).abs()
        total = self._reduce(diff_in, "input", loss_mask, training)
        # 2) VGG features
        ims = torch.cat([gt, pr], dim=0)
        ims = self._preprocess_vgg(ims)
        self.vgg.eval()
        feats = self.vgg(ims)
        B = gt.size(0)
        for name,fmap in feats.items():
            f_gt,f_pr = fmap[:B],fmap[B:]
            diff = (f_gt - f_pr).pow(2) if self.cfg.l2 else (f_gt - f_pr).abs()
            total = total + self._reduce(diff, name, loss_mask, training)
        return 1000.0 * total

