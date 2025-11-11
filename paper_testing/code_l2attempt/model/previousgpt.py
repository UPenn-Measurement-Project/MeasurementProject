#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#vgg loss
##vgg class
class VGG(nn.Module):
    def __init__(self, used_layers, use_input):
        super().__init__()
        self.used_layers = used_layers
        self.use_input = use_input
        vgg = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.backbone = vgg.features.eval()
        for i, m in enumerate(self.backbone):
            if isinstance(m, nn.ReLU):
                self.backbone[i] = nn.ReLU(inplace = False)
        for p in self.backbone.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent = False)
        self.register_buffer("std", std, persistent = False)
    
    def forward(self, x):
        x = (x - self.mean) / self.std

        feats = []

        if self.use_input:
            feats.append(x)

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.used_layers:
                feats.append(x)
        return feats

class PerceptualLoss(nn.Module):
    def __init__(self, used_layers = [3, 8, 15, 22, 29], use_input = True, weights = [100.0, 1.6, 2.3, 1.8, 2.8, 100.0], alpha = 0.99, fin_scale = 1000):
        super().__init__()
        self.alpha = alpha
        self.fin_scale = fin_scale
        
        self.vgg = VGG(used_layers, use_input)
        self.l2 = nn.MSELoss(reduction = "mean")

        w = torch.tensor(weights)
        self.register_buffer("weights", w, persistent = True)
    
    def forward(self, y_pred, y_act):
        # targets don't need grads
        feat_act  = self.vgg(y_act.detach())
        feat_pred = self.vgg(y_pred)          # grads flow to y_pred/model

        # 1) compute per-feature losses (part of graph)
        per_feat = [self.l2(a, p) for a, p in zip(feat_act, feat_pred)]

        # 2) build updated weights for THIS step (detached, not in graph)
        with torch.no_grad():
            new_w = self.alpha * self.weights + (1 - self.alpha) * torch.stack([l.detach() for l in per_feat])

        # 3) use the updated weights to compute the loss (no in-place on buffers)
        eps = 1e-12
        loss = sum(l / (new_w[i] + eps) for i, l in enumerate(per_feat)) * self.fin_scale

        # 4) now persist the update (not part of the graph)
        with torch.no_grad():
            self.weights.copy_(new_w)

        return loss
