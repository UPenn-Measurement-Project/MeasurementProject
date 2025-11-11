#imports
import numpy as np
import pandas as pd
import os
import pydicom

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

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

#datapoint class
class DataPoint:
    def __init__(self, img_dir, img_file, img_flip, img_width, img_height):
        self.img_path = img_dir + "/" + img_file
        self.img_flip = img_flip
        self.img_width = img_width
        self.img_height = img_height

        ds = pydicom.dcmread(self.img_path)
        img = ds.pixel_array

        img = torch.from_numpy(img).float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = img.unsqueeze(0).unsqueeze(0)

        #flip image if needed
        if self.img_flip:
            img = TF.hflip(img)

        #downscale image
        img = F.interpolate(img, size = (self.img_height, self.img_width), mode = "bilinear", align_corners = False)

        self.img = img.squeeze(0)

#torch inherited dataset class
class ImageDataset(Dataset):
    def __init__(self, data_points):
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx].img

#class for creating dataset
class DataProcessor:
    def __init__(self, img_dir, train_val_split, batch_size, img_width, img_height, seed):
        self.img_dir = "../data/" + img_dir
        self.train_split = train_val_split[0]
        self.val_split = train_val_split[1]
        self.train_batch = batch_size[0]
        self.val_batch = batch_size[1]
        self.test_batch = batch_size[2]
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed

        #all images in directory
        all_img = [i for i in os.listdir(self.img_dir) if i.lower().endswith("dcm")]
        print(f"Images in use: {len(all_img)}\n")

        #train and val sizes
        train_sz = int(self.train_split * len(all_img))
        val_sz = int(self.val_split * len(all_img))

        #splits
        torch.manual_seed(self.seed)
        all_idx = torch.randperm(len(all_img))
        self.train_img = [all_img[i] for i in all_idx[:train_sz]]
        self.val_img = [all_img[i] for i in all_idx[train_sz:train_sz + val_sz]]
        self.test_img = [all_img[i] for i in all_idx[train_sz + val_sz:]]
    
    def create_ds(self, ds_name):
        if ds_name == "train":
            ds = self.train_img
            batch_size = self.train_batch
            shuffle = True
        elif ds_name == "valid":
            ds = self.val_img
            batch_size = self.val_batch
            shuffle = False
        elif ds_name == "test":
            ds = self.test_img
            batch_size = self.test_batch
            shuffle = False
        else:
            raise ValueError(f"Unknown dataset name: {ds_name}")

        data_points = []
        for img_file in ds:
            for img_flip in range(2):
                data_points.append(DataPoint(self.img_dir, img_file, img_flip, self.img_width, self.img_height))
        print(f"Loaded {ds_name} with {len(data_points)} points")

        dataset = ImageDataset(data_points)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

        return dataset, dataloader
