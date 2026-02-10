#imports
import numpy as np
import pandas as pd
import os
import pydicom
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

#datapoint class
class DataPoint:
    def __init__(self, img_dir, img_file, img_flip, img_width, img_height, aug_rot = 0, aug_scale = 1, aug_crop = 1):
        self.img_path = img_dir + "/" + img_file
        self.img_flip = img_flip
        self.img_width = img_width
        self.img_height = img_height
        self.aug_rot = aug_rot
        self.aug_scale = aug_scale
        self.aug_crop = aug_crop

        #additionally required aug_scale given aug_rot
        abscos = np.abs(np.cos(self.aug_rot * np.pi / 180))
        abssin = np.abs(np.sin(self.aug_rot * np.pi / 180))
        self.aug_scale *= min(1 / (abscos + self.img_height / self.img_width * abssin), 1 / (self.img_width / self.img_height * abssin + abscos))

        ds = pydicom.dcmread(self.img_path)
        img = ds.pixel_array

        img = torch.from_numpy(img).float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = img.unsqueeze(0).unsqueeze(0)

        #flip image if needed
        if self.img_flip:
            img = TF.hflip(img)

        #crop image
        _, _, orig_height, orig_width = img.shape
        img = img[:, :, int(orig_height * (1 - self.aug_crop)):, int(orig_width * (1 - self.aug_crop)):]

        #downscale image
        new_height = int(self.img_height * self.aug_scale)
        new_width = int(self.img_width * self.aug_scale)
        img = F.interpolate(img, size = (new_height, new_width), mode = "bilinear", align_corners = False)

        #pad back to target size
        pad_height = self.img_height - new_height
        pad_width = self.img_width - new_width
        pad_u = pad_height // 2
        pad_d = pad_height - pad_u
        pad_l = pad_width // 2
        pad_r = pad_width - pad_l
        img = F.pad(img, (pad_l, pad_r, pad_u, pad_d), mode = "constant")

        #rotate image
        img = TF.rotate(img, angle = self.aug_rot, interpolation = TF.InterpolationMode.BILINEAR)

        self.data = img.squeeze(0)

#torch inherited dataset class
class ImageDataset(Dataset):
    def __init__(self, data_points):
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx].data

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

        #####image data#####

        #all images in directory
        all_img = [i for i in os.listdir(self.img_dir) if i.lower().endswith("dcm")]

        #train and val sizes
        train_sz = int(self.train_split * len(all_img))
        val_sz = int(self.val_split * len(all_img))

        #splits
        torch.manual_seed(self.seed)
        all_idx = torch.randperm(len(all_img))
        self.train_img = [all_img[i] for i in all_idx[:train_sz]]
        self.val_img = [all_img[i] for i in all_idx[train_sz:train_sz + val_sz]]
        self.test_img = [all_img[i] for i in all_idx[train_sz + val_sz:]]
    
    def create_ds(self, ds_name, rot_rng = (0, 0, 1), scale_rng = (1, 1, 1), crop_rng = (1, 1, 1)):
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
        for img_file in tqdm(ds, unit = "image"):
            for img_flip in range(2):
                for aug_rot in np.arange(rot_rng[0], rot_rng[1] + 1e-8, rot_rng[2]):
                    for aug_scale in np.arange(scale_rng[0], scale_rng[1] + 1e-8, scale_rng[2]):
                        for aug_crop in np.arange(crop_rng[0], crop_rng[1] + 1e-8, crop_rng[2]):
                            data_points.append(DataPoint(self.img_dir, img_file, img_flip, self.img_width, self.img_height, aug_rot, aug_scale, aug_crop))
        print(f"Loaded {ds_name} with {len(data_points)} points")

        dataset = ImageDataset(data_points)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

        return dataset, dataloader
