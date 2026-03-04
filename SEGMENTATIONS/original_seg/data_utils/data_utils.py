#imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import pydicom
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#datapoint class
class DataPoint:
    def __init__(self, xrayfi, segfi, img_width, img_height):
        self.xrayfi = xrayfi

        ds = pydicom.dcmread(xrayfi)
        xrayimg = ds.pixel_array

        segimg = nib.load(segfi).get_fdata().transpose(1,0,2).squeeze()

        xrayimg = torch.from_numpy(xrayimg).float()
        xrayimg = (xrayimg - xrayimg.min()) / (xrayimg.max() - xrayimg.min() + 1e-5)
        xrayimg = xrayimg.unsqueeze(0).unsqueeze(0)

        segimg = torch.from_numpy(segimg).float()
        segime = (segimg - segimg.min()) / (segimg.max() - segimg.min() + 1e-5)
        segimg = segimg.unsqueeze(0).unsqueeze(0)

        #downscale image
        try:
            xrayimg = F.interpolate(xrayimg, size = (img_height, img_width), mode = "bilinear", align_corners = False)
            segimg = F.interpolate(segimg, size = (img_height, img_width), mode = "bilinear", align_corners = False)
        except:
            raise SystemError(xrayfi)

        self.data = (xrayimg.squeeze(0), segimg.squeeze(0))

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
    def __init__(self, train_val_split, batch_size, img_width, img_height, seed):
        self.xrayfi = "../../data/segmentation/data/xray/"
        self.segfi = "../../data/segmentation/data/seg/"

        self.train_split = train_val_split[0]
        self.val_split = train_val_split[1]
        self.train_batch = batch_size[0]
        self.val_batch = batch_size[1]
        self.test_batch = batch_size[2]
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed

        all_xray_img = [i[:-4] for i in os.listdir(self.xrayfi) if i.lower().endswith("dcm")]
        all_seg_img = [i[:-7] for i in os.listdir(self.segfi) if i.lower().endswith("nii.gz")]
        all_img = sorted(list(set(all_xray_img) & set(all_seg_img)))
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
        for img_file in tqdm(ds, unit = "image"):
            data_points.append(DataPoint(self.xrayfi + img_file + ".dcm", self.segfi + img_file + ".nii.gz", self.img_width, self.img_height))
        print(f"Loaded {ds_name} with {len(data_points)} points")

        dataset = ImageDataset(data_points)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

        return dataset, dataloader

    def get_fi_name(self, ds_name, bsz, bidx, fidx):
        if ds_name == "train":
            return self.train_img[bsz * bidx + fidx]
        elif ds_name == "valid":
            return self.val_img[bsz * bidx + fidx]
        elif ds_name == "test":
            return self.test_img[bsz * bidx + fidx]
        else:
            raise ValueError(f"Unknown dataset name: {ds_name}")
