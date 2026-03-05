#imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import pydicom
import nibabel as nib

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

#datapoint class
class DataPoint:
    def __init__(self, xrayfi, segfi, og_width, og_height, tar_width, tar_height, fill_noise, mirror):
        self.og_width = og_width
        self.og_height = og_height
        self.tar_width = tar_width
        self.tar_height = tar_height
        self.fill_noise = fill_noise
        self.mirror = mirror

        ds = pydicom.dcmread(xrayfi)
        xrayimg = ds.pixel_array
        xrayimg = self.proc_img(xrayimg, fill_noise, mirror)

        segimg = nib.load(segfi).get_fdata().transpose(1,0,2).squeeze()
        segimg = self.proc_img(segimg, False, False)

        self.data = (xrayimg.squeeze(0), segimg.squeeze(0))

    def proc_img(self, img, fill_noise, mirror):
        img = torch.from_numpy(img).float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = img.unsqueeze(0).unsqueeze(0)

        msk = torch.ones_like(img)
        mirmsk = torch.ones_like(img)

        #flip
        mirimg = img.clone()
        img = TF.hflip(img)

        #scale down (to match height)
        scale_fact = self.tar_height / self.og_height
        curwidth = int(self.og_width * scale_fact)
        img = F.interpolate(img, size = (self.tar_height, curwidth), mode = "bilinear", align_corners = False)
        mirimg = F.interpolate(mirimg, size = (self.tar_height, curwidth), mode = "bilinear", align_corners = False)
        msk = F.interpolate(msk, size = (self.tar_height, curwidth), mode = "bilinear", align_corners = False)
        mirmsk = F.interpolate(mirmsk, size = (self.tar_height, curwidth), mode = "bilinear", align_corners = False)

        #padding (on left for normal, right for mirror)
        pad_amt = self.tar_width - curwidth
        img = F.pad(img, (pad_amt, 0, 0, 0), mode = "constant")
        mirimg = F.pad(mirimg, (0, pad_amt, 0, 0), mode = "constant")
        msk = F.pad(msk, (pad_amt, 0, 0, 0), mode = "constant")
        mirmsk = F.pad(mirmsk, (0, pad_amt, 0, 0), mode = "constant")

        #mirror
        if mirror:
            img = img + mirimg
            msk = msk + mirmsk

        #fill noise
        if fill_noise:
            msk = 1 - msk
            img = msk * torch.rand_like(img) + img
        
        return img


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
    def __init__(self, train_val_split, batch_size, og_width, og_height, tar_width, tar_height, seed, fill_noise, mirror):
        self.xrayfi = "../../data/segmentation/data/xray/"
        self.segfi = "../../data/segmentation/data/seg/"

        self.train_split = train_val_split[0]
        self.val_split = train_val_split[1]
        self.train_batch = batch_size[0]
        self.val_batch = batch_size[1]
        self.test_batch = batch_size[2]
        self.og_width = og_width
        self.og_height = og_height
        self.tar_width = tar_width
        self.tar_height = tar_height
        self.seed = seed
        self.fill_noise = fill_noise
        self.mirror = mirror

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
            data_points.append(DataPoint(self.xrayfi + img_file + ".dcm", self.segfi + img_file + ".nii.gz", self.og_width, self.og_height, self.tar_width, self.tar_height, self.fill_noise, self.mirror))
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
