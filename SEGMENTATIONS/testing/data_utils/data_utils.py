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

        if self.img_flip:
            img = TF.hflip(img)

        #downscale image
        img = F.interpolate(img, size = (img_height, img_width), mode = "bilinear", align_corners = False)

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
    def __init__(self, img_width, img_height, tc_cnt):
        self.img_dir = "./data/box_images"
        self.img_width = img_width
        self.img_height = img_height
        self.tc_cnt = tc_cnt

        #####image data#####

        self.all_img = [i for i in os.listdir(self.img_dir) if i.lower().endswith("dcm")]
        if self.tc_cnt != -1:
            if self.tc_cnt > len(self.all_img):
                raise ValueError(f'Requested test case count ({self.tc_cnt}) exceeds total files ({len(self.all_img)})')
            self.all_img = self.all_img[:self.tc_cnt]
        
        print(f"Images in use: {len(self.all_img)}\n")
    
    def create_ds(self):
        data_points = []
        for img_file in tqdm(self.all_img, unit = "image"):
            for img_flip in range(2):
                    data_points.append(DataPoint(self.img_dir, img_file, img_flip, self.img_width, self.img_height))
        print(f"Loaded {len(data_points)} points")

        dataset = ImageDataset(data_points)
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)

        return dataset, dataloader
