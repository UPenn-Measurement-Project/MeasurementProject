#imports
import os
import random
from tqdm import tqdm

import pydicom
import nibabel as nib

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader


class DataPoint:
    def __init__(self, xrayfi, segfi, og_width, og_height, tar_width, tar_height):
        self.og_width = og_width
        self.og_height = og_height
        self.tar_width = tar_width
        self.tar_height = tar_height

        ds = pydicom.dcmread(xrayfi)
        xrayimg = ds.pixel_array
        xrayimg = self._proc_img(xrayimg)

        segimg = nib.load(segfi).get_fdata().transpose(1, 0, 2).squeeze()
        segimg = self._proc_img(segimg)

        # store as (1, H, W) tensors
        self.data = (xrayimg.squeeze(0), segimg.squeeze(0))

    def _proc_img(self, img):
        img = torch.from_numpy(img).float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        img = TF.hflip(img)

        scale_fact = self.tar_height / self.og_height
        curwidth = int(self.og_width * scale_fact)
        img = F.interpolate(img, size=(self.tar_height, curwidth), mode='bilinear', align_corners=False)

        pad_amt = self.tar_width - curwidth
        img = F.pad(img, (pad_amt, 0, 0, 0), mode='constant')

        return img  # (1, 1, H, W)


class ImageDataset(Dataset):
    def __init__(self, data_points, augment=False):
        self.data_points = data_points
        self.augment = augment

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        img, mask = self.data_points[idx].data  # each (1, H, W)

        if self.augment:
            angle = random.uniform(-12, 12)
            scale = random.uniform(0.85, 1.15)
            tx = int(random.uniform(-0.05, 0.05) * img.shape[-1])
            ty = int(random.uniform(-0.05, 0.05) * img.shape[-2])

            img = TF.affine(img, angle=angle, translate=[tx, ty], scale=scale, shear=0,
                            interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle=angle, translate=[tx, ty], scale=scale, shear=0,
                             interpolation=InterpolationMode.NEAREST, fill=0)

        return img, mask


class DataProcessor:
    def __init__(self, train_val_split, batch_sizes, og_width, og_height, tar_width, tar_height, seed):
        self.xrayfi = "../../data/segmentation/data/xray/"
        self.segfi = "../../data/segmentation/data/seg/"
        self.train_split = train_val_split[0]
        self.val_split = train_val_split[1]
        self.train_batch = batch_sizes[0]
        self.val_batch = batch_sizes[1]
        self.test_batch = batch_sizes[2]
        self.og_width = og_width
        self.og_height = og_height
        self.tar_width = tar_width
        self.tar_height = tar_height
        self.seed = seed

        all_xray = [i[:-4] for i in os.listdir(self.xrayfi) if i.lower().endswith('.dcm')]
        all_seg = [i[:-7] for i in os.listdir(self.segfi) if i.lower().endswith('.nii.gz')]
        all_img = sorted(list(set(all_xray) & set(all_seg)))
        print(f"Images in use: {len(all_img)}")

        train_sz = int(self.train_split * len(all_img))
        val_sz = int(self.val_split * len(all_img))

        torch.manual_seed(self.seed)
        all_idx = torch.randperm(len(all_img))
        self.train_img = [all_img[i] for i in all_idx[:train_sz]]
        self.val_img = [all_img[i] for i in all_idx[train_sz:train_sz + val_sz]]
        self.test_img = [all_img[i] for i in all_idx[train_sz + val_sz:]]

    def create_ds(self, ds_name, augment=False):
        if ds_name == 'train':
            img_list = self.train_img
            batch_size = self.train_batch
            shuffle = True
        elif ds_name == 'valid':
            img_list = self.val_img
            batch_size = self.val_batch
            shuffle = False
        elif ds_name == 'test':
            img_list = self.test_img
            batch_size = self.test_batch
            shuffle = False
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

        data_points = []
        for fi in tqdm(img_list, unit='image'):
            data_points.append(DataPoint(
                self.xrayfi + fi + '.dcm',
                self.segfi + fi + '.nii.gz',
                self.og_width, self.og_height, self.tar_width, self.tar_height
            ))
        print(f"Loaded {ds_name}: {len(data_points)} samples")

        dataset = ImageDataset(data_points, augment=augment)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader
