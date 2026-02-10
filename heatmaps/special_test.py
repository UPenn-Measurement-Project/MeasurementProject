#imports
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import DataProcessor
from model.model import HeatmapsModel, measurements_to_coord, coord_to_measurements, abs_kp_to_coord, get_ab

#==========#

#device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

#parser

parser = argparse.ArgumentParser(description = "Model training")

parser.add_argument("--mdata", type = str, required = True, help = "Path to measurement data file (from ../data/measurements)")
parser.add_argument("--idata", type = str, required = True, help = "Path to image data directory (from ../data)")
parser.add_argument("--ds", type = str, required = True, help = "Dataset to test on (train, valid, test)")
parser.add_argument("--model", type = str, required = True, help = "Model type (hm)")

parser.add_argument("--path", type = str, default = "current_best.pth", required = False, help = "Model path (from ./model_saves/)")
parser.add_argument("--bn", type = str, default = "none", required = False, help = "Batch norm setting (none, before, after)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 1, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 1, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 1, required = False, help = "Testiing set batch size")

args = parser.parse_args()

#settings

pix_per_mm = 2400 / 408 
img_scale_factor = 0.1 
img_width = int(2400 * img_scale_factor)
img_height = int(1920 * img_scale_factor)

measurement_file = args.mdata
img_dir = args.idata
test_set_name = args.ds.lower()
model_name = args.model

model_path = args.path
batch_norm_setting = args.bn
seed = args.seed
train_split = args.train_split
val_split = args.val_split
train_batch_size = args.train_bs
val_batch_size = args.val_bs
test_batch_size = args.test_bs
batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

#checks
if test_set_name not in ["train", "valid", "test"]:
    raise ValueError(f"Unknown dataset name \"{test_set_name}\"")
if model_name not in ["hm"]:
    raise ValueError(f"Unknown model type: {model_name}")
if batch_norm_setting not in ["none", "before", "after"]:
    raise ValueError(f"Unknown batch norm setting: {batch_norm_setting}")

print("\nSelected settings:\n")
print(f"Measurement file: {measurement_file}\nImage directory: {img_dir}\nSelected dataset: {test_set_name}\n")
print(f"Batch norm setting: {batch_norm_setting}\nTorch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(measurement_file, img_dir, (train_split, val_split), batch_sizes, img_width, img_height, seed)
dataset, dataloader = data_processor.create_ds(test_set_name)

#==========#

#model set up
if model_name == "hm":
    model = HeatmapsModel(img_width, img_height, 13) #include A

model.load_state_dict(torch.load(f"./model_saves/{model_path}"))
print(f"\n==========\n\nModel loaded from./model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

with torch.no_grad():
    for idx, (img, yvals, aug_scales) in enumerate(tqdm(dataloader, unit = "batch")):
        img = img.to(device)
        yvals = yvals.to(device)
        aug_scales = aug_scales.to(device)

        plt.figure()

        #original image
        abs_coord = model(img)
        model_coord = abs_kp_to_coord(abs_coord)
        ab = get_ab(model_coord)
        ypred = coord_to_measurements(model_coord, pix_per_mm, img_scale_factor)
        print("Original:")
        print(torch.abs(yvals - ypred) / yvals)
        plt.subplot(1, 4, 1)
        plt.imshow(img[0][0].detach().cpu(), cmap = 'gray')
        plt.scatter(abs_coord[0, :, 0].detach().cpu(), abs_coord[0, :, 1].detach().cpu(), s = 10)

        plt.subplot(1, 4, 4)
        plt.scatter(abs_coord[0, :, 0].detach().cpu(), abs_coord[0, :, 1].detach().cpu(), s = 10, c = 'b')

        '''
        for i, label in enumerate(['A', 'D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'B']):
            curx, cury = abs_coord[0, i]
            curx = curx.detach().cpu()
            cury = cury.detach().cpu()
            plt.text(curx + 0.5, cury + 0.5, label, c = 'r')
        '''

        #all noise
        noise = torch.rand_like(img)
        abs_coord = model(noise)
        model_coord =  abs_kp_to_coord(abs_coord)
        ab = get_ab(model_coord)
        ypred = coord_to_measurements(model_coord, pix_per_mm, img_scale_factor)
        print("All noise:")
        print(torch.abs(yvals - ypred) / yvals)
        plt.subplot(1, 4, 2)
        plt.imshow(noise[0][0].detach().cpu(), cmap = 'gray')
        plt.scatter(abs_coord[0, :, 0].detach().cpu(), abs_coord[0, :, 1].detach().cpu(), s = 10)

        plt.subplot(1, 4, 4)
        plt.scatter(abs_coord[0, :, 0].detach().cpu(), abs_coord[0, :, 1].detach().cpu(), s = 10, c = 'black')

        #part noise
        msk = torch.ones_like(img)
        msk[:, :, img_height//2:, img_width//2:] = 0
        inv_msk = 1 - msk
        part_noise = img * inv_msk + noise * msk
        abs_coord = model(part_noise)
        model_coord = abs_kp_to_coord(abs_coord)
        ab = get_ab(model_coord)
        ypred = coord_to_measurements(model_coord, pix_per_mm, img_scale_factor)
        print("Part noise:")
        print(torch.abs(yvals - ypred) / yvals)
        plt.subplot(1, 4, 3)
        plt.imshow(part_noise[0][0].detach().cpu(), cmap = 'gray')
        plt.scatter(abs_coord[0, :, 0].detach().cpu(), abs_coord[0, :, 1].detach().cpu(), s = 10)

        plt.subplot(1, 4, 4)
        plt.scatter(abs_coord[0, :, 0].detach().cpu(), abs_coord[0, :, 1].detach().cpu(), s = 10, c = 'r')
        plt.xlim(0, 240)
        plt.ylim(192, 0)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.show()
