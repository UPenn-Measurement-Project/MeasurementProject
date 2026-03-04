#imports
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import DataProcessor
from model.model import SegUNet

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

parser.add_argument("--ds", type = str, required = True, help = "Dataset to test on (train, valid, test)")

parser.add_argument("--path", type = str, default = "current_best.pth", required = False, help = "Model path (from ./model_saves/)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 1, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 1, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 1, required = False, help = "Testiing set batch size")

args = parser.parse_args()

#settings

img_scale_factor = 0.5 
img_width = int(512 * img_scale_factor)
img_height = int(1024 * img_scale_factor)

test_set_name = args.ds.lower()

model_path = args.path
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

print("\nSelected settings:\n")
print(f"Selected dataset: {test_set_name}\n")
print(f"Torch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor((train_split, val_split), batch_sizes, img_width, img_height, seed)
dataset, dataloader = data_processor.create_ds(test_set_name)

#==========#

#model set up
model = SegUNet()

model.load_state_dict(torch.load(f"./model_saves/{model_path}"))
print(f"\n==========\n\nModel loaded from./model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

with torch.no_grad():
    for idx, (xrays, segs) in enumerate(tqdm(dataloader, unit = "batch")):
        xrays = xrays.to(device)
        segs = segs.to(device)

        pred = model(xrays)

        plt.figure(figsize = (12, 3))
        plt.subplot(1, 4, 1)
        plt.imshow(xrays[0][0].detach().cpu(), cmap = 'gray')

        plt.subplot(1, 4, 2)
        plt.imshow(segs[0][0].detach().cpu(), cmap = 'gray')
        
        plt.subplot(1, 4, 3)
        plt.imshow(pred[0][0].detach().cpu(), cmap = 'gray')

        plt.subplot(1, 4, 4)
        emptychan = torch.zeros_like(segs)
        colordiff = torch.cat((pred, emptychan, segs), dim = 1)
        plt.imshow(colordiff[0].detach().cpu().permute(1, 2, 0))

        plt.savefig(f"./test_results/img/{idx}.png")

print("\n==========\n\nDone\n")
