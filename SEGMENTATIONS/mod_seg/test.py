#imports
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from SEGMENTATIONS.mod_seg.data_utils.data_utils import DataProcessor
from SEGMENTATIONS.mod_seg.model.model import SegUNet

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
parser.add_argument("--noise", action = "store_true", help = "Fill image will noise")
parser.add_argument("--mirror", action = "store_true", help = "Mirror femur to left")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 1, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 1, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 1, required = False, help = "Testiing set batch size")

args = parser.parse_args()

#settings

og_width = 512
og_height = 1024
tar_width = 240
tar_height = 192

test_set_name = args.ds.lower()

model_path = args.path
fill_noise = args.noise
mirror = args.mirror
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
data_processor = DataProcessor((train_split, val_split), batch_sizes, og_width, og_height, tar_width, tar_height, seed, fill_noise, mirror)
dataset, dataloader = data_processor.create_ds(test_set_name)

#==========#

#model set up
model = SegUNet()

model.load_state_dict(torch.load(f"SEGMENTATIONS/mod_seg/model_saves/{model_path}"))
print(f"\n==========\n\nModel loaded from SEGMENTATIONS/mod_seg/model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

dice_scores = []

with torch.no_grad():
    for idx, (xrays, segs) in enumerate(tqdm(dataloader, unit = "batch")):
        xrays = xrays.to(device)
        segs = segs.to(device)

        pred = model(xrays)

        pred_bin = (pred > 0.5).float()
        intersection = (pred_bin * segs).sum(dim = (1, 2, 3))
        dice = (2.0 * intersection) / (pred_bin.sum(dim = (1, 2, 3)) + segs.sum(dim = (1, 2, 3)) + 1e-8)
        dice_scores.extend(dice.cpu().tolist())

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

        plt.savefig(f"SEGMENTATIONS/mod_seg/test_results/img/{idx}.png")

mean_dice = sum(dice_scores) / len(dice_scores)
print(f"\n==========\n\nDice Coefficient Results ({test_set_name} set):")
print(f"  Mean:  {mean_dice:.4f}")
print(f"  Min:   {min(dice_scores):.4f}")
print(f"  Max:   {max(dice_scores):.4f}")
print(f"  Count: {len(dice_scores)}")
print("\n==========\n\nDone\n")
