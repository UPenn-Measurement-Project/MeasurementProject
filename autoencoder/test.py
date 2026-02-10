#imports
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import DataProcessor
from model.model import Autoencoder

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

parser.add_argument("--idata", type = str, required = True, help = "Path to image data directory (from ../data)")
parser.add_argument("--ds", type = str, required = True, help = "Dataset to test on (train, valid, test)")

parser.add_argument("--resid", action = "store_true", help = "Residual connections for autoencoder")

parser.add_argument("--path", type = str, default = "current_best.pth", required = False, help = "Model path (from ./model_saves/)")
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

img_dir = args.idata
test_set_name = args.ds.lower()

model_resid = args.resid

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
print(f"Image directory: {img_dir}\nSelected dataset: {test_set_name}\n")
print(f"Autoencoder residual connections: {model_resid}\nTorch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(img_dir, (train_split, val_split), batch_sizes, img_width, img_height, seed)
dataset, dataloader = data_processor.create_ds(test_set_name)

#==========#

#model set up
model = Autoencoder(model_resid, channel_cnt = [1, 64, 128, 32, 1])

model.load_state_dict(torch.load(f"./model_saves/{model_path}"))
print(f"\n==========\n\nModel loaded from./model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

with torch.no_grad():
    for idx, images in enumerate(tqdm(dataloader, unit = "batch")):
        images = images.to(device)

        pred = model(images)

        plt.figure(figsize = (6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(images[0][0].detach().cpu(), cmap = 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(pred[0][0].detach().cpu(), cmap = 'gray')
        plt.savefig(f"./test_results/{idx}.png")

        #testing
        #predenc = model.enc(images)
        #raise SystemExit(predenc[-1].shape)

print("\n==========\n\nDone\n")
