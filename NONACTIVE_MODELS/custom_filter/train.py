#imports
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import DataProcessor

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

parser.add_argument("--model", type = str, default = "basic", required = False, help = "Model type (basic, alex)") #FIXME

parser.add_argument("--aug", action = "store_true", help = "Augment training and validation sets")

parser.add_argument("--epoch", type = int, default = 100, required = False, help = "Max number of epochs")
parser.add_argument("--bn", type = str, default = "none", required = False, help = "Batch norm setting (none, before, after)")
parser.add_argument("--lr", type = float, default = 1e-3, required = False, help = "Learning rate")
parser.add_argument("--esl", type = int, default = 5, required = False, help = "Number of epochs with no improvement to early stop training")
parser.add_argument("--load", type = str, default = "", required = False, help = "Load a model to continue training (./model_saves)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 64, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 64, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 32, required = False, help = "Testiing set batch size")

args = parser.parse_args()

#settings

pix_per_mm = 2400 / 408
img_scale_factor = 0.1
img_width = int(2400 * img_scale_factor)
img_height = int(1920 * img_scale_factor)

measurement_file = args.mdata
img_dir = args.idata
model_name = args.model

aug_data = args.aug

epoch_cnt = args.epoch
batch_norm_setting = args.bn
learning_rate = args.lr
early_stop_lim = args.esl
load_path = None if args.load == "" else args.load
seed = args.seed
train_split = args.train_split
val_split = args.val_split
train_batch_size = args.train_bs
val_batch_size = args.val_bs
test_batch_size = args.test_bs
batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

if model_name not in ["basic", "alex"]:
    raise ValueError(f"Unknown model type: {model_name}")
if batch_norm_setting not in ["none", "before", "after"]:
    raise ValueError(f"Unknown batch norm setting: {batch_norm_setting}")

print("\nSelected settings:\n")
print(f"Measurement file: {measurement_file}\nImage directory: {img_dir}\nModel: {model_name}\n")
print(f"Augment data: {aug_data}\nMax epoch cnt: {epoch_cnt}\nBatch norm setting: {batch_norm_setting}\nLearning rate: {learning_rate}\nEarly stopping limit: {early_stop_lim}\nLoad pretrained: {load_path}\nTorch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

#data
print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(measurement_file, img_dir, (train_split, val_split), batch_sizes, img_width, img_height, seed)
if aug_data:
    train_set, train_loader = data_processor.create_ds("train", (-10, 10, 10), (0.5, 1, 0.25), (0.6, 1, 0.2)) #tuples are augmentation values
    val_set, val_loader = data_processor.create_ds("valid", (-10, 10, 10), (0.5, 1, 0.25), (0.6, 1, 0.2)) #tuples are augmentation values
else:
    train_set, train_loader = data_processor.create_ds("train") #tuples are augmentation values
    val_set, val_loader = data_processor.create_ds("valid") #tuples are augmentation values

#==========#

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

sana, _, _ = next(iter(train_loader))
sana = sana[0, 0].numpy()
#plt.imshow(sana, cmap = 'gray')
#plt.show()

sha = (sana * 255).astype('uint8')
sha = cv2.medianBlur(sha,3)
plt.imshow(sha, cmap = 'gray')
plt.show()

circ = cv2.HoughCircles(sha, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

##
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(sana, cmap='gray')
if circ is not None:
    circ = np.round(circ[0]).astype("int")
    for (x, y, r) in circ:
        cur = patches.Circle((x, y), r, fill=False, color='red', linewidth=2)
        ax.add_patch(cur)
plt.show()





