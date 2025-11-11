#imports
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import DataProcessor, warp_image
from model.model import KeypointDetectionModel

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

parser.add_argument("--kcnt", type = int, default = 30, required = False, help = "Number of keypoints in trained model")
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

keypoint_cnt = args.kcnt
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
print(f"Image directory: {img_dir}\nSelected dataset: {test_set_name}\nKeypoint cnt: {keypoint_cnt}\n")
print(f"Torch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(img_dir, (train_split, val_split), batch_sizes, img_width, img_height, seed)
dataset, dataloader = data_processor.create_ds(test_set_name)

#==========#

#model set up
model = KeypointDetectionModel(keypoint_cnt, img_height, img_width)
model.load_state_dict(torch.load(f"./model_saves/{model_path}"))
print(f"Model loaded from./model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

with torch.no_grad():
    for idx, source_images in enumerate(tqdm(dataloader, unit = "batch")):
        source_images = source_images.to(device)
        tar_images = warp_image(source_images)

        model_out, src_keypoints, tar_keypoints = model(source_images, tar_images, True)

        plt.figure(figsize = (15, 3))

        plt.subplot(1, 5, 1)
        plt.imshow(source_images[0, 0].cpu().numpy(), cmap = "gray")

        plt.subplot(1, 5, 2)
        plt.imshow(tar_images[0, 0].cpu().numpy(), cmap = "gray")

        plt.subplot(1, 5, 3)
        plt.imshow(model_out[0, 0].cpu().numpy(), cmap = "gray")

        plt.subplot(1, 5, 4)
        plt.imshow(source_images[0, 0].cpu().numpy(), cmap = "gray")
        plt.scatter(src_keypoints[0, :, 0].cpu().numpy(), src_keypoints[0, :, 1].cpu().numpy(), c = "r")

        plt.subplot(1, 5, 5)
        plt.imshow(tar_images[0, 0].cpu().numpy(), cmap = "gray")
        plt.scatter(tar_keypoints[0, :, 0].cpu().numpy(), tar_keypoints[0, :, 1].cpu().numpy(), c = "r")

        plt.savefig(f"./test_results/{idx}.png")

print("\n==========\n\nDone")
