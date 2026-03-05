#imports
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from SEGMENTATIONS.testing.data_utils.data_utils import DataProcessor
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

parser.add_argument("--path", type = str, default = "segunet.pth", required = False, help = "Model path (from ./SEGMENTATIONS/mod_seg/model_saves/)")
parser.add_argument("--tccnt", type = int, default = -1, required = False, help = "Number of test cases")

args = parser.parse_args()

#settings

img_width = 240
img_height = 192

model_path = args.path
tc_cnt = args.tccnt

#==========#

print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(img_width, img_height, tc_cnt)
dataset, dataloader = data_processor.create_ds()

#==========#

#model set up
model = SegUNet()

model.load_state_dict(torch.load(f"./SEGMENTATIONS/mod_seg/model_saves/{model_path}"))
print(f"\n==========\n\nModel loaded from ./SEGMENTATIONS/mod_seg/model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

with torch.no_grad():
    for idx, xrays in enumerate(tqdm(dataloader, unit = "batch")):
        xrays = xrays.to(device)
        pred = model(xrays)

        plt.figure(figsize = (6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(xrays[0][0].detach().cpu(), cmap = 'gray')

        plt.subplot(1, 2, 2)
        plt.imshow(pred[0][0].detach().cpu(), cmap = 'gray')

        plt.savefig(f"./SEGMENTATIONS/testing/test_results/img/{idx}.png")

print("\n==========\n\nDone\n")
