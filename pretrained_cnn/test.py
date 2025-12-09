#imports
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv

from data_utils.data_utils import DataProcessor
from femur_model.model import FemurModel, model_to_coord, measurements_to_coord, coord_to_measurements

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
sq_dim = 224
img_scale_factor = sq_dim / 2400
img_width = int(2400 * img_scale_factor)
img_height = int(1920 * img_scale_factor)

measurement_file = args.mdata
img_dir = args.idata
test_set_name = args.ds.lower()

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
if batch_norm_setting not in ["none", "before", "after"]:
    raise ValueError(f"Unknown batch norm setting: {batch_norm_setting}")

print("\nSelected settings:\n")
print(f"Measurement file: {measurement_file}\nImage directory: {img_dir}\nSelected dataset: {test_set_name}\n")
print(f"Batch norm setting: {batch_norm_setting}\nTorch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(measurement_file, img_dir, (train_split, val_split), batch_sizes, img_width, img_height, sq_dim, seed)
dataset, dataloader = data_processor.create_ds(test_set_name)

#==========#

#model set up
model = FemurModel()

model.load_state_dict(torch.load(f"./model_saves/{model_path}"))
print(f"\n==========\n\nModel loaded from./model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

all_errs = None
total_percent_err = torch.zeros(10).to(device)

with torch.no_grad():
    for idx, (images, yvals, aug_scales) in enumerate(tqdm(dataloader, unit = "batch")):
        images = images.to(device)
        yvals = yvals.to(device)
        aug_scales = aug_scales.to(device)

        model_out = model(images)
        model_coord, ab = model_to_coord(model_out)
        real_coord = measurements_to_coord(yvals, ab, pix_per_mm, img_scale_factor)

        errs = torch.abs(yvals - coord_to_measurements(model_coord, pix_per_mm, img_scale_factor))
        if all_errs is None:
            all_errs = errs
        else:
            all_errs = torch.cat((all_errs, errs), dim = 0)
        total_percent_err += torch.sum(torch.abs(yvals - coord_to_measurements(model_coord, pix_per_mm, img_scale_factor)) / yvals, dim = 0)

        plt.figure(figsize = (6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(images[0][0].detach().cpu(), cmap = 'gray')

        plt.subplot(1, 2, 2)
        plt.scatter(model_coord[0, :, 0].detach().cpu(), model_coord[0, :, 1].detach().cpu(), c = 'r')
        for i, label in enumerate(['D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'C']):
            curx, cury = model_coord[0, i]
            curx = curx.detach().cpu()
            cury = cury.detach().cpu()
            plt.text(curx + 0.5, cury + 0.5, label)
        plt.scatter(real_coord[0, :, 0].detach().cpu(), real_coord[0, :, 1].detach().cpu(), c = 'b')
        for i, label in enumerate(['D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'C']):
            curx, cury = real_coord[0, i]
            curx = curx.detach().cpu()
            cury = cury.detach().cpu()
            plt.text(curx + 0.5, cury + 0.5, label)
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')

        plt.savefig(f"./test_results/{idx}.png")

print("\n==========\n\nDone\n")

print(f'Percent error for each measurement:')
for i in total_percent_err:
    print(f'{(i.item() / len(dataset)):.4f}', end = ' ')
print()
print(all_errs.mean(dim = 0))
print(all_errs.std(dim = 0, unbiased = True))
