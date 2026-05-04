#imports
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
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

parser.add_argument("--epoch", type = int, default = 100, required = False, help = "Max number of epochs")
parser.add_argument("--lr", type = float, default = 1e-3, required = False, help = "Learning rate")
parser.add_argument("--esl", type = int, default = 5, required = False, help = "Number of epochs with no improvement to early stop training")

parser.add_argument("--load", type = str, default = "", required = False, help = "Load a model to continue training (./model_saves)")

parser.add_argument("--loss", type = str, default = "bce", choices = ["bce", "dice", "bce_dice"], help = "Loss function")
parser.add_argument("--aug", action = "store_true", help = "Enable geometric augmentation (rotation, scale, translation)")
parser.add_argument("--iaug", action = "store_true", help = "Enable intensity augmentation (noise, contrast, gamma)")
parser.add_argument("--scheduler", action = "store_true", help = "Enable ReduceLROnPlateau LR scheduler")

parser.add_argument("--noise", action = "store_true", help = "Fill image will noise")
parser.add_argument("--mirror", action = "store_true", help = "Mirror femur to left")

parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 64, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 64, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 32, required = False, help = "Testiing set batch size")

args = parser.parse_args()

#settings

og_width = 512
og_height = 1024
tar_width = 240
tar_height = 192

epoch_cnt = args.epoch
learning_rate = args.lr
early_stop_lim = args.esl
load_path = None if args.load == "" else args.load
loss_fn_name = args.loss
use_aug = args.aug
use_iaug = args.iaug
use_scheduler = args.scheduler
fill_noise = args.noise
mirror = args.mirror
seed = args.seed
train_split = args.train_split
val_split = args.val_split
train_batch_size = args.train_bs
val_batch_size = args.val_bs
test_batch_size = args.test_bs
batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

print("\nSelected settings:\n")
print(f"Max epoch cnt: {epoch_cnt}\nLearning rate: {learning_rate}\nEarly stopping limit: {early_stop_lim}\nLoad pretrained: {load_path}\nTorch seed: {seed}")
print(f"Loss: {loss_fn_name} | Aug: {use_aug} | IAug: {use_iaug} | Scheduler: {use_scheduler}")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

#data
print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor((train_split, val_split), batch_sizes, og_width, og_height, tar_width, tar_height, seed, fill_noise, mirror)
train_set, train_loader = data_processor.create_ds("train", augment = use_aug, iaugment = use_iaug)
val_set, val_loader = data_processor.create_ds("valid")

#==========#

#loss functions

bce_fn = nn.BCELoss()

def dice_loss(pred, target, smooth = 1e-6):
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim = 1)
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum(dim = 1) + tgt_flat.sum(dim = 1) + smooth)
    return 1.0 - dice_coeff.mean()

def compute_loss(pred, target):
    if loss_fn_name == "bce":
        return bce_fn(pred, target)
    elif loss_fn_name == "dice":
        return dice_loss(pred, target)
    elif loss_fn_name == "bce_dice":
        return 0.5 * bce_fn(pred, target) + 0.5 * dice_loss(pred, target)

#model set up
model = SegUNet()

if load_path:
    model.load_state_dict(torch.load(f"SEGMENTATIONS/mod_seg/model_saves/{load_path}"))

#optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#scheduler
scheduler = None
if use_scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", patience = 2, factor = 0.5)

#training
early_stop_cnt = 0
best_val_loss = float("inf")

print("\n==========\n\nTraining started\n\n")
for epoch in range(epoch_cnt):
    model.to(device)
    model.train()

    total_loss = 0

    for xrays, segs in tqdm(train_loader, unit = "batch"):
        xrays = xrays.to(device)
        segs = segs.to(device)

        pred = model(xrays)

        loss = compute_loss(pred, segs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xrays.shape[0]

    print(f'Epoch: {epoch}\n')
    print(f'Loss (in sample): {total_loss / len(train_set)}')

    total_loss = 0

    model.eval()
    with torch.no_grad():
        for xrays, segs in tqdm(val_loader, unit = "batch"):
            xrays = xrays.to(device)
            segs = segs.to(device)

            pred = model(xrays)

            loss = compute_loss(pred, segs)

            total_loss += loss.item() * xrays.shape[0]

        print(f'Loss (validation): {total_loss / len(val_set)}\n')
        print('\n===\n')

    if scheduler:
        scheduler.step(total_loss / len(val_set))

    if total_loss > best_val_loss:
        early_stop_cnt += 1
    else:
        early_stop_cnt = 0
        best_val_loss = total_loss
        torch.save(model.state_dict(), 'SEGMENTATIONS/mod_seg/model_saves/current_best.pth')

    if early_stop_cnt >= early_stop_lim:
        print('EARLY STOPPED')
        break

model.load_state_dict(torch.load("SEGMENTATIONS/mod_seg/model_saves/current_best.pth"))
torch.save(model.state_dict(), "SEGMENTATIONS/mod_seg/model_saves/final_model.pth")
