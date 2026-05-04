#imports
import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_utils.data_utils import DataProcessor
from model.model import SegUNet

#==========#

#device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using MPS')
else:
    device = torch.device('cpu')
    print('Using CPU')

#parser
parser = argparse.ArgumentParser(description='Segmentation experiment training')

parser.add_argument('--name',      type=str,   required=True,  help='Experiment name (used for model save filename)')
parser.add_argument('--loss',      type=str,   default='bce',  choices=['bce', 'dice', 'bce_dice', 'focal'], help='Loss function')
parser.add_argument('--aug',       action='store_true',        help='Enable geometric augmentation during training')
parser.add_argument('--scheduler', action='store_true',        help='Enable ReduceLROnPlateau LR scheduler')
parser.add_argument('--wd',        type=float, default=0.0,    help='Adam weight decay')
parser.add_argument('--lr',        type=float, default=1e-3,   help='Learning rate')
parser.add_argument('--esl',       type=int,   default=5,      help='Early stop patience (epochs with no improvement)')
parser.add_argument('--epoch',     type=int,   default=500,    help='Max epochs')
parser.add_argument('--seed',      type=int,   default=42,     help='Random seed')
parser.add_argument('--train_split', type=float, default=0.8)
parser.add_argument('--val_split',   type=float, default=0.1)
parser.add_argument('--train_bs',    type=int,   default=64)
parser.add_argument('--val_bs',      type=int,   default=64)
parser.add_argument('--test_bs',     type=int,   default=32)

args = parser.parse_args()

os.makedirs('model_saves', exist_ok=True)

#==========#

#loss functions

bce_fn = nn.BCELoss()

def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim=1)
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) + smooth)
    return 1.0 - dice_coeff.mean()

def focal_loss(pred, target, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    return ((1.0 - pt) ** gamma * bce).mean()

def compute_loss(pred, target):
    if args.loss == 'bce':
        return bce_fn(pred, target)
    elif args.loss == 'dice':
        return dice_loss(pred, target)
    elif args.loss == 'bce_dice':
        return 0.5 * bce_fn(pred, target) + 0.5 * dice_loss(pred, target)
    elif args.loss == 'focal':
        return focal_loss(pred, target)

#==========#

#settings
og_width, og_height = 512, 1024
tar_width, tar_height = 240, 192

print(f"\nExperiment: {args.name}")
print(f"Loss: {args.loss} | Aug: {args.aug} | Scheduler: {args.scheduler} | WD: {args.wd} | LR: {args.lr}")
print(f"ESL: {args.esl} | Max epochs: {args.epoch} | Seed: {args.seed}\n")

#==========#

#data
print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(
    (args.train_split, args.val_split),
    (args.train_bs, args.val_bs, args.test_bs),
    og_width, og_height, tar_width, tar_height, args.seed
)
train_set, train_loader = data_processor.create_ds('train', augment=args.aug)
val_set, val_loader = data_processor.create_ds('valid', augment=False)

#==========#

#model
torch.manual_seed(args.seed)
model = SegUNet()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = None
if args.scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

#==========#

#training
early_stop_cnt = 0
best_val_loss = float('inf')
save_path = f'model_saves/{args.name}_best.pth'

print("\n==========\n\nTraining started\n")

for epoch in range(args.epoch):
    model.to(device)
    model.train()
    train_loss = 0.0

    for xrays, segs in tqdm(train_loader, unit='batch', desc=f'Epoch {epoch:03d} train'):
        xrays = xrays.to(device)
        segs = segs.to(device)

        pred = model(xrays)
        loss = compute_loss(pred, segs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xrays.size(0)

    print(f'Epoch {epoch:03d} | Train loss: {train_loss / len(train_set):.6f}')

    #validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xrays, segs in val_loader:
            xrays = xrays.to(device)
            segs = segs.to(device)
            pred = model(xrays)
            val_loss += compute_loss(pred, segs).item() * xrays.size(0)

    print(f'Epoch {epoch:03d} | Val loss:   {val_loss / len(val_set):.6f}')

    if scheduler:
        scheduler.step(val_loss / len(val_set))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_cnt = 0
        torch.save(model.state_dict(), save_path)
        print(f'  -> New best, saved to {save_path}')
    else:
        early_stop_cnt += 1
        if early_stop_cnt >= args.esl:
            print(f'\nEARLY STOPPED at epoch {epoch}')
            break

    print()

print(f'\nTraining complete.')
print(f'Best val loss: {best_val_loss / len(val_set):.6f}')
print(f'Model saved to {save_path}')
