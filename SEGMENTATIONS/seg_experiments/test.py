#imports
import argparse
import csv
import os
from tqdm import tqdm

import torch

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
parser = argparse.ArgumentParser(description='Segmentation experiment evaluation')

parser.add_argument('--name',        type=str,   required=True,  help='Experiment name (loads model_saves/{name}_best.pth)')
parser.add_argument('--ds',          type=str,   default='test', choices=['train', 'valid', 'test'])
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--train_split', type=float, default=0.8)
parser.add_argument('--val_split',   type=float, default=0.1)
parser.add_argument('--bs',          type=int,   default=32,     help='Batch size')

args = parser.parse_args()

os.makedirs('results', exist_ok=True)

#==========#

og_width, og_height = 512, 1024
tar_width, tar_height = 240, 192

print(f"\nEvaluating: {args.name} on {args.ds} set\n")

#data
print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(
    (args.train_split, args.val_split),
    (args.bs, args.bs, args.bs),
    og_width, og_height, tar_width, tar_height, args.seed
)
dataset, dataloader = data_processor.create_ds(args.ds, augment=False)

#==========#

#model
model_path = f'model_saves/{args.name}_best.pth'
model = SegUNet()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
model.eval()
print(f"\nLoaded model from {model_path}")

#==========#

#evaluation
print("\n==========\n\nEvaluating...\n")

dice_scores = []

with torch.no_grad():
    for xrays, segs in tqdm(dataloader, unit='batch'):
        xrays = xrays.to(device)
        segs = segs.to(device)

        pred = model(xrays)
        pred_bin = (pred > 0.5).float()

        intersection = (pred_bin * segs).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection) / (pred_bin.sum(dim=(1, 2, 3)) + segs.sum(dim=(1, 2, 3)) + 1e-8)
        dice_scores.extend(dice.cpu().tolist())

mean_dice = sum(dice_scores) / len(dice_scores)
min_dice  = min(dice_scores)
max_dice  = max(dice_scores)
count     = len(dice_scores)

print(f"\n==========\n\nResults — {args.name} ({args.ds} set):")
print(f"  Mean Dice: {mean_dice:.4f}")
print(f"  Min Dice:  {min_dice:.4f}")
print(f"  Max Dice:  {max_dice:.4f}")
print(f"  Count:     {count}")

#save to CSV
results_file = 'results/results.csv'
write_header = not os.path.exists(results_file)

with open(results_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(['experiment', 'dataset', 'mean_dice', 'min_dice', 'max_dice', 'count'])
    writer.writerow([args.name, args.ds, f'{mean_dice:.4f}', f'{min_dice:.4f}', f'{max_dice:.4f}', count])

print(f"\nAppended to {results_file}\n")
