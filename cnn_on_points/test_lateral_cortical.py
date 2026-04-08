#imports
import argparse
import pandas as pd
from tqdm import tqdm

import torch

from data_utils.data_utils import DataProcessor
from model.model import SimpleCNNModel, AlexNet, model_to_coord, measurements_to_coord, coord_to_measurements

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

parser = argparse.ArgumentParser(description = "Lateral cortical thickness diagnostic")

parser.add_argument("--mdata", type = str, required = True, help = "Path to measurement data file (from ../data/measurements)")
parser.add_argument("--idata", type = str, required = True, help = "Path to image data directory (from ../data)")
parser.add_argument("--ds", type = str, required = True, help = "Dataset to test on (train, valid, test)")
parser.add_argument("--model", type = str, required = True, help = "Model type (basic, alex)")

parser.add_argument("--path", type = str, default = "current_best.pth", required = False, help = "Model path (from ./model_saves/)")
parser.add_argument("--bn", type = str, default = "none", required = False, help = "Batch norm setting (none, before, after)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")

args = parser.parse_args()

#settings

pix_per_mm = 2400 / 408
img_scale_factor = 0.2
img_width = int(2400 * img_scale_factor)
img_height = int(1920 * img_scale_factor)

measurement_file = args.mdata
img_dir = args.idata
test_set_name = args.ds.lower()
model_name = args.model

model_path = args.path
batch_norm_setting = args.bn
seed = args.seed
train_split = args.train_split
val_split = args.val_split
batch_sizes = (1, 1, 1)

#==========#

print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(measurement_file, img_dir, (train_split, val_split), batch_sizes, img_width, img_height, seed)
dataset, dataloader = data_processor.create_ds(test_set_name)

#==========#

#model set up
if model_name == "basic":
    model = SimpleCNNModel(img_width, img_height, batch_norm_setting)
elif model_name == "alex":
    model = AlexNet(img_width, img_height, batch_norm_setting)

model.load_state_dict(torch.load(f"./model_saves/{model_path}"))

model.to(device)
model.eval()

#lateral cortical = measurement index 1 = Z_x - Y_x (keypoint indices 7 and 6)

rows = []

with torch.no_grad():
    for idx, (images, yvals, aug_scales) in enumerate(tqdm(dataloader, unit = "batch")):
        images = images.to(device)
        yvals = yvals.to(device)

        model_out = model(images)
        model_coord, ab = model_to_coord(model_out)

        #predicted keypoints Y and Z (in relative pixel coords)
        pred_Y_x = model_coord[0, 6, 0].item()
        pred_Z_x = model_coord[0, 7, 0].item()
        pred_px = pred_Z_x - pred_Y_x

        #predicted mm
        pred_mm = pred_px / (pix_per_mm * img_scale_factor)

        #ground truth keypoints
        real_coord = measurements_to_coord(yvals, ab, pix_per_mm, img_scale_factor)
        real_Y_x = real_coord[0, 6, 0].item()
        real_Z_x = real_coord[0, 7, 0].item()
        real_px = real_Z_x - real_Y_x

        #ground truth mm
        real_mm = yvals[0, 1].item()

        rows.append({
            "idx": idx,
            "pred_Y_x (px)": round(pred_Y_x, 3),
            "pred_Z_x (px)": round(pred_Z_x, 3),
            "pred_lat_cortical (px)": round(pred_px, 3),
            "pred_lat_cortical (mm)": round(pred_mm, 3),
            "real_Y_x (px)": round(real_Y_x, 3),
            "real_Z_x (px)": round(real_Z_x, 3),
            "real_lat_cortical (px)": round(real_px, 3),
            "real_lat_cortical (mm)": round(real_mm, 3),
            "error (px)": round(pred_px - real_px, 3),
            "error (mm)": round(pred_mm - real_mm, 3),
            "pct_error": round(abs(pred_mm - real_mm) / real_mm * 100, 2),
        })

df = pd.DataFrame(rows)
df.to_csv("./test_results/lateral_cortical_diagnostic.csv", index = False)

print(f"\nSaved to ./test_results/lateral_cortical_diagnostic.csv ({len(rows)} rows)")
print(f"\nSummary:")
print(f"  Mean real (mm):   {df['real_lat_cortical (mm)'].mean():.3f}")
print(f"  Mean pred (mm):   {df['pred_lat_cortical (mm)'].mean():.3f}")
print(f"  Mean real (px):   {df['real_lat_cortical (px)'].mean():.3f}")
print(f"  Mean pred (px):   {df['pred_lat_cortical (px)'].mean():.3f}")
print(f"  Mean abs err (mm): {df['error (mm)'].abs().mean():.3f}")
print(f"  Mean abs err (px): {df['error (px)'].abs().mean():.3f}")
print(f"  Mean pct error:   {df['pct_error'].mean():.2f}%")
