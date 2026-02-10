#imports
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import DataProcessor
#from data_utils.old_du import DataProcessor
from model.model import HeatmapsModel, measurements_to_coord, coord_to_measurements, abs_kp_to_coord, get_ab

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
parser.add_argument("--model", type = str, required = True, help = "Model type (hm)")

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

if model_name not in ["hm"]:
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
NOISE_AUG = (0, 0.5, 0.25)
if aug_data:
    train_set, train_loader = data_processor.create_ds("train", (-10, 10, 10), (0.5, 1, 0.25), (0.6, 1, 0.2), NOISE_AUG) #tuples are augmentation values
    val_set, val_loader = data_processor.create_ds("valid", (-10, 10, 10), (0.5, 1, 0.25), (0.6, 1, 0.2), NOISE_AUG) #tuples are augmentation values
else:
    train_set, train_loader = data_processor.create_ds("train", noise_rng = NOISE_AUG) #tuples are augmentation values
    val_set, val_loader = data_processor.create_ds("valid", noise_rng = NOISE_AUG) #tuples are augmentation values

#==========#

#model set up
if model_name == "hm":
    model = HeatmapsModel(img_width, img_height, 13) #include A

if load_path:
    model.load_state_dict(torch.load(f"./model_saves/{load_path}"))

#loss function
lossfn = lambda ypred, yvals, aug_scales : torch.mean(torch.sum(torch.norm(ypred - yvals, dim = 2), dim = 1) / aug_scales)

#optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#training
early_stop_cnt = 0
best_val_loss = float("inf")

print("\n==========\n\nTraining started\n\n")
for epoch in range(epoch_cnt):
    model.to(device)
    model.train()

    total_loss = 0
    
    for images, yvals, aug_scales in tqdm(train_loader, unit = "batch"):
        images = images.to(device)
        yvals = yvals.to(device)
        aug_scales = aug_scales.to(device)

        abs_coord = model(images)
        model_coord = abs_kp_to_coord(abs_coord)
        ab = get_ab(model_coord)
        real_coord = measurements_to_coord(yvals, ab, pix_per_mm, img_scale_factor)
        
        loss = lossfn(model_coord, real_coord, aug_scales)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]

    print(f'Epoch: {epoch}\n')
    print(f'Loss (in sample): {total_loss / len(train_set)}')

    total_loss = 0
    total_percent_err = torch.zeros(10).to(device)

    model.eval()
    with torch.no_grad():
        for images, yvals, aug_scales in val_loader:
            images = images.to(device)
            yvals = yvals.to(device)
            aug_scales = aug_scales.to(device)
            
            abs_coord = model(images)
            model_coord = abs_kp_to_coord(abs_coord)
            ab = get_ab(model_coord)
            real_coord = measurements_to_coord(yvals, ab, pix_per_mm, img_scale_factor)

            loss = lossfn(model_coord, real_coord, aug_scales)

            total_loss += loss.item() * images.shape[0]

            total_percent_err += torch.sum(torch.abs(yvals - coord_to_measurements(model_coord, pix_per_mm, img_scale_factor)) / yvals, dim = 0)

        print(f'Loss (validation): {total_loss / len(val_set)}\n')
        print(f'Percent error for each measurement (validation):')
        for i in total_percent_err:
            print(f'{(i.item() / len(val_set)):.4f}', end = ' ')
        print()

        print('\n===\n')

    if total_loss > best_val_loss:
        early_stop_cnt += 1
    else:
        early_stop_cnt = 0
        best_val_loss = total_loss
        torch.save(model.state_dict(), './model_saves/current_best.pth')

    if early_stop_cnt >= early_stop_lim:
        print('EARLY STOPPED')
        break

model.load_state_dict(torch.load("./model_saves/current_best.pth"))
torch.save(model.state_dict(), "./model_saves/final_model.pth")
