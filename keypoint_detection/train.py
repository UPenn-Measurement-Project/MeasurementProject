#imports
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import DataProcessor, warp_image
from model.model import KeypointDetectionModel
from model.vgg_loss import VggPerceptualConfig, VggPerceptualLoss

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

parser.add_argument("--kcnt", type = int, default = 30, required = False, help = "Number of keypoints")
parser.add_argument("--epoch", type = int, default = 100, required = False, help = "Max number of epochs")
parser.add_argument("--lr", type = float, default = 1e-3, required = False, help = "Learning rate")
parser.add_argument("--esl", type = int, default = 5, required = False, help = "Number of epochs with no improvement to early stop training")
parser.add_argument("--loss", type = str, default = "vgg", required = False, help = "Loss function (vgg, l2)")
parser.add_argument("--load", type = str, default = "", required = False, help = "Load a model to continue training (./model_saves)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 64, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 32, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 32, required = False, help = "Testiing set batch size")

args = parser.parse_args()

#settings

pix_per_mm = 2400 / 408
img_scale_factor = 0.1
img_width = int(2400 * img_scale_factor)
img_height = int(1920 * img_scale_factor)

img_dir = args.idata

keypoint_cnt = args.kcnt
epoch_cnt = args.epoch
learning_rate = args.lr
early_stop_lim = args.esl
loss_name = args.loss
load_path = None if args.load == "" else args.load
seed = args.seed
train_split = args.train_split
val_split = args.val_split
train_batch_size = args.train_bs
val_batch_size = args.val_bs
test_batch_size = args.test_bs
batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

if loss_name not in ["vgg", "l2"]:
    raise ValueError(f"Unknown loss function: {loss_name}")

print("\nSelected settings:\n")
print(f"Image directory: {img_dir}\n")
print(f"Keypoint cnt: {keypoint_cnt}\nMax epoch cnt: {epoch_cnt}\nLearning rate: {learning_rate}\nEarly stopping limit: {early_stop_lim}\nLoss function: {loss_name}\nLoad pretrained: {load_path}\nTorch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

#data
print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(img_dir, (train_split, val_split), batch_sizes, img_width, img_height, seed)
train_set, train_loader = data_processor.create_ds("train")
val_set, val_loader = data_processor.create_ds("valid")

#==========#

#model set up
model = KeypointDetectionModel(keypoint_cnt, img_height, img_width)
if load_path:
    model.load_state_dict(torch.load(f"./model_saves/{load_path}"))

#loss function
if loss_name == "vgg":
    cfg = VggPerceptualConfig(
        comp=['input','conv1_2','conv2_2','conv3_2','conv4_2','conv5_2'],
        init_ws=[100.0, 1.6, 2.3, 1.8, 2.8, 100.0],   # 6 entries for 6 layers
        l2=True,
        ema_momentum=0.99,
        net_file=None,                  # use torchvision pretrained
        input_feature_mode="raw"        # pixel term before VGG normalization
    )
    lossfn = VggPerceptualLoss(cfg).to(device)
elif loss_name == "l2":
    default_mse = nn.MSELoss()
    lossfn = lambda ypred, yreal: 1000 * default_mse(ypred, yreal)

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

    for source_images in tqdm(train_loader, unit = "batch"):
        source_images = source_images.to(device)
        tar_images = warp_image(source_images)

        model_out, _ = model(source_images, tar_images)

        loss = lossfn(model_out, tar_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch: {epoch}\n")
    print(f"Loss (in sample): {total_loss / len(train_set)}")

    if loss_name == "vgg":
        for name, ema in lossfn.ema.items():
            print(f"{name}: {ema.value.item():.4f}", end="  ")
        print()

    total_loss = 0

    model.eval()
    with torch.no_grad():
        for source_images in val_loader:
            source_images = source_images.to(device)
            tar_images = warp_image(source_images)

            model_out, _ = model(source_images, tar_images)

            loss = lossfn(model_out, tar_images)
            total_loss += loss.item()

    print(f"Loss (validation): {total_loss / len(val_set)} (best {best_val_loss / len(val_set)})\n\n===\n")

    if total_loss > best_val_loss:
        early_stop_cnt += 1
    else:
        early_stop_cnt = 0
        best_val_loss = total_loss
        torch.save(model.state_dict(), "./model_saves/current_best.pth")

    if early_stop_cnt >= early_stop_lim:
        print("EARLY STOPPED")
        break

model.load_state_dict(torch.load("./model_saves/current_best.pth"))
torch.save(model.state_dict(), "./model_saves/final_model.pth")
