#imports
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


from model.model import KeypointDetectionModel
from model.loss import PerceptualConfig, PerceptualLoss
from data_utils.data_utils import warp_image

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


parser.add_argument("--loss", type = str, required = True, help = "Loss function (vgg, l2)")

parser.add_argument("--kcnt", type = int, default = 10, required = False, help = "Number of keypoints")
parser.add_argument("--epoch", type = int, default = 100, required = False, help = "Max number of epochs")
parser.add_argument("--lr", type = float, default = 1e-3, required = False, help = "Learning rate")
parser.add_argument("--esl", type = int, default = 5, required = False, help = "Number of epochs with no improvement to early stop training")
parser.add_argument("--load", type = str, default = "", required = False, help = "Load a model to continue training")

args = parser.parse_args()

#settings

img_height = 128
img_width = 128
keypoint_cnt = args.kcnt
epoch_cnt = args.epoch
learning_rate = args.lr
early_stop_lim = args.esl
loss_name = args.loss
load_path = None if args.load == "" else args.load

if loss_name not in ["vgg", "l2"]:
    raise ValueError(f"Unknown loss function: {loss_name}")

print(f"\nSelected settings:\nKeypoint cnt: {keypoint_cnt}\nMax epoch cnt: {epoch_cnt}\nLearning rate: {learning_rate}\nEarly stopping limit: {early_stop_lim}\nLoss function: {loss_name}\nLoad pretrained: {load_path}\n")

#==========#

#dataset
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

celeba_train = CelebA(
    root = "../data",
    split = "train",
    transform = transform,
    download = False
)

celeba_val = CelebA(
    root = "../data",
    split = "valid",
    transform = transform,
    download = False
)

print("Datasets loaded")

#dataloader
if device.type == "cuda":
    train_loader = DataLoader(celeba_train, batch_size = 64, shuffle = True, num_workers = 4, pin_memory = True, pin_memory_device = "cuda", prefetch_factor = 2)
    val_loader = DataLoader(celeba_val, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True, pin_memory_device = "cuda", prefetch_factor = 2)
else:
    train_loader = DataLoader(celeba_train, batch_size = 64)
    val_loader = DataLoader(celeba_val, batch_size = 64)

print("Dataloaders created")

#model set up
model = KeypointDetectionModel(n_maps = keypoint_cnt, max_size = (img_height, img_width))
if load_path:
    model.load_state_dict(torch.load(f"./model_saves/{load_path}"))

#loss function
if loss_name == "vgg":
    cfg = PerceptualConfig(
        comp=['input','conv1_2','conv2_2','conv3_2','conv4_2','conv5_2'],
        init_ws=[100.0, 1.6, 2.3, 1.8, 2.8, 100.0],   # 6 entries for 6 layers
        l2=True,
        ema_momentum=0.99,
        net_file=None,                  # use torchvision pretrained
        input_feature_mode="raw"        # pixel term before VGG normalization
    )
    lossfn = PerceptualLoss(cfg).to(device)
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

    for source_images, _ in tqdm(train_loader, unit = "batch"):
        source_images = source_images.to(device)
        tar_images = warp_image(source_images)

        model_out, _, _ = model(source_images, tar_images)

        loss = lossfn(model_out, tar_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch: {epoch}\n")
    print(f"Loss (in sample): {total_loss / len(celeba_train)}")

    if loss_name == "vgg":
        for name, ema in lossfn.ema.items():
            print(f"{name}: {ema.value.item():.4f}", end="  ")
        print()

    total_loss = 0

    model.eval()
    with torch.no_grad():
        for source_images, _ in val_loader:
            source_images = source_images.to(device)
            tar_images = warp_image(source_images)

            model_out, _, _ = model(source_images, tar_images)

            loss = lossfn(model_out, tar_images)
            total_loss += loss.item()

    print(f"Loss (validation): {total_loss / len(celeba_val)} (best {best_val_loss / len(celeba_val)})\n\n===\n")

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
