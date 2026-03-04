#imports
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


from model.model import KeypointDetectionModel
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

parser.add_argument("--ds", type = str, required = True, help = "Dataset to test on (train, valid, test)")

parser.add_argument("--kcnt", type = int, default = 10, required = False, help = "Number of keypoints in trained model")
parser.add_argument("--path", type = str, default = "current_best.pth", required = False, help = "Model path (from ./model_saves/)")

args = parser.parse_args()

#settings

img_height = 128
img_width = 128
test_set_name = args.ds.lower()
keypoint_cnt = args.kcnt
model_path = args.path

#checks
if test_set_name not in ["train", "valid", "test"]:
    raise ValueError(f"Unknown dataset name \"{test_set_name}\"")

print(f"\nSelected settings:\nKeypoint cnt: {keypoint_cnt}\nSelected dataset: {test_set_name}\n")

#==========#

#dataset
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

celeba_test = CelebA(
    root = "../data",
    split = test_set_name,
    transform = transform,
    download = False
)

print("Datasets loaded")

#dataloader

if device.type == "cuda":
    test_loader = DataLoader(celeba_test, batch_size = 64, shuffle = True, num_workers = 4, pin_memory = True, pin_memory_device = "cuda", prefetch_factor = 2)
else:
    tese_loader = DataLoader(celeba_test, batch_size = 64)

print("Dataloaders created")

#model set up
model = KeypointDetectionModel(keypoint_cnt, img_height, img_width, [3, 32, 64, 128, 256])
model.load_state_dict(torch.load(f"./model_saves/{model_path}"))
print(f"Model loaded from./model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

with torch.no_grad():
    for idx, (source_images, _) in enumerate(tqdm(test_loader, unit = "batch")):
        source_images = source_images.to(device)
        tar_images = warp_image(source_images)

        model_out, tar_keypoints = model(source_images, tar_images)

        plt.figure(figsize = (12, 3))

        plt.subplot(1, 4, 1)
        plt.imshow(source_images[0, :, :].permute(1, 2, 0).cpu().numpy())

        plt.subplot(1, 4, 2)
        plt.imshow(tar_images[0, :, :].permute(1, 2, 0).cpu().numpy())

        plt.subplot(1, 4, 3)
        plt.imshow(model_out[0, :, :].permute(1, 2, 0).cpu().numpy())

        plt.subplot(1, 4, 4)
        plt.imshow(tar_images[0, :, :].permute(1, 2, 0).cpu().numpy())
        plt.scatter(tar_keypoints[0, :, 0].cpu().numpy(), tar_keypoints[0, :, 1].cpu().numpy(), c = "r")

        plt.savefig(f"./test_results/{idx}.png")

print("\n==========\n\nDone")
