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
    test_loader = DataLoader(celeba_test, batch_size = 64)

print("Dataloaders created")

#model set up
model = KeypointDetectionModel(n_maps = keypoint_cnt, max_size = (img_height, img_width))
model.load_state_dict(torch.load(f"./model_saves/{model_path}"))
print(f"Model loaded from./model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

with torch.no_grad():
    for idx, (source_images, _) in enumerate(tqdm(test_loader, unit="batch")):
        source_images = source_images.to(device)
        tar_images = warp_image(source_images)  # create target/geometry view

        # IMM_L2 forward: returns (y_pred, mu_norm, g_small)
        y_pred, mu_norm, _ = model(source_images, tar_images)

        # ---- keypoints: normalized [-1,1] (y, x) -> pixel (x_px, y_px)
        H, W = img_height, img_width
        x_px = ((mu_norm[0, :, 1] + 1.0) * 0.5 * W).detach().cpu().numpy()
        y_px = ((mu_norm[0, :, 0] + 1.0) * 0.5 * H).detach().cpu().numpy()

        # tensors -> HxWxC numpy for imshow
        src_np = source_images[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
        tgt_np = tar_images[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
        rec_np = y_pred[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()

        plt.figure(figsize=(12, 3))

        plt.subplot(1, 4, 1)
        plt.imshow(src_np)
        plt.title("Source")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(tgt_np)
        plt.title("Target")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(rec_np)
        plt.title("Reconstruction")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(tgt_np)
        # scatter expects (x, y)
        plt.scatter(x_px, y_px, s=20, marker="o")
        plt.title("Target + keypoints")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"./test_results/{idx}.png", dpi=150)
        plt.close()

