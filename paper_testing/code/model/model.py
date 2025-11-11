#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#heatmaps (reduced H, W) to keypoints (normalized)
def heatmaps_to_keypoints(heatmaps):
    device = heatmaps.device

    B, K, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, K, -1)
    softmax_heatmaps = F.softmax(heatmaps, dim=-1)
    softmax_heatmaps = softmax_heatmaps.view(B, K, H, W)

    xs = torch.linspace(-1, 1, W, device = device)
    ys = torch.linspace(-1, 1, H, device = device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing = 'ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    expected_x = (softmax_heatmaps * grid_x).sum(dim= [-2, -1])
    expected_y = (softmax_heatmaps * grid_y).sum(dim= [-2, -1])

    return torch.stack([expected_x, expected_y], dim = 2)

#keypoints (normalized) to gaussian (normalized)
def keypoints_to_gaussian(keypoints, height, width, sigma = 0.1):
    device = keypoints.device

    B, K, _ = keypoints.shape

    xs = torch.linspace(-1, 1, width, device = device)
    ys = torch.linspace(-1, 1, height, device = device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing = 'ij')
    grid = torch.stack([grid_x, grid_y], dim = -1).view(1, 1, height, width, 2)

    keypoints = keypoints.view(B, K, 1, 1, 2)
    sq_dist = torch.sum((grid - keypoints) ** 2, dim = -1)

    return torch.exp(-sq_dist / (2 * sigma ** 2))

#get keypoints in original pixel space from normalized keypoints
def get_keypoints(keypoints, img_height, img_width):
    keypoints = keypoints.clone()
    keypoints = (keypoints + 1) / 2
    keypoints[:, :, 0] = keypoints[:, :, 0] * img_width
    keypoints[:, :, 1] = keypoints[:, :, 1] * img_height
    return keypoints

#encoder for keypoint detection
class Encoder(nn.Module):
    def __init__(self, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_cnt[0], channel_cnt[1], kernel_size = 7, stride = 1, padding = 3), #192, 240
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[1], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1), #192, 240
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_cnt[1], channel_cnt[2], kernel_size = 3, stride = 2, padding = 1), #96, 120
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[2], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1), #96, 120
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_cnt[2], channel_cnt[3], kernel_size = 3, stride = 2, padding = 1), #48, 60
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[3], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1), #48, 60
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_cnt[3], channel_cnt[4], kernel_size = 3, stride = 2, padding = 1), #24, 30
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1), #24, 30
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU()
        )

    def forward(self, x):
        skips = []

        x = self.conv1(x)
        skips.append(x) #C1, 192, 240

        x = self.conv2(x)
        skips.append(x) #C2, 96, 120

        x = self.conv3(x)
        skips.append(x) #C3, 48, 60

        x = self.conv4(x)
        skips.append(x) #C4, 24, 30

        return skips

#image encoder for keypoint detection (embedding of source image)
class ImageEncoder(nn.Module):
    def __init__(self, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()
        self.encoder = Encoder(channel_cnt)

    def forward(self, x):
        enc = self.encoder(x)
        return enc[-1]

#location encoder for keypoint detection (points on target image)
class LocationEncoder(nn.Module):
    def __init__(self, keypoint_cnt, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()
        self.encoder = Encoder(channel_cnt)
        self.conv = nn.Conv2d(channel_cnt[-1], keypoint_cnt, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        enc = self.encoder(x)
        x = enc[-1]
        x = self.conv(x)
        return x

#keypoint detection decoder
class Decoder(nn.Module):
    def __init__(self, keypoint_cnt, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(keypoint_cnt + channel_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1), #24, 30
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_cnt[4], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1), #48, 60
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[3], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_cnt[3], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1), #96, 120
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[2], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_cnt[2], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1), #192, 240
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[1], channel_cnt[0], kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)

        x = self.conv4(x)

        return x

#full model
class KeypointDetectionModel(nn.Module):
    def __init__(self, keypoint_cnt, img_height, img_width, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width

        self.img_encoder = ImageEncoder(channel_cnt)
        self.location_encoder = LocationEncoder(keypoint_cnt, channel_cnt)
        self.decoder = Decoder(keypoint_cnt, channel_cnt)

    def forward(self, source_img, tar_img):
        src_enc = self.img_encoder(source_img)
        tar_heatmaps = self.location_encoder(tar_img)

        tar_keypoints = heatmaps_to_keypoints(tar_heatmaps)
        tar_gaussians = keypoints_to_gaussian(tar_keypoints, tar_heatmaps.shape[-2], tar_heatmaps.shape[-1])

        joint = torch.cat([src_enc, tar_gaussians], dim = 1)

        recon = self.decoder(joint)

        tar_keypoints = get_keypoints(tar_keypoints, self.img_height, self.img_width)

        return recon, tar_keypoints
