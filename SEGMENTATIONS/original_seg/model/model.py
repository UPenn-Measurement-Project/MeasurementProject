#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#encoder for keypoint detection
class Encoder(nn.Module):
    def __init__(self, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_cnt[0], channel_cnt[1], kernel_size = 7, stride = 1, padding = 3),
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[1], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_cnt[1], channel_cnt[2], kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[2], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_cnt[2], channel_cnt[3], kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[3], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_cnt[3], channel_cnt[4], kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU()
        )

    def forward(self, x):
        skips = []

        skips.append(x)

        x = self.conv1(x)
        skips.append(x)

        x = self.conv2(x)
        skips.append(x)

        x = self.conv3(x)
        skips.append(x)

        x = self.conv4(x)
        skips.append(x)

        return skips

#keypoint detection decoder
class Decoder(nn.Module):
    def __init__(self, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[4], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * channel_cnt[3], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[3], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * channel_cnt[2], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[2], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * channel_cnt[1], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[1], channel_cnt[0], kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, skips):
        x = skips[4]

        x = self.conv1(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = torch.cat((x, skips[3]), dim = 1)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = torch.cat((x, skips[2]), dim = 1)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = torch.cat((x, skips[1]), dim = 1)

        x = self.conv4(x)

        return x

#full model
class SegUNet(nn.Module):
    def __init__(self, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.encoder = Encoder(channel_cnt)
        self.decoder = Decoder(channel_cnt)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
