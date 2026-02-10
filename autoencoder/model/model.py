#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#encoder
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
        resid = []

        x = self.conv1(x)
        resid.append(x)
        
        x = self.conv2(x)
        resid.append(x)

        x = self.conv3(x)
        resid.append(x)

        x = self.conv4(x)
        resid.append(x)

        return resid

#decoder
class Decoder(nn.Module):
    def __init__(self, has_resid = False, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.has_resid = has_resid

        first_ch_cnt = channel_cnt
        if has_resid:
            first_ch_cnt[:-1] = [2 * i for i in first_ch_cnt[:-1]]

        self.conv1 = nn.Sequential(
            nn.Conv2d(first_ch_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1), #24, 30
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[4], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(first_ch_cnt[3], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1), #48, 60
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[3], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(first_ch_cnt[2], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1), #96, 120
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[2], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(first_ch_cnt[1], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1), #192, 240
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[1], channel_cnt[0], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[0]),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(first_ch_cnt[0], channel_cnt[0], kernel_size = 3, stride = 1, padding = 1), #192, 240
            nn.Sigmoid()
        )

    def forward(self, x, resid = None):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)

        if self.has_resid:
            x = torch.cat([x, resid[3]], dim = 1)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)

        if self.has_resid:
            x = torch.cat([x, resid[2]], dim = 1)
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)

        if self.has_resid:
            x = torch.cat([x, resid[1]], dim = 1)
        x = self.conv4(x)

        if self.has_resid:
            x = torch.cat([x, resid[0]], dim = 1)
        x = self.conv5(x)

        return x

#autoencoder
class Autoencoder(nn.Module):
    def __init__(self, has_resid = False, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.enc = Encoder(channel_cnt)
        self.dec = Decoder(has_resid, channel_cnt)

    def forward(self, x):
        resid = self.enc(x)
        return self.dec(resid[-1], resid)
