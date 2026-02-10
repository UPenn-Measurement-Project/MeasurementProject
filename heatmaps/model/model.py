#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#rotate points (around origin)
def rot_coord(coord):
    #W: 4
    #Z: 7
    
    B = coord.shape[0]
    K = coord.shape[1]

    vec = coord[:, 7, :] - coord[:, 4, :]
    ang = torch.atan2(vec[:, 1], vec[:, 0])
    sinang = torch.sin(ang)
    cosang = torch.cos(ang)
    rotmat = torch.stack([
        torch.stack([cosang, -sinang], dim = -1),
        torch.stack([sinang, cosang], dim = -1)
    ], dim = -1).view(B, 1, 2, 2)
    res = rotmat @ coord.view(B, K, 2, 1)

    return res.view(B, K, 2)

#translate and rotate absolute coordinates to relative coordinates
def abs_kp_to_coord(abs_kp):
    #abs_kp shape: (batch_size, 13, 2)
    #refer to powerpoint: https://docs.google.com/presentation/d/1FPTIKSnscUQzuTzK1f5JnABffidgGuaxPcGEeSBpgRY/edit?slide=id.g362ec749174_0_26#slide=id.g362ec749174_0_26

    #translate
    trans_kp = abs_kp[:, 1:, :] - abs_kp[:, 0:1, :]

    #rotate
    trans_kp = rot_coord(trans_kp)

    return trans_kp

#get ab from model relative coord
def get_ab(coord):
    #B: 11
    ab = torch.norm(coord[:, 11], dim = 1)
    return ab

#turn measurements into points
def measurements_to_coord(measurements, ab, pix_per_mm, img_scale_factor, DEBUG_MODE = False):
    #units: mm -> pixels
    #measurements shape: (batch_size, 10)
    #ab shape: (batch_size,)
    #refer to powerpoint: https://docs.google.com/presentation/d/1FPTIKSnscUQzuTzK1f5JnABffidgGuaxPcGEeSBpgRY/edit?slide=id.g362ec749174_0_26#slide=id.g362ec749174_0_26

    measurements = measurements.clone()
    measurements[:, :9] *= pix_per_mm * img_scale_factor

    ang = (270 - measurements[:, 9]) * torch.pi / 180
    cos = torch.cos(ang)
    sin = torch.sin(ang)

    b_x = -cos * ab
    b_y = -sin * ab
    c_x = measurements[:, 4]
    c_y = -measurements[:, 5]

    all_x_y = [None for _ in range(12)]
    all_x_y[0] = torch.stack([cos * measurements[:, 3] / 2, sin * measurements[:, 3] / 2], dim = -1) #D
    all_x_y[2] = torch.stack([all_x_y[0][:, 0] - cos * measurements[:, 8], all_x_y[0][:, 1] - sin * measurements[:, 8]], dim = -1) #G
    all_x_y[1] = torch.stack([all_x_y[2][:, 0] + cos * measurements[:, 7], all_x_y[2][:, 1] + sin * measurements[:, 7]], dim = -1) #F
    all_x_y[3] = torch.stack([b_x + sin * measurements[:, 6] / 2, b_y - cos * measurements[:, 6] / 2], dim = -1) #H
    all_x_y[4] = torch.stack([c_x - measurements[:, 2] / 2, c_y], dim = -1) #W
    all_x_y[7] = torch.stack([c_x + measurements[:, 2] / 2, c_y], dim = -1) #Z
    all_x_y[5] = torch.stack([all_x_y[4][:, 0] + measurements[:, 0], c_y], dim = -1) #X
    all_x_y[6] = torch.stack([all_x_y[7][:, 0] - measurements[:, 1], c_y], dim = -1) #Y

    all_x_y[8] = torch.stack([b_x - sin * measurements[:, 6] / 2, b_y + cos * measurements[:, 6] / 2], dim = -1) #H'
    all_x_y[9] = torch.stack([sin * measurements[:, 3] / 2, -cos * measurements[:, 3] / 2], dim = -1) #A'
    all_x_y[10] = torch.stack([-sin * measurements[:, 3] / 2, cos * measurements[:, 3] / 2], dim = -1) #A''
    all_x_y[11] = torch.stack([b_x, b_y], dim = -1) #B

    coord = torch.stack(all_x_y, dim = 1)

    #flip y
    coord[:, :, 1] *= -1

    ##DEBUG BEGIN##
    if DEBUG_MODE:
        for i in range(measurements.shape[0]):
            plt.scatter(coord[i, :, 0].cpu(), coord[i, :, 1].cpu(), c = 'r')
            for j, label in enumerate(['D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'B']):
                curx, cury = coord[i, j]
                plt.text(curx + 0.5, cury + 0.5, label)
            plt.show()
            print(coord[i])
    ##DEBUG END##

    return coord

#turn points into measurements
def coord_to_measurements(og_coord, pix_per_mm, img_scale_factor):
    #units: pixels -> mm
    #shape (batch_size, 9, 2)
    #refer to powerpoint: https://docs.google.com/presentation/d/1FPTIKSnscUQzuTzK1f5JnABffidgGuaxPcGEeSBpgRY/edit?slide=id.g362ec749174_0_26#slide=id.g362ec749174_0_26

    #flip y
    coord = og_coord.clone()
    coord[:, :, 1] *= -1

    eps = 1e-6

    slope = coord[:, 0, 1] / (coord[:, 0, 0] + eps)
    b_x = (coord[:, 3, 0] / (slope + eps) + coord[:, 3, 1]) / (slope + 1 / (slope + eps) + eps)
    b_y = slope * b_x


    measurements = [None for _ in range(10)]
    measurements[0] = coord[:, 5, 0] - coord[:, 4, 0]
    measurements[1] = coord[:, 7, 0] - coord[:, 6, 0]
    measurements[2] = coord[:, 7, 0] - coord[:, 4, 0]
    measurements[3] = 2 * torch.norm(coord[:, 0], dim = 1)
    measurements[4] = (coord[:, 4, 0] + coord[:, 7, 0]) / 2
    measurements[5] = -coord[:, 4, 1]
    measurements[6] = 2 * torch.sqrt((coord[:, 3, 0] - b_x) ** 2 + (coord[:, 3, 1] - b_y) ** 2)
    measurements[7] = torch.norm(coord[:, 1] - coord[:, 2], dim = 1)
    measurements[8] = torch.norm(coord[:, 0] - coord[:, 2], dim = 1)
    measurements[9] = torch.arctan(-slope) * 180 / torch.pi + 90

    measurements = torch.stack(measurements, dim = -1).clone()
    measurements[:, :9] /= pix_per_mm * img_scale_factor

    return measurements

#heatmaps to keypoints
def heatmaps_to_keypoints(heatmaps, img_height, img_width):
    device = heatmaps.device

    B, K, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, K, -1)
    softmax_heatmaps = F.softmax(heatmaps, dim=-1)
    softmax_heatmaps = softmax_heatmaps.view(B, K, H, W)

    xs = torch.linspace(0, img_width, W, device = device)
    ys = torch.linspace(0, img_height, H, device = device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing = 'ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    expected_x = (softmax_heatmaps * grid_x).sum(dim= [-2, -1])
    expected_y = (softmax_heatmaps * grid_y).sum(dim= [-2, -1])

    return torch.stack([expected_x, expected_y], dim = 2)

#heatmap model
class HeatmapsModel(nn.Module):
    def __init__(self, img_width, img_height, kp_cnt, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height

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

        self.conv5 = nn.Conv2d(channel_cnt[-1], kp_cnt, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = heatmaps_to_keypoints(x, self.img_height, self.img_width)
        return x
