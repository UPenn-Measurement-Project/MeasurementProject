#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#turn model output into (coordinate, length ab)
def model_to_coord(model_out, DEBUG_MODE = False):
    #units: pixels -> pixels
    #shape: (batch_size, 11)
    #refer to powerpoint: https://docs.google.com/presentation/d/1FPTIKSnscUQzuTzK1f5JnABffidgGuaxPcGEeSBpgRY/edit?slide=id.g362ec749174_0_26#slide=id.g362ec749174_0_26

    eps = 1e-6

    slope = model_out[:, 1] / (model_out[:, 0] + eps)

    all_x_y = [None for _ in range(12)]
    all_x_y[0] = torch.stack([model_out[:, 0], model_out[:, 1]], dim = -1) #D
    all_x_y[1] = torch.stack([model_out[:, 2], slope * model_out[:, 2]], dim = -1) #F
    all_x_y[2] = torch.stack([model_out[:, 3], slope * model_out[:, 3]], dim = -1) #G
    all_x_y[3] = torch.stack([model_out[:, 4], model_out[:, 5]], dim = -1) #H
    all_x_y[4] = torch.stack([model_out[:, 6], model_out[:, 7]], dim = -1) #W
    all_x_y[5] = torch.stack([model_out[:, 8], model_out[:, 7]], dim = -1) #X
    all_x_y[6] = torch.stack([model_out[:, 9], model_out[:, 7]], dim = -1) #Y
    all_x_y[7] = torch.stack([model_out[:, 10], model_out[:, 7]], dim = -1) #Z

    b_x = (model_out[:, 4] / (slope + eps) + model_out[:, 5]) / (slope + 1 / (slope + eps) + eps)
    b_y = slope * b_x
    ab = torch.sqrt(b_x ** 2 + b_y ** 2)

    all_x_y[8] = torch.stack([2 * b_x - model_out[:, 4], 2 * b_y - model_out[:, 5]], dim = -1) #H'
    all_x_y[9] = torch.stack([model_out[:, 1], -model_out[:, 0]], dim = -1) #A'
    all_x_y[10] = torch.stack([-model_out[:, 1], model_out[:, 0]], dim = -1) #A''
    all_x_y[11] = torch.stack([(model_out[:, 6] + model_out[:, 10]) / 2, model_out[:, 7]], dim = -1) #C

    coord = torch.stack(all_x_y, dim = 1)

    ##DEBUG BEGIN##
    if DEBUG_MODE:
        for i in range(model_out.shape[0]):
            plt.scatter(coord[i, :, 0].cpu(), coord[i, :, 1].cpu(), c = 'r')
            plt.scatter(b_x[i].cpu(), b_y[i].cpu())
            for j, label in enumerate(['D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'C']):
                curx, cury = coord[i, j]
                plt.text(curx + 0.5, cury + 0.5, label)
            plt.text(b_x[i].cpu(), b_y[i].cpu(), 'B')
            plt.show()
            print(f'AB: {ab[i].item()}')
            print(coord[i])
            print(b_x[i].item(), b_y[i].item())
    ##DEBUG END##

    return coord, ab

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
    all_x_y[11] = torch.stack([c_x, c_y], dim = -1) #C

    coord = torch.stack(all_x_y, dim = 1)

    ##DEBUG BEGIN##
    if DEBUG_MODE:
        for i in range(measurements.shape[0]):
            plt.scatter(coord[i, :, 0].cpu(), coord[i, :, 1].cpu(), c = 'r')
            plt.scatter(b_x[i].cpu(), b_y[i].cpu())
            for j, label in enumerate(['D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'C']):
                curx, cury = coord[i, j]
                plt.text(curx + 0.5, cury + 0.5, label)
            plt.text(b_x[i].cpu(), b_y[i].cpu(), 'B')
            plt.show()
            print(coord[i])
    ##DEBUG END##

    return coord

#turn points into measurements
def coord_to_measurements(coord, pix_per_mm, img_scale_factor):
    #units: pixels -> mm
    #shape (batch_size, 9, 2)
    #refer to powerpoint: https://docs.google.com/presentation/d/1FPTIKSnscUQzuTzK1f5JnABffidgGuaxPcGEeSBpgRY/edit?slide=id.g362ec749174_0_26#slide=id.g362ec749174_0_26

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

#batchnorm relu class
class BNRelu(nn.Module):
    def __init__(self, chcnt, bn):
        super().__init__()
        
        if bn == "none":
            self.seq = nn.ReLU()
        elif bn == "before":
            self.seq = nn.Sequential(nn.BatchNorm2d(chcnt), nn.ReLU())
        elif bn == "after":
            self.seq = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(chcnt))
        else:
            raise ValueError(f"Unknown batch norm setting: {bn}")

    def forward(self, x):
        x = self.seq(x)
        return x

#simple cnn pipeline
class SimpleCNNModel(nn.Module):
    def __init__(self, img_width, img_height, bn):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            BNRelu(32, bn),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            BNRelu(64, bn),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            BNRelu(128, bn),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            BNRelu(256, bn),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(256 * (img_width // 16) * (img_height // 16), 4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, 11)
        )

    def forward(self, x):
        x = self.seq(x)
        return x

#alexnet pipeline
class AlexNet(nn.Module):
    def __init__(self, img_width, img_height, bn):
        super().__init__()

        conv_sz = lambda dim, ksz, str, pd : (dim + 2 * pd - ksz) // str + 1
        pool_sz = lambda dim, ksz, str : (dim - ksz) // str + 1

        fin_w = img_width
        fin_h = img_height

        fin_w = pool_sz(conv_sz(fin_w, 11, 4, 2), 3, 2)
        fin_h = pool_sz(conv_sz(fin_h, 11, 4, 2), 3, 2)

        fin_w = pool_sz(conv_sz(fin_w, 5, 1, 2), 3, 2)
        fin_h = pool_sz(conv_sz(fin_h, 5, 1, 2), 3, 2)

        fin_w = conv_sz(fin_w, 3, 1, 1)
        fin_h = conv_sz(fin_h, 3, 1, 1)

        fin_w = conv_sz(fin_w, 3, 1, 1)
        fin_h = conv_sz(fin_h, 3, 1, 1)

        fin_w = pool_sz(conv_sz(fin_w, 3, 1, 1), 3, 2)
        fin_h = pool_sz(conv_sz(fin_h, 3, 1, 1), 3, 2)

        self.seq = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size = 11, stride = 4, padding = 2),
            BNRelu(96, bn),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            BNRelu(256, bn),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            BNRelu(384, bn),

            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            BNRelu(384, bn),

            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            BNRelu(256, bn),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Flatten(),
            #nn.Dropout(),
            nn.Linear(256 * fin_w * fin_h, 4096),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 11)

        )

    def forward(self, x):
        x = self.seq(x)
        return x
