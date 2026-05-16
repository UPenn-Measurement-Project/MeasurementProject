import torch
import torch.nn as nn


def test_measurements_to_coord(measurements, pix_per_mm, img_scale_factor, DEBUG_MODE = False):
    #units: mm -> pixels
    #measurements shape: (batch_size, 10)
    #ab shape: (batch_size,)
    #refer to powerpoint: https://docs.google.com/presentation/d/1FPTIKSnscUQzuTzK1f5JnABffidgGuaxPcGEeSBpgRY/edit?slide=id.g362ec749174_0_26#slide=id.g362ec749174_0_26

    measurements = measurements.clone()
    measurements[:, :9] *= pix_per_mm * img_scale_factor

    ang = (270 - measurements[:, 9]) * torch.pi / 180
    cos = torch.cos(ang)
    sin = torch.sin(ang)

    
    c_x = measurements[:, 4]
    c_y = -measurements[:, 5]

    all_x_y = [None for _ in range(10)]
    all_x_y[0] = torch.stack([cos * measurements[:, 3] / 2, sin * measurements[:, 3] / 2], dim = -1) #D
    all_x_y[2] = torch.stack([all_x_y[0][:, 0] - cos * measurements[:, 8], all_x_y[0][:, 1] - sin * measurements[:, 8]], dim = -1) #G
    all_x_y[1] = torch.stack([all_x_y[2][:, 0] + cos * measurements[:, 7], all_x_y[2][:, 1] + sin * measurements[:, 7]], dim = -1) #F
    all_x_y[3] = torch.stack([c_x - measurements[:, 2] / 2, c_y], dim = -1) #W
    all_x_y[6] = torch.stack([c_x + measurements[:, 2] / 2, c_y], dim = -1) #Z
    all_x_y[4] = torch.stack([all_x_y[3][:, 0] + measurements[:, 0], c_y], dim = -1) #X
    all_x_y[5] = torch.stack([all_x_y[6][:, 0] - measurements[:, 1], c_y], dim = -1) #Y
    all_x_y[7] = torch.stack([sin * measurements[:, 3] / 2, -cos * measurements[:, 3] / 2], dim = -1) #A'
    all_x_y[8] = torch.stack([-sin * measurements[:, 3] / 2, cos * measurements[:, 3] / 2], dim = -1) #A''
    all_x_y[9] = torch.stack([c_x, c_y], dim = -1) #C

    coord = torch.stack(all_x_y, dim = 1)

    return coord

    
#input: 1 channel 
#output: femoral head center
class testCNNModel(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        self.filter = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            )
        fSize = 256 * (img_height // 16) * (img_width // 16)
        self.lineList = nn.Linear(fSize, 9)
        self.angleElement = nn.Linear(fSize, 1)
        
        
    def forward(self, x):
        features = self.filter(x)
        measurements = self.lineList(features)
        angle = self.angleElement(features)
        return torch.cat([measurements, angle], dim = 1)
        #returns single tensor of ten elements (last one being angle)

class testCNNModel_02(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        self.filter = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            )
        fSize = 256 * (img_height // 32) * (img_width // 32)
        self.lineList = nn.Linear(fSize, 9)
        self.angleElement = nn.Linear(fSize, 1)
        
        
    def forward(self, x):
        features = self.filter(x)
        measurements = self.lineList(features)
        angle = self.angleElement(features)
        return torch.cat([measurements, angle], dim = 1)
        #returns single tensor of ten elements (last one being angle)
    
