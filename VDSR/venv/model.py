import torch
import torch.nn as nn
from math import sqrt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        layers = []
        # 一共20层
        for i in range(18):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.residual_layer = nn.Sequential(*layers)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    # 残差模式
    def forward(self, x):
        residuel = x
        out = self.input(x)
        out = self.relu(out)
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residuel)
        return out
