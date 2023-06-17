import sys
import torch
import torch.nn as nn
import torchvision
from ..separableconv.nn import SeparableConv1d, SeparableConv2d, SeparableConv3d


class SeparableAttention(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(SeparableAttention, self).__init__()

        self.W_g = nn.Sequential(
            SeparableConv2d(in_channels=F_g, out_channels=F_int, kernel_size=1,
                            stride=1, padding='valid'),
            nn.BatchNorm2d(num_features=F_int)
        )

        self.W_x = nn.Sequential(
            SeparableConv2d(in_channels=F_l, out_channels=F_int, kernel_size=1, 
                            stride=1, padding='valid'),
            nn.BatchNorm2d(num_features=F_int)
        )

        self.psi = nn.Sequential(
            SeparableConv2d(in_channels=F_int, out_channels=1, kernel_size=1,
                            stride=1, padding='valid'),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'W_g': self.W_g,
            'W_x': self.W_x,
            'psi': self.psi,
            'relu': self.relu
        })
        return config
