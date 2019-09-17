import torch
from torch import nn


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(), # try inplace? Try leaky? Try sub_(-0.5)?
        )

class ConvNet(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.seq = nn.Sequential(
            Conv_BN_ReLU(in_ch, 16, 3),
            Conv_BN_ReLU(16, 32, 3, stride=2),
            Conv_BN_ReLU(32, 64, 3),
            Conv_BN_ReLU(64, 64, 3, stride=2),
        )
