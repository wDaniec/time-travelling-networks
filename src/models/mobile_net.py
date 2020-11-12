# -*- coding: utf-8 -*-
"""
SimpleCNN model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import gin


@gin.configurable
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = torchvision.models.mobilenet_v2()
        conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        fc = nn.Linear(1280, 10)
        self.model.features[0][0] = conv
        self.model.classifier[1] = fc

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
