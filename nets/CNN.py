import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt





class ModelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2, 2) # output: 256 x 4 x 4
            )
        
        self.feature_extractor[0].weight.data = torch.nn.init.xavier_normal_(self.feature_extractor[0].weight.data,
                                                                     gain = torch.nn.init.calculate_gain("leaky_relu"))
        
        ## Bias --> Standard distribution
        self.feature_extractor[0].bias.data = torch.randn(self.feature_extractor[0].bias.data.shape)
        
        self.flat = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, x):
        x = self.feature_extractor(x)
        # x = self.flat(x)
        x = x.view(-1,256*4*4)
        x = self.classifier(x)
        return x