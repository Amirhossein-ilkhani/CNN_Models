
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary as Model_summary

import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class Resnext_block(nn.Module):
    def __init__(self, in_channel, out_channel, b, g):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = g
        self.b = b
        self.branch = []

        self.conv1 = nn.Sequential(
        nn.Conv2d(self.in_channel, self.b, kernel_size=1),
        nn.BatchNorm2d(self.b),
        nn.ReLU() 
        )

        self.conv3 = nn.Sequential(
        nn.Conv2d(self.b, int(self.b / self.g), kernel_size=3, padding=1),
        nn.BatchNorm2d(int(self.b / self.g)),
        nn.ReLU() 
        )

        for i in range(g):
            self.branch.append(self.conv3)

        self.conv1_2 = nn.Sequential(
        nn.Conv2d(self.b, self.out_channel, kernel_size=1),
        nn.BatchNorm2d(self.out_channel)
        )
        
    def forward(self, x):
        passway = x
        x = self.conv1(x)
        out = self.branch[0](x)
        for i in range(self.g - 1):
            out = torch.cat([out, self.branch[i+1](x)], dim=1)
        out = self.conv1_2(out)
        out+= passway
        return out
    

class Resnext_block_1(nn.Module):
    def __init__(self, in_channel, out_channel, b, g):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = g
        self.b = b
        self.branch = []

        self.conv1 = nn.Sequential(
        nn.Conv2d(self.in_channel, self.b, kernel_size=1),
        nn.BatchNorm2d(self.b),
        nn.ReLU() 
        )

        self.conv3 = nn.Sequential(
        nn.Conv2d(self.b, int(self.b / self.g), kernel_size=3, padding=1),
        nn.BatchNorm2d(int(self.b / self.g)),
        nn.ReLU() 
        )

        for i in range(g):
            self.branch.append(self.conv3)

        self.conv1_2 = nn.Sequential(
        nn.Conv2d(self.b, self.out_channel, kernel_size=1),
        nn.BatchNorm2d(self.out_channel)
        )

        self.conv1_3 = nn.Sequential(
        nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1),
        nn.BatchNorm2d(self.out_channel),
        nn.ReLU() 
        )
        
        

    def forward(self, x):
        passway = self.conv1_3(x)
        x = self.conv1(x)
        out = self.branch[0](x)
        for i in range(self.g - 1):
            out = torch.cat([out, self.branch[i+1](x)], dim=1)
        out = self.conv1_2(out)
        out+= passway
        return out



class Resnext(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # output 64*32*32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output 128*16*16

            Resnext_block_1(128, 128, b=32, g=4), # output 128*16*16

            Resnext_block_1(128, 256, b=64, g=4), # output 256*16*16
            nn.MaxPool2d(kernel_size=2), # output 256*8*8

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # output 512*8*8
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output 512*4*4

            Resnext_block_1(512, 512, b=128, g=4), # output 512*4*4
            nn.AvgPool2d(kernel_size=4) # output 512
            )
        

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1,512*1*1)
        x = self.classifier(x)
        return x
    

def main():
    s = Resnext()
    s.to('cuda')
    Model_summary(s, (3, 32, 32))
if __name__ == '__main__':main()