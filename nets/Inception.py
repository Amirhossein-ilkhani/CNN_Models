
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


class Inception_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = int(out_channel/4)

        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_channel, int(self.out_channel/4), kernel_size=1),
            nn.BatchNorm2d(int(self.out_channel/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.out_channel/4), int(self.out_channel/2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(self.out_channel/2)),
            nn.ReLU(),
            nn.Conv2d(int(self.out_channel/2), self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(self.in_channel, int(self.out_channel/2), kernel_size=1),
            nn.BatchNorm2d(int(self.out_channel/2)),
            nn.ReLU(),  
            nn.Conv2d(int(self.out_channel/2), self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
        )

        self.branch3 = nn.Sequential(
        nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1),
        nn.BatchNorm2d(self.out_channel),
        nn.ReLU(),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=1),
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
        )
        

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return out



class Inception(nn.Module):
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

            Inception_Block(128, 128), # output 128*16*16

            Inception_Block(128, 256), # output 256*16*16
            nn.MaxPool2d(kernel_size=2), # output 256*8*8

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # output 512*8*8
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output 512*4*4

            Inception_Block(512, 512), # output 512*4*4
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
    s = Inception()
    s.to('cuda')
    Model_summary(s, (3, 32, 32))
if __name__ == '__main__':main()