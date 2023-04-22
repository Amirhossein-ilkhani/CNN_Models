
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


class Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.path = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )
        
    def forward(self, x):
        out = self.path(x)
        return out
    

    
class SE_block(nn.Module):
    def __init__(self, in_channel, out_channel, r):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.r = r

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channel, self.in_channel // self.r),
            nn.ReLU(),
            nn.Linear(self.in_channel // self.r, self.out_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.SE(x)
        return out
    
    

class SE_Resnet(nn.Module):
    def __init__(self, in_channel, out_channel, r):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.r = r

        self.residual = Residual_block(self.in_channel, self.out_channel)
        self.SE = SE_block(self.out_channel, self.out_channel, r=self.r)

        self.passway = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel,  kernel_size=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )


    def forward(self, x):
        bypass = self.passway(x)
        x = self.residual(x)
        bypass_1 = x
        x = self.SE(x)
        x = x[:, :, None,None]

        scale = bypass_1 * x 
        out = scale + bypass
        
        return out

        


class SE(nn.Module):
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
            
            SE_Resnet(128, 128, 4), # output 128*16*16

            SE_Resnet(128, 256, 4), # output 256*16*16
            nn.MaxPool2d(kernel_size=2), # output 256*8*8

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # output 512*8*8
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output 512*4*4

            SE_Resnet(512, 512, 4), # output 512*4*4
            nn.AvgPool2d(kernel_size=4) # output 512
            )
        

        

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        # x = self.flat(x)
        x = x.view(-1,512*1*1)
        x = self.classifier(x)
        return x
    

def main():
    s = SE()
    s.to('cuda')
    Model_summary(s, (3, 32, 32))
if __name__ == '__main__':main()