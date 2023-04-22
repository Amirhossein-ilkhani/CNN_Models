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
import os


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def test(
    test_loader,
    model,
    device,
    load_saved_model,
    ckpt_path,
):

    torch.backends.cudnn.benchmark = True
    model = model.to(device, non_blocking=True)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )

    y_pred = []
    model.eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
            print(labels)
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            labels_pred = model(images) 
            y_pred.append(torch.argmax(labels_pred))

    return y_pred