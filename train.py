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

import dataloaders
import nets
import deeplearning
import utils


def plotting(path):
    df = pd.read_csv(path)
    
    acc_train = df[['epoch','avg_train_top1_acc_till_current_batch']]
    acc_train = acc_train.dropna()
    loss_train = df[['epoch',"avg_train_loss_till_current_batch"]]
    loss_train = loss_train.dropna()

    acc_valid = df[['epoch','avg_val_top1_acc_till_current_batch']]
    acc_valid = acc_valid.dropna()
    loss_valid = df[['epoch',"avg_val_loss_till_current_batch"]]
    loss_valid = loss_valid.dropna()

   
    epoch = acc_train.iloc[-1,0]
    print("Epoch={}".format(epoch))
    train_size = len(acc_train[acc_train['epoch'] == 1])
    print(train_size)
    valid_size = len(acc_valid[acc_valid['epoch'] == 1])
    print(valid_size)


    acc_t =[]
    for i in range(epoch):
        df_temp = acc_train[acc_train['epoch'] == i+1]
        df_temp = df_temp.reset_index(drop=True)
        acc_t.append(df_temp.iloc[-1,-1])

    acc_v =[]
    for i in range(epoch):
        df_temp = acc_valid[acc_valid['epoch'] == i+1]
        df_temp = df_temp.reset_index(drop=True)
        acc_v.append(df_temp.iloc[-1,-1])
    

    loss_t =[]
    for i in range(epoch):
        df_temp = loss_train[loss_train['epoch'] == i+1]
        df_temp = df_temp.reset_index(drop=True)
        loss_t.append(df_temp.iloc[-1,-1])

    loss_v =[]
    for i in range(epoch):
        df_temp = loss_valid[loss_valid['epoch'] == i+1]
        df_temp = df_temp.reset_index(drop=True)
        loss_v.append(df_temp.iloc[-1,-1])
        


    plt.figure()
    plt.subplot(121)
    plt.plot(acc_t, color='black')
    plt.plot(acc_v, color='red')
    plt.xlabel('epoch ')
    plt.ylabel('acc')

    plt.subplot(122)
    plt.plot(loss_t, color='black')
    plt.plot(loss_v, color='red')
    plt.xlabel('epoch')
    plt.ylabel('error')

    plt.legend(["train", "valid"], loc ="upper right")
    plt.show()




def main():

    yaml = utils.load()
    print(yaml)
    
    
    batch_size = yaml['batch_size']
    epochs = yaml['epochs']
    learning_rate = yaml['learning_rate']
    gamma = yaml['gamma']
    step_size = yaml['step_size']
    ckpt_save_freq = yaml['ckpt_save_freq']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name())

    if (yaml['model'] == "CNN"):
        custom_model = nets.CNN()
    elif (yaml['model'] == "Resnet"):
        custom_model = nets.Resnet()
    elif (yaml['model'] == "Inception"):
        custom_model = nets.Inception()
    elif (yaml['model'] == "Resnext"):
        custom_model = nets.Resnext()
    elif (yaml['model'] == "SE"):
        custom_model = nets.SE()
    elif (yaml['model'] == "Mixed"):
        custom_model = nets.Mixed()
    else:
        print("Unknown model")
        exit()


    custom_model.to(device,  non_blocking=True)
    Model_summary(custom_model, (3, 32, 32))

    cifar_train_loader, cifar_val_loader = dataloaders.cifa10_data(batch_size=batch_size)

    trainer = deeplearning.train(
        train_loader=cifar_train_loader,
        val_loader=cifar_val_loader,
        model = custom_model,
        model_name= yaml['model'],
        epochs=epochs,
        learning_rate=learning_rate,
        gamma = gamma,
        step_size = step_size,
        device=device,
        load_saved_model= yaml["load"],
        ckpt_save_freq=ckpt_save_freq,
        ckpt_save_path= yaml["ckpt_save_path"],
        ckpt_path= yaml["ckpt_path"],
        report_path= yaml["report_path"]
    ) 

if __name__ == '__main__':main()