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



class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res




def train(
    train_loader,
    val_loader,
    model,
    model_name,
    epochs,
    learning_rate,
    gamma,
    step_size,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
):

    torch.backends.cudnn.benchmark = True
    model = model.to(device, non_blocking=True)
    

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )
        s = ckpt_path.find('.ckpt')
        extra_epoch = int(ckpt_path[s-2:s])
    else:
        extra_epoch = 0
    

    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_top1_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_top1_acc_till_current_batch"])

    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        loss_avg_val = AverageMeter()
        i = 0

        model.train()
        mode = "train"
      
        loop_train = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc="train",
            position=0,
            leave=True)
        for batch_idx, (images, labels) in loop_train:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(labels_pred, labels)
            top1_acc_train.update(acc1[0], images.size(0))
            loss_avg_train.update(loss.detach().item(), images.size(0))


            new_row = pd.DataFrame(
                {"model_name": model_name,
                "mode": mode,
                "image_type":"original",
                "epoch": epoch + extra_epoch,
                "learning_rate":optimizer.param_groups[0]["lr"],
                "batch_size": images.size(0),
                "batch_index": batch_idx,
                "loss_batch": loss.detach().item(),
                "avg_train_loss_till_current_batch":loss_avg_train.avg,
                "avg_train_top1_acc_till_current_batch":top1_acc_train.avg,
                "avg_val_loss_till_current_batch":None,
                "avg_val_top1_acc_till_current_batch":None},index=[0])
            
            report.loc[len(report)] = new_row.values[0]


            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                max_len=2,
                refresh=True,
            )

        lr_scheduler.step()
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch + extra_epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )

        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (images, labels) in loop_val:
                optimizer.zero_grad()
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)
                labels_pred = model(images)
                loss = criterion(labels_pred, labels)
                acc1 = accuracy(labels_pred, labels)
                top1_acc_val.update(acc1[0], images.size(0))
                loss_avg_val.update(loss.item(), images.size(0))
                

                new_row = pd.DataFrame(
                    {"model_name": model_name,
                    "mode": mode,
                    "image_type":"original",
                    "epoch": epoch + extra_epoch,
                    "learning_rate":optimizer.param_groups[0]["lr"],
                    "batch_size": images.size(0),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch":None,
                    "avg_train_top1_acc_till_current_batch":None,
                    "avg_val_loss_till_current_batch":loss_avg_val.avg,
                    "avg_val_top1_acc_till_current_batch":top1_acc_val.avg},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True,
                )

        
    if(load_saved_model):
        report.to_csv(f"{report_path}/{model_name}_report.csv", mode = 'a')
    else:
        report.to_csv(f"{report_path}/{model_name}_report.csv")
    return model, optimizer, report
