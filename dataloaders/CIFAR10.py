import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import random
import matplotlib.pyplot as plt



def load_cifar10():
    DIR_TRAIN = "./dataset/cifar10/train/"
    DIR_VAL = "./dataset/cifar10/test/" 

    classes = os.listdir(DIR_TRAIN)
    print("Total Classes: ", len(classes))

    train_imgs = []
    val_imgs  = []
    for _class in classes:
        a = glob.glob(DIR_TRAIN + _class + '/*.png')
        a_1 = [filename.replace("\\", "/") for filename in a]
        train_imgs += a_1
        b = glob.glob(DIR_VAL + _class + '/*.png')
        b_1 = [filename.replace("\\", "/") for filename in b]
        val_imgs += b_1

    print("\nTotal train images: ", len(train_imgs))
    print("Total test images: ", len(val_imgs))

    return train_imgs,  val_imgs, classes


class CIFAR10Dataset(Dataset):
    
    def __init__(self, imgs_list, classes, transforms=None):
        super(CIFAR10Dataset, self).__init__()
        self.imgs_list = imgs_list
        self.class_to_int = {classes[i] : i for i in range(len(classes))}
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.imgs_list[index]

        # Reading image
        image = Image.open(image_path)
        
        # Retriving class label
        label = image_path.split("/")[-2]
        label = self.class_to_int[label]
        
        # Applying transforms on image
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label
        

    def __len__(self):
        return len(self.imgs_list)


def cifar10_data(batch_size= 16):

    Data_train, Valid_train, classes = load_cifar10()
    
    cifar_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

    cifar_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])


    # train_dataset = CIFAR10Dataset(imgs_list = Data_train, classes = classes, transforms = cifar_transforms_train)
    # val_dataset = CIFAR10Dataset(imgs_list = Valid_train, classes = classes, transforms = cifar_transforms_val)

    train_dataset = torchvision.datasets.CIFAR10(root="CIFAR", download=True, train=True, transform=cifar_transforms_train)
    val_dataset = torchvision.datasets.CIFAR10(root="CIFAR", download=True, train=False, transform=cifar_transforms_val)

    cifar_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size= batch_size,
                                                shuffle=True, pin_memory = True, num_workers = 4)

    cifar_val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size= batch_size,
                                                shuffle=False, pin_memory = True, num_workers = 4)
    
    return cifar_train_loader, cifar_val_loader



if __name__ == '__main__':cifar10_data()
