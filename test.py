import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import random
import matplotlib.pyplot as plt
import random
import tqdm as tqdm

import nets
import deeplearning

def load_cifar10_test():
    DIR_VAL = "./dataset/cifar10/test/" 

    classes = os.listdir(DIR_VAL)
    print("Total Classes: ", len(classes))
    print(classes)

    val_imgs  = []
    for _class in classes:
        b = glob.glob(DIR_VAL + _class + '/*.png')
        b_1 = [filename.replace("\\", "/") for filename in b]
        val_imgs += b_1

    pic = [0,0,0]
    for i in range(3):
        r = random.randint(1, len(val_imgs)-1)
        pic[i] = val_imgs[r]

    return pic, classes


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



def main():
    Data_test, classes = load_cifar10_test()
    

    cifar_transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
    
    test_dataset = CIFAR10Dataset(imgs_list = Data_test, classes = classes, transforms = cifar_transforms_test)

    cifar_test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size= 1,
                                                shuffle=False, pin_memory = True, num_workers = 4)

    device = 'cuda'
    print(device)
    print(torch.cuda.get_device_name())

    # model = nets.CNN()
    # model = nets.Resnet()
    # model = nets.Inception()
    # model = nets.Resnext()
    # model = nets.SE()
    model = nets.Mixed()


    y_pred = deeplearning.test(
    test_loader= cifar_test_loader,
    model = model,
    device = device,
    load_saved_model = True,
    ckpt_path = "./ckpt_Mixed_epoch60.ckpt",
    )


    plt.figure()
    plt.subplot(131)
    image = Image.open(Data_test[0])
    label = Data_test[0].split("/")[-2]
    plt.imshow(image)
    # plt.xlabel(label)
    plt.xlabel(classes[y_pred[0]])

    plt.subplot(132)
    image = Image.open(Data_test[1])
    label = Data_test[1].split("/")[-2]
    plt.imshow(image)
    # plt.xlabel(label)
    plt.xlabel(classes[y_pred[1]])

    plt.subplot(133)
    image = Image.open(Data_test[2])
    label = Data_test[2].split("/")[-2]
    plt.imshow(image)
    # plt.xlabel(label)
    plt.xlabel(classes[y_pred[2]])

    plt.tight_layout()
    plt.show()
    


if __name__ == '__main__':main()