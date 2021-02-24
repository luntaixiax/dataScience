import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from customDataset import get_CatDogDataSet
from torchvision.utils import save_image

# load data
my_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness = 0.5),
        transforms.RandomRotation(degrees = 45),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.05),
        transforms.RandomGrayscale(p = 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0, 0, 0], std = [1, 1, 1])  # do nothing
    ]
)

dataset = get_CatDogDataSet(my_transforms)
