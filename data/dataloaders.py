from __future__ import absolute_import, division

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from data import data 

# statistics for data sampling
mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'imagenet': (0.485, 0.456, 0.406),
    'mnist':(0.1307,),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'imagenet': (0.229, 0.224, 0.225),
    'mnist': (0.3081,),
}


'''
CIFAR-X dataloader
'''

def CIFARX(name:str, data_path: str, sample: int=None):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name]),
    ])

    trainset, testset = None, None
    trainset = data.CIFAR(root=data_path, train=True, trans=transform_train, sample=sample, name=name)
    testset  = data.CIFAR(root=data_path, train=False, trans=transform_test, sample=sample, name=name)
    
    return trainset, testset


'''
ROTMNIST dataloader
'''
def ROTMNIST(data_path: str, tmode: int):
    trans1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean["mnist"], std["mnist"]),
    ])

    trans2 = transforms.Compose([transforms.ToTensor(),])

    if tmode == 0: 
        trainset = data.ROTMNIST(root=data_path + '/mnist_all_rotation_normalized_float_train_valid.amat', trans=trans2) 
        testset  = data.ROTMNIST(root=data_path + '/mnist_all_rotation_normalized_float_test.amat', trans=trans2) 
    elif tmode == 1: 
        trainset = data.ROTMNIST(root=data_path + '/mnist_all_rotation_normalized_float_train_valid.amat', trans=trans2)
        testset  = data.MNIST(root=data_path, train=False, trans=trans1)

    return trainset, testset


'''
Imagenet dataloader
'''
def IMAGENET(data_path:str):
    train_imagenet=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet']),
    ])

    test_imagenet=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet']),
    ])
    trainset = data.IMAGENET(root=data_path, split='train', trans=train_imagenet)    
    testset  = data.IMAGENET(root=data_path, split='val', trans=test_imagenet) 

    return trainset, testset
