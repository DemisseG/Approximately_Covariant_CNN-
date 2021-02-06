from __future__ import absolute_import, division
from typing import Tuple, List, Union
import os

import numpy
import torch
import torch.utils.data as Data
from torchvision import datasets, transforms
from PIL import Image


class ROTMNIST(Data.Dataset):
    _ROUNDING_NUM =  100
    def __init__(self,
                root: str,
                trans: transforms.Compose=None):
        super(ROTMNIST, self).__init__()

        assert os.path.exists(root), "Error: the path \"" + root + "\" to ROTMNIST dataset does not exisit!" 
        self.main_path = root
        self.transform = trans
        self.data, self.label = self.readAMAT()

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        im1 = numpy.array((self.data[id] * ROTMNIST._ROUNDING_NUM), dtype=numpy.uint8)
        if self.transform:
            dataPoint = self.transform(Image.fromarray(im1)) / float(ROTMNIST._ROUNDING_NUM)
        else:
            dataPoint = (torch.as_tensor(im1, dtype=torch.float32) / float(ROTMNIST._ROUNDING_NUM)).unsqueeze(0)
        return dataPoint, self.label[id]

    def __len__(self) -> int:
        return len(self.data)

    def readAMAT(self) -> Tuple[List[numpy.array], torch.Tensor]:
        path = self.main_path
        data_points = []
        labels = []
        with open(path, 'rb') as f1:
            for line in f1:
                try:
                    float_line = [float(l) for l in line.split()]
                    data_points += [numpy.array(float_line[:-1]).reshape(28, 28)]
                    labels += [torch.tensor(int(float_line[-1]), dtype=torch.int64)]
                except Exception:
                    raise
                    exit(1)

        labels = torch.stack(labels, 0)
        return data_points, labels


class MNIST(Data.Dataset):
    def __init__(self,
                root: str=None,
                train: bool=False,
                trans: transforms.Compose=None
                ):
        super(MNIST, self).__init__()

        self.transform = trans
        temp = datasets.MNIST(root, train=train, download=(not os.path.exists(root + '/MNIST')))
        self.data = temp.data
        self.label = temp.targets

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataPoint = self.data[id]
        if self.transform:
            if not isinstance(dataPoint, Image.Image):
                dataPoint = Image.fromarray(numpy.array(dataPoint, dtype=numpy.uint8))
            dataPoint = self.transform(dataPoint)
        dataPoint = torch.as_tensor(dataPoint)
        
        return dataPoint, self.label[id]  

    def __len__(self) -> int:
        return len(self.data)


class CIFAR(Data.Dataset):
    def __init__(self,
                root: str = None,
                train: bool=True, 
                trans: transforms.Compose=None, 
                sample: int=None, 
                name: str='cifar10'):
        super(CIFAR, self).__init__()

        self.name = name
        self.transform = trans
        Temp = datasets.CIFAR10(root, train=train, download=(not os.path.exists(root + 'cifar-10-python.tar.gz'))) if self.name == 'cifar10' \
             else datasets.CIFAR100(root, train=train, download=(not os.path.exists(root + 'cifar-100-python.tar.gz')))
        self.data = Temp.data
        self.label = torch.as_tensor(Temp.targets)
        self.index = list(numpy.arange(0, len(self.data), 1)) if sample is None else self.sample_complexity(sample)

    def sample_complexity(self, sample: int) -> List[int]:
        """
        Method for randomly subsampling each label: 
        Usage: mainly to test a model's performance aginist sample complexity.
        """

        label_num =  10 if self.name == 'cifar10' else 100
        indexes = []
        for i in range(label_num):
            indexL = torch.nonzero(self.label == i).flatten()
            if indexL.shape[0] > sample:
                randi = torch.randint(0, indexL.shape[0], (sample,)).tolist()
                indexes += indexL[randi].tolist()
            else:
                indexes += indexL.tolist()
        return indexes

    def __getitem__(self, key: int) -> Tuple[torch.Tensor, torch.Tensor]:
        id  = self.index[key]
        dataPoint = self.data[id]
        if self.transform:
            if not isinstance(dataPoint, Image.Image):
                dataPoint = Image.fromarray(dataPoint)
            dataPoint = self.transform(dataPoint)
        dataPoint = torch.as_tensor(dataPoint)
        return dataPoint, self.label[id]


    def __len__(self) -> int:
        return len(self.data)



class IMAGENET(Data.Dataset):
    NUM_CLASSES = 1000
    def __init__(self,
                root: str,
                split:str='train',
                trans: transforms.Compose=None):
        super(IMAGENET, self).__init__()

        self.data = self._load_dataset(root, split)
        self.target_names = None
        self.transform = trans
        self.get_target_names(root)

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transform:
            img_sample = self.transform(self.data[id][0])
        else:
            img_sample = self.data[id][0]
        return img_sample, self.data[id][1]

    def _load_dataset(self, path: str, split: str) -> Tuple[Image.Image, int] :
        img_dir = os.path.join(path, split)
        dataset = datasets.ImageFolder(img_dir)
        return dataset

    def get_target_names(self, path):
        if self.target_names is not None:
            return self.target_names

        labels_file = os.path.join(path, "labels.txt")
        with open(labels_file, "r") as fh:
            class_names = []
            synset_names = []
            for line in fh:
                synset_name, class_name = line.strip().split(",")
                synset_names.append(synset_name)
                class_names.append(class_name)
        assert len(synset_names) == IMAGENET.NUM_CLASSES
        assert len(class_names) == IMAGENET.NUM_CLASSES
        self.target_names = (synset_names, class_names)

    def __len__(self) -> int:
        return len(self.data)
