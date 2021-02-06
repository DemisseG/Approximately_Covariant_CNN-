from __future__ import absolute_import
import unittest
import os

import torch
from torchvision import transforms

from data import data 
from data import dataloaders 
  

class test_data(unittest.TestCase):
    def setUp(self):
        self.path=os.getcwd() + '/data'
        self.name='cifar10'
        self.trans= transforms.Compose([transforms.RandomResizedCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])

    def test_loading_cifar(self):
        data_class = data.CIFAR(root=self.path, train=True, trans=self.trans, sample=None, name=self.name)
        self.assertEqual(len(data_class), 50000)
        self.assertTrue(isinstance(data_class[0][0], torch.Tensor))
        self.assertTrue(isinstance(data_class[0][1], torch.Tensor))

    def test_subsampling(self):
        test_size = torch.randint(0, 50000, (1,))
        data_class = data.CIFAR(root=self.path, train=True, trans=self.trans, sample=test_size, name=self.name)
        for i in range(10):
            index = torch.nonzero((data_class.label[data_class.index] == i)).flatten()
            self.assertLessEqual(index.shape[0], test_size)

    def test_dataloader(self):
        # test loader 
        train, test = dataloaders.CIFARX(self.name, self.path)
        self.assertIsNotNone(train)
        self.assertIsNotNone(test)

    def test_loading_rotMNIST(self):
        datapath = os.getcwd() + '/data/mnist_all_rotation_normalized_float_test.amat'
        data_class = data.ROTMNIST(datapath, trans= transforms.Compose([transforms.ToTensor(),
                                                                        ]))
        self.assertEqual(len(data_class), 50000)

