from __future__ import absolute_import, division
import sys 
import os 
import unittest

import torch 
from ac import ac_conv as ac 
from ac import utils 

class test_ac(unittest.TestCase):
    def setUp(self):
        self.filter = torch.tensor([[-1.0, 2.0, -1.0], 
                                    [-1.0, 4.5, -1.0],
                                    [-1.0, 2.0, -1.0]], dtype=torch.float32)
        self.input  = torch.randn((1, 1, 28, 28))
        self.ac_module = ac.ac_conv(1, 3, kernel_size=3, stride=1,
                        padding=0, groups=1, bias=False, dilation=1)
        
    def test_selectivity(self):
        """
        Simple case: For a filter that is invariant to rotation by 180 degree,
        the method makes sure the covariance measure is the highest for 180, except for the identity.
        """
        self.ac_module.weight = torch.nn.Parameter(self.filter.unsqueeze(0).unsqueeze(0))
        _, score = self.ac_module.response_variation(self.input)
        self.assertEqual(torch.max(score.flatten()[1:],0)[1], 1)

    def test_entropy_tuning(self):
        ## (AC ~ C) equivalence of AC to convolution at a very low entropy and difference at high 
        self.ac_module.entropy = torch.tensor([100.0], dtype = torch.float32)
        self.ac_module.weight = torch.nn.Parameter(self.filter.unsqueeze(0).unsqueeze(0))
        res1 = self.ac_module(self.input)

        conv_module = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        conv_module.weight = torch.nn.Parameter(self.filter.unsqueeze(0).unsqueeze(0)) 
        res2 = conv_module(self.input)

        self.ac_module.entropy = torch.tensor([1.0], dtype = torch.float32)
        res3 = self.ac_module(self.input)
       
        self.assertEqual(torch.floor(torch.norm(res1 - res2)).item(), 0.0)
        self.assertNotEqual(torch.floor(torch.norm(res3 - res2)).item(), 0.0)

