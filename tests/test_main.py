from __future__ import absolute_import, division
import sys 
import os 
import unittest

import torch 
import trainlogic as tl 

class main_test(unittest.TestCase):
    def test_prepare_data(self):
        tl.config()
        tl.args.datasets = 'rotmnist'
        tl.args.datapath = os.getcwd() + '/data'
        tl.prepare_data()

        self.assertTrue(tl.trainloader, torch.utils.data.DataLoader)
        self.assertTrue(tl.testloader, torch.utils.data.DataLoader)

        for batch in tl.trainloader:
            self.assertLessEqual(len(batch), tl.args.batch)

    def test_accuracy(self):
        # simple case
        target = torch.tensor([1, 1, 1]) 
        pred =  torch.tensor([[0.1, 0.5, 0.3, 0.1], [0.2, 0.1, 0.1, 0.5], [0.4, 0.3, 0.1,0.2]])
        res = tl.accuracy(pred, target, (1,2))
        self.assertEqual(res[0] / target.shape[0], 1.0 / 3.0)
        self.assertEqual(res[1] / target.shape[0], 2.0 / 3.0 )
        
        
   



