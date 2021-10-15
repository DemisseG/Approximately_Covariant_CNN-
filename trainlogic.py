from __future__ import absolute_import, division
import os
import sys
import time
import argparse
import datetime
import pickle5 as pickle
from typing import Tuple, List

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from data import dataloaders
from data import data
from ac import utils
from models import resnet_backbone as models
import config

use_cuda = torch.cuda.is_available()

def parse():
    # TODO: convert to config file.
    parser = argparse.ArgumentParser(description="==> Training and testing script for Approximately covariant (AC) convolutional network <==")
    parser.add_argument('-lr', '--l_rate', metavar='\b', default=0.1, type=float, help='  Learning_rate')
    parser.add_argument('-bs', '--batch', metavar='\b', default=128, type=int, help='  Batch size')
    parser.add_argument('-opt', '--optim', metavar='\b', default="adam", help='optim = [adam / sgd]  ... ')
    parser.add_argument('-ep', '--epoch', metavar='\b', default=100, type=int, help='Epoch training')
    parser.add_argument('-d', '--depth', metavar='\b', default=18, type=int, help='Depth = [ 8 / 10 / 18 / 36 / 50 / 101]')
    parser.add_argument('-ty', '--trainType', metavar='\b',  default=0, type=int, help='Training type for the discriminative experiment.\
         ty = [0 = train and test on rot-MNIST, 1 = train on rot-MNIST and test on MNIST]')
    parser.add_argument('-im', '--imode', metavar='\b',  default=0, type=int, help='Inference mode = [0 (exact inference) / 1 (efficient inference)]:')
    parser.add_argument('-ct', '--convType', metavar='\b' , default='conv', help='Convolution type = [\'conv\'  \'covar\']')
    parser.add_argument('-ss', '--symset', metavar='\b', default=0, type=int, help='Supported types of symmetry sets. \
         ss = [0 = rotation, 1 = densly sampled rotation, 2 = horizontal flip, 3 = scaling, 4 = composition of scaling and reflection]' )
    parser.add_argument('-n', '--normalize', metavar='\b', default=1, type=int, help='Normalize exponetiated covariace measures')
    parser.add_argument('-ds', '--datasets', metavar='\b', default='cifar10', type=str, help='dataset = [cifar10 / cifar100 / rotmnist / imagenet]')
    parser.add_argument('-dp', '--datapath', metavar='\b', type=str, help='directory path to the dataset location')
    parser.add_argument('-fp', '--filepath', metavar='\b', default=os.getcwd(), type=str, help='File path for saving checkpoints and experimental results. DEFAULT =  current working directory.')
    parser.add_argument('-fx', '--fixed_init', metavar='\b', default=None, type=str, help='Path to fixed inital parameters to start the model training. DEFAULT = current working directory.')
    config.args = parser.parse_args()


def prepare_data():
    config.args.datapath = os.getcwd() + '/data' if config.args.datapath == None else config.args.datapath

    if config.args.datasets == 'cifar10' or config.args.datasets == 'cifar100':
        trainset, testset = dataloaders.CIFARX(config.args.datasets, config.args.datapath)
        config.num_classes = 10 if config.args.datasets == 'cifar10' else 100
        config.INPUT_CHANNEL = 3
    elif config.args.datasets == 'rotmnist':
        trainset, testset = dataloaders.ROTMNIST(config.args.datapath, config.args.trainType)
        config.num_classes = 10
        config.INPUT_CHANNEL = 1
    elif config.args.datasets == 'imagenet':
        trainset, testset = dataloaders.IMAGENET(config.args.datapath)
        config.num_classes = 1000
        config.INPUT_CHANNEL = 3
    else:
        print('Error : Dataset should be either [ cifar10 / cifar100 / rotmnist / imagenet ]')
        sys.exit(0)
    config.trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.args.batch, shuffle=True, num_workers=4)
    config.testloader = torch.utils.data.DataLoader(testset, batch_size=config.args.batch, shuffle=False, num_workers=4)


def load_network():
    assert config.args.depth in [8, 10, 18, 34, 50, 101], 'Error: Choose depth as [8, 10, 18, 34, 50, 101]'
    net = models.getResNet(config.args.convType, 
                        config.args.depth, 
                        config.num_classes,
                        config.INPUT_CHANNEL,
                        config.args.symset)
    return net


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int,int] = (1,)) -> List[torch.Tensor]:
    # Returns accuracy matching counts at different cutoff.
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res += [correct[:k].flatten().float().sum(0)]
    return res


def manageModelParameters(net: models.ResNetCustom , parameter: torch.Tensor, name: str):
    if use_cuda:
        net.module.manageHyperparameters(parameter.cuda(), name)
    else:
        net.manageHyperparameters(parameter, name)


def train(net: models.ResNetCustom, 
         current_epoch: int, 
         criterion: torch.nn.modules.loss.CrossEntropyLoss):

    net.train()
    manageModelParameters(net, torch.tensor(0), 'efficient_inference')
    
    train_loss = 0
    prec1, prec5, total = 0, 0, 0
    assert config.args.optim in ['sgd', 'adam'], "Error. System only support [sgd / adam] optimizers"

    if config.args.optim == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=utils.scheduled_learning(config.args.l_rate, current_epoch, config.args.epoch), momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optimizer = optim.Adam(net.parameters(), lr=config.args.l_rate, amsgrad=True)
    
    for _, (inputs, targets) in enumerate(config.trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        res = accuracy(outputs,targets,topk=(1,5))
        prec1 += res[0]
        prec5 += res[1]

    prec1 = 100.*prec1/total
    prec5 = 100.*prec5/total
    config.report['Progress']['train_loss'] += [train_loss]
    config.report['Progress']['train_acc'] += [[prec1, prec5]]

    if config.args.imode:
        manageModelParameters(net, utils.squeeze_entropy(current_epoch), 'entropy')

    print('\r')
    if config.args.convType == 'covar':
        print('| Training epoch %d, entropy scale =%.4f' %(current_epoch, net.module.conv1.entropy.item()))
    print('| Training epoch %3d\t\tLoss: %.4f Acc@1: %.3f%%' %(current_epoch, loss.item(), prec1))
    print('| Training epoch %3d\t\tLoss: %.4f Acc@5: %.3f%%' %(current_epoch, loss.item(), prec5))


def test(net: models.ResNetCustom, 
        current_epoch:int, 
        criterion: torch.nn.modules.loss.CrossEntropyLoss, 
        final: bool=False):

    if config.args.imode:
        manageModelParameters(net, torch.tensor(1), 'efficient_inference')

    net.eval()

    prec1, prec5, test_loss = 0, 0, 0
    total = 0
    OutpredProb, Outpred, Target = [], [], []

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(config.testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total += targets.size(0)

            test_loss += loss.item()
            res = accuracy(outputs,targets,topk=(1,5))
            prec1 += res[0]
            prec5 += res[1]

            if final:
                _, predicted = torch.max(outputs.data, 1)
                OutpredProb += [outputs]
                Outpred += [predicted]
                Target += [targets]

    prec1 = 100.*prec1/total
    prec5 = 100.*prec5/total
    config.report['Progress']['test_loss'] += [test_loss]
    config.report['Progress']['test_acc'] += [[prec1, prec5]]

    print("\n| Validation Epoch %d\t\tLoss: %.4f Acc@1: %.2f%%" %(current_epoch, loss.item(), prec1))
    print("| Validation Epoch %d\t\tLoss: %.4f Acc@5: %.2f%%" %(current_epoch, loss.item(), prec5))

    if prec1 > config.best_acc:
        print('\n| Best accuracy! So far.')
        print('\tTop1 = %.2f%%' %(prec1))
        print('\tTop5 = %.2f%%' %(prec5))
        config.best_acc = prec1

    if (current_epoch % 20  == 0):
        try:
            state_dict = net.module.state_dict()
        except AttributeError:
            state_dict = net.state_dict()
        state = {
                'acc':prec1,
                'epoch':current_epoch,
                'lr': config.args.l_rate
        }
        # save checkpoint every 20 epoch
        save_point = config.args.filepath + os.sep + 'checkpoint_' +  str(current_epoch) + '_'
        torch.save(state_dict, save_point +'.pt')
        hyper_point = config.args.filepath + os.sep + 'checkpoint_metadata_' + str(current_epoch) + '_.p'
        with open(hyper_point, 'wb') as f1:
            pickle.dump(state, f1)

    if final:
        config.report['Final']['output_pre'] = torch.cat(Outpred, 0).cpu()
        config.report['Final']['pred_prob'] = torch.cat(OutpredProb, 0).cpu()
        config.report['Final']['target'] = torch.cat(Target, 0).cpu()

def main():
    # training and testing logic
    start_epoch = 0
    criterion = nn.CrossEntropyLoss()
    net = load_network()

    if config.args.fixed_init:
        config.args.fixed_init += '/init.pt'
        if not os.path.exists(config.args.fixed_init):
            torch.save(net.state_dict(), config.args.fixed_init)
            print("\n Saving current inital state for later ....")
        else:
            try:
                print("\n Loading initalization ... ")
                net.load_state_dict(torch.load(config.args.fixed_init))
            except:
                print("\n Error: There is a mismatch between model specification and saved inital parameters! ")
                raise SystemExit

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    if not config.args.normalize:
        manageModelParameters(net, torch.tensor(0), 'normalize')
       
    for e in range(start_epoch, start_epoch+config.args.epoch):
        train(net, e, criterion)
        test(net, e, criterion, (e == config.args.epoch-1))
 
    with open(config.args.filepath + '/Result_report_on_' + config.args.datasets +'.p', 'wb') as f1:
        pickle.dump(config.report, f1)

    print('\n: Best model accuracy for testing')
    print('* Test results : Acc@1 = %.2f%%' %(config.best_acc))

    if config.PERFORMANCE_DETAILS:
        final_performance() 


def verbose():
    print('\n [ Training model ]')
    print(' ================== ')
    print('| Training Epochs = ' + str(config.args.epoch))
    print('| Initial Learning Rate = ' + str(config.args.l_rate))
    print('| Batch size = ' + str(config.args.batch))
    print('| Optimizer = ' + str(config.args.optim))
    print('| Filter type = ' + str(config.args.convType))
    if config.args.convType == 'covar':
        print('| Symmetry set = ' + config.AUGMENTED_TRANS_SET[config.args.symset])
        print('| Normalization of covariance measure = ' + ('YES' if config.args.normalize else 'NO'))
        print('| Inference Mode = ' + ('Efficient / Approximate' if config.args.imode else 'Non-efficient / Exact'))
    print(' ================== \n ')



def final_performance():
    import meter.perfMeter as pm 
    import numpy 
    import warnings

    with open(config.args.filepath + '/Result_report_on_' + config.args.datasets +'.p', 'rb') as f1:
        report = pickle.load(f1)

    if config.args.datasets == 'rotmnist':    
        label_set = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine')
    elif config.args.datasets == 'cifar10':
        label_set = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        warnings.warn('Label names is assigned to NONE! Figures will display numbers instead.')
        label_set = None 


    perf_args={'target': report['Final']['target'],
    'pred' : report['Final']['output_pre'].cpu(),
    'pred_prob': report['Final']['pred_prob'].cpu(),
    'label_names': label_set,
    }
    perf_train = numpy.asarray(report['Progress']['train_acc'])
    perf_test = numpy.asarray(report['Progress']['test_acc'])
    data = [perf_train[:,0], perf_test[:,0]]
    labels = ['Precision@1_train', 'Precision@1_test']

    meter = pm.model_meter(perf_args) 
    meter.conditoinal_measure()

    print('\n [ Model performance summary ]')
    print(' ============================= ')
    print('| Mean model accuracy: ', meter.accuracy)
    print('| Mean precision: ', numpy.mean(meter.precision))
    print('| Mean recall: ', numpy.mean(meter.sensitivity))

    meter.visualze()
    pm.plotSequence(data, labels=labels)
