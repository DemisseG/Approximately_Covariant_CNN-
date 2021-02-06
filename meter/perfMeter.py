from __future__ import absolute_import, division
import sys, os 
from typing import Dict

import torch 
import numpy
import matplotlib
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import matplotlib.pyplot as plt


font = {'family': 'monospace',
        'size' : 10,
        'weight':  'bold'} 

plt.rc('axes', linewidth=1.5)
plt.rc('font', **font)

AXIS_FONT_SIZE = 25


class model_meter():
    
    def __init__(self, args: Dict={'target': None,
                             'pred' : None,
                             'pred_prob': None,
                             'label_names': None,
                            }):
        
        self.target = args['target']
        self.pred = args['pred'] 
        self.pred_prob = args['pred_prob']
        self.lNames = args['label_names']
        self.labelNums = numpy.zeros((1,self.pred_prob.shape[1]))

        # label conditional measures
        self.confusion = numpy.zeros((self.pred_prob.shape[1], self.pred_prob.shape[1]))
        self.sensitivity = numpy.zeros((1,self.pred_prob.shape[1]))     # recall, True postive rate
        self.specifcity = numpy.zeros((1,self.pred_prob.shape[1]))      # True negative rate
        self.precision = numpy.zeros((1,self.pred_prob.shape[1]))
         
        # aggregated performance measures
        self.accuracy = None 
        self.balanced_accuracy = None 
        

    def conditoinal_measure(self):
        # NOTE: The confusion matrix has rows as True values and columns as predictions.
        # FIXME: Include support for different cutoff 

        for k in range(self.pred_prob.shape[1]): 
            pred_val = (self.pred == k).int()
            for j in range(self.pred_prob.shape[1]):
                true_val = (self.target == j).int()
                match = (pred_val & true_val).int()
                self.confusion[j,k] = torch.sum(match).item()
                
            self.labelNums[0,k] = torch.sum((self.target == k).int(), 0).item()
            self.sensitivity[0,k] = float(self.confusion[k,k]) / float(self.labelNums[0,k])
            
            if float(torch.sum(pred_val).item()) != 0: 
                self.precision[0,k] = float(self.confusion[k,k]) / float(torch.sum(pred_val).item())
            
        temp = numpy.sum(numpy.diag(self.confusion))
        for j in range(self.pred_prob.shape[1]):
            self.specifcity[0,j] = float(temp - self.confusion[j,j])/\
                                    float(numpy.sum(self.labelNums, 1) - self.labelNums[0,j])
     
        # aggregated accuracy 
        self.accuracy = float(temp) / float(numpy.sum(self.confusion))
        self.balanced_accuracy = (numpy.mean(self.sensitivity) + numpy.mean(self.specifcity)) / 2.0
    
    def visualze(self, path:str=None):
        
        # plot confusion 
        f1 = plt.figure()
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        
        nlabel = self.pred_prob.shape[1]
        plt.imshow(self.confusion, cmap='plasma') 
        plt.xticks(numpy.arange(nlabel), self.lNames, rotation=90)
        plt.yticks(numpy.arange(nlabel), self.lNames) 
        plt.colorbar()
        plt.grid(False) 
        if path:
            f1.savefig(os.path.join(path, 'Confusion'), bbox_inches='tight', edgecolor='w', format='eps', dpi=500)
        plt.show() 
        plt.close()   

        # plot precision and recall
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
        
        f1, ax = plt.subplots(figsize=(10,5))
        width = 0.2
        ax.bar(numpy.arange(nlabel) - (2*width), self.precision[0,:], width, label='Precision')
        ax.bar(numpy.arange(nlabel) - (width), self.sensitivity[0,:], width, label='Recall')
        ax.legend(prop={'size': 15})
        
        plt.yticks(numpy.array([0.0, 0.3, 0.6, 0.9]))
        plt.xticks(numpy.arange(nlabel) - (width), self.lNames)
        plt.xlabel('Labels', fontsize=AXIS_FONT_SIZE)
        plt.ylabel('PR(%)', fontsize=AXIS_FONT_SIZE)
        if path:
            f1.savefig(os.path.join(path, 'PR'), bbox_inches='tight', format='eps', dpi=500)
        plt.show()
        plt.close()
        
        
  
def plotSequence(data, path:str = None, labels = None):
    # Used to plot error / accuracy progression during training.

    f1, ax = plt.subplots()
    if isinstance(data, list):
        for i in range(len(data)):
            if labels != None:
                ax.plot(numpy.asarray(data[i]), label=labels[i], linewidth=3.0)
            else:
                ax.plot(numpy.asarray(data[i]), linewidth=3.0)
    else:
        if labels != None:
            ax.plot(numpy.asarray(data), linewidth=3.0)
        else:
            ax.plot(numpy.asarray(data), label=labels, linewidth=3.0)

    plt.grid(True)
    plt.legend(prop={'size': 15})
    plt.xlabel('Epoch', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Accuracy', fontsize=AXIS_FONT_SIZE)
    if path:
        f1.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps', dpi=500)
    plt.show()
    plt.close() 
    
    