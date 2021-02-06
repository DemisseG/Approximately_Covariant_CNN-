"""
The baseline model exetnds Pytorch's implementation of ResNet. 
"""


from __future__ import absolute_import, division
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy

from ac import ac_conv


TRACK = False           # batch statistics tracker--- affects mainly ineference
INPUT_CHANNEL = 1       # size of the input data channel 

def conv3x3(in_planes, out_planes, stride=1, conv_type='conv', groups=1, dilation=1):
    if conv_type == 'conv':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        return ac_conv.ac_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    conv_type = 'conv'
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=TRACK)
        self.conv2 = conv3x3(width, width, stride, Bottleneck.conv_type, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=TRACK)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=TRACK)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    conv_type = 'conv'

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride, self.conv_type)
        self.bn1 = norm_layer(planes, track_running_stats=TRACK)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, self.conv_type)
        self.bn2 = norm_layer(planes, track_running_stats=TRACK)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

"""
 Basic Resnet model with AC-based training support. 
"""

class ResNetCustom(nn.Module):
    def __init__(self, name, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.name = name
        self.inplanes = 20 if name == 'resnet8' else 64
        self.dilation = 1
        self.conv_type = block.conv_type

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if block.conv_type == 'conv':
            self.conv1 = nn.Conv2d(INPUT_CHANNEL, self.inplanes, kernel_size=3, stride=1, padding=1,
                                  bias=False) 
        else:
            self.conv1 = ac_conv.ac_conv(INPUT_CHANNEL, self.inplanes, kernel_size=3, stride=1, padding=0,
                                  bias=False)  

        self.bn1 = norm_layer(self.inplanes, track_running_stats=TRACK)
        self.relu = nn.ReLU(inplace=True)
        
        if self.name != 'resnet8':
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        else:
            self.layer1 = self._make_layer(block, 20, 2)
            self.layer2 = self._make_layer(block, 30, 1, stride=2,
                                           dilate=replace_stride_with_dilation[0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.name != 'resnet8':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(30 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, ac_conv.ac_conv)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=TRACK),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def manageHyperparameters(self, value: torch.Tensor, param_name: str):
        """
            The method is useful to slowly tune the conv-level enetropy hyperparameter,
            and to swicth betweetn different inference modes.
        """

        assert param_name in ['entropy', 'efficient_inference', 'normalize'], "unknown hyper parameter name"
        def assigner(module, value):
            children = list(module.children())
            for i in range(len(children)):
                if self.name in ['resnet8', 'resnet10', 'resnet18', 'resnet34']:
                    setattr(children[i].conv1, param_name, value)
                    setattr(children[i].conv2, param_name, value)
                else:
                    setattr(children[i].conv2, param_name, value)

        if self.conv_type != 'conv':
            setattr(self.conv1, param_name, value)
            assigner(self.layer1, value)
            assigner(self.layer2, value)
            if self.name != 'resnet8':
                assigner(self.layer3, value)
                assigner(self.layer4, value)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.name != 'resnet8':
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            
        x = self.avgpool(x)      
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def getResNet(conv_type, depth, num_classes, num_channel=1, sym_type=0):
    from ac import utils
    
    global INPUT_CHANNEL 
    INPUT_CHANNEL = num_channel

    ac_conv.domain_sym = utils.AUGMENTED_TRANS_SET[sym_type]
    
    if depth == 8:
        BasicBlock.conv_type = conv_type
        return ResNetCustom('resnet8', BasicBlock, None, num_classes)
    elif depth == 10:
        BasicBlock.conv_type = conv_type
        return ResNetCustom('resnet10', BasicBlock, [1, 1, 1, 1], num_classes)
    elif depth == 18:
        BasicBlock.conv_type = conv_type
        return ResNetCustom('resnet18', BasicBlock, [2, 2, 2, 2], num_classes)
    elif depth == 34:
        BasicBlock.conv_type = conv_type
        return ResNetCustom('resnet34', BasicBlock, [3, 4, 6, 3], num_classes)
    elif depth == 50:
        Bottleneck.conv_type = conv_type
        return ResNetCustom('resnet50', Bottleneck, [3, 4, 6, 3], num_classes)
    elif depth == 101:
        Bottleneck.conv_type = conv_type
        return ResNetCustom('resnet101', Bottleneck, [3, 4, 23, 3], num_classes)
