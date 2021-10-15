from __future__ import absolute_import, division
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from ac import utils
import config 

domain_sym = config.AUGMENTED_TRANS_SET[0]

class ac_conv(nn.Conv2d):
    _pd = 2
    def __init__(self,*args, **kwargs):
        super(ac_conv, self).__init__(*args, **kwargs)

        self.register_buffer('efficient_inference', torch.tensor([0.0], dtype=torch.float32), persistent=False)
        self.register_buffer('entropy', torch.tensor([1.0],dtype=torch.float32), persistent=False)
        self.register_buffer('normalize', torch.tensor([1.0], dtype=torch.float32), persistent=False)
        self._generate_trans()

    def _generate_trans(self):
        
        trans, trans_inv = utils.domain_trans(domain_sym)
        self.register_buffer('domain_trans', torch.stack(trans, 0) if len(trans) > 0 else None, persistent=False)
        self.register_buffer('domain_trans_inv', torch.stack(trans_inv, 0) if len(trans_inv) > 0 else None, persistent=False)
        
    def _apply_conv(self, data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        sw = ((self.weight.shape[-1]) // 2) + ac_conv._pd
        return F.conv2d(data, weight, self.bias, self.stride, sw, self.dilation, self.groups)

    def _covar_measure(self, orbit1: torch.Tensor, orbit2: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(torch.flatten(orbit1, -2), torch.flatten(orbit2, -2), -1)

    def stable_response(self, data: torch.Tensor) -> torch.Tensor:
        intersections, score = self.response_variation(data)
        score = torch.softmax(self.entropy * score, 0) if self.normalize else torch.exp(self.entropy * score)
        score_ext = torch.unsqueeze(torch.unsqueeze(score, -1), -1)
        return torch.sum(torch.mul(intersections, score_ext), 0)

    def response_variation(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        score, intersections = [], []
        i = 0
        while self.domain_trans is not None and i < len(self.domain_trans):
            response2transforms = self._apply_conv(self.domain_transform(data, self.domain_trans[i]), self.pdstream(self.weight))
            response2originals  = self.domain_transform(self._apply_conv(data, self.pdstream(self.weight)), self.domain_trans[i])
            score.append(self._covar_measure(response2transforms, response2originals))
            intersections.append(self._apply_conv(data, self.domain_transform(self.pdstream(self.weight), self.domain_trans_inv[i])))
            i += 1
        return torch.stack(intersections, 0), torch.stack(score, 0)

    def domain_transform(self, data: torch.Tensor, trans: torch.Tensor):
        g2 = trans.expand(data.shape[0], -1, -1)
        Res = F.affine_grid(g2, data.size())
        return F.grid_sample(data, Res)  # interpolation with bilinear method

    def pdstream(self, input: torch.Tensor) -> torch.Tensor:
        sw = ac_conv._pd
        return F.pad(input, pad=(sw,sw,sw,sw))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.efficient_inference:
            return F.conv2d(data, self.weight, self.bias, self.stride, 1, self.dilation, self.groups)
        else:
            return self.stable_response(data)


"""
 Extended AC: derived class from ac_conv
"""