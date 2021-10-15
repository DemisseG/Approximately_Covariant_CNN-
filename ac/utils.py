from __future__ import absolute_import, division
from typing import Tuple, List

import math
import numpy
import torch


def scheduled_learning(init: float, current_epoch: int, total_epoch: int) -> int:
    scale = lambda x, y :x * math.pow(0.2, y)
    if(current_epoch > int(total_epoch*0.8)):
        return scale(init, 3)
    elif(current_epoch > int(total_epoch*0.6)):
        return scale(init, 2)
    elif(current_epoch > int(total_epoch*0.3)):
        return scale(init, 1)
    else:
        return init


def squeeze_entropy(epoch: int) -> torch.Tensor:
    if epoch < 25:
        return torch.tensor(0.1, dtype=torch.float32)
    elif epoch < 50 and epoch >= 25:
        return torch.tensor(1.0, dtype=torch.float32) 
    elif epoch < 75 and epoch >= 50:
        return torch.tensor(10.0, dtype=torch.float32)
    elif epoch < 100 and epoch >= 75:
        return torch.tensor(100.0, dtype=torch.float32)


def rotation(period: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # returns counterclockwise rotation with mod(\theta, period * math.pi) = 0
    trans, i_trans = [], []
    for j in numpy.arange(0.0, 2.0, period): 
        angle=torch.tensor(math.pi) * j
        temp = torch.tensor([[torch.cos(angle), -1 * torch.sin(angle)],
                             [torch.sin(angle), torch.cos(angle)]])
        i_trans.append(torch.cat([temp.t(), torch.zeros(2,1)], 1))
        trans.append(torch.cat([temp, torch.zeros(2,1)], 1))
    return trans, i_trans


def scaling(min_s: float=0.75, max_s: float=2.0, nums: int=4) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # returns unfirom scaling
    res, Ires = [], []
    steps = float(max_s - min_s) / float(nums)
    scales = list(numpy.arange(min_s, max_s, steps))
    for i in scales:
        trans = torch.tensor([[1.0 / float(i), 0.0, 0.0],[0.0, 1.0 / float(i), 0.0]], dtype=torch.float32)
        itrans = torch.tensor([[float(i), 0.0, 0.0],[0.0,float(i), 0.0]], dtype=torch.float32)
        res += [trans]
        Ires += [itrans]

    return res, Ires


def reflection() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    ref = torch.tensor([[-1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=torch.float32)
    return [ref], [ref]


def scale_ref() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # Non-separable transformation composition of scaling and reflection.
    scale, i_scale = scaling()
    ref, _ = reflection()
    scale_ref, i_scale_ref = scale , i_scale
    
    for i in range(len(scale)):
        ## non-commutative composition
        scale_ref[i][:,:2] = torch.mm(ref[0][:,:2], scale[i][:,:2])
        i_scale_ref[i][:,:2] = torch.mm(i_scale[i][:,:2], ref[0][:,:2])

    return scale_ref, i_scale_ref


def domain_trans(tra_type: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    import config 
    
    def get_trans(key_id):
        if key_id == 'rot':
            t, it =  rotation(0.5)
        elif key_id == 'rot_ext':
            t, it = rotation(0.25)
        elif key_id == 'ref':
            t, it =  reflection()
        elif key_id == 'scale':
            t, it = scaling()
        elif key_id == 'scale_ref':
            t, it = scale_ref()
        return t, it

    assert tra_type in config.AUGMENTED_TRANS_SET, "Error: unkown transformation!" 
    return get_trans(tra_type)
