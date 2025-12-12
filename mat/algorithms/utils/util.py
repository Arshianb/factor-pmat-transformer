import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def reindex_tensor(input_tensor, index_tensor):
    # input_tensor = input_tensor.to(index_tensor.device)
    index_tensor = index_tensor.to(input_tensor.device, dtype=torch.long)
    expanded_indices = index_tensor.unsqueeze(-1).expand(-1, -1, input_tensor.size(-1))
    reindexed_tensor = torch.gather(input_tensor, dim=1, index=expanded_indices)
    return reindexed_tensor

def cal_restore_index(index_tensor):
    index_range = torch.arange(index_tensor.size(1)).repeat(index_tensor.size(0), 1).to(index_tensor.device)
    argsort_indices = index_tensor.argsort(dim=1)
    restore_index = index_range.gather(1, argsort_indices)
    return restore_index
