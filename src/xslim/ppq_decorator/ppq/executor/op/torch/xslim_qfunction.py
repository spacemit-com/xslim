import operator
import os
from functools import reduce
from typing import List

import torch
import torch.nn.functional as F
from torch import _VF


def ceil_div(a, b):
    return (a + b - 1) // b

def quantize_channel_blockwise(data, block=16, bit_size=4):
    M, K = data.shape
    align_k = ceil_div(K, block) * block
    pad_k = align_k - K
    if pad_k > 0:
        data = F.pad(data, [0, pad_k, 0, 0], "constant", 0.0)
    quant_range = (2 ** bit_size - 1) / 2
    quant_min = -2 ** (bit_size - 1)
    quant_max = 2 ** (bit_size - 1) - 1

    data_pack = data.reshape(M, align_k // block, block)
    data_pack = data_pack.permute(1, 0, 2).reshape(align_k // block, -1)
    data_pack_max = torch.max(torch.abs(data_pack), dim=1)[0]
    data_pack_scale = (data_pack_max / quant_range).reshape(-1, 1)
    data_pack_quant = torch.clip(torch.round(data_pack / data_pack_scale), quant_min, quant_max)
    data_pack_quant = data_pack_quant * data_pack_scale
    data_pack_quant = data_pack_quant.reshape(align_k // block, M, block).permute(1, 0, 2).reshape(M, align_k)
    if pad_k > 0:
        data_pack_quant = data_pack_quant[:, :K]
    return data_pack_quant

def quantize_tensorwise(data, block=16, bit_size=4):
    quant_range = (2 ** bit_size - 1) / 2
    quant_min = -2 ** (bit_size - 1)
    quant_max = 2 ** (bit_size - 1) - 1
    data_pack_max = torch.max(torch.abs(data))
    data_pack_scale = (data_pack_max / quant_range)
    data_pack_quant = torch.clip(torch.round(data / data_pack_scale), quant_min, quant_max)
    data_pack_quant = data_pack_quant * data_pack_scale
    return data_pack_quant

def quantize_blockwise(data, block=16, bit_size=4):
    # M, K
    M, K = data.shape
    align_m = ceil_div(M, block) * block
    align_k = ceil_div(K, block) * block
    pad_m = align_m - M
    pad_k = align_k - K
    if pad_m > 0 or pad_k > 0:
        data = F.pad(data, [0, pad_k, 0, pad_m], "constant", 0.0)
    quant_range = (2 ** bit_size - 1) / 2
    quant_min = -2 ** (bit_size - 1)
    quant_max = 2 ** (bit_size - 1) - 1
    data_pack = data.reshape(align_m // block, block, align_k // block, block)
    data_pack = data_pack.permute(0, 2, 1, 3).reshape(-1, block * block)
    data_pack_max = torch.max(torch.abs(data_pack), dim=1)[0]
    data_pack_scale = (data_pack_max / quant_range).reshape(-1, 1)
    data_pack_quant = torch.clip(torch.round(data_pack / data_pack_scale), quant_min, quant_max)
    data_pack_quant = data_pack_quant * data_pack_scale
    data_pack_quant = data_pack_quant.reshape(align_m // block, align_k // block, block, block).permute(0, 2, 1, 3).reshape(align_m, align_k)
    if pad_m > 0 or pad_k > 0:
        data_pack_quant = data_pack_quant[:M, :K]
    return data_pack_quant

def quantize_groupwise(data, group=16, bit_size=4):
    # M, K
    M, K = data.shape
    align_k = ceil_div(data.shape[1], group) * group
    pad_k = align_k - data.shape[1]
    if pad_k > 0:
        data = F.pad(data, [0, pad_k, 0, 0], "constant", 0.0)
    data_pack = data.reshape(-1, group)
    quant_range = (2 ** bit_size - 1) / 2
    quant_min = -2 ** (bit_size - 1)
    quant_max = 2 ** (bit_size - 1) - 1
    data_pack_max = torch.max(torch.abs(data_pack), dim=1)[0]
    data_pack_scale = (data_pack_max / quant_range).reshape(-1, 1)
    data_pack_quant = torch.clip(torch.round(data_pack / data_pack_scale), quant_min, quant_max)
    data_pack_quant = (data_pack_quant * data_pack_scale).reshape(-1, align_k)
    if pad_k > 0:
        data_pack_quant = data_pack_quant[:, :K]
    return data_pack_quant
