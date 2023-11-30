from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
import time
import math
import numpy as np
from ppq.lib import (
    register_operation_handler,
)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.core import TargetPlatform


def _get_quant_min_max(num_of_bits: int, signed: bool = True):
    if signed:
        return -(2 ** (num_of_bits - 1)), 2 ** (num_of_bits - 1) - 1
    else:
        return 0, 2**num_of_bits - 1


def QuantizeLinear_Forward(op: Operation, values, ctx, **kwargs):
    axis_ = op.attributes.get("axis", 0)
    x, scale, zp = values[:3]
    new_shape = x.dim() * [1]
    if len(new_shape) > axis_:
        new_shape[axis_] = -1
    if scale.numel() > 1:
        scale = scale.reshape(new_shape)
    if zp.numel() > 1:
        zp = zp.reshape(new_shape)
    y = x / scale + zp
    y = torch.round(y)

    quant_min, quant_max = _get_quant_min_max(8)
    if zp.dtype == torch.int8:
        pass
    elif zp.dtype == torch.uint8:
        quant_min, quant_max = _get_quant_min_max(8, False)
    elif zp.dtype == torch.int16:
        quant_min, quant_max = _get_quant_min_max(12)
    elif zp.dtype == torch.int32:
        quant_min, quant_max = _get_quant_min_max(32)
    else:
        raise NotImplementedError(zp.dtype)
    y = torch.clip(y, quant_min, quant_max)
    return y


def DequantizeLinear_Forward(op: Operation, values, ctx, **kwargs):
    axis_ = op.attributes.get("axis", 0)
    x, scale, zp = values[:3]
    new_shape = x.dim() * [1]
    if len(new_shape) > axis_:
        new_shape[axis_] = -1
    if scale.numel() > 1:
        scale = scale.reshape(new_shape)
    if zp.numel() > 1:
        zp = zp.reshape(new_shape)
    y = (x.to(torch.float32) - zp) * scale
    return y


def DynamicQuantizeLinear_Forward(op: Operation, values, ctx, **kwargs):
    min, max = values[0].min(), values[0].max()
    quant_min, quant_max = _get_quant_min_max(8)
    scale = (max - min) / (quant_max - quant_min)
    zp = torch.round(quant_min - min / scale).to(torch.int8)
    y_quant = torch.round(values[0] / scale + zp)
    y_quant = torch.clip(y_quant, quant_min, quant_max)
    return [y_quant, scale, zp]


def LRN_Forward(op: Operation, values, ctx, **kwargs):
    input = values[0]
    size = op.attributes.get("size")
    alpha = op.attributes.get("alpha")
    beta = op.attributes.get("beta")
    k = op.attributes.get("bias")
    output_value = torch.nn.functional.local_response_norm(input, size, alpha, beta, k)
    return [output_value]


def Dropout_Forward(op: Operation, values, ctx, **kwargs):
    return [values[0], values[0]]


register_operation_handler(
    Dropout_Forward,
    operation_type="Dropout",
    platform=TargetPlatform.UNSPECIFIED,
)
register_operation_handler(
    LRN_Forward,
    operation_type="LRN",
    platform=TargetPlatform.UNSPECIFIED,
)
register_operation_handler(
    QuantizeLinear_Forward,
    operation_type="QuantizeLinear",
    platform=TargetPlatform.UNSPECIFIED,
)
register_operation_handler(
    DynamicQuantizeLinear_Forward,
    operation_type="DynamicQuantizeLinear",
    platform=TargetPlatform.UNSPECIFIED,
)
register_operation_handler(
    DequantizeLinear_Forward,
    operation_type="DequantizeLinear",
    platform=TargetPlatform.UNSPECIFIED,
)
