from typing import Iterable, List, Set, Union, Dict, Callable, Tuple, Sequence
import time
import math
import numpy as np
from ppq.lib import (
    register_operation_handler,
)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.core import TargetPlatform
from ppq.executor.op.torch import Conv_forward
from ppq.executor.op.torch.base import *
from ppq.utils import process_attribute

import torch
import torch.nn.functional as F


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


def Conv_Forward_Wrapper(op: Operation, values, ctx, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    groups = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="group", default=1)
    dilation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="dilations", default=1)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="strides", default=1)
    x, w = values[:2]
    b = values[2] if len(values) > 2 else None
    ndim = w.ndim

    if ndim == 4:
        process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:])
        onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="pads", default=0)
        # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            # torch does not support padding contains 4 value, there is a fix of it.
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0

        output = F.conv2d(input=x, weight=w, bias=b, groups=groups, padding=onnx_pads, dilation=dilation, stride=stride)
    else:
        output = Conv_forward(op, values, ctx, **kwargs)
    return output


def Resize_forward_Wrapper(
    op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs
) -> torch.Tensor:
    value = values[0]
    # Not used roi
    # roi  = input_value[1] if len(input_value) > 1 else None
    if len(values) > 2 and values[2] is not None:
        scale_factor = values[2].cpu()
    else:
        scale_factor = None
    size = values[-1].cpu().tolist() if (len(values) == 4 and values[-1] != None) else None
    mode = op.attributes.get("mode", "nearest")
    if mode == "cubic":
        mode = "bicubic"
    # onnx resize 'linear' model include N-linear interpolate for N-D tensor
    linear_mode_map = {1: "linear", 2: "bilinear", 3: "trilinear"}

    # If 'size' is specified, then set scales to empty data (zero shape) in this operator's input list.
    if size is None or len(size) == 0:
        size = None
        if scale_factor.numel() == 1:
            scale_factor = scale_factor.item()
        else:
            scale_factor = scale_factor.tolist()
            if len(scale_factor) == 2:
                # 大家相安无事，和平共处
                pass
            elif len(scale_factor) == 4:
                if scale_factor[:2] != [1, 1]:
                    raise NotImplementedError(
                        "Can not resize your image with current op, "
                        "cause 4-dimension resize is not implemented with pytorch."
                    )
                scale_factor = scale_factor[2:]
            else:
                raise NotImplementedError(
                    "Can not resize your image with current op, "
                    f"cause {len(scale_factor)}-dimension resize is not implemented with pytorch."
                )
    else:
        # the sizes in onnx is 4-D while in pytorch is 2-D
        # check the dim.0 & dim.1 is equal, then remain dim.2 and dim.3
        scale_factor = None
        assert size[:2] == list(value.shape[:2])
        size = size[2:]
        mode = linear_mode_map[len(size)] if mode == "linear" else mode

        if mode == "cubic":
            assert len(size[2:]) == 2
            mode = "bicubic"

    # PATCH 2022.04.22
    # ONNX DO NOT HAVE BILINEAR MODE, FOR 4D INPUT, WE OVERRIDE MODE TO BILINEAR
    if len(value.shape) == 4 and mode == "linear":
        mode = "bilinear"

    trans_mode = op.attributes.get("coordinate_transformation_mode", "half_pixel")
    if trans_mode == "align_corners":
        output = F.interpolate(value, size, scale_factor, mode, align_corners=True)
    else:
        output = F.interpolate(value, size, scale_factor, mode)
    return output


def register_operation_handler_merge(
    handler: Callable,
    operation_type: str,
    platforms: Sequence[TargetPlatform] = [
        TargetPlatform.UNSPECIFIED,
        TargetPlatform.ONNXRUNTIME,
        TargetPlatform.INT8,
        TargetPlatform.FP32,
    ],
):
    for plat in platforms:
        register_operation_handler(
            handler,
            operation_type,
            platform=plat,
        )


register_operation_handler_merge(
    Dropout_Forward,
    operation_type="Dropout",
)
register_operation_handler_merge(
    LRN_Forward,
    operation_type="LRN",
)
register_operation_handler_merge(
    QuantizeLinear_Forward,
    operation_type="QuantizeLinear",
)
register_operation_handler_merge(
    DynamicQuantizeLinear_Forward,
    operation_type="DynamicQuantizeLinear",
)
register_operation_handler_merge(
    DequantizeLinear_Forward,
    operation_type="DequantizeLinear",
)
register_operation_handler_merge(
    Conv_Forward_Wrapper,
    operation_type="Conv",
)
register_operation_handler_merge(
    Resize_forward_Wrapper,
    operation_type="Resize",
)
