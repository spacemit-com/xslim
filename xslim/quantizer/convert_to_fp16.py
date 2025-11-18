#!/usr/bin/env python3
# Copyright (c) 2025 SpacemiT. All rights reserved.
from typing import Sequence, Set, Tuple, Union

import onnx
from onnxconverter_common import float16 as convert_float_to_float16
from xslim.logger import logger

from ..onnx_graph_helper import format_onnx_model


def convert_to_fp16_onnx_model(
    file_or_model: Union[str, onnx.ModelProto],
    ignore_op_types_list: Sequence[str],
    ignore_node_names_list: Sequence[str],
    sim_en: bool = True,
):
    if isinstance(file_or_model, onnx.ModelProto):
        onnx_model = file_or_model
    elif isinstance(file_or_model, str):
        onnx_model = onnx.load(file_or_model)
    else:
        raise TypeError("type of file_or_model error, {} .vs str or modelproto".format(type(file_or_model)))

    model_opt = format_onnx_model(onnx_model, sim_en)
    # 转换为FP16精度，保持输入输出为FP32
    logger.info("convert onnx model to fp16.")
    try:
        model_fp16 = convert_float_to_float16.convert_float_to_float16(
            model_opt,
            keep_io_types=True,  # 保持输入输出为FP32
            disable_shape_infer=False,
            op_block_list=ignore_op_types_list,  # 可以指定某些操作不转换
            node_block_list=ignore_node_names_list,  # 可以指定某些节点不转换
        )
    except Exception as e:
        logger.info(f"FP16 Convert Failed!: {e}")
        raise

    model_fp16 = format_onnx_model(model_fp16)

    return model_fp16
