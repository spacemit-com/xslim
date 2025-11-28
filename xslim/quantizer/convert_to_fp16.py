#!/usr/bin/env python3
# Copyright (c) 2025 SpacemiT. All rights reserved.
from typing import Sequence, Set, Tuple, Union

import onnx
import numpy as np
from onnxconverter_common import float16 as convert_float_to_float16
from xslim.logger import logger
import onnx_graphsurgeon as osg
from ..onnx_graph_helper import format_onnx_model

def legalize_fp16_graph(osg_graph : osg.Graph):
    for node in osg_graph.nodes:
        if node.op in {"Resize", "Upsample"}:
            for input_var in node.inputs:
                if isinstance(input_var, osg.Constant) and input_var.dtype == np.float16:
                    input_var.values = input_var.values.astype(np.float32)
    return osg_graph

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
        raise TypeError("type of file_or_model error, {} .vs str or modelproto".format(
            type(file_or_model)))

    model_opt = format_onnx_model(onnx_model, sim_en)

    logger.info("convert onnx model to fp16.")
    try:
        model_fp16 = convert_float_to_float16.convert_float_to_float16(
            model_opt,
            keep_io_types=True,
            disable_shape_infer=False,
            op_block_list=ignore_op_types_list,
            node_block_list=ignore_node_names_list,
        )
    except Exception as e:
        logger.info(f"FP16 Convert Failed!: {e}")
        raise

    osg_graph = osg.import_onnx(model_fp16)
    osg_graph = legalize_fp16_graph(osg_graph)
    model_fp16 = osg.export_onnx(osg_graph)

    model_fp16 = format_onnx_model(model_fp16)

    return model_fp16
