#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import functools
import math
import os
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from typing import Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx_graphsurgeon as osg
from onnxruntime.quantization.quant_utils import \
    quantize_data as ort_quantize_data
from xslim.defs import XQUANT_CONFIG
from xslim.logger import logger

from ..onnx_graph_helper import format_onnx_model


def dynamic_weight_only_quantize(onnx_model, ignore_op_types_list, ignore_op_names_list, quant_bits=8):
    quant_min = -(2 ** (quant_bits - 1))
    quant_max = 2 ** (quant_bits - 1) - 1
    new_nodes = []

    def make_dq_node(weight, scale, zp, axis=0):
        dq_node_index = len(new_nodes)
        weight_tensor = osg.Constant("insert_dq_node_weight_tensor_{}".format(dq_node_index), weight)
        scale_tensor = osg.Constant("insert_dq_node_scale_tensor_{}".format(dq_node_index), scale)
        zp_tensor = osg.Constant("insert_dq_node_zp_tensor_{}".format(dq_node_index), zp)
        out_tensor = osg.Variable("insert_dq_node_out_tensor_{}".format(dq_node_index), np.float32)
        attrs = {}
        attrs["axis"] = axis
        new_node = osg.Node(
            "DequantizeLinear",
            "insert_dq_node_{}".format(dq_node_index),
            attrs,
            inputs=[weight_tensor, scale_tensor, zp_tensor],
            outputs=[out_tensor],
        )
        return new_node

    def make_dynamic_q_node(node, input_idx):
        dyn_q_node_index = len(new_nodes)
        dyn_quantize_node = osg.Node(
            "DynamicQuantizeLinear",
            "insert_dyn_q_node_{}_{}_{}".format(input_idx, node.name, dyn_q_node_index),
            inputs=[node.inputs[input_idx]],
            outputs=[],
        )

        dyn_quantize_node.outputs.append(
            osg.Variable("insert_dyn_q_node_out_{}".format(dyn_quantize_node.name), np.uint8)
        )
        dyn_quantize_node.outputs.append(
            osg.Variable("insert_dyn_q_node_out_scale_{}".format(dyn_quantize_node.name), np.float32)
        )
        dyn_quantize_node.outputs.append(
            osg.Variable("insert_dyn_q_node_out_zp_{}".format(dyn_quantize_node.name), np.uint8)
        )

        dyn_dequantize_node = osg.Node(
            "DequantizeLinear",
            "insert_dyn_dq_node_{}_{}_{}".format(input_idx, node.name, dyn_q_node_index),
            attrs={"axis": 0},
            inputs=dyn_quantize_node.outputs,
            outputs=[],
        )
        dyn_dequantize_node.outputs.append(
            osg.Variable("dyn_deq_node_out_{}".format(dyn_dequantize_node.name), np.float32)
        )
        return dyn_quantize_node, dyn_dequantize_node

    def get_scale_zp(weight_value):
        weight_scale_list = []
        weight_zp_list = []
        weight_quant_list = []
        for i in range(weight_value.shape[0]):
            weight_value_i = weight_value[i]
            # _, _, zero_point, scale, quantized_per_channel_data = ort_quantize_data(
            #     weight_value_i,
            #     onnx.TensorProto.INT8,
            #     True,
            #     False
            # )
            zero_point, scale, quantized_per_channel_data = ort_quantize_data(
                weight_value_i, onnx.TensorProto.INT8, True, False
            )
            weight_scale_list.append(scale)
            weight_zp_list.append(zero_point)
            weight_quant_list.append(quantized_per_channel_data)

        return np.array(weight_scale_list), np.array(weight_zp_list), np.array(weight_quant_list)

    def cosine_error(lhs_value, rhs_value):
        lhs_value = lhs_value.reshape(-1)
        rhs_value = rhs_value.reshape(-1)
        dot_product = np.sum(lhs_value * lhs_value)

        norm_a = np.linalg.norm(lhs_value, axis=-1)
        norm_b = np.linalg.norm(rhs_value, axis=-1)
        epsilon = 1e-10
        denominator = np.maximum(norm_a * norm_b, epsilon)
        similarity = dot_product / denominator
        return similarity

    osg_graph = osg.import_onnx(onnx_model)

    for node in osg_graph.nodes:

        def eval_error(weight_value, weight_q_value, weight_value_scale, weight_value_zp):
            requant_weight_value = (
                weight_q_value.astype(np.float32) - weight_value_zp.reshape(-1, 1)
            ) * weight_value_scale.reshape(-1, 1)
            error = np.sum(np.abs(weight_value - requant_weight_value)) / weight_value.size
            cosine = cosine_error(weight_value, requant_weight_value)
            # print(f"node: [{node.op}]({node.name}), error: {error}, cosine: {cosine}")

        if node.op in {"Conv", "Gemm", "MatMul", "ConvTranspose"}:
            if node.op in ignore_op_types_list:
                continue

            if node.name in ignore_op_names_list:
                continue

            axis = 0
            if isinstance(node.inputs[1], osg.Constant):
                weight_value = node.inputs[1].values
                weight_shape = weight_value.shape

                if np.allclose(
                    weight_value.astype(np.float32),
                    weight_value.astype(np.int32).astype(np.float32),
                    rtol=1e-03,
                    atol=1e-05,
                ):
                    continue

                if node.op == "Conv" or (node.op == "Gemm" and node.attrs.get("transB", 0) == 1):
                    weight_value = weight_value.reshape(weight_shape[0], -1)
                    weight_value_scale, weight_value_zp, quant_weight_value = get_scale_zp(weight_value)
                    # eval_error(weight_value, quant_weight_value, weight_value_scale, weight_value_zp)
                    quant_weight_value = quant_weight_value.reshape(weight_shape)
                elif node.op == "MatMul" or (node.op == "Gemm" and node.attrs.get("transB", 0) == 0):
                    axis = 1
                    permute_weight_value = np.transpose(weight_value, (1, 0))
                    permute_weight_value = weight_value.reshape(permute_weight_value.shape[0], -1)
                    weight_value_scale, weight_value_zp, permute_quant_weight_value = get_scale_zp(permute_weight_value)
                    # eval_error(permute_weight_value, permute_quant_weight_value, weight_value_scale, weight_value_zp)
                    quant_weight_value = np.transpose(permute_quant_weight_value, (1, 0)).reshape(weight_shape)
                elif node.op == "ConvTranspose" and node.attrs.get("group", 1) == 1:
                    axis = 1
                    group = node.attrs.get("group", 1)
                    weight_value = weight_value.reshape(group, weight_shape[0] // group, weight_shape[1], -1)
                    permute_weight_value = np.transpose(weight_value, (0, 2, 1, 3))
                    permute_weight_value = permute_weight_value.reshape(group * weight_shape[1], -1)

                    weight_value_scale, weight_value_zp, permute_quant_weight_value = get_scale_zp(permute_weight_value)
                    permute_quant_weight_value = permute_quant_weight_value.reshape(
                        (group, weight_shape[1], weight_shape[0] // group, -1)
                    )
                    quant_weight_value = np.transpose(permute_quant_weight_value, (0, 2, 1, 3))
                    quant_weight_value = quant_weight_value.reshape(weight_shape)
                else:
                    continue

                new_nodes.append(
                    make_dq_node(
                        quant_weight_value,
                        weight_value_scale.reshape(-1).astype(np.float32),
                        weight_value_zp.reshape(-1).astype(np.int8),
                        axis,
                    )
                )
                node.inputs[1] = new_nodes[-1].outputs[0]

    osg_graph.nodes.extend(new_nodes)
    osg_graph.toposort()
    new_onnx_model = osg.export_onnx(osg_graph)
    return new_onnx_model


def dynamic_quantize_onnx_model(
    file_or_model: Union[str, onnx.ModelProto],
    ignore_op_types_list: Sequence[str],
    ignore_op_names_list: Sequence[str],
    sim_en: bool = True,
):
    from onnxruntime.quantization import QuantizationMode, QuantType
    from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer

    if isinstance(file_or_model, onnx.ModelProto):
        onnx_model = file_or_model
    elif isinstance(file_or_model, str):
        onnx_model = onnx.load(file_or_model)
    else:
        raise TypeError("type of file_or_model error, {} .vs str or modelproto".format(type(file_or_model)))

    onnx_model = format_onnx_model(onnx_model, sim_en)

    dynamic_q_op_types = {"Attention", "LSTM", "MatMul"}

    for ignore_op in ignore_op_types_list:
        if ignore_op in dynamic_q_op_types:
            dynamic_q_op_types.remove(ignore_op)

    nodes_to_exclude = ignore_op_names_list

    extra_options = {"WeightSymmetric": True, "MatMulConstBOnly": True}
    quantizer = ONNXQuantizer(
        onnx_model,
        True,  # per channel
        False,  # reduce_range
        QuantizationMode.IntegerOps,
        False,  # static
        QuantType.QInt8,
        QuantType.QUInt8,
        None,
        [],
        nodes_to_exclude,
        list(dynamic_q_op_types),
        extra_options,
    )

    logger.info("quantize onnx model dynamic...")
    quantizer.quantize_model()
    quantized_model = quantizer.model.model
    quantized_model = dynamic_weight_only_quantize(quantized_model, ignore_op_types_list, ignore_op_names_list)
    quantized_model = format_onnx_model(quantized_model, True)

    quantized_model.producer_name = "xslim"
    export_time = quantized_model.metadata_props.add()
    export_time.key = "xslim_export_time"
    export_time.value = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    xslim_version = quantized_model.metadata_props.add()
    xslim_version.key = "xslim_version"
    xslim_version.value = XQUANT_CONFIG.version

    return quantized_model
