#!/usr/bin/env python3
# Copyright (c) 2025 SpacemiT. All rights reserved.
from typing import Sequence, Union

import onnx
import numpy as np
from onnxconverter_common import float16 as convert_float_to_float16
from xslim.logger import logger
import onnx_graphsurgeon as osg
from ..onnx_graph_helper import format_onnx_model

    # Compatibility fallback blocked ops for onnxconverter-common releases without
    # DEFAULT_OP_BLOCK_LIST.
    # NOTE:
    #   This list is intended to mirror `onnxconverter_common.float16.DEFAULT_OP_BLOCK_LIST`
    #   from the onnx/onnxconverter-common project and is used only when that attribute
    #   is not available (older onnxconverter-common versions).
    #   Upstream source (for reference):
    #   https://github.com/onnx/onnxconverter-common/blob/main/onnxconverter_common/float16.py
    #   If you update this list, please compare against the upstream definition above and
    #   keep the two in sync where appropriate.
    LEGACY_DEFAULT_OP_BLOCK_LIST = {
        "ArrayFeatureExtractor",
        "Binarizer",
        "CastMap",
        "CategoryMapper",
        "CumSum",
        "DictVectorizer",
        "FeatureVectorizer",
        "Imputer",
        "LabelEncoder",
        "LinearClassifier",
        "LinearRegressor",
        "Max",
        "Min",
        "NonMaxSuppression",
        "Normalizer",
        "OneHotEncoder",
        "RandomUniformLike",
        "Range",
        "Resize",
        "RoiAlign",
        "SVMClassifier",
        "SVMRegressor",
        "Scaler",
        "TopK",
        "TreeEnsembleClassifier",
        "TreeEnsembleRegressor",
        "Upsample",
        "ZipMap",
    }


def get_default_fp16_block_list():
    """Return the upstream FP16 blocked-op list or the legacy compatibility fallback."""
    return set(getattr(convert_float_to_float16, "DEFAULT_OP_BLOCK_LIST", LEGACY_DEFAULT_OP_BLOCK_LIST))


def legalize_fp16_graph(osg_graph: osg.Graph):
    for node in osg_graph.nodes:
        if node.op in {"Resize", "Upsample"}:
            for input_var in node.inputs:
                if isinstance(input_var, osg.Constant) and input_var.dtype == np.float16:
                    input_var.values = input_var.values.astype(np.float32)
        elif node.op in {"Cast"}:
            to_dtype = onnx.helper.tensor_dtype_to_np_dtype(node.attrs["to"])
            if node.outputs[0].dtype != to_dtype:
                node.attrs["to"] = onnx.helper.np_dtype_to_tensor_dtype(
                    node.outputs[0].dtype)
        elif node.op in {"Equal", "NotEqual", "Greater", "Less", "GreaterEqual", "LessEqual", "Add", "Sub", "Mul", "Div"}:
            if node.inputs[0].dtype == np.float16 and node.inputs[1].dtype != np.float16:
                if isinstance(node.inputs[1], osg.Constant):
                    node.inputs[1].values = node.inputs[1].values.astype(
                        np.float16)
                else:
                    raise RuntimeError("Unsupported op {} with fp16 inputs".format(node.op))
            elif node.inputs[0].dtype != np.float16 and node.inputs[1].dtype == np.float16:
                if isinstance(node.inputs[0], osg.Constant):
                    node.inputs[0].values = node.inputs[0].values.astype(
                        np.float16)
                else:
                    raise RuntimeError("Unsupported op {} with fp16 inputs".format(node.op))
        elif node.op in {"Range"}:
            remove_var = []
            add_var = []
            remove_idx = []
            for input_var in node.inputs:
                if isinstance(input_var, osg.Constant) and input_var.dtype == np.float16:
                    new_var = osg.Constant("{}_to_fp32".format(
                        input_var.name), input_var.values.astype(np.float32))
                    remove_idx.append(node.inputs.index(input_var))
                    remove_var.append(input_var)
                    add_var.append(new_var)

            for r_var, add_var, r_idx in zip(remove_var, add_var, remove_idx):
                node.inputs[r_idx] = add_var

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

    default_ignore_op_types = get_default_fp16_block_list()

    try:
        model_fp16 = convert_float_to_float16.convert_float_to_float16(
            model_opt,
            keep_io_types=True,
            disable_shape_infer=False,
            op_block_list=sorted(default_ignore_op_types.union(ignore_op_types_list or [])),
            node_block_list=ignore_node_names_list,
        )
    except Exception as e:
        logger.info(f"FP16 Convert Failed!: {e}")
        raise

    try:
        model_fp16 = onnx.shape_inference.infer_shapes(model_fp16, data_prop=True)
    except Exception as e:
        logger.warning("shape_inference error with {}, skipped".format(e))

    osg_graph = osg.import_onnx(model_fp16)
    osg_graph = legalize_fp16_graph(osg_graph)
    model_fp16 = osg.export_onnx(osg_graph)

    model_fp16 = format_onnx_model(model_fp16)

    return model_fp16
