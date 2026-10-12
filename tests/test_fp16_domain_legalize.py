#!/usr/bin/env python3
# Copyright (c) 2025 SpacemiT. All rights reserved.
"""Regression tests for FP16 conversion of models with Microsoft custom operators.

Reproduces the HuggingFace EmbeddingGemma-2 export, which declares
``SimplifiedLayerNormalization`` without an ai.onnx schema and ``RotaryEmbedding`` in the
``com.microsoft`` domain. Before the fix, ``xslim --fp16`` raised
``IndexError: unordered_map::at`` and then produced a model ONNX Runtime rejected with
``Type parameter (T) ... bound to different types``.
"""

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto

from xslim.quantizer.convert_to_fp16 import convert_to_fp16_onnx_model
from xslim.quantizer.fp16_legalize import register_extension_op_schemas


def _make_layernorm_like_model():
    """Small graph with a schema-less SimplifiedLayerNormalization + a Where mixing types."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    W = helper.make_tensor("W", TensorProto.FLOAT, [4], [1.0, 1.0, 1.0, 1.0])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
    neg = helper.make_tensor("neg", TensorProto.FLOAT, [1], [-1.0])
    nodes = [
        helper.make_node("SimplifiedLayerNormalization", ["X", "W"], ["ln"], name="sln", epsilon=1e-5, axis=-1),
        helper.make_node("Greater", ["ln", "zero"], ["cond"], name="gt"),
        helper.make_node("Where", ["cond", "zero", "neg"], ["wm"], name="wh"),
        helper.make_node("Add", ["ln", "wm"], ["Y"], name="add"),
    ]
    graph = helper.make_graph(nodes, "sln_model", [X], [Y], [W, zero, neg])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def test_extension_schema_registered():
    """The schema-less operator gets an extension schema (kept, not renamed)."""
    register_extension_op_schemas()
    schema = onnx.defs.get_schema("SimplifiedLayerNormalization", 21, "")
    assert schema.name == "SimplifiedLayerNormalization"
    assert [i.name for i in schema.inputs] == ["X", "scale"]


def test_convert_model_with_schema_less_operator():
    """Conversion succeeds and the operator keeps its name and default domain."""
    model = _make_layernorm_like_model()
    fp16_model = convert_to_fp16_onnx_model(model, [], [], sim_en=True, target_onnx_opset=24)
    sln = [n for n in fp16_model.graph.node if n.op_type == "SimplifiedLayerNormalization"]
    assert len(sln) == 1
    assert sln[0].domain == ""
    # The graph passes strict type checking (no fp32/fp16 mismatch left).
    probe = onnx.ModelProto()
    probe.CopyFrom(fp16_model)
    del probe.graph.value_info[:]
    onnx.shape_inference.infer_shapes(probe, strict_mode=True, check_type=True)


def test_convert_keeps_numerics():
    """The fp16 model stays numerically close to fp32 on the CPU execution provider."""
    ort = __import__("pytest").importorskip("onnxruntime")
    model = _make_layernorm_like_model()
    fp16_model = convert_to_fp16_onnx_model(model, [], [], sim_en=True, target_onnx_opset=24)
    feed = {"X": np.random.default_rng(0).standard_normal((1, 4)).astype(np.float32)}
    ref = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    got = ort.InferenceSession(fp16_model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    a, b = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    assert cosine > 0.999
