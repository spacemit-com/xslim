"""Regression tests for the local onnxslim wrapper."""

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import onnx
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnx import TensorProto, helper
from onnxconverter_common import float16 as convert_float_to_float16


def _load_onnxslim_pass_module():
    repo_root = os.path.join(os.path.dirname(__file__), "..", "xslim")

    package = types.ModuleType("xslim")
    package.__path__ = [repo_root]
    sys.modules["xslim"] = package

    logger_spec = importlib.util.spec_from_file_location(
        "xslim.logger", os.path.join(repo_root, "logger.py")
    )
    logger_module = importlib.util.module_from_spec(logger_spec)
    sys.modules["xslim.logger"] = logger_module
    logger_spec.loader.exec_module(logger_module)

    onnxslim_pass_spec = importlib.util.spec_from_file_location(
        "xslim.onnxslim_pass",
        os.path.join(repo_root, "onnxslim_pass", "__init__.py"),
        submodule_search_locations=[os.path.join(repo_root, "onnxslim_pass")],
    )
    onnxslim_pass_module = importlib.util.module_from_spec(onnxslim_pass_spec)
    sys.modules["xslim.onnxslim_pass"] = onnxslim_pass_module
    onnxslim_pass_spec.loader.exec_module(onnxslim_pass_module)
    return onnxslim_pass_module


def _load_onnx_graph_helper_module():
    repo_root = os.path.join(os.path.dirname(__file__), "..", "xslim")

    package = types.ModuleType("xslim")
    package.__path__ = [repo_root]
    sys.modules["xslim"] = package

    logger_spec = importlib.util.spec_from_file_location(
        "xslim.logger", os.path.join(repo_root, "logger.py")
    )
    logger_module = importlib.util.module_from_spec(logger_spec)
    sys.modules["xslim.logger"] = logger_module
    logger_spec.loader.exec_module(logger_module)

    defs_spec = importlib.util.spec_from_file_location(
        "xslim.defs", os.path.join(repo_root, "defs.py")
    )
    defs_module = importlib.util.module_from_spec(defs_spec)
    sys.modules["xslim.defs"] = defs_module
    defs_spec.loader.exec_module(defs_module)

    onnxslim_pass_module = _load_onnxslim_pass_module()
    sys.modules["xslim.onnxslim_pass"] = onnxslim_pass_module

    onnx_graph_helper_spec = importlib.util.spec_from_file_location(
        "xslim.onnx_graph_helper",
        os.path.join(repo_root, "onnx_graph_helper.py"),
    )
    onnx_graph_helper_module = importlib.util.module_from_spec(
        onnx_graph_helper_spec
    )
    sys.modules["xslim.onnx_graph_helper"] = onnx_graph_helper_module
    onnx_graph_helper_spec.loader.exec_module(onnx_graph_helper_module)
    return onnx_graph_helper_module


class TestOnnxSlimPass(unittest.TestCase):
    """Test compatibility behavior in the onnxslim wrapper."""

    @staticmethod
    def _build_add_model():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
        bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])
        nodes = [
            helper.make_node("Add", ["x", "bias"], ["z"], name="add"),
            helper.make_node("Identity", ["z"], ["y"]),
        ]
        graph = helper.make_graph(
            nodes,
            "add_graph",
            [x],
            [y],
            [bias],
            value_info=[z],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        return onnx.shape_inference.infer_shapes(model)

    @staticmethod
    def _build_pad_pool_model(pool_op_type, pad_value=None):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
        pads = helper.make_tensor(
            "pads", TensorProto.INT64, [8], [0, 0, 1, 1, 0, 0, 1, 1]
        )

        initializers = [pads]
        pad_inputs = ["x", "pads"]
        if pad_value is not None:
            pad_value_tensor = helper.make_tensor(
                "pad_value", TensorProto.FLOAT, [], [pad_value]
            )
            initializers.append(pad_value_tensor)
            pad_inputs.append("pad_value")

        nodes = [
            helper.make_node("Pad", pad_inputs, ["padded"], name="pad"),
            helper.make_node(
                pool_op_type,
                ["padded"],
                ["y"],
                name="pool",
                kernel_shape=[3, 3],
                strides=[1, 1],
            ),
        ]
        graph = helper.make_graph(
            nodes, "pad_pool_graph", [x], [y], initializers
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        return onnx.shape_inference.infer_shapes(model)

    @staticmethod
    def _build_conv_model_without_kernel_shape(op_type):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 6, 6])

        if op_type == "Conv":
            weight_shape = [4, 3, 3, 3]
        else:
            weight_shape = [3, 4, 3, 3]

        weight_count = (
            weight_shape[0]
            * weight_shape[1]
            * weight_shape[2]
            * weight_shape[3]
        )
        weight_values = [0.1] * int(weight_count)
        weight = helper.make_tensor(
            "w", TensorProto.FLOAT, weight_shape, weight_values
        )
        node = helper.make_node(
            op_type, ["x", "w"], ["y"], name=f"{op_type.lower()}_node"
        )
        graph = helper.make_graph(
            [node], f"{op_type.lower()}_graph", [x], [y], [weight]
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )

    @staticmethod
    def _build_model_with_duplicate_node_names():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
        y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [2, 2])
        y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [2, 2])
        shape = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 2])
        nodes = [
            helper.make_node(
                "Reshape", ["x", "shape"], ["y0"], name="Reshape_49"
            ),
            helper.make_node(
                "Reshape", ["z", "shape"], ["y1"], name="Reshape_49"
            ),
        ]
        graph = helper.make_graph(
            nodes, "duplicate_node_names_graph", [x, z], [y0, y1], [shape]
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )

    @staticmethod
    def _build_conv_model_with_bad_spatial_attrs():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 6])
        weight = helper.make_tensor(
            "w", TensorProto.FLOAT, [4, 3, 3], [0.1] * (4 * 3 * 3)
        )
        node = helper.make_node(
            "Conv",
            ["x", "w"],
            ["y"],
            name="conv_bad_attrs",
            kernel_shape=[3],
            strides=[1, 1],
            dilations=[1, 1],
            pads=[0, 0, 0, 0],
        )
        graph = helper.make_graph(
            [node], "conv_bad_attrs_graph", [x], [y], [weight]
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )

    @staticmethod
    def _build_layernorm_pattern_model_with_axes_input():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4])

        axes = helper.make_tensor("axes", TensorProto.INT64, [1], [2])
        exponent = helper.make_tensor("exponent", TensorProto.FLOAT, [], [2.0])
        epsilon = helper.make_tensor("epsilon", TensorProto.FLOAT, [], [1e-5])
        scale = helper.make_tensor(
            "scale", TensorProto.FLOAT, [4], [1.1, 1.2, 1.3, 1.4]
        )
        bias = helper.make_tensor(
            "bias", TensorProto.FLOAT, [4], [0.1, -0.2, 0.3, -0.4]
        )

        nodes = [
            helper.make_node(
                "ReduceMean",
                ["x", "axes"],
                ["mean0"],
                name="reduce_mean_0",
                keepdims=1,
            ),
            helper.make_node("Sub", ["x", "mean0"], ["sub0"], name="sub_0"),
            helper.make_node(
                "Pow",
                ["sub0", "exponent"],
                ["pow0"],
                name="pow_0",
            ),
            helper.make_node(
                "ReduceMean",
                ["pow0", "axes"],
                ["mean1"],
                name="reduce_mean_1",
                keepdims=1,
            ),
            helper.make_node(
                "Add",
                ["mean1", "epsilon"],
                ["add0"],
                name="add_0",
            ),
            helper.make_node("Sqrt", ["add0"], ["sqrt0"], name="sqrt_0"),
            helper.make_node(
                "Div",
                ["sub0", "sqrt0"],
                ["div0"],
                name="div_0",
            ),
            helper.make_node(
                "Mul",
                ["div0", "scale"],
                ["mul0"],
                name="mul_0",
            ),
            helper.make_node("Add", ["mul0", "bias"], ["y"], name="add_1"),
        ]
        graph = helper.make_graph(
            nodes,
            "layernorm_axes_input_graph",
            [x],
            [y],
            [axes, exponent, epsilon, scale, bias],
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 24)]
        )

    @staticmethod
    def _build_yolo_decode_pattern_model():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 18, 4])

        reshape_0_shape = helper.make_tensor(
            "reshape_0_shape", TensorProto.INT64, [4], [1, 4, 16, 1]
        )
        conv_weight = helper.make_tensor(
            "conv_weight",
            TensorProto.FLOAT,
            [1, 16, 1, 1],
            [float(index) / 16.0 for index in range(16)],
        )
        reshape_1_shape = helper.make_tensor(
            "reshape_1_shape", TensorProto.INT64, [3], [1, 2, 4]
        )
        slice_0_starts = helper.make_tensor(
            "slice_0_starts", TensorProto.INT64, [1], [0]
        )
        slice_0_ends = helper.make_tensor(
            "slice_0_ends", TensorProto.INT64, [1], [1]
        )
        slice_1_starts = helper.make_tensor(
            "slice_1_starts", TensorProto.INT64, [1], [1]
        )
        slice_1_ends = helper.make_tensor(
            "slice_1_ends", TensorProto.INT64, [1], [2]
        )
        slice_axes = helper.make_tensor("slice_axes", TensorProto.INT64, [1], [1])
        slice_steps = helper.make_tensor(
            "slice_steps", TensorProto.INT64, [1], [1]
        )
        sub_const = helper.make_tensor(
            "sub_const",
            TensorProto.FLOAT,
            [1, 1, 4],
            [10.0, 20.0, 30.0, 40.0],
        )
        add_const = helper.make_tensor(
            "add_const",
            TensorProto.FLOAT,
            [1, 1, 4],
            [11.0, 21.0, 31.0, 41.0],
        )
        div_const = helper.make_tensor("div_const", TensorProto.FLOAT, [], [2.0])
        mul_const = helper.make_tensor(
            "mul_const",
            TensorProto.FLOAT,
            [1, 1, 4],
            [8.0, 8.0, 16.0, 16.0],
        )

        nodes = [
            helper.make_node("Sigmoid", ["x"], ["sigmoid_out"], name="sigmoid_0"),
            helper.make_node(
                "Reshape",
                ["x", "reshape_0_shape"],
                ["reshape_0_out"],
                name="reshape_0",
            ),
            helper.make_node(
                "Transpose",
                ["reshape_0_out"],
                ["transpose_0_out"],
                name="transpose_0",
                perm=[0, 2, 1, 3],
            ),
            helper.make_node(
                "Softmax",
                ["transpose_0_out"],
                ["softmax_0_out"],
                name="softmax_0",
                axis=1,
            ),
            helper.make_node(
                "Conv",
                ["softmax_0_out", "conv_weight"],
                ["conv_0_out"],
                name="conv_0",
                pads=[0, 0, 0, 0],
            ),
            helper.make_node(
                "Reshape",
                ["conv_0_out", "reshape_1_shape"],
                ["reshape_1_out"],
                name="reshape_1",
            ),
            helper.make_node(
                "Slice",
                [
                    "reshape_1_out",
                    "slice_0_starts",
                    "slice_0_ends",
                    "slice_axes",
                    "slice_steps",
                ],
                ["slice_0_out"],
                name="slice_0",
            ),
            helper.make_node(
                "Slice",
                [
                    "reshape_1_out",
                    "slice_1_starts",
                    "slice_1_ends",
                    "slice_axes",
                    "slice_steps",
                ],
                ["slice_1_out"],
                name="slice_1",
            ),
            helper.make_node(
                "Sub",
                ["sub_const", "slice_0_out"],
                ["sub_0_out"],
                name="sub_0",
            ),
            helper.make_node(
                "Add",
                ["add_const", "slice_1_out"],
                ["add_0_out"],
                name="add_0",
            ),
            helper.make_node(
                "Sub",
                ["add_0_out", "sub_0_out"],
                ["sub_1_out"],
                name="sub_1",
            ),
            helper.make_node(
                "Add",
                ["sub_0_out", "add_0_out"],
                ["add_1_out"],
                name="add_1",
            ),
            helper.make_node(
                "Div",
                ["add_1_out", "div_const"],
                ["div_0_out"],
                name="div_0",
            ),
            helper.make_node(
                "Concat",
                ["div_0_out", "sub_1_out"],
                ["concat_0_out"],
                name="concat_0",
                axis=1,
            ),
            helper.make_node(
                "Mul",
                ["concat_0_out", "mul_const"],
                ["mul_0_out"],
                name="mul_0",
            ),
            helper.make_node(
                "Concat",
                ["mul_0_out", "sigmoid_out"],
                ["y"],
                name="concat_1",
                axis=1,
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "yolo_decode_pattern_graph",
            [x],
            [y],
            [
                reshape_0_shape,
                conv_weight,
                reshape_1_shape,
                slice_0_starts,
                slice_0_ends,
                slice_1_starts,
                slice_1_ends,
                slice_axes,
                slice_steps,
                sub_const,
                add_const,
                div_const,
                mul_const,
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 24)]
        )
        return onnx.shape_inference.infer_shapes(model)

    @staticmethod
    def _build_yolo_decode_split_pattern_model():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 72, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 12, 4])
        split_sizes = helper.make_tensor(
            "split_sizes", TensorProto.INT64, [2], [64, 8]
        )

        split_outputs = [
            helper.make_tensor_value_info("split_0_out", TensorProto.FLOAT, [1, 64, 4]),
            helper.make_tensor_value_info("split_1_out", TensorProto.FLOAT, [1, 8, 4]),
        ]
        graph = onnx.helper.make_graph(
            [
                helper.make_node(
                    "Split",
                    ["x", "split_sizes"],
                    ["split_0_out", "split_1_out"],
                    name="split_0",
                    axis=1,
                ),
                helper.make_node("Sigmoid", ["split_1_out"], ["sigmoid_out"], name="sigmoid_0"),
                helper.make_node(
                    "Reshape",
                    ["split_0_out", "reshape_0_shape"],
                    ["reshape_0_out"],
                    name="reshape_0",
                ),
                helper.make_node(
                    "Transpose",
                    ["reshape_0_out"],
                    ["transpose_0_out"],
                    name="transpose_0",
                    perm=[0, 2, 1, 3],
                ),
                helper.make_node(
                    "Softmax",
                    ["transpose_0_out"],
                    ["softmax_0_out"],
                    name="softmax_0",
                    axis=1,
                ),
                helper.make_node(
                    "Conv",
                    ["softmax_0_out", "conv_weight"],
                    ["conv_0_out"],
                    name="conv_0",
                    pads=[0, 0, 0, 0],
                ),
                helper.make_node(
                    "Reshape",
                    ["conv_0_out", "reshape_1_shape"],
                    ["reshape_1_out"],
                    name="reshape_1",
                ),
                helper.make_node(
                    "Slice",
                    [
                        "reshape_1_out",
                        "slice_0_starts",
                        "slice_0_ends",
                        "slice_axes",
                        "slice_steps",
                    ],
                    ["slice_0_out"],
                    name="slice_0",
                ),
                helper.make_node(
                    "Slice",
                    [
                        "reshape_1_out",
                        "slice_1_starts",
                        "slice_1_ends",
                        "slice_axes",
                        "slice_steps",
                    ],
                    ["slice_1_out"],
                    name="slice_1",
                ),
                helper.make_node(
                    "Sub",
                    ["sub_const", "slice_0_out"],
                    ["sub_0_out"],
                    name="sub_0",
                ),
                helper.make_node(
                    "Add",
                    ["add_const", "slice_1_out"],
                    ["add_0_out"],
                    name="add_0",
                ),
                helper.make_node(
                    "Sub",
                    ["add_0_out", "sub_0_out"],
                    ["sub_1_out"],
                    name="sub_1",
                ),
                helper.make_node(
                    "Add",
                    ["sub_0_out", "add_0_out"],
                    ["add_1_out"],
                    name="add_1",
                ),
                helper.make_node(
                    "Div",
                    ["add_1_out", "div_const"],
                    ["div_0_out"],
                    name="div_0",
                ),
                helper.make_node(
                    "Concat",
                    ["div_0_out", "sub_1_out"],
                    ["concat_0_out"],
                    name="concat_0",
                    axis=1,
                ),
                helper.make_node(
                    "Mul",
                    ["concat_0_out", "mul_const"],
                    ["mul_0_out"],
                    name="mul_0",
                ),
                helper.make_node(
                    "Concat",
                    ["mul_0_out", "sigmoid_out"],
                    ["y"],
                    name="concat_1",
                    axis=1,
                ),
            ],
            "yolo_decode_split_pattern_graph",
            [x],
            [y],
            [
                split_sizes,
                helper.make_tensor("reshape_0_shape", TensorProto.INT64, [4], [1, 4, 16, 4]),
                helper.make_tensor(
                    "conv_weight",
                    TensorProto.FLOAT,
                    [1, 16, 1, 1],
                    [float(index) / 16.0 for index in range(16)],
                ),
                helper.make_tensor("reshape_1_shape", TensorProto.INT64, [3], [1, 4, 4]),
                helper.make_tensor("slice_0_starts", TensorProto.INT64, [1], [0]),
                helper.make_tensor("slice_0_ends", TensorProto.INT64, [1], [2]),
                helper.make_tensor("slice_1_starts", TensorProto.INT64, [1], [2]),
                helper.make_tensor("slice_1_ends", TensorProto.INT64, [1], [4]),
                helper.make_tensor("slice_axes", TensorProto.INT64, [1], [1]),
                helper.make_tensor("slice_steps", TensorProto.INT64, [1], [1]),
                helper.make_tensor("sub_const", TensorProto.FLOAT, [1, 2, 4], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
                helper.make_tensor("add_const", TensorProto.FLOAT, [1, 2, 4], [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0]),
                helper.make_tensor("div_const", TensorProto.FLOAT, [], [2.0]),
                helper.make_tensor("mul_const", TensorProto.FLOAT, [1, 4, 4], [8.0, 8.0, 16.0, 16.0, 8.0, 8.0, 16.0, 16.0, 8.0, 8.0, 16.0, 16.0, 8.0, 8.0, 16.0, 16.0]),
            ],
            value_info=split_outputs,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 24)]
        )
        return onnx.shape_inference.infer_shapes(model)

    @staticmethod
    def _build_matmul_qkv_split_pattern_model():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        q_out = helper.make_tensor_value_info("q_out", TensorProto.FLOAT, [1, 2])
        k_out = helper.make_tensor_value_info("k_out", TensorProto.FLOAT, [1, 2])
        v_out = helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [1, 2])

        weight_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        bias_values = [0.1, 1.1, 0.2, 1.2, 0.3, 1.3]

        weight = helper.make_tensor(
            "weight", TensorProto.FLOAT, [2, 6], weight_values
        )
        bias = helper.make_tensor(
            "bias", TensorProto.FLOAT, [6], bias_values
        )
        reshape_shape = helper.make_tensor(
            "reshape_shape", TensorProto.INT64, [3], [1, 2, 3]
        )
        gather_index_0 = helper.make_tensor("gather_index_0", TensorProto.INT64, [1], [0])
        gather_index_1 = helper.make_tensor("gather_index_1", TensorProto.INT64, [1], [1])
        gather_index_2 = helper.make_tensor("gather_index_2", TensorProto.INT64, [1], [2])

        nodes = [
            helper.make_node("MatMul", ["x", "weight"], ["matmul_out"], name="matmul_0"),
            helper.make_node("Add", ["bias", "matmul_out"], ["add_out"], name="add_0"),
            helper.make_node(
                "Reshape",
                ["add_out", "reshape_shape"],
                ["reshape_out"],
                name="reshape_0",
            ),
            helper.make_node(
                "Gather",
                ["reshape_out", "gather_index_0"],
                ["q_out"],
                name="gather_0",
                axis=2,
            ),
            helper.make_node(
                "Gather",
                ["reshape_out", "gather_index_1"],
                ["k_out"],
                name="gather_1",
                axis=2,
            ),
            helper.make_node(
                "Gather",
                ["reshape_out", "gather_index_2"],
                ["v_out"],
                name="gather_2",
                axis=2,
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "matmul_qkv_split_pattern_graph",
            [x],
            [q_out, k_out, v_out],
            [weight, bias, reshape_shape, gather_index_0, gather_index_1, gather_index_2],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        return onnx.shape_inference.infer_shapes(model)

    def test_infer_onnx_model_restores_shape_info_for_float16_model(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_add_model()
        model = convert_float_to_float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=False
        )
        model.graph.ClearField("value_info")
        self.assertEqual(len(model.graph.value_info), 0)

        optimized_model = onnxslim_pass_module.infer_onnx_model(model)

        self.assertGreater(len(optimized_model.graph.value_info), 0)
        self.assertIn(
            TensorProto.FLOAT16,
            {
                value_info.type.tensor_type.elem_type
                for value_info in optimized_model.graph.value_info
            },
        )

    def test_optimize_onnx_model_skips_fusion_gemm(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_add_model()

        with mock.patch.object(
            onnxslim_pass_module.onnxslim,
            "slim",
            return_value=model,
        ) as slim_mock:
            optimized_model = onnxslim_pass_module.optimize_onnx_model(model)

        self.assertIs(optimized_model, model)
        slim_mock.assert_called_once_with(
            model, skip_fusion_patterns=["FusionGemm"]
        )

    def test_optimize_onnx_model_fuses_pad_average_pool(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_pad_pool_model("AveragePool")

        optimized_model = onnxslim_pass_module.optimize_onnx_model(model)
        optimized_model = onnxslim_pass_module.infer_onnx_model(
            optimized_model
        )

        onnx.checker.check_model(optimized_model)
        self.assertEqual(
            [node.op_type for node in optimized_model.graph.node],
            ["AveragePool"],
        )

        pool_node = optimized_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in pool_node.attribute
        }
        self.assertEqual(list(attrs["pads"]), [1, 1, 1, 1])
        self.assertEqual(attrs["count_include_pad"], 1)

    def test_optimize_onnx_model_fuses_pad_max_pool_with_neg_inf_padding(
        self,
    ):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_pad_pool_model("MaxPool", pad_value=float("-inf"))

        optimized_model = onnxslim_pass_module.optimize_onnx_model(model)
        optimized_model = onnxslim_pass_module.infer_onnx_model(
            optimized_model
        )

        onnx.checker.check_model(optimized_model)
        self.assertEqual(
            [node.op_type for node in optimized_model.graph.node], ["MaxPool"]
        )

        pool_node = optimized_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in pool_node.attribute
        }
        self.assertEqual(list(attrs["pads"]), [1, 1, 1, 1])

    def test_optimize_onnx_model_keeps_zero_pad_before_max_pool(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_pad_pool_model("MaxPool")

        optimized_model = onnxslim_pass_module.optimize_onnx_model(model)
        optimized_model = onnxslim_pass_module.infer_onnx_model(
            optimized_model
        )

        onnx.checker.check_model(optimized_model)
        self.assertEqual(
            [node.op_type for node in optimized_model.graph.node],
            ["Pad", "MaxPool"],
        )

    def test_optimize_onnx_model_fuses_layernorm_with_axes_input(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_layernorm_pattern_model_with_axes_input()

        optimized_model = onnxslim_pass_module.optimize_onnx_model(model)
        optimized_model = onnxslim_pass_module.infer_onnx_model(
            optimized_model
        )

        onnx.checker.check_model(optimized_model)
        self.assertEqual(
            [node.op_type for node in optimized_model.graph.node],
            ["LayerNormalization"],
        )

        layernorm_node = optimized_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in layernorm_node.attribute
        }
        self.assertEqual(attrs["axis"], 2)
        self.assertAlmostEqual(attrs["epsilon"], 1e-5, places=7)
        self.assertEqual(layernorm_node.domain, "")

    def test_optimize_onnx_model_fuses_yolo_decode_pattern(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_yolo_decode_pattern_model()

        optimized_model = onnxslim_pass_module.optimize_onnx_model(model)

        self.assertEqual(
            [node.op_type for node in optimized_model.graph.node],
            ["YoloDecode"],
        )

        yolo_decode_node = optimized_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in yolo_decode_node.attribute
        }
        initializers = {
            initializer.name: onnx.numpy_helper.to_array(initializer)
            for initializer in optimized_model.graph.initializer
        }

        self.assertEqual(yolo_decode_node.domain, "spacemit_functions")
        self.assertEqual(yolo_decode_node.name, "yolo_decode_concat_1")
        self.assertEqual(list(yolo_decode_node.input), [
            "x",
            "conv_weight_flat",
            "sub_const",
            "add_const",
            "mul_const",
        ])
        self.assertEqual(attrs["num_class"], -1)
        self.assertEqual(attrs["reg_max"], 16)
        self.assertEqual(initializers["conv_weight_flat"].shape, (16,))
        self.assertIn(
            ("spacemit_functions", 1),
            {(item.domain, item.version) for item in optimized_model.opset_import},
        )
        self.assertEqual(
            [(function.domain, function.name) for function in optimized_model.functions],
            [("spacemit_functions", "YoloDecode")],
        )

        yolo_decode_function = optimized_model.functions[0]
        self.assertEqual(
            list(yolo_decode_function.input),
            ["input", "flat_weight", "sub_const", "add_const", "mul_const"],
        )
        self.assertEqual(list(yolo_decode_function.output), ["output"])
        self.assertEqual(
            list(yolo_decode_function.attribute),
            ["num_class", "reg_max"],
        )

        onnx.checker.check_model(optimized_model)

    def test_optimize_onnx_model_fuses_yolo_decode_split_pattern(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_yolo_decode_split_pattern_model()

        optimized_model = onnxslim_pass_module.optimize_onnx_model(model)

        self.assertEqual(
            [node.op_type for node in optimized_model.graph.node],
            ["YoloDecode"],
        )

        yolo_decode_node = optimized_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in yolo_decode_node.attribute
        }
        self.assertEqual(yolo_decode_node.domain, "spacemit_functions")
        self.assertEqual(list(yolo_decode_node.input), [
            "x",
            "conv_weight_flat",
            "sub_const",
            "add_const",
            "mul_const",
        ])
        self.assertEqual(attrs["num_class"], 8)
        self.assertEqual(attrs["reg_max"], 16)

        self.assertIn(("", 24), {(item.domain, item.version) for item in optimized_model.opset_import})
        self.assertIn(("spacemit_functions", 1), {(item.domain, item.version) for item in optimized_model.opset_import})
        self.assertEqual(
            [(function.domain, function.name) for function in optimized_model.functions],
            [("spacemit_functions", "YoloDecode")],
        )
        self.assertEqual(
            list(optimized_model.functions[0].attribute),
            ["num_class", "reg_max"],
        )

        onnx.checker.check_model(optimized_model)

    def test_optimize_onnx_model_fuses_matmul_qkv_split_pattern(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_matmul_qkv_split_pattern_model()

        optimized_model = onnxslim_pass_module.optimize_onnx_model(model)
        optimized_model = onnxslim_pass_module.infer_onnx_model(
            optimized_model
        )

        onnx.checker.check_model(optimized_model)
        self.assertEqual(
            [node.op_type for node in optimized_model.graph.node],
            ["MatMul", "MatMul", "MatMul", "Add", "Add", "Add"],
        )
        self.assertEqual(
            [node.name for node in optimized_model.graph.node],
            [
                "matmul_0_q",
                "matmul_0_k",
                "matmul_0_v",
                "add_0_q",
                "add_0_k",
                "add_0_v",
            ],
        )

        initializers = {
            initializer.name: onnx.numpy_helper.to_array(initializer).tolist()
            for initializer in optimized_model.graph.initializer
        }
        self.assertEqual(
            initializers["matmul_0_q_weight"],
            [[1.0, 4.0], [7.0, 10.0]],
        )
        self.assertEqual(
            initializers["matmul_0_k_weight"],
            [[2.0, 5.0], [8.0, 11.0]],
        )
        self.assertEqual(
            initializers["matmul_0_v_weight"],
            [[3.0, 6.0], [9.0, 12.0]],
        )
        self.assertTrue(
            all(
                abs(actual - expected) < 1e-6
                for actual, expected in zip(
                    initializers["add_0_q_bias"], [0.1, 1.2]
                )
            )
        )
        self.assertTrue(
            all(
                abs(actual - expected) < 1e-6
                for actual, expected in zip(
                    initializers["add_0_k_bias"], [1.1, 0.3]
                )
            )
        )
        self.assertTrue(
            all(
                abs(actual - expected) < 1e-6
                for actual, expected in zip(
                    initializers["add_0_v_bias"], [0.2, 1.3]
                )
            )
        )

    def test_build_yolo_decode_function_uses_dynamic_bbox_reshape_shape(self):
        _load_onnxslim_pass_module()
        yolo_decode_module = sys.modules["xslim.onnxslim_pass.yolo_decode"]

        function_proto = yolo_decode_module.build_yolo_decode_function()
        nodes_by_name = {node.name: node for node in function_proto.node}

        def _constant_values(node_name):
            tensor = helper.get_attribute_value(nodes_by_name[node_name].attribute[0])
            return onnx.numpy_helper.to_array(tensor).reshape(-1).tolist()

        self.assertEqual(list(function_proto.input), [
            "input",
            "flat_weight",
            "sub_const",
            "add_const",
            "mul_const",
        ])
        self.assertEqual(list(function_proto.output), ["output"])
        self.assertEqual(
            list(function_proto.attribute),
            ["num_class", "reg_max"],
        )

        self.assertEqual(
            list(nodes_by_name["bbox_reshape"].input),
            ["bbox_input", "bbox_reshape_shape"],
        )
        self.assertEqual(
            list(nodes_by_name["bbox_reshape_shape_concat"].input),
            ["batch_dim_vec", "four_vec", "reg_max_vec", "spatial_dim_vec"],
        )
        self.assertEqual(
            list(nodes_by_name["conv_weight_shape_concat"].input),
            ["one_vec", "reg_max_vec", "one_vec", "one_vec"],
        )
        self.assertEqual(
            list(nodes_by_name["bbox_output_shape_concat"].input),
            ["batch_dim_vec", "four_vec", "spatial_dim_vec"],
        )
        self.assertEqual(
            list(nodes_by_name["gather_batch_dim"].input),
            ["bbox_shape", "shape_idx_0"],
        )
        self.assertEqual(
            list(nodes_by_name["gather_spatial_dim"].input),
            ["bbox_shape", "shape_idx_2"],
        )
        self.assertEqual(_constant_values("shape_idx_0_const"), [0])
        self.assertEqual(_constant_values("shape_idx_2_const"), [2])
        self.assertEqual(
            helper.get_attribute_value(nodes_by_name["shape_idx_0_const"].attribute[0]).dims,
            [],
        )
        self.assertEqual(
            helper.get_attribute_value(nodes_by_name["shape_idx_2_const"].attribute[0]).dims,
            [],
        )

        reg_max_attr = nodes_by_name["reg_max_const"].attribute[0]
        num_class_attr = nodes_by_name["num_class_const"].attribute[0]
        self.assertEqual(reg_max_attr.ref_attr_name, "reg_max")
        self.assertEqual(num_class_attr.ref_attr_name, "num_class")

        self.assertEqual(nodes_by_name["bbox_dfl"].op_type, "Conv")
        self.assertEqual(
            list(nodes_by_name["bbox_output_reshape"].input),
            ["bbox_dfl", "bbox_output_shape"],
        )
        self.assertEqual(
            list(nodes_by_name["bbox_slice_0"].input),
            ["bbox_output", "slice_0_starts", "slice_0_ends", "axes1", "slice_steps"],
        )
        self.assertEqual(
            list(nodes_by_name["bbox_slice_1"].input),
            ["bbox_output", "slice_1_starts", "slice_1_ends", "axes1", "slice_steps"],
        )
        self.assertEqual(
            list(nodes_by_name["bbox_concat"].input),
            ["bbox_div", "bbox_adjusted"],
        )
        self.assertEqual(
            list(nodes_by_name["bbox_scaled"].input),
            ["bbox_concat", "mul_const"],
        )

        transpose_attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in nodes_by_name["bbox_transpose"].attribute
        }
        softmax_attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in nodes_by_name["bbox_softmax"].attribute
        }
        concat_attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in nodes_by_name["bbox_concat"].attribute
        }
        output_concat_attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in nodes_by_name["output_concat"].attribute
        }
        self.assertEqual(list(transpose_attrs["perm"]), [0, 2, 1, 3])
        self.assertEqual(softmax_attrs["axis"], 1)
        self.assertEqual(concat_attrs["axis"], 1)
        self.assertEqual(output_concat_attrs["axis"], 1)

    def test_merge_onnx_model_reuses_existing_named_tensors(self):
        onnx_graph_helper_module = _load_onnx_graph_helper_module()
        model = self._build_add_model()

        truncated_model, truncate_left_graph, truncate_vars = (
            onnx_graph_helper_module.truncate_onnx_model(model, ["z"])
        )
        merged_model = onnx_graph_helper_module.merge_onnx_model(
            truncated_model,
            truncate_left_graph,
            truncate_vars,
        )

        merged_graph = osg.import_onnx(merged_model)
        tensors_by_name = {}
        duplicate_names = []
        merged_tensors = list(merged_graph.inputs)
        for node in merged_graph.nodes:
            merged_tensors.extend(node.inputs)
            merged_tensors.extend(node.outputs)
        merged_tensors.extend(merged_graph.outputs)

        for tensor in merged_tensors:
            if tensor.is_empty():
                continue
            if (
                tensor.name in tensors_by_name
                and tensors_by_name[tensor.name] is not tensor
            ):
                duplicate_names.append(tensor.name)
            else:
                tensors_by_name[tensor.name] = tensor

        self.assertEqual(duplicate_names, [])
        self.assertEqual([node.op_type for node in merged_model.graph.node], ["Add", "Identity"])
        onnx.checker.check_model(merged_model)

    def test_format_onnx_model_fills_missing_conv_kernel_shape(self):
        onnx_graph_helper_module = _load_onnx_graph_helper_module()
        model = self._build_conv_model_without_kernel_shape("Conv")

        with mock.patch.object(
            onnx_graph_helper_module,
            "optimize_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ), mock.patch.object(
            onnx_graph_helper_module,
            "infer_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ):
            formatted_model = onnx_graph_helper_module.format_onnx_model(
                model, sim_en=False
            )

        conv_node = formatted_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in conv_node.attribute
        }
        self.assertEqual(list(attrs["kernel_shape"]), [3, 3])

    def test_format_onnx_model_fills_missing_convtranspose_kernel_shape(self):
        onnx_graph_helper_module = _load_onnx_graph_helper_module()
        model = self._build_conv_model_without_kernel_shape("ConvTranspose")

        with mock.patch.object(
            onnx_graph_helper_module,
            "optimize_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ), mock.patch.object(
            onnx_graph_helper_module,
            "infer_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ):
            formatted_model = onnx_graph_helper_module.format_onnx_model(
                model, sim_en=False
            )

        conv_transpose_node = formatted_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in conv_transpose_node.attribute
        }
        self.assertEqual(list(attrs["kernel_shape"]), [3, 3])

    def test_format_onnx_model_renames_duplicated_node_names(self):
        onnx_graph_helper_module = _load_onnx_graph_helper_module()
        model = self._build_model_with_duplicate_node_names()

        with mock.patch.object(
            onnx_graph_helper_module,
            "optimize_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ), mock.patch.object(
            onnx_graph_helper_module,
            "infer_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ):
            formatted_model = onnx_graph_helper_module.format_onnx_model(
                model, sim_en=False
            )

        node_names = [node.name for node in formatted_model.graph.node]
        onnx.checker.check_model(formatted_model)
        self.assertEqual(node_names[0], "Reshape_49")
        self.assertEqual(len(node_names), len(set(node_names)))
        self.assertRegex(node_names[1], r"^Reshape_49_\d+$")

    def test_format_onnx_model_normalizes_conv_spatial_attrs(self):
        onnx_graph_helper_module = _load_onnx_graph_helper_module()
        model = self._build_conv_model_with_bad_spatial_attrs()

        with mock.patch.object(
            onnx_graph_helper_module,
            "optimize_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ), mock.patch.object(
            onnx_graph_helper_module,
            "infer_onnx_model",
            side_effect=lambda model_arg: model_arg,
        ):
            formatted_model = onnx_graph_helper_module.format_onnx_model(
                model, sim_en=False
            )

        conv_node = formatted_model.graph.node[0]
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in conv_node.attribute
        }
        onnx.checker.check_model(formatted_model)
        self.assertEqual(list(attrs["kernel_shape"]), [3])
        self.assertEqual(list(attrs["strides"]), [1])
        self.assertEqual(list(attrs["dilations"]), [1])
        self.assertEqual(list(attrs["pads"]), [0, 0])


if __name__ == "__main__":
    unittest.main()
