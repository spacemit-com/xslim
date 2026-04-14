"""Regression tests for the local onnxslim wrapper."""

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import onnx
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

    def test_optimize_onnx_model_restores_shape_info_before_slim(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_add_model()
        model = convert_float_to_float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=False
        )
        model.graph.ClearField("value_info")
        self.assertEqual(len(model.graph.value_info), 0)

        captured = {}

        def _capture_and_return(model_arg):
            captured["model"] = model_arg
            return model_arg

        with mock.patch.object(
            onnxslim_pass_module.onnxslim,
            "slim",
            side_effect=_capture_and_return,
        ):
            optimized_model = onnxslim_pass_module.optimize_onnx_model(model)
            optimized_model = onnxslim_pass_module.infer_onnx_model(
                optimized_model
            )

        self.assertIs(optimized_model, captured["model"])
        self.assertGreater(len(captured["model"].graph.value_info), 0)
        self.assertIn(
            TensorProto.FLOAT16,
            {
                value_info.type.tensor_type.elem_type
                for value_info in captured["model"].graph.value_info
            },
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
