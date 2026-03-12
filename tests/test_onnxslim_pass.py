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

    logger_spec = importlib.util.spec_from_file_location("xslim.logger", os.path.join(repo_root, "logger.py"))
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


class TestOnnxSlimPass(unittest.TestCase):
    """Test compatibility behavior in the onnxslim wrapper."""

    @staticmethod
    def _build_add_model():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
        bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])
        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "bias"], ["z"], name="add"), helper.make_node("Identity", ["z"], ["y"])],
            "add_graph",
            [x],
            [y],
            [bias],
            value_info=[z],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        return onnx.shape_inference.infer_shapes(model)

    def test_optimize_onnx_model_restores_shape_info_before_slim(self):
        onnxslim_pass_module = _load_onnxslim_pass_module()
        model = self._build_add_model()
        model = convert_float_to_float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
        model.graph.ClearField("value_info")
        self.assertEqual(len(model.graph.value_info), 0)

        captured = {}

        def _capture_and_return(model_arg):
            captured["model"] = model_arg
            return model_arg

        with mock.patch.object(onnxslim_pass_module.onnxslim, "slim", side_effect=_capture_and_return):
            optimized_model = onnxslim_pass_module.optimize_onnx_model(model)

        self.assertIs(optimized_model, captured["model"])
        self.assertGreater(len(captured["model"].graph.value_info), 0)
        self.assertIn(
            TensorProto.FLOAT16,
            {value_info.type.tensor_type.elem_type for value_info in captured["model"].graph.value_info},
        )


if __name__ == "__main__":
    unittest.main()
