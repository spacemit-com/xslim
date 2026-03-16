"""Regression tests for FP16 ONNX conversion."""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import onnx
import onnx_graphsurgeon as osg
import onnxruntime as ort
from onnx import TensorProto, helper
from onnxconverter_common import float16 as convert_float_to_float16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xslim.quantizer import convert_to_fp16 as convert_to_fp16_module

MAX_TRANSFORMER_FP16_ERROR = 1e-3


class TestConvertToFp16(unittest.TestCase):
    """Test dtype preservation around FP16 conversion legalization."""

    @staticmethod
    def _build_add_cast_model():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
        bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])

        nodes = [
            helper.make_node("Add", ["x", "bias"], ["z"], name="add"),
            helper.make_node("Cast", ["z"], ["y"], name="cast", to=TensorProto.FLOAT),
        ]
        graph = helper.make_graph(nodes, "add_cast_graph", [x], [y], [bias], value_info=[z])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        return onnx.shape_inference.infer_shapes(model)

    @staticmethod
    def _build_layernorm_softmax_model():
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
        scale = helper.make_tensor("scale", TensorProto.FLOAT, [8], [1.0] * 8)
        bias = helper.make_tensor("bias", TensorProto.FLOAT, [8], [0.0] * 8)

        nodes = [
            helper.make_node(
                "LayerNormalization",
                ["x", "scale", "bias"],
                ["norm"],
                name="layernorm",
                axis=-1,
                epsilon=1e-12,
            ),
            helper.make_node("Softmax", ["norm"], ["y"], name="softmax", axis=-1),
        ]
        graph = helper.make_graph(
            nodes,
            "layernorm_softmax_graph",
            [x],
            [y],
            [scale, bias],
            value_info=[helper.make_tensor_value_info("norm", TensorProto.FLOAT, [1, 8])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        return onnx.shape_inference.infer_shapes(model)

    def test_convert_to_fp16_restores_tensor_dtypes_before_legalize(self):
        model = self._build_add_cast_model()
        converted_model = convert_float_to_float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=False
        )
        converted_model.graph.ClearField("value_info")

        with mock.patch.object(
            convert_to_fp16_module.convert_float_to_float16,
            "convert_float_to_float16",
            return_value=converted_model,
        ):
            fp16_model = convert_to_fp16_module.convert_to_fp16_onnx_model(model, [], [], sim_en=False)

        osg_graph = osg.import_onnx(fp16_model)
        cast_nodes = [node for node in osg_graph.nodes if node.op == "Cast" and node.name == "cast"]

        self.assertEqual(len(cast_nodes), 1)
        self.assertEqual(cast_nodes[0].attrs["to"], TensorProto.FLOAT)
        self.assertEqual(cast_nodes[0].inputs[0].dtype, onnx.helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT16))
        self.assertEqual(cast_nodes[0].outputs[0].dtype, onnx.helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT))
        self.assertGreater(len(fp16_model.graph.value_info), 0)

    def test_convert_to_fp16_merges_custom_ignore_op_types_with_defaults(self):
        model = self._build_add_cast_model()
        observed = {}
        original_convert = convert_float_to_float16.convert_float_to_float16

        def _capture_convert(*args, **kwargs):
            observed["op_block_list"] = kwargs["op_block_list"]
            return original_convert(*args, **kwargs)

        with mock.patch.object(
            convert_to_fp16_module.convert_float_to_float16,
            "convert_float_to_float16",
            side_effect=_capture_convert,
        ):
            convert_to_fp16_module.convert_to_fp16_onnx_model(
                model, ["CustomStableOp"], [], sim_en=False
            )

        self.assertIn("CustomStableOp", observed["op_block_list"])
        self.assertIn("Softmax", observed["op_block_list"])
        self.assertIn("LayerNormalization", observed["op_block_list"])

    def test_convert_to_fp16_keeps_transformer_sensitive_ops_numerically_stable(self):
        model = self._build_layernorm_softmax_model()
        fp16_model = convert_to_fp16_module.convert_to_fp16_onnx_model(model, [], [], sim_en=False)

        fp32_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        fp16_session = ort.InferenceSession(
            fp16_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        inputs = {
            "x": np.array(
                [[1000.0, 1000.1, 999.9, 1000.05, 1000.02, 999.98, 1000.03, 999.97]],
                dtype=np.float32,
            )
        }
        fp32_output = fp32_session.run(None, inputs)[0]
        fp16_output = fp16_session.run(None, inputs)[0]

        self.assertLess(np.max(np.abs(fp32_output - fp16_output)), MAX_TRANSFORMER_FP16_ERROR)


if __name__ == "__main__":
    unittest.main()
