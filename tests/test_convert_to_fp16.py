"""Regression tests for FP16 ONNX conversion."""

import os
import sys
import unittest
from unittest import mock

import onnx
import onnx_graphsurgeon as osg
from onnx import TensorProto, helper
from onnxconverter_common import float16 as convert_float_to_float16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xslim.quantizer import convert_to_fp16 as convert_to_fp16_module


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


if __name__ == "__main__":
    unittest.main()
