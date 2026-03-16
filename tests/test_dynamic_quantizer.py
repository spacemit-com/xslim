"""Regression tests for dynamic ONNX quantization helpers."""

import os
import sys
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.quant_utils import quantize_data as ort_quantize_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xslim.quantizer.dynamic_quantizer import dynamic_weight_only_quantize


class TestDynamicQuantizer(unittest.TestCase):
    """Validate dynamic weight-only quantization edge cases."""

    @staticmethod
    def _build_matmul_model(weight_value):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, weight_value.shape[0]])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, weight_value.shape[1]])
        weight = numpy_helper.from_array(weight_value.astype(np.float32), name="weight")
        node = helper.make_node("MatMul", ["x", "weight"], ["y"], name="matmul")
        graph = helper.make_graph([node], "matmul_graph", [x], [y], [weight])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_dynamic_weight_only_quantize_preserves_matmul_channel_order(self):
        weight_value = np.array([[0.1, 10.2], [0.3, 20.4], [0.5, 30.6]], dtype=np.float32)
        model = self._build_matmul_model(weight_value)

        quantized_model = dynamic_weight_only_quantize(model, [], [])

        dq_node = next(node for node in quantized_model.graph.node if node.op_type == "DequantizeLinear")
        initializers = {initializer.name: numpy_helper.to_array(initializer) for initializer in quantized_model.graph.initializer}
        quantized_weight = initializers[dq_node.input[0]]
        scales = initializers[dq_node.input[1]]
        zero_points = initializers[dq_node.input[2]]

        expected_scales = []
        expected_zero_points = []
        expected_quantized_columns = []
        for output_channel in weight_value.T:
            zero_point, scale, quantized_column = ort_quantize_data(
                output_channel, onnx.TensorProto.INT8, True, False
            )
            expected_scales.append(scale)
            expected_zero_points.append(zero_point)
            expected_quantized_columns.append(quantized_column)

        expected_quantized_weight = np.stack(expected_quantized_columns, axis=1)

        np.testing.assert_array_equal(scales, np.array(expected_scales, dtype=np.float32))
        np.testing.assert_array_equal(zero_points, np.array(expected_zero_points, dtype=np.int8))
        np.testing.assert_array_equal(quantized_weight, expected_quantized_weight)


if __name__ == "__main__":
    unittest.main()
