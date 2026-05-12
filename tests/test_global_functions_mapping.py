import os
import sys
import unittest

import onnx
import torch
from onnx import TensorProto, helper


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from xslim.defs import GLOBAL_FUNCTIONS_MAPPING
from xslim.optimizer import GraphLegalized
from xslim.ppq_decorator.ppq.executor.torch import TorchExecutor
from xslim.ppq_decorator.ppq.parser.onnx_parser import OnnxParser
from xslim.ppq_decorator.ppq.parser.onnxruntime_exporter import ONNXRUNTIMExporter


def _build_batch_matmul_function(opset_version=13):
    return helper.make_function(
        domain="spacemit_functions",
        fname="BatchMatMul",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        nodes=[
            helper.make_node("MatMul", ["A", "B"], ["MatMul_temp"], name="MatMul"),
            helper.make_node("Add", ["MatMul_temp", "C"], ["Y"], name="Add"),
        ],
        opset_imports=[helper.make_opsetid("", opset_version)],
        attributes=["transA", "transB", "transY"],
    )


def _build_batch_matmul_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [2, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    node = helper.make_node(
        "BatchMatMul",
        ["x", "w", "b"],
        ["y"],
        name="custom_batch_matmul",
        domain="spacemit_functions",
        transA=0,
        transB=0,
        transY=0,
    )
    model = helper.make_model(
        helper.make_graph([node], "batch_matmul_graph", [x, w, b], [y]),
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("spacemit_functions", 1)],
    )
    model.functions.append(_build_batch_matmul_function())
    return model


def _build_matmul_add_pattern_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [2, 2],
        [1.0, 3.0, 2.0, 4.0],
    )
    bias = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        [2],
        [0.5, -1.0],
    )
    nodes = [
        helper.make_node("MatMul", ["x", "weight"], ["matmul_out"], name="matmul"),
        helper.make_node("Add", ["matmul_out", "bias"], ["y"], name="add"),
    ]
    return helper.make_model(
        helper.make_graph(nodes, "matmul_add_graph", [x], [y], [weight, bias]),
        opset_imports=[helper.make_opsetid("", 13)],
    )


class TestGlobalFunctionsMapping(unittest.TestCase):
    def test_parser_stores_function_proto_in_global_mapping(self):
        graph = OnnxParser().build(_build_batch_matmul_model())

        mapping = graph._detail[GLOBAL_FUNCTIONS_MAPPING]
        self.assertEqual(list(mapping.keys()), ["spacemit_functions.BatchMatMul"])

        function_proto = mapping["spacemit_functions.BatchMatMul"]
        self.assertIsInstance(function_proto, onnx.FunctionProto)
        self.assertEqual(function_proto.domain, "spacemit_functions")
        self.assertEqual(function_proto.name, "BatchMatMul")
        self.assertEqual(list(function_proto.attribute), ["transA", "transB", "transY"])

    def test_legalized_registers_function_proto_for_export(self):
        graph = OnnxParser().build(_build_matmul_add_pattern_model())

        legalized = GraphLegalized(graph)
        legalized._formatter.convert_to_tensor()
        legalized.fuse_matmul_bias()

        function_proto = graph._detail[GLOBAL_FUNCTIONS_MAPPING]["spacemit_functions.BatchMatMul"]
        self.assertIsInstance(function_proto, onnx.FunctionProto)
        self.assertEqual(graph.operations["matmul"].type, "BatchMatMul")

        exported_model = ONNXRUNTIMExporter().export(graph)
        self.assertEqual(
            [(function.domain, function.name) for function in exported_model.functions],
            [("spacemit_functions", "BatchMatMul")],
        )
        self.assertEqual(exported_model.graph.node[0].domain, "spacemit_functions")

    def test_torch_executor_uses_registered_batch_matmul_forward(self):
        graph = OnnxParser().build(_build_batch_matmul_model())
        graph._detail[GLOBAL_FUNCTIONS_MAPPING]["spacemit_functions.BatchMatMul"] = helper.make_function(
            domain="spacemit_functions",
            fname="BatchMatMul",
            inputs=["A", "B", "C"],
            outputs=["Y"],
            nodes=[helper.make_node("Sub", ["A", "B"], ["Y"], name="Sub")],
            opset_imports=[helper.make_opsetid("", 13)],
            attributes=["transA", "transB", "transY"],
        )

        executor = TorchExecutor(graph, device="cpu")
        outputs = executor.forward(
            inputs={
                "x": torch.tensor([[1.0, 2.0]]),
                "w": torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
                "b": torch.tensor([[0.5, -1.0]]),
            },
            output_names=["y"],
        )

        self.assertTrue(torch.allclose(outputs[0], torch.tensor([[5.5, 10.0]])))

    def test_batch_matmul_forward_honors_transpose_attrs(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 2])
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 3])
        node = helper.make_node(
            "BatchMatMul",
            ["x", "w", "b"],
            ["y"],
            name="custom_batch_matmul_transposed",
            domain="spacemit_functions",
            transA=1,
            transB=1,
            transY=1,
        )
        model = helper.make_model(
            helper.make_graph([node], "batch_matmul_transpose_graph", [x, w, b], [y]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("spacemit_functions", 1)],
        )
        model.functions.append(_build_batch_matmul_function())

        graph = OnnxParser().build(model)
        executor = TorchExecutor(graph, device="cpu")

        lhs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rhs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        bias = torch.tensor(0.5)

        outputs = executor.forward(
            inputs={"x": lhs, "w": rhs, "b": bias},
            output_names=["y"],
        )

        expected = torch.matmul(lhs.transpose(-1, -2), rhs.transpose(-1, -2))
        expected = (expected + bias).transpose(-1, -2)
        self.assertTrue(torch.allclose(outputs[0], expected))
        self.assertEqual(list(outputs[0].shape), [4, 3])


if __name__ == "__main__":
    unittest.main()
