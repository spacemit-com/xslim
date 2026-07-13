"""Lightweight unit tests for torch executor operator execution."""

import os
import sys
import unittest

import onnx
import torch
from onnx import TensorProto, helper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xslim.optimizer.fusion import FlattenGemmFusionPass
from xslim.ppq_decorator.ppq.core import (DataType, QuantizationPolicy,
                                          QuantizationProperty,
                                          QuantizationStates, RoundingPolicy,
                                          TargetPlatform,
                                          TensorQuantizationConfig)
from xslim.ppq_decorator.ppq.executor.op import (DEFAULT_BACKEND_TABLE,
                                                 TorchBackendContext)
from xslim.ppq_decorator.ppq.executor.torch import TorchExecutor
from xslim.ppq_decorator.ppq.IR import BaseGraph, Operation, Variable
from xslim.ppq_decorator.ppq.IR.base.opdef import DEFAULT_SOCKET_TABLE, Opset
from xslim.ppq_decorator.ppq.parser.onnx_parser import OnnxParser
from xslim.quantizer.xslim import XSlimQuantizer


def make_op(name, op_type, attributes=None, num_inputs=1, num_outputs=1):
    """Helper to create a minimal Operation with linked Variables."""
    attributes = attributes or {}
    inputs = [Variable(name=f"{name}_in_{i}") for i in range(num_inputs)]
    outputs = [Variable(name=f"{name}_out_{i}") for i in range(num_outputs)]
    op = Operation(
        name=name,
        op_type=op_type,
        attributes=attributes,
        platform=TargetPlatform.UNSPECIFIED,
        inputs=inputs,
        outputs=outputs,
    )
    for v in inputs:
        v._dest_ops.append(op)
    for v in outputs:
        v._source_op = op
    return op


CTX = TorchBackendContext(executing_device="cpu")


class TestArithmeticOps(unittest.TestCase):
    """Test arithmetic operator forward functions."""

    def test_add(self):
        op = make_op("add", "Add", num_inputs=2)
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = DEFAULT_BACKEND_TABLE["Add"](op, [a, b], CTX)
        torch.testing.assert_close(result, a + b)

    def test_add_broadcast(self):
        op = make_op("add_bc", "Add", num_inputs=2)
        a = torch.randn(2, 3)
        b = torch.randn(3)
        result = DEFAULT_BACKEND_TABLE["Add"](op, [a, b], CTX)
        torch.testing.assert_close(result, a + b)

    def test_mul(self):
        op = make_op("mul", "Mul", num_inputs=2)
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        result = DEFAULT_BACKEND_TABLE["Mul"](op, [a, b], CTX)
        torch.testing.assert_close(result, a * b)

    def test_sub(self):
        op = make_op("sub", "Sub", num_inputs=2)
        a = torch.tensor([5.0, 3.0])
        b = torch.tensor([1.0, 2.0])
        result = DEFAULT_BACKEND_TABLE["Sub"](op, [a, b], CTX)
        torch.testing.assert_close(result, (a - b).float())

    def test_div(self):
        op = make_op("div", "Div", num_inputs=2)
        a = torch.tensor([6.0, 8.0])
        b = torch.tensor([2.0, 4.0])
        result = DEFAULT_BACKEND_TABLE["Div"](op, [a, b], CTX)
        torch.testing.assert_close(result, a / b)


class TestActivationOps(unittest.TestCase):
    """Test activation operator forward functions."""

    def test_relu(self):
        op = make_op("relu", "Relu")
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        result = DEFAULT_BACKEND_TABLE["Relu"](op, [x], CTX)
        torch.testing.assert_close(result, torch.relu(x))

    def test_sigmoid(self):
        op = make_op("sigmoid", "Sigmoid")
        x = torch.tensor([-2.0, 0.0, 2.0])
        result = DEFAULT_BACKEND_TABLE["Sigmoid"](op, [x], CTX)
        torch.testing.assert_close(result, torch.sigmoid(x))

    def test_exp(self):
        op = make_op("exp", "Exp")
        x = torch.tensor([0.0, 1.0, 2.0])
        result = DEFAULT_BACKEND_TABLE["Exp"](op, [x], CTX)
        torch.testing.assert_close(result, torch.exp(x))

    def test_tanh(self):
        op = make_op("tanh", "Tanh")
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = DEFAULT_BACKEND_TABLE["Tanh"](op, [x], CTX)
        torch.testing.assert_close(result, torch.tanh(x))

    def test_softmax(self):
        op = make_op("softmax", "Softmax", attributes={"axis": -1})
        op.opset = Opset(version=13)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = DEFAULT_BACKEND_TABLE["Softmax"](op, [x], CTX)
        torch.testing.assert_close(result, torch.softmax(x, dim=-1))

    def test_leaky_relu(self):
        op = make_op("leaky_relu", "LeakyRelu", attributes={"alpha": 0.01})
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0])
        result = DEFAULT_BACKEND_TABLE["LeakyRelu"](op, [x], CTX)
        torch.testing.assert_close(result, torch.nn.functional.leaky_relu(x, 0.01))

    def test_additional_activation_ops(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        cases = [
            ("Celu", {"alpha": 0.75}, torch.nn.functional.celu(x, alpha=0.75)),
            ("Mish", {}, x * torch.tanh(torch.nn.functional.softplus(x))),
            ("Shrink", {"bias": 0.25, "lambd": 0.5}, torch.tensor([-1.75, 0.0, 0.0, 0.0, 1.75])),
            ("Softsign", {}, x / (1 + torch.abs(x))),
            ("ThresholdedRelu", {"alpha": 0.5}, torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])),
        ]
        for op_type, attributes, expected in cases:
            with self.subTest(op_type=op_type):
                op = make_op(op_type.lower(), op_type, attributes=attributes)
                result = DEFAULT_BACKEND_TABLE[op_type](op, [x], CTX)
                torch.testing.assert_close(result, expected)

    def test_hardmax(self):
        op = make_op("hardmax", "Hardmax", attributes={"axis": 1})
        x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
        result = DEFAULT_BACKEND_TABLE["Hardmax"](op, [x], CTX)
        expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(result, expected)


class TestUnaryOps(unittest.TestCase):
    """Test unary operator forward functions."""

    def test_abs(self):
        op = make_op("abs", "Abs")
        x = torch.tensor([-3.0, -1.0, 0.0, 2.0])
        result = DEFAULT_BACKEND_TABLE["Abs"](op, [x], CTX)
        torch.testing.assert_close(result, x.abs())

    def test_sqrt(self):
        op = make_op("sqrt", "Sqrt")
        x = torch.tensor([1.0, 4.0, 9.0])
        result = DEFAULT_BACKEND_TABLE["Sqrt"](op, [x], CTX)
        torch.testing.assert_close(result, torch.sqrt(x))

    def test_neg(self):
        op = make_op("neg", "Neg")
        x = torch.tensor([1.0, -2.0, 3.0])
        result = DEFAULT_BACKEND_TABLE["Neg"](op, [x], CTX)
        torch.testing.assert_close(result, -x)

    def test_log(self):
        op = make_op("log", "Log")
        x = torch.tensor([1.0, 2.0, 3.0])
        result = DEFAULT_BACKEND_TABLE["Log"](op, [x], CTX)
        torch.testing.assert_close(result, torch.log(x))

    def test_floor(self):
        op = make_op("floor", "Floor")
        x = torch.tensor([1.5, 2.7, -0.3])
        result = DEFAULT_BACKEND_TABLE["Floor"](op, [x], CTX)
        torch.testing.assert_close(result, torch.floor(x))

    def test_reciprocal(self):
        op = make_op("reciprocal", "Reciprocal")
        x = torch.tensor([1.0, 2.0, 4.0])
        result = DEFAULT_BACKEND_TABLE["Reciprocal"](op, [x], CTX)
        torch.testing.assert_close(result, 1.0 / x)

    def test_additional_unary_math_ops(self):
        x = torch.tensor([1.25, 1.5, 2.0])
        cases = [
            ("Acos", torch.tensor([-0.5, 0.0, 0.5]), torch.acos(torch.tensor([-0.5, 0.0, 0.5]))),
            ("Acosh", x, torch.acosh(x)),
            ("Asin", torch.tensor([-0.5, 0.0, 0.5]), torch.asin(torch.tensor([-0.5, 0.0, 0.5]))),
            ("Asinh", x, torch.asinh(x)),
            ("Atan", x, torch.atan(x)),
            ("Atanh", torch.tensor([-0.5, 0.0, 0.5]), torch.atanh(torch.tensor([-0.5, 0.0, 0.5]))),
            ("Ceil", torch.tensor([-1.2, 0.2, 1.8]), torch.tensor([-1.0, 1.0, 2.0])),
            ("Cosh", x, torch.cosh(x)),
            ("Round", torch.tensor([-1.5, 0.6, 2.4]), torch.tensor([-2.0, 1.0, 2.0])),
            ("Sign", torch.tensor([-3.0, 0.0, 2.0]), torch.tensor([-1.0, 0.0, 1.0])),
            ("Sinh", x, torch.sinh(x)),
        ]
        for op_type, value, expected in cases:
            with self.subTest(op_type=op_type):
                op = make_op(op_type.lower(), op_type)
                result = DEFAULT_BACKEND_TABLE[op_type](op, [value], CTX)
                torch.testing.assert_close(result, expected)

    def test_isinf_and_isnan(self):
        x = torch.tensor([float("-inf"), -1.0, float("nan"), float("inf")])
        isinf = DEFAULT_BACKEND_TABLE["IsInf"](make_op("isinf", "IsInf"), [x], CTX)
        isnan = DEFAULT_BACKEND_TABLE["IsNaN"](make_op("isnan", "IsNaN"), [x], CTX)
        torch.testing.assert_close(isinf, torch.tensor([True, False, False, True]))
        torch.testing.assert_close(isnan, torch.tensor([False, False, True, False]))


class TestReductionOps(unittest.TestCase):
    """Test reduction operator forward functions."""

    def test_reduce_mean_with_axes_input(self):
        op = make_op("reduce_mean", "ReduceMean", attributes={"keepdims": 1}, num_inputs=2)
        op.opset = Opset(version=18)
        x = torch.randn(1, 2048, 9, 9)
        axes = torch.tensor([-1, -2], dtype=torch.int64)

        result = DEFAULT_BACKEND_TABLE["ReduceMean"](op, [x, axes], CTX)

        torch.testing.assert_close(result, torch.mean(x, dim=(-1, -2), keepdim=True))

    def test_reduce_mean_flattens_axes_input(self):
        op = make_op("reduce_mean_nested_axes", "ReduceMean", attributes={"keepdims": 1}, num_inputs=2)
        op.opset = Opset(version=18)
        x = torch.randn(2, 3, 4)
        axes = torch.tensor([[1.0]], dtype=torch.float32)

        result = DEFAULT_BACKEND_TABLE["ReduceMean"](op, [x, axes], CTX)

        torch.testing.assert_close(result, torch.mean(x, dim=(1,), keepdim=True))

    def test_reduce_l2_with_axes_input(self):
        op = make_op("reduce_l2", "ReduceL2", attributes={"keepdims": 1}, num_inputs=2)
        op.opset = Opset(version=18)
        x = torch.arange(1.0, 49.0).reshape(2, 2, 3, 4)
        axes = torch.tensor([1, 2, 3], dtype=torch.int64)

        result = DEFAULT_BACKEND_TABLE["ReduceL2"](op, [x, axes], CTX)

        torch.testing.assert_close(
            result,
            torch.linalg.vector_norm(x, ord=2, dim=(1, 2, 3), keepdim=True),
        )

    def test_reduce_max_with_axes_input(self):
        op = make_op("reduce_max", "ReduceMax", attributes={"keepdims": 1}, num_inputs=2)
        op.opset = Opset(version=18)
        x = torch.tensor(
            [[[1.0, 5.0], [4.0, 3.0]], [[2.0, 0.0], [6.0, 1.0]]]
        )
        axes = torch.tensor([1, 2], dtype=torch.int64)

        result = DEFAULT_BACKEND_TABLE["ReduceMax"](op, [x, axes], CTX)

        torch.testing.assert_close(result, torch.amax(x, dim=(1, 2), keepdim=True))

    def test_reduce_max_noop_with_empty_axes(self):
        op = make_op(
            "reduce_max_noop",
            "ReduceMax",
            attributes={"keepdims": 1, "noop_with_empty_axes": 1},
            num_inputs=2,
        )
        op.opset = Opset(version=18)
        x = torch.randn(2, 3)
        axes = torch.tensor([], dtype=torch.int64)

        result = DEFAULT_BACKEND_TABLE["ReduceMax"](op, [x, axes], CTX)

        torch.testing.assert_close(result, x)

    def test_reduce_ops_noop_with_omitted_axes(self):
        x = torch.tensor([[-2.0, 3.0], [4.0, 5.0]])
        cases = [
            ("ReduceL1", torch.abs(x)),
            ("ReduceL2", torch.abs(x)),
            ("ReduceLogSum", torch.log(x)),
            ("ReduceLogSumExp", x),
            ("ReduceMax", x),
            ("ReduceMean", x),
            ("ReduceMin", x),
            ("ReduceProd", x),
            ("ReduceSum", x),
            ("ReduceSumSquare", torch.square(x)),
        ]
        for op_type, expected in cases:
            with self.subTest(op_type=op_type):
                op = make_op(
                    op_type.lower(),
                    op_type,
                    attributes={"noop_with_empty_axes": 1},
                    num_inputs=2,
                )
                op.opset = Opset(version=13 if op_type == "ReduceSum" else 18)
                result = DEFAULT_BACKEND_TABLE[op_type](op, [x], CTX)
                torch.testing.assert_close(result, expected, equal_nan=True)

    def test_reduce_prod_with_multiple_negative_axes(self):
        op = make_op(
            "reduce_prod",
            "ReduceProd",
            attributes={"keepdims": 0},
            num_inputs=2,
        )
        op.opset = Opset(version=18)
        x = torch.arange(1.0, 25.0).reshape(2, 3, 4)
        axes = torch.tensor([-1, -2], dtype=torch.int64)

        result = DEFAULT_BACKEND_TABLE["ReduceProd"](op, [x, axes], CTX)

        torch.testing.assert_close(result, torch.prod(x.reshape(2, -1), dim=1))

    def test_additional_reduce_ops(self):
        x = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
        axes = torch.tensor([1], dtype=torch.int64)
        cases = [
            ("ReduceL1", torch.sum(torch.abs(x), dim=1, keepdim=True)),
            ("ReduceLogSum", torch.log(torch.sum(x, dim=1, keepdim=True))),
            ("ReduceLogSumExp", torch.logsumexp(x, dim=1, keepdim=True)),
            ("ReduceMin", torch.amin(x, dim=1, keepdim=True)),
            ("ReduceProd", torch.prod(x, dim=1, keepdim=True)),
            ("ReduceSumSquare", torch.sum(torch.square(x), dim=1, keepdim=True)),
        ]
        for op_type, expected in cases:
            with self.subTest(op_type=op_type):
                op = make_op(op_type.lower(), op_type, attributes={"keepdims": 1}, num_inputs=2)
                op.opset = Opset(version=18)
                result = DEFAULT_BACKEND_TABLE[op_type](op, [x, axes], CTX)
                torch.testing.assert_close(result, expected)

    def test_reduce_ops_accept_scalar_input(self):
        x = torch.tensor(2.0)
        cases = [
            ("ReduceL1", torch.sum(torch.abs(x))),
            ("ReduceLogSum", torch.log(torch.sum(x))),
            ("ReduceLogSumExp", torch.logsumexp(x.reshape(-1), dim=0)),
            ("ReduceMax", torch.max(x)),
            ("ReduceMean", torch.mean(x)),
            ("ReduceMin", torch.min(x)),
            ("ReduceProd", torch.prod(x)),
            ("ReduceSumSquare", torch.sum(torch.square(x))),
        ]
        for op_type, expected in cases:
            with self.subTest(op_type=op_type):
                op = make_op(op_type.lower(), op_type)
                op.opset = Opset(version=18)
                result = DEFAULT_BACKEND_TABLE[op_type](op, [x], CTX)
                torch.testing.assert_close(result, expected)


class TestTensorManipulationOps(unittest.TestCase):
    """Test tensor manipulation operator forward functions."""

    def test_reshape(self):
        op = make_op("reshape", "Reshape", num_inputs=2)
        data = torch.randn(2, 3, 4)
        shape = torch.tensor([2, 12], dtype=torch.int64)
        result = DEFAULT_BACKEND_TABLE["Reshape"](op, [data, shape], CTX)
        self.assertEqual(result.shape, torch.Size([2, 12]))

    def test_transpose(self):
        op = make_op("transpose", "Transpose", attributes={"perm": [0, 2, 1]})
        x = torch.randn(2, 3, 4)
        result = DEFAULT_BACKEND_TABLE["Transpose"](op, [x], CTX)
        self.assertEqual(result.shape, torch.Size([2, 4, 3]))
        torch.testing.assert_close(result, x.permute(0, 2, 1))

    def test_concat(self):
        op = make_op("concat", "Concat", attributes={"axis": 0}, num_inputs=2)
        a = torch.randn(2, 3)
        b = torch.randn(3, 3)
        result = DEFAULT_BACKEND_TABLE["Concat"](op, [a, b], CTX)
        self.assertEqual(result.shape, torch.Size([5, 3]))

    def test_flatten(self):
        op = make_op("flatten", "Flatten", attributes={"axis": 1})
        x = torch.randn(2, 3, 4)
        result = DEFAULT_BACKEND_TABLE["Flatten"](op, [x], CTX)
        self.assertEqual(result.shape, torch.Size([2, 12]))

    def test_squeeze(self):
        op = make_op("squeeze", "Squeeze", num_inputs=2)
        op.opset = Opset(version=13)
        x = torch.randn(1, 3, 1, 4)
        axes = torch.tensor([0], dtype=torch.int64)
        result = DEFAULT_BACKEND_TABLE["Squeeze"](op, [x, axes], CTX)
        self.assertEqual(result.shape, torch.Size([3, 1, 4]))

    def test_unsqueeze(self):
        op = make_op("unsqueeze", "Unsqueeze", num_inputs=2)
        op.opset = Opset(version=13)
        x = torch.randn(3, 4)
        axes = torch.tensor([0], dtype=torch.int64)
        result = DEFAULT_BACKEND_TABLE["Unsqueeze"](op, [x, axes], CTX)
        self.assertEqual(result.shape, torch.Size([1, 3, 4]))


class TestMatrixOps(unittest.TestCase):
    """Test matrix operation forward functions."""

    def test_matmul_2d(self):
        op = make_op("matmul", "MatMul", num_inputs=2)
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        result = DEFAULT_BACKEND_TABLE["MatMul"](op, [a, b], CTX)
        torch.testing.assert_close(result, torch.matmul(a, b))

    def test_matmul_batch(self):
        op = make_op("matmul_batch", "MatMul", num_inputs=2)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        result = DEFAULT_BACKEND_TABLE["MatMul"](op, [a, b], CTX)
        torch.testing.assert_close(result, torch.matmul(a, b))

    def test_gemm(self):
        op = make_op("gemm", "Gemm", attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}, num_inputs=3)
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.randn(5)
        result = DEFAULT_BACKEND_TABLE["Gemm"](op, [a, b, c], CTX)
        expected = torch.matmul(a, b) + c
        torch.testing.assert_close(result, expected)


class TestCustomOps(unittest.TestCase):
    """Test custom operator forward functions."""

    @staticmethod
    def _build_yolo_decode_inputs():
        input_value = torch.tensor(
            [
                [
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [3.0, 0.0],
                    [0.0, 3.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [0.2, -0.2],
                    [0.4, -0.4],
                ]
            ],
            dtype=torch.float32,
        )
        flat_weight = torch.tensor([0.0, 1.0], dtype=torch.float32)
        sub_const = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]], dtype=torch.float32)
        add_const = torch.tensor([[[11.0, 21.0], [31.0, 41.0]]], dtype=torch.float32)
        mul_const = torch.tensor(
            [[[8.0, 8.0], [8.0, 8.0], [16.0, 16.0], [16.0, 16.0]]],
            dtype=torch.float32,
        )
        return input_value, flat_weight, sub_const, add_const, mul_const

    @staticmethod
    def _reference_yolo_decode(input_value, flat_weight, sub_const, add_const, mul_const):
        batch_size, _, spatial_dim = input_value.shape
        reg_max = int(flat_weight.numel())

        bbox_input = input_value[:, : reg_max * 4, :].reshape(batch_size, 4, reg_max, spatial_dim)
        bbox_input = bbox_input.permute(0, 2, 1, 3)
        bbox_input = torch.softmax(bbox_input, dim=1)

        conv_weight = flat_weight.reshape(1, reg_max, 1, 1)
        bbox_output = torch.nn.functional.conv2d(bbox_input, conv_weight).reshape(batch_size, 4, spatial_dim)

        bbox_sub = sub_const - bbox_output[:, :2, :]
        bbox_add = add_const + bbox_output[:, 2:4, :]
        bbox_adjusted = bbox_add - bbox_sub
        bbox_div = (bbox_sub + bbox_add) / 2.0
        bbox_scaled = torch.cat([bbox_div, bbox_adjusted], dim=1) * mul_const
        cls_sigmoid = torch.sigmoid(input_value[:, reg_max * 4 :, :])
        return torch.cat([bbox_scaled, cls_sigmoid], dim=1)

    def test_yolo_decode(self):
        op = make_op(
            "yolo_decode",
            "YoloDecode",
            attributes={"reg_max": 2, "num_class": 2},
            num_inputs=5,
        )
        values = list(self._build_yolo_decode_inputs())

        result = DEFAULT_BACKEND_TABLE["YoloDecode"](op, values, CTX)
        expected = self._reference_yolo_decode(*values)

        self.assertEqual(result.shape, torch.Size([1, 6, 2]))
        torch.testing.assert_close(result, expected)

    def test_yolo_decode_infers_num_class(self):
        op = make_op(
            "yolo_decode_infer_cls",
            "YoloDecode",
            attributes={"reg_max": 2, "num_class": -1},
            num_inputs=5,
        )
        values = list(self._build_yolo_decode_inputs())

        result = DEFAULT_BACKEND_TABLE["YoloDecode"](op, values, CTX)
        expected = self._reference_yolo_decode(*values)

        torch.testing.assert_close(result, expected)


class TestOtherOps(unittest.TestCase):
    """Test miscellaneous operator forward functions."""

    def test_clip(self):
        op = make_op("clip", "Clip", num_inputs=3)
        x = torch.tensor([-5.0, -1.0, 0.0, 3.0, 10.0])
        min_val = torch.tensor(-2.0)
        max_val = torch.tensor(5.0)
        result = DEFAULT_BACKEND_TABLE["Clip"](op, [x, min_val, max_val], CTX)
        torch.testing.assert_close(result, torch.clamp(x, -2.0, 5.0))

    def test_identity(self):
        op = make_op("identity", "Identity")
        x = torch.randn(3, 4)
        result = DEFAULT_BACKEND_TABLE["Identity"](op, [x], CTX)
        torch.testing.assert_close(result, x)

    def test_shape(self):
        op = make_op("shape", "Shape")
        x = torch.randn(2, 3, 4)
        result = DEFAULT_BACKEND_TABLE["Shape"](op, [x], CTX)
        expected = torch.tensor([2, 3, 4], dtype=torch.long)
        torch.testing.assert_close(result, expected)

    def test_where(self):
        op = make_op("where", "Where", num_inputs=3)
        cond = torch.tensor([True, False, True])
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = DEFAULT_BACKEND_TABLE["Where"](op, [cond, a, b], CTX)
        torch.testing.assert_close(result, torch.where(cond, a, b))

    def test_additional_logical_ops(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 2.0, 1.0])
        cases = [
            ("GreaterOrEqual", [a, b], torch.ge(a, b)),
            ("LessOrEqual", [a, b], torch.le(a, b)),
            ("Xor", [a > 1, b > 1], torch.logical_xor(a > 1, b > 1)),
        ]
        for op_type, values, expected in cases:
            with self.subTest(op_type=op_type):
                op = make_op(op_type.lower(), op_type, num_inputs=2)
                result = DEFAULT_BACKEND_TABLE[op_type](op, values, CTX)
                torch.testing.assert_close(result, expected)

    def test_cast_to_float(self):
        op = make_op("cast", "Cast", attributes={"to": DataType.FP32})
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = DEFAULT_BACKEND_TABLE["Cast"](op, [x], CTX)
        self.assertEqual(result.dtype, torch.float32)

    def test_parsed_onnx_cast_uses_internal_data_type(self):
        x_info = helper.make_tensor_value_info("x", TensorProto.INT64, [3])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])
        cast = helper.make_node(
            "Cast", ["x"], ["y"], name="cast", to=TensorProto.FLOAT
        )
        model = helper.make_model(
            helper.make_graph([cast], "cast_graph", [x_info], [y_info]),
            opset_imports=[helper.make_opsetid("", 18)],
        )

        graph = OnnxParser().build(model)
        self.assertEqual(graph.operations["cast"].attributes["to"], DataType.FP32)
        result = TorchExecutor(graph=graph, device="cpu").forward(
            torch.tensor([1, 2, 3], dtype=torch.int64)
        )[0]
        self.assertEqual(result.dtype, torch.float32)

    def test_constant_of_shape(self):
        op = make_op("const_shape", "ConstantOfShape", attributes={"value": torch.tensor([0.0])})
        shape = torch.tensor([2, 3], dtype=torch.int64)
        result = DEFAULT_BACKEND_TABLE["ConstantOfShape"](op, [shape], CTX)
        self.assertEqual(result.shape, torch.Size([2, 3]))


class TestReduceOps(unittest.TestCase):
    """Test reduce operator forward functions."""

    def test_reduce_mean(self):
        op = make_op("reduce_mean", "ReduceMean", attributes={"axes": [1], "keepdims": 1})
        op.opset = Opset(version=11)
        x = torch.randn(2, 3, 4)
        result = DEFAULT_BACKEND_TABLE["ReduceMean"](op, [x], CTX)
        expected = torch.mean(x, dim=1, keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_reduce_sum(self):
        op = make_op("reduce_sum", "ReduceSum", attributes={"keepdims": 1}, num_inputs=2)
        op.opset = Opset(version=13)
        x = torch.randn(2, 3, 4)
        axes = torch.tensor([1], dtype=torch.int64)
        result = DEFAULT_BACKEND_TABLE["ReduceSum"](op, [x, axes], CTX)
        expected = torch.sum(x, dim=1, keepdim=True)
        torch.testing.assert_close(result, expected)


class TestQuantizerConfig(unittest.TestCase):
    """Test quantizer config generation for operator socket metadata."""

    def test_new_operator_sockets_are_registered(self):
        socket_cases = [
            ("Celu", 1, [TargetPlatform.UNSPECIFIED]),
            ("GreaterOrEqual", 2, [TargetPlatform.UNSPECIFIED, TargetPlatform.UNSPECIFIED]),
            ("LessOrEqual", 2, [TargetPlatform.UNSPECIFIED, TargetPlatform.UNSPECIFIED]),
            ("Mish", 1, [TargetPlatform.UNSPECIFIED]),
            ("Softsign", 1, [TargetPlatform.UNSPECIFIED]),
            ("ThresholdedRelu", 1, [TargetPlatform.UNSPECIFIED]),
            ("Dropout", 3, [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI]),
            ("IsNaN", 1, [TargetPlatform.UNSPECIFIED]),
        ]
        for op_type, num_inputs, expected_in_plat in socket_cases:
            with self.subTest(op_type=op_type):
                op = make_op(op_type.lower(), op_type, num_inputs=num_inputs)
                op.opset = Opset(domain="", version=18)
                self.assertEqual(op.socket.in_plat, expected_in_plat)

    def test_quantize_linear_ops_use_default_socket(self):
        for op_type in ("QuantizeLinear", "DequantizeLinear"):
            with self.subTest(op_type=op_type):
                op = make_op(op_type.lower(), op_type, num_inputs=3)
                op.opset = Opset(domain="", version=18)
                self.assertNotIn(op_type, DEFAULT_SOCKET_TABLE)
                self.assertEqual(op.socket.in_plat, [TargetPlatform.UNSPECIFIED] * 3)

    def test_reduce_axes_input_stays_fp32(self):
        reduce_cases = [
            ("ReduceL1", 18),
            ("ReduceL2", 18),
            ("ReduceLogSum", 18),
            ("ReduceLogSumExp", 18),
            ("ReduceMax", 18),
            ("ReduceMean", 18),
            ("ReduceMin", 18),
            ("ReduceProd", 18),
            ("ReduceSum", 18),
            ("ReduceSumSquare", 18),
        ]

        for op_type, opset_version in reduce_cases:
            with self.subTest(op_type=op_type, opset_version=opset_version):
                operation = make_op("reduce", op_type, num_inputs=2)
                operation.opset = Opset(domain="", version=opset_version)

                config = XSlimQuantizer.create_default_quant_config(operation)

                self.assertEqual(
                    config.input_quantization_config[0].state,
                    QuantizationStates.INITIAL,
                )
                self.assertEqual(
                    config.input_quantization_config[1].state,
                    QuantizationStates.FP32,
                )


class TestConvOps(unittest.TestCase):
    """Test convolution operator forward functions."""

    def test_conv2d(self):
        op = make_op(
            "conv",
            "Conv",
            attributes={
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "pads": [1, 1, 1, 1],
                "dilations": [1, 1],
                "group": 1,
            },
            num_inputs=3,
        )
        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(16, 3, 3, 3)
        b = torch.randn(16)
        result = DEFAULT_BACKEND_TABLE["Conv"](op, [x, w, b], CTX)
        self.assertEqual(result.shape, torch.Size([1, 16, 8, 8]))

    def test_conv2d_no_bias(self):
        op = make_op(
            "conv_nb",
            "Conv",
            attributes={
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "pads": [0, 0, 0, 0],
                "dilations": [1, 1],
                "group": 1,
            },
            num_inputs=2,
        )
        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(16, 3, 3, 3)
        result = DEFAULT_BACKEND_TABLE["Conv"](op, [x, w], CTX)
        self.assertEqual(result.shape, torch.Size([1, 16, 6, 6]))


class TestTorchExecutorGraph(unittest.TestCase):
    """Test TorchExecutor with a minimal computation graph."""

    def test_quantize_function_skips_none_optional_input(self):
        executor = TorchExecutor(BaseGraph(name="none_input_graph"), device="cpu")
        config = TensorQuantizationConfig(
            policy=QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL
                + QuantizationProperty.PER_TENSOR
                + QuantizationProperty.LINEAR
            ),
            rounding=RoundingPolicy.ROUND_HALF_EVEN,
            num_of_bits=8,
            quant_min=-127,
            quant_max=128,
            scale=torch.tensor(1.0),
            offset=torch.tensor(0.0),
            state=QuantizationStates.ACTIVATED,
        )

        self.assertIsNone(executor.quantize_function(None, config))

    def _build_simple_graph(self):
        """Build: input -> Relu -> Add(with param) -> output."""
        graph = BaseGraph(name="test_graph")
        graph.set_extension_attrib("IS_DISPATCHED_GRAPH", True)

        # Variables
        input_var = Variable(name="input", shape=[1, 3], dtype=DataType.FP32)
        relu_out = Variable(name="relu_out", shape=[1, 3], dtype=DataType.FP32)
        param_var = Variable(name="param", value=torch.ones(1, 3), is_parameter=True, shape=[1, 3], dtype=DataType.FP32)
        output_var = Variable(name="output", shape=[1, 3], dtype=DataType.FP32)

        # Operations
        relu_op = Operation(name="relu_1", op_type="Relu", attributes={}, platform=TargetPlatform.UNSPECIFIED)
        add_op = Operation(name="add_1", op_type="Add", attributes={}, platform=TargetPlatform.UNSPECIFIED)

        # Link variables to operations
        input_var._dest_ops = [relu_op]
        relu_op._input_vars = [input_var]
        relu_op._output_vars = [relu_out]
        relu_out._source_op = relu_op

        relu_out._dest_ops = [add_op]
        param_var._dest_ops = [add_op]
        add_op._input_vars = [relu_out, param_var]
        add_op._output_vars = [output_var]
        output_var._source_op = add_op

        # Add variables first (those without dest_ops referencing ops not yet in graph)
        graph._variables["input"] = input_var
        graph._variables["param"] = param_var
        graph._variables["relu_out"] = relu_out
        graph._variables["output"] = output_var

        # Add operations
        graph._operations["relu_1"] = relu_op
        graph._operations["add_1"] = add_op

        # Set graph inputs and outputs
        graph._graph_inputs["input"] = input_var
        graph._graph_outputs["output"] = output_var

        return graph

    def test_executor_forward(self):
        graph = self._build_simple_graph()
        executor = TorchExecutor(graph=graph, device="cpu")

        x = torch.tensor([[-1.0, 2.0, -3.0]])
        results = executor.forward(inputs={"input": x}, output_names=["output"])

        expected = torch.relu(x) + torch.ones(1, 3)
        torch.testing.assert_close(results[0], expected)

    def test_executor_default_outputs(self):
        graph = self._build_simple_graph()
        executor = TorchExecutor(graph=graph, device="cpu")

        x = torch.tensor([[1.0, -1.0, 0.5]])
        results = executor.forward(inputs={"input": x})

        expected = torch.relu(x) + torch.ones(1, 3)
        torch.testing.assert_close(results[0], expected)

    def test_executor_list_input(self):
        graph = self._build_simple_graph()
        executor = TorchExecutor(graph=graph, device="cpu")

        x = torch.tensor([[0.5, -0.5, 1.0]])
        results = executor.forward(inputs=[x])

        expected = torch.relu(x) + torch.ones(1, 3)
        torch.testing.assert_close(results[0], expected)

    def test_executor_tensor_input(self):
        graph = self._build_simple_graph()
        executor = TorchExecutor(graph=graph, device="cpu")

        x = torch.tensor([[2.0, -2.0, 0.0]])
        results = executor.forward(inputs=x)

        expected = torch.relu(x) + torch.ones(1, 3)
        torch.testing.assert_close(results[0], expected)


class TestFusionPasses(unittest.TestCase):
    def _build_flatten_gemm_graph(self):
        graph = BaseGraph(name="flatten_gemm_graph")

        input_var = Variable(
            name="input",
            shape=[1, 3, 8],
            dtype=DataType.FP32,
        )
        flatten_out = Variable(
            name="flatten_out",
            shape=[1, 24],
            dtype=DataType.FP32,
        )
        weight_var = Variable(
            name="weight",
            value=torch.randn(4, 24),
            is_parameter=True,
            shape=[4, 24],
            dtype=DataType.FP32,
        )
        gemm_out = Variable(
            name="gemm_out",
            shape=[1, 4],
            dtype=DataType.FP32,
        )

        flatten_op = Operation(
            name="flatten",
            op_type="Flatten",
            attributes={"axis": 1},
            platform=TargetPlatform.UNSPECIFIED,
        )
        gemm_op = Operation(
            name="gemm",
            op_type="Gemm",
            attributes={"alpha": 1, "transA": 0, "transB": 1},
            platform=TargetPlatform.UNSPECIFIED,
        )

        input_var._dest_ops = [flatten_op]
        flatten_op._input_vars = [input_var]
        flatten_op._output_vars = [flatten_out]
        flatten_out._source_op = flatten_op
        flatten_out._dest_ops = [gemm_op]

        weight_var._dest_ops = [gemm_op]
        gemm_op._input_vars = [flatten_out, weight_var]
        gemm_op._output_vars = [gemm_out]
        gemm_out._source_op = gemm_op

        graph._variables[input_var.name] = input_var
        graph._variables[flatten_out.name] = flatten_out
        graph._variables[weight_var.name] = weight_var
        graph._variables[gemm_out.name] = gemm_out

        graph._operations[flatten_op.name] = flatten_op
        graph._operations[gemm_op.name] = gemm_op

        graph._graph_inputs[input_var.name] = input_var
        graph._graph_outputs[gemm_out.name] = gemm_out
        return graph, flatten_op, gemm_op

    def test_flatten_gemm_fusion_uses_spatial_rank_for_conv_attrs(self):
        graph, flatten_op, gemm_op = self._build_flatten_gemm_graph()
        input_var = graph.variables["input"]

        FlattenGemmFusionPass().optimize(
            graph=graph,
            dataloader=[],
            executor=None,
        )

        self.assertEqual(gemm_op.type, "Conv")
        self.assertEqual(gemm_op.attributes["kernel_shape"], [8])
        self.assertEqual(gemm_op.attributes["strides"], [1])
        self.assertEqual(gemm_op.attributes["dilations"], [1])
        self.assertEqual(gemm_op.inputs[0], input_var)
        self.assertIn(gemm_op, input_var.dest_ops)


class TestGridSampleOp(unittest.TestCase):
    """Test GridSample operator forward function."""

    def test_grid_sample_default(self):
        op = make_op("gs", "GridSample", num_inputs=2)
        x = torch.randn(1, 1, 4, 4)
        grid = torch.randn(1, 3, 3, 2).clamp(-1, 1)
        result = DEFAULT_BACKEND_TABLE["GridSample"](op, [x, grid], CTX)
        expected = torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        torch.testing.assert_close(result, expected)

    def test_grid_sample_nearest(self):
        op = make_op("gs_n", "GridSample", attributes={"mode": "nearest"}, num_inputs=2)
        x = torch.randn(1, 1, 4, 4)
        grid = torch.randn(1, 3, 3, 2).clamp(-1, 1)
        result = DEFAULT_BACKEND_TABLE["GridSample"](op, [x, grid], CTX)
        expected = torch.nn.functional.grid_sample(
            x, grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
        torch.testing.assert_close(result, expected)

    def test_grid_sample_align_corners(self):
        op = make_op(
            "gs_ac", "GridSample",
            attributes={"align_corners": 1, "padding_mode": "border"},
            num_inputs=2,
        )
        x = torch.randn(1, 1, 4, 4)
        grid = torch.randn(1, 3, 3, 2).clamp(-1, 1)
        result = DEFAULT_BACKEND_TABLE["GridSample"](op, [x, grid], CTX)
        expected = torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        torch.testing.assert_close(result, expected)


class TestDepthToSpaceOp(unittest.TestCase):
    """Test DepthToSpace operator forward function."""

    def test_depth_to_space_dcr(self):
        op = make_op("d2s", "DepthToSpace", attributes={"blocksize": 2, "mode": "DCR"})
        x = torch.randn(1, 8, 2, 3)
        result = DEFAULT_BACKEND_TABLE["DepthToSpace"](op, [x], CTX)
        expected = torch.nn.functional.pixel_shuffle(x, 2)
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, (1, 2, 4, 6))

    def test_depth_to_space_crd(self):
        op = make_op("d2s_crd", "DepthToSpace", attributes={"blocksize": 2, "mode": "CRD"})
        x = torch.randn(1, 8, 2, 3)
        result = DEFAULT_BACKEND_TABLE["DepthToSpace"](op, [x], CTX)
        self.assertEqual(result.shape, (1, 2, 4, 6))
        # Verify CRD mode manually
        b, c, h, w = x.shape
        blocksize = 2
        tmp = x.reshape(b, c // (blocksize * blocksize), blocksize, blocksize, h, w)
        tmp = tmp.permute(0, 1, 4, 2, 5, 3)
        expected = tmp.reshape(b, c // (blocksize * blocksize), h * blocksize, w * blocksize)
        torch.testing.assert_close(result, expected)

    def test_depth_to_space_default_mode(self):
        op = make_op("d2s_def", "DepthToSpace", attributes={"blocksize": 2})
        x = torch.randn(1, 4, 3, 3)
        result = DEFAULT_BACKEND_TABLE["DepthToSpace"](op, [x], CTX)
        expected = torch.nn.functional.pixel_shuffle(x, 2)
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, (1, 1, 6, 6))


if __name__ == "__main__":
    unittest.main()
