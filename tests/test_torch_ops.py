"""Lightweight unit tests for torch executor operator execution."""

import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xslim.ppq_decorator.ppq.core import DataType, TargetPlatform
from xslim.ppq_decorator.ppq.IR import BaseGraph, Operation, Variable
from xslim.ppq_decorator.ppq.IR.base.opdef import Opset
from xslim.ppq_decorator.ppq.executor.op import TorchBackendContext, DEFAULT_BACKEND_TABLE
from xslim.ppq_decorator.ppq.executor.torch import TorchExecutor


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
        op._opset = Opset(version=13)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = DEFAULT_BACKEND_TABLE["Softmax"](op, [x], CTX)
        torch.testing.assert_close(result, torch.softmax(x, dim=-1))

    def test_leaky_relu(self):
        op = make_op("leaky_relu", "LeakyRelu", attributes={"alpha": 0.01})
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0])
        result = DEFAULT_BACKEND_TABLE["LeakyRelu"](op, [x], CTX)
        torch.testing.assert_close(result, torch.nn.functional.leaky_relu(x, 0.01))


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
        op._opset = Opset(version=13)
        x = torch.randn(1, 3, 1, 4)
        axes = torch.tensor([0], dtype=torch.int64)
        result = DEFAULT_BACKEND_TABLE["Squeeze"](op, [x, axes], CTX)
        self.assertEqual(result.shape, torch.Size([3, 1, 4]))

    def test_unsqueeze(self):
        op = make_op("unsqueeze", "Unsqueeze", num_inputs=2)
        op._opset = Opset(version=13)
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

    def test_cast_to_float(self):
        op = make_op("cast", "Cast", attributes={"to": DataType.FP32})
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = DEFAULT_BACKEND_TABLE["Cast"](op, [x], CTX)
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
        op._opset = Opset(version=11)
        x = torch.randn(2, 3, 4)
        result = DEFAULT_BACKEND_TABLE["ReduceMean"](op, [x], CTX)
        expected = torch.mean(x, dim=1, keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_reduce_sum(self):
        op = make_op("reduce_sum", "ReduceSum", attributes={"keepdims": 1}, num_inputs=2)
        op._opset = Opset(version=13)
        x = torch.randn(2, 3, 4)
        axes = torch.tensor([1], dtype=torch.int64)
        result = DEFAULT_BACKEND_TABLE["ReduceSum"](op, [x, axes], CTX)
        expected = torch.sum(x, dim=1, keepdim=True)
        torch.testing.assert_close(result, expected)


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
