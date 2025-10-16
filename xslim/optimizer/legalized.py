#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import onnx
import torch
from xslim.defs import GLOBAL_FUNCTIONS_MAPPING, MIN_ONNX_OPSET_VERSION
from xslim.logger import logger

from ..ppq_decorator import (
    BaseGraph,
    DataType,
    GraphFormatter,
    GraphMerger,
    Operation,
    Opset,
    SearchableGraph,
    Variable,
    convert_any_to_torch_tensor,
)


class GraphLegalized:
    def __init__(self, graph) -> None:
        self._graph = graph
        self._merger = GraphMerger(self._graph)
        self._formatter = GraphFormatter(self._graph)

    def __call__(self) -> Any:
        self._formatter.remove_constant_input()
        self.remove_empty_sequence_input()
        self._formatter.convert_to_tensor()
        self.format_cast()
        self._formatter.format_parameter()
        self._merger.fuse_bias_add()
        self._merger.fuse_bn()
        self._formatter.format_slice()
        self._formatter.format_clip()
        self._formatter.format_pad()
        self._formatter.format_resize()
        self._formatter.remove_identity()
        self._formatter.delete_isolated()
        self.format_reshape_squeeze()
        self._merger.fuse_layernorm()
        self._merger.fuse_gelu()
        self.format_div()
        self._merger.fuse_bias_add()
        self.remove_dropout()
        self.format_ms_domain()
        self.fuse_mul_add()
        self.fuse_mul_add()
        # self.fuse_matmul_bias()

    def remove_empty_sequence_input(self) -> None:
        removing_ops = []
        for op in self._graph.operations.values():
            if op.type == "SequenceEmpty":
                removing_ops.append(op)

        for const_op in removing_ops:
            assert isinstance(const_op, Operation)
            dtype = DataType(const_op.attributes.get("dtype", DataType.FP32))
            constant_value = np.empty(0, dtype=DataType.to_numpy(dtype))
            output_var = const_op.outputs[0]
            output_var._is_parameter = True
            output_var.value = constant_value
            self._graph.remove_operation(removing_op=const_op)

    def fuse_matmul_bias(self):
        function_impl = BaseGraph(name="BatchMatMul")
        var_A = Variable(name="A", dtype=DataType.FP32)
        var_B = Variable(name="B", dtype=DataType.FP32)
        var_C = Variable(name="C", dtype=DataType.FP32)
        var_temp = Variable(name="MatMul_temp", dtype=DataType.FP32)
        var_Y = Variable(name="Y", dtype=DataType.FP32)
        function_impl.operations["MatMul"] = Operation(
            name="MatMul",
            op_type="MatMul",
            attributes={},
            inputs=[var_A, var_B],
            outputs=[var_temp],
        )
        function_impl.operations["Add"] = Operation(
            name="Add",
            op_type="Add",
            attributes={},
            inputs=[var_temp, var_C],
            outputs=[var_Y],
        )
        var_A.dest_ops.append(function_impl.operations["MatMul"])
        var_B.dest_ops.append(function_impl.operations["MatMul"])
        var_C.dest_ops.append(function_impl.operations["Add"])
        var_temp.dest_ops.append(function_impl.operations["Add"])
        var_temp.source_op = function_impl.operations["MatMul"]
        var_Y.source_op = function_impl.operations["Add"]
        for op in function_impl.operations.values():
            for in_var in op.inputs:
                function_impl.variables[in_var.name] = in_var
            for out_var in op.outputs:
                function_impl.variables[out_var.name] = out_var

        function_impl.inputs[var_A.name] = var_A
        function_impl.inputs[var_B.name] = var_B
        function_impl.inputs[var_C.name] = var_C
        function_impl.outputs[var_Y.name] = var_Y
        function_impl._detail["function_input"] = [var_A.name, var_B.name, var_C.name]
        function_impl._detail["function_output"] = [var_Y.name]
        function_impl._detail["function_domain"] = "spacemit_functions"
        function_impl._detail["function_opset_import"] = [{"domain": "", "version": MIN_ONNX_OPSET_VERSION}]
        function_impl._detail["function_attribute"] = ["transA", "transB", "transY"]

        add_function_impl = False
        for current_op in [_ for _ in self._graph.operations.values()]:
            if current_op.type != "MatMul":
                continue

            next_ops = self._graph.get_downstream_operations(current_op)
            if len(next_ops) != 1:
                continue
            if next_ops[0].type != "Add":
                continue

            fusing_op = next_ops[0]
            if fusing_op.num_of_parameter == 1:
                bias = fusing_op.parameters[0].value
                if bias.ndim in {0, 1}:
                    add_function_impl = True
                    self._graph.remove_operation(fusing_op, keep_coherence=True)
                    self._graph.create_variable(value=bias, is_parameter=True, dest_ops=[current_op])
                    current_op.type = "BatchMatMul"
                    current_op.attributes["domain"] = "spacemit_functions"
                    current_op.attributes["transA"] = 0
                    current_op.attributes["transB"] = 0
                    current_op.attributes["transY"] = 0
                    current_op.opset = Opset("spacemit_functions", version=1)

        if add_function_impl:
            self._graph._detail[GLOBAL_FUNCTIONS_MAPPING]["spacemit_functions.BatchMatMul"] = function_impl

    def format_cast(self):
        interested_ops = []
        for _, operation in self._graph.operations.items():
            assert isinstance(operation, Operation)
            if operation.type == "Cast":
                interested_ops.append(operation)
        for operation in interested_ops:
            assert isinstance(operation, Operation)
            assert "to" in operation.attributes
            if isinstance(operation.attributes["to"], np.dtype):
                operation.attributes["to"] = DataType.convert_from_numpy(operation.attributes["to"])

    def format_div(self):
        for op in self._graph.operations.values():
            if op.type == "Div" and op.inputs[1].is_parameter:
                if op.inputs[1].value.dtype in {torch.float32, torch.float64, torch.float16}:
                    op.type = "Mul"
                    op.inputs[1].value = 1 / op.inputs[1].value

    def format_reshape_squeeze(self):
        search_engine = SearchableGraph(graph=self._graph)
        paths = search_engine.path_matching(
            sp_expr=lambda x: x.type in {"Reshape"}
            and len(x.inputs) > 1
            and x.inputs[1].is_parameter
            and len(x.outputs[0].dest_ops) == 1
            and x.outputs[0].name not in self._graph.outputs,
            rp_expr=lambda x, y: False,
            ep_expr=lambda x: x.type in {"Squeeze"},
            direction="down",
        )

        for path in paths:
            path = path.tolist()
            assert len(path) == 2, "Oops seems we got something unexpected."
            reshape_op, squeeze_op = path
            assert isinstance(reshape_op, Operation) and isinstance(squeeze_op, Operation)
            reshape_size = reshape_op.inputs[1].value
            squeeze_axes = squeeze_op.attributes.get("axes", None)
            if len(squeeze_op.inputs) > 1 and squeeze_op.inputs[1].is_parameter:
                squeeze_axes = squeeze_op.inputs[1].value.numpy().tolist()

            if squeeze_axes is not None and all([reshape_size[axes] == 1 for axes in squeeze_axes]):
                new_shape = [int(s) for i, s in enumerate(reshape_size) if i not in squeeze_axes]

                reshape_op.outputs[0] = squeeze_op.outputs[0]
                squeeze_op.outputs[0].source_op = reshape_op
                reshape_op.inputs[1].value = convert_any_to_torch_tensor(
                    new_shape, device=reshape_size.device, dtype=reshape_size.dtype
                )

                for in_var in squeeze_op.inputs:
                    in_var.dest_ops.clear()
                    in_var.source_op = None
                    self._graph.remove_variable(in_var)

                squeeze_op.inputs.clear()
                squeeze_op.outputs.clear()
                self._graph.remove_operation(squeeze_op)

    def remove_dropout(self):
        removing_ops = []
        for op in self._graph.operations.values():
            if op.type == "Dropout":
                removing_ops.append(op)

        for op in removing_ops:
            in_var = op.inputs[0]
            out_var = op.outputs[0]

            in_var.dest_ops.remove(op)
            out_var.source_op = None

            for var in op.inputs[1:]:
                var.dest_ops.remove(op)
                self._graph.remove_variable(var)

            for var in op.outputs[1:]:
                var.source_op = None
                self._graph.remove_variable(var)

            op.inputs.clear()
            op.outputs.clear()

            for dest_op, dest_idx in zip([_ for _ in out_var.dest_ops], [_ for _ in out_var.dest_idx]):
                dest_op.inputs[dest_idx] = in_var
                in_var.dest_ops.append(dest_op)

            self._graph.remove_operation(op)
            self._graph.remove_variable(out_var)

    def format_ms_domain(self):
        has_ms_domain = False
        for op in self._graph.operations.values():
            if op.type in {"Gelu"}:
                op.attributes["domain"] = "com.microsoft"
                has_ms_domain = True
        if has_ms_domain:
            self._graph._detail["pb_opset_import"].append({"domain": "com.microsoft", "version": 1})

    def fuse_mul_add(self):
        search_engine = SearchableGraph(graph=self._graph)
        paths = search_engine.path_matching(
            sp_expr=lambda x: x.type in {"Conv", "Gemm", "ConvTranspose"} and x.num_of_parameter > 0,
            rp_expr=lambda x, y: False,
            ep_expr=lambda x: x.type in {"Mul", "Div", "Add", "Sub"}
            and any([in_var.is_parameter for in_var in x.inputs]),
            direction="down",
        )

        for path in paths:
            path = path.tolist()
            assert len(path) == 2, "Oops seems we got something unexpected."

            computing_op, mul_op = path
            assert isinstance(computing_op, Operation) and isinstance(mul_op, Operation)

            if (
                len(self._graph.get_downstream_operations(computing_op)) != 1
                or len(self._graph.get_upstream_operations(mul_op)) != 1
            ):
                continue

            parameter_index = [in_var.is_parameter for in_var in mul_op.inputs].index(True)
            feature_index = 1 - parameter_index
            parameter = mul_op.inputs[parameter_index]
            feature = mul_op.inputs[feature_index]

            if computing_op.num_of_parameter == 1:
                w = computing_op.parameters[0].value  # no bias.
                assert isinstance(w, torch.Tensor), "values of parameters are assumed as torch Tensor"
                if computing_op.type == "ConvTranspose":
                    b = torch.zeros(w.shape[1] * computing_op.attributes.get("group", 1)).to(w.device)
                elif computing_op.type == "Gemm" and computing_op.attributes.get("transB", 0) == 0:
                    b = torch.zeros(w.shape[1]).to(w.device)
                else:
                    b = torch.zeros(w.shape[0]).to(w.device)
            else:
                w, b = [var.value for var in computing_op.parameters[:2]]  # has bias.

            if parameter.value.numel() == 1 or parameter.value.ndim == w.ndim:
                pass
            else:
                continue

            if parameter.value.numel() == 1 or (
                parameter.value.ndim >= 2 and parameter.value.shape[1] == parameter.value.numel()
            ):
                pass
            else:
                continue

            if mul_op.type in {"Mul", "Div"}:
                alpha = parameter.value
                if alpha.dtype not in {torch.float32, torch.float64, torch.float16}:
                    continue
                if mul_op.type == "Div":
                    if parameter_index != 1:
                        continue
                    alpha = 1.0 / alpha

                if computing_op.type == "Conv":
                    # calculate new weight and bias
                    scale = alpha
                    w = w * scale.reshape([-1] + [1] * (w.ndim - 1))
                    b = alpha.reshape(-1) * b

                elif computing_op.type == "Gemm":
                    # calculate new weight and bias
                    scale = alpha
                    if computing_op.attributes.get("transB", 0):
                        w = w * scale.reshape([-1, 1])
                    else:
                        w = w * scale.reshape([1, -1])
                    b = alpha.reshape(-1) * b

                elif computing_op.type == "ConvTranspose":
                    scale = alpha
                    group = computing_op.attributes.get("group", 1)
                    scale = scale.reshape([group, 1, -1] + [1] * (w.ndim - 2))
                    w = w.reshape([group, -1] + list(w.shape[1:])) * scale
                    w = w.reshape([w.shape[0] * w.shape[1]] + list(w.shape[2:]))
                    b = alpha.reshape(-1) * b
                else:
                    raise TypeError(
                        f"Unexpected op type {computing_op.type}. "
                        f"Can not merge {computing_op.name} with {mul_op.name}"
                    )
            else:
                beta = parameter.value
                if mul_op.type == "Div":
                    if parameter_index != 1:
                        continue
                    beta = -beta
                b = beta.reshape(-1) + b

            # create new op and variable
            merged_op = Operation(
                computing_op.name, op_type=computing_op.type, attributes=computing_op.attributes.copy()
            )
            weight_var = Variable(computing_op.name + "_weight", w, True, [merged_op])
            bias_var = Variable(computing_op.name + "_bias", b, True, [merged_op])

            # replace & dirty work
            input_var = computing_op.inputs[0]
            output_var = mul_op.outputs[0]

            input_var.dest_ops.remove(computing_op)
            input_var.dest_ops.append(merged_op)

            output_var.source_op = merged_op

            # delete old operations
            computing_op.inputs.pop(0)
            mul_op.outputs.clear()
            self._graph.remove_operation(computing_op)
            self._graph.remove_operation(mul_op)

            # insert new
            self._graph.append_operation(merged_op)
            merged_op.inputs.extend([input_var, weight_var, bias_var])
            merged_op.outputs.extend([output_var])

            self._graph.append_variable(weight_var)
            self._graph.append_variable(bias_var)
