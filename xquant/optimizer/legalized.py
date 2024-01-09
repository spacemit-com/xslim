#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Any, Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
from ppq.core import (
    ppq_warning,
    convert_any_to_torch_tensor,
)
from ppq.IR import Operation, Variable
from ppq.IR import GraphMerger
from ppq.IR.search import SearchableGraph


class GraphLegalized:
    def __init__(self, graph) -> None:
        self._graph = graph
        self._merger = GraphMerger(self._graph)

    def __call__(self) -> Any:
        self.format_reshape_squeeze()
        self.fuse_layernorm()
        self.fuse_gelu()
        self._merger.fuse_bias_add()
        self.remove_dropout()
        self.format_ms_domain()
        self.fuse_matmul_add()
        self.fuse_mul_add()
        self.fuse_mul_add()
        self.format_gemm()
        self.format_gemm()

    def fuse_matmul_add(self):
        graph = self._graph
        pass
        # for current_op in [_ for _ in graph.operations.values()]:
        #    if current_op.type != "MatMul":
        #        continue

    #
    #    if current_op.inputs[1].is_parameter:
    #        pass

    # check down-stream op is add
    # next_ops = graph.get_downstream_operations(current_op)
    # if len(next_ops) != 1:
    #    continue
    # if next_ops[0].type != "Add":
    #    continue
    ## check if is a constant add
    # fusing_op = next_ops[0]
    # if current_op.inputs[1].is_parameter and fusing_op.num_of_parameter == 1:
    #    pass
    # elif fusing_op.num_of_parameter == 1:
    #    # do graph fusion
    #    bias = fusing_op.parameters[0].value
    #    graph.remove_operation(fusing_op, keep_coherence=True)
    #    graph.create_variable(value=bias, is_parameter=True, dest_ops=[current_op])
    #    current_op.type = "PPQBiasFusedMatMul"

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
            squeeze_axes = squeeze_op.attributes["axes"]

            if all([reshape_size[axes] == 1 for axes in squeeze_axes]):
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

    def format_gemm(self):
        search_engine = SearchableGraph(graph=self._graph)
        paths = search_engine.path_matching(
            sp_expr=lambda x: x.type in {"Flatten"} and x.attributes.get("axis", 0) == 1,
            rp_expr=lambda x, y: False,
            ep_expr=lambda x: x.type in {"Gemm"}
            and x.attributes.get("alpha", 1) == 1
            and x.attributes.get("transA", 0) == 0
            and len(x.inputs) >= 2
            and x.inputs[1].is_parameter,
            direction="down",
        )

        for path in paths:
            path = path.tolist()
            assert len(path) == 2, "Oops seems we got something unexpected."

            flatten_op, gemm_op = path
            assert isinstance(flatten_op, Operation) and isinstance(gemm_op, Operation)

            transB = gemm_op.attributes.get("transB", 0)

            w = gemm_op.parameters[0].value
            if transB != 1:
                w = torch.permute(w, [1, 0])

            w = torch.unsqueeze(w, -1)
            w = torch.unsqueeze(w, -1)

            gemm_op.inputs[1].value = w
            conv_attributes = {"dilations": [1, 1], "kernel_shape": [1, 1], "strides": [1, 1], "group": 1}
            gemm_op.type = "Conv"
            gemm_op.attributes.clear()
            for k, v in conv_attributes.items():
                gemm_op.attributes[k] = v

            gemm_op.inputs[0] = flatten_op.inputs[0]
            gemm_op.inputs[0].dest_ops.remove(flatten_op)
            gemm_op.inputs[0].dest_ops.append(gemm_op)

            temp_var = flatten_op.outputs[0]
            flatten_op.outputs[0] = gemm_op.outputs[0]
            gemm_op.outputs[0] = temp_var
            flatten_op.inputs[0] = gemm_op.outputs[0]

            flatten_op.inputs[0].dest_ops.remove(gemm_op)
            flatten_op.inputs[0].dest_ops.append(flatten_op)

            gemm_op.outputs[0].source_op = gemm_op
            flatten_op.outputs[0].source_op = flatten_op

        for op_name, op in self._graph.operations.items():
            if op.type in {"Gemm"}:
                if op.inputs[1].is_parameter and op.attributes.get("transB", 0) == 0:
                    op.attributes["transB"] = 1
                    op.inputs[1].value = torch.permute(op.inputs[1].value, [1, 0])

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
            self._graph._detail["GRAPH_OPSET"].append({"domain": "com.microsoft", "version": 1})

    def fuse_layernorm(self, exclusive_search: bool = False):
        """Fuse Layernormalization with pattern matching."""

        def _fuse(
            rm1: Operation,
            rm2: Operation,
            eps: Operation,
            scale: torch.Tensor,
            bias: torch.Tensor,
            layernorm_input_var: Variable,
            layernorm_output_var: Variable,
        ) -> Operation:
            if rm2.type == rm1.type == "ReduceMean":
                if "axes" not in rm1.attributes:
                    return None
                if "axes" not in rm2.attributes:
                    return None
                if rm1.attributes["axes"] != rm2.attributes["axes"]:
                    return None
                layernorm_axis = rm1.attributes["axes"]
                if isinstance(layernorm_axis, list):
                    if len(layernorm_axis) != 1:
                        return None
                    layernorm_axis = layernorm_axis[0]
                if not isinstance(layernorm_axis, int):
                    return None
            else:
                layernorm_axis = -1

            if not eps.inputs[-1].is_parameter:
                return None
            value = eps.inputs[-1].value
            value = convert_any_to_torch_tensor(value).cpu()
            if value.numel() != 1:
                return None
            layernorm_eps = value.item()

            layernorm_output_var.source_op.outputs.clear()
            layernorm = self._graph.create_operation(
                op_type="LayerNormalization",
                attributes={"axis": layernorm_axis, "epsilon": layernorm_eps, "stash_type": 0},
                inputs=[layernorm_input_var, self._graph.create_variable(value=scale, is_parameter=True)],
                outputs=[layernorm_output_var],
            )

            if bias is not None:
                self._graph.create_link_with_op(
                    variable=self._graph.create_variable(value=bias, is_parameter=True), A=None, B=layernorm
                )
            return layernorm

        search_engine = SearchableGraph(graph=self._graph)
        fused = False

        # pattern 1:
        #                                 ---     ---     ---      ---        ---       ---    ---    --
        #                               |                                                              |
        # ***(0) --- ReduceMean(1) --- Sub(2) --- Pow(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Div(7) --- Mul(8) --- (Add)(9)
        #      |                     |
        #       ---   ---   ---   ---
        matches = search_engine.pattern_matching(
            patterns=[
                lambda x: True,
                lambda x: x.type in {"ReduceMean", "GlobalAveragePool"},
                "Sub",
                "Pow",
                lambda x: x.type in {"ReduceMean", "GlobalAveragePool"},
                "Add",
                "Sqrt",
                "Div",
                "Mul",
            ],
            edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]],
            exclusive=exclusive_search,
        )

        for _, rm1, sub, pow, rm2, add, sqrt, div, mul in matches:
            layernorm_ops = [rm1, sub, pow, rm2, add, sqrt, div, mul]

            layernorm_scale = mul.inputs[-1].value
            layernorm_output_var = div.outputs[0]
            layernorm_input_var = sub.inputs[0]

            # bias check
            layernorm_bias = None
            next_op = self._graph.get_downstream_operations(mul)
            if len(next_op) == 1 and (next_op[0].type == "Add"):
                bias_op = next_op[0]
                if bias_op.inputs[-1].is_parameter:
                    layernorm_bias = bias_op.inputs[-1].value
                    layernorm_output_var = bias_op.outputs[0]
                    layernorm_ops.append(bias_op)

            layernorm = _fuse(
                rm1=rm1,
                rm2=rm2,
                eps=add,
                scale=layernorm_scale,
                bias=layernorm_bias,
                layernorm_input_var=layernorm_input_var,
                layernorm_output_var=layernorm_output_var,
            )

            if layernorm is not None:
                # delete merged ops
                for op in layernorm_ops:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var != layernorm_input_var and var != layernorm_output_var:
                            self._graph.remove_variable(var)
                    self._graph.remove_operation(op)

        # pattern 2:
        #  ---   ---  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  -- Mul(11)   ---   ---   Add(12) ---
        #  ^                             |                                                                                ^                     ^
        #  |                             v                                                                                |                     |
        # ***(0) --- ReduceMean(1) --- Sub(2) --- Mul(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Reciprocal(7) --- Mul(8) --- Mul(9) --- Sub(10)
        #             |                                                                                                             |
        #             v                                                                                                             ^
        #             --   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  ---  --- --
        matches = search_engine.pattern_matching(
            patterns=[
                lambda x: True,
                lambda x: x.type in {"ReduceMean", "GlobalAveragePool"},
                "Sub",
                "Mul",
                lambda x: x.type in {"ReduceMean", "GlobalAveragePool"},
                "Add",
                "Sqrt",
                "Reciprocal",
                "Mul",
                "Mul",
                "Sub",
                "Mul",
                "Add",
            ],
            edges=[
                [0, 1],
                [0, 2],
                [0, 11],
                [1, 2],
                [1, 9],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [8, 11],
                [9, 10],
                [11, 12],
                [10, 12],
            ],
            exclusive=exclusive_search,
        )

        for _, rm1, sub1, mul, rm2, add1, sqrt, recipro, mul2, mul3, sub2, mul4, add2 in matches:
            layernorm_ops = [rm1, sub1, mul, rm2, add1, sqrt, recipro, mul2, mul3, sub2, mul4, add2]

            # mul check
            if not mul2.inputs[-1].is_parameter:
                continue
            layernorm_scale = mul2.inputs[-1].value
            layernorm_output_var = add2.outputs[0]
            layernorm_input_var = sub1.inputs[0]
            layernorm_bias = sub2.inputs[0].value

            layernorm = _fuse(
                rm1=rm1,
                rm2=rm2,
                eps=add1,
                scale=layernorm_scale,
                bias=layernorm_bias,
                layernorm_input_var=layernorm_input_var,
                layernorm_output_var=layernorm_output_var,
            )

            if layernorm is not None:
                # delete merged ops
                for op in layernorm_ops:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var != layernorm_input_var and var != layernorm_output_var:
                            self._graph.remove_variable(var)
                    self._graph.remove_operation(op)

    def fuse_gelu(self):
        search_engine = SearchableGraph(graph=self._graph)

        matches = search_engine.pattern_matching(
            patterns=[lambda x: True, "Div", "Erf", "Add", "Mul", "Mul"],
            edges=[[0, 1], [1, 2], [2, 3], [3, 4], [0, 4], [4, 5]],
            exclusive=True,
        )

        for _, div, erf, add, mul1, mul2 in matches:
            removing_var = []
            removing_var.extend(div.outputs)
            removing_var.extend(erf.outputs)
            removing_var.extend(add.outputs)
            removing_var.extend(mul1.outputs)

            self._graph.remove_operation(div)
            self._graph.remove_operation(erf)
            self._graph.remove_operation(add)
            self._graph.remove_operation(mul1)
            for var in removing_var:
                self._graph.remove_variable(var)

            input_vars = _.outputs.copy()
            output_vars = mul2.outputs.copy()

            self._graph.remove_operation(mul2)
            self._graph.create_operation(op_type="Gelu", inputs=input_vars, outputs=output_vars)
            assert len(input_vars) == 1, "Fusion failed, Pattern unrecognized."

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
                ppq_warning(
                    f"PPQ can not merge operation {computing_op.name} and {mul_op.name}, "
                    "this is not suppose to happen with your network, "
                    "network with batchnorm inside might not be able to quantize and deploy."
                )
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
