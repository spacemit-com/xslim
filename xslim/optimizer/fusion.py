#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, Union

import torch
from xslim.logger import logger

from ..defs import BIAS_CORRECTION_INTERST_TYPE, PASSIVE_OPERATIONS, XQUANT_CONFIG
from ..ppq_decorator import (
    BaseGraph,
    BaseGraphExecutor,
    Operation,
    QuantableOperation,
    QuantizationOptimizationPass,
    QuantizationStates,
    SearchableGraph,
    TargetPlatform,
    Variable,
)


class FlattenGemmFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__("XSlim FlattenGemmFusionPass")

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs,
    ) -> None:
        search_engine = SearchableGraph(graph)
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

            if not isinstance(flatten_op.inputs[0].shape, Sequence):
                continue

            weight_new_shape = [*w.shape, *flatten_op.inputs[0].shape[2:]]
            weight_new_shape[1] = -1
            w = w.reshape(weight_new_shape)
            gemm_op.inputs[1].value = w
            conv_attributes = {"dilations": [1, 1], "kernel_shape": weight_new_shape[2:], "strides": [1, 1], "group": 1}
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
            flatten_op.platform = TargetPlatform.UNSPECIFIED
            gemm_op.platform = TargetPlatform.UNSPECIFIED

        for op_name, op in graph.operations.items():
            if op.type in {"Gemm"}:
                if op.inputs[1].is_parameter and op.attributes.get("transB", 0) == 0:
                    op.attributes["transB"] = 1
                    op.inputs[1].value = torch.permute(op.inputs[1].value, [1, 0])


class FormatBatchNormalizationPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__("XSlim FormatBatchNormalizationPass")

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs,
    ) -> None:
        op_list = [op for op in graph.operations.values()]
        for op in op_list:
            if op.type == "BatchNormalization" and op.inputs[1].is_parameter and op.inputs[2].is_parameter:
                if not isinstance(op.inputs[0].shape, Sequence):
                    raise RuntimeError("remove BatchNormalization error, shape not found.")
                weight_shape = [op.inputs[0].shape[1]] + [1] * (len(op.inputs[0].shape) - 2)
                alpha = op.parameters[0].value
                beta = op.parameters[1].value
                mean = op.parameters[2].value
                var = op.parameters[3].value
                epsilon = op.attributes.get("epsilon", 1e-5)

                with torch.no_grad():
                    w = alpha / torch.sqrt(var + epsilon)
                    w = w.reshape(weight_shape)
                    b = alpha * (-mean) / torch.sqrt(var + epsilon) + beta
                    b = b.reshape(weight_shape)

                op.type = "Mul"
                op.attributes.clear()

                graph.remove_variable(op.inputs[-1])
                graph.remove_variable(op.inputs[-1])
                bias_var = op.inputs.pop()
                bias_var.dest_ops.remove(op)
                with torch.no_grad():
                    op.inputs[1].value = w
                    bias_var.value = b
                op.platform = TargetPlatform.UNSPECIFIED
                bias_out_var = op.outputs[0]
                inner_var = Variable("{}_bn_inner".format(op.name), shape=op.inputs[0].shape)
                graph.append_variable(inner_var)
                op.outputs[0] = inner_var
                inner_var.source_op = op

                bias_op = Operation(
                    "{}_bias".format(op.name),
                    "Add",
                    attributes={},
                    inputs=[inner_var, bias_var],
                    outputs=[bias_out_var],
                    platform=TargetPlatform.UNSPECIFIED,
                )
                inner_var.dest_ops.append(bias_op)
                bias_var.dest_ops.append(bias_op)
                bias_out_var.source_op = bias_op
                graph.append_operation(bias_op)


class HardSwishFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__("XSlim HardSwish Fusion")

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs,
    ) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(
            patterns=[lambda x: x.is_computing_op, "HardSigmoid", "Mul"],
            edges=[[0, 1], [1, 2], [0, 2]],
            exclusive=True,
        )

        for pattern in patterns:
            if any([not isinstance(op, QuantableOperation) for op in pattern]):
                logger.warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            if any([op.platform != pattern[0].platform for op in pattern]):
                logger.warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            computing, sigmoid, mul = pattern

            assert isinstance(computing, QuantableOperation)
            assert isinstance(sigmoid, QuantableOperation)
            assert isinstance(mul, QuantableOperation)

            computing_config = computing.config.output_quantization_config[0]
            sigmoid.config.input_quantization_config[0].dominated_by = computing_config
            sigmoid.config.output_quantization_config[0].state = QuantizationStates.FP32
            mul.config.input_quantization_config[0].dominated_by = computing_config
            mul.config.input_quantization_config[1].state = QuantizationStates.FP32


class SwishFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__("XSlim Swish Fusion")

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs,
    ) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(
            patterns=[lambda x: x.is_computing_op, "Sigmoid", "Mul"],
            edges=[[0, 1], [1, 2], [0, 2]],
            exclusive=True,
        )

        for pattern in patterns:
            if any([not isinstance(op, QuantableOperation) for op in pattern]):
                logger.warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            if any([op.platform != pattern[0].platform for op in pattern]):
                logger.warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            computing, sigmoid, mul = pattern

            assert isinstance(computing, QuantableOperation)
            assert isinstance(sigmoid, QuantableOperation)
            assert isinstance(mul, QuantableOperation)

            computing_config = computing.config.output_quantization_config[0]
            sigmoid.config.input_quantization_config[0].dominated_by = computing_config
            sigmoid.config.output_quantization_config[0].state = QuantizationStates.FP32
            mul.config.input_quantization_config[0].dominated_by = computing_config
            mul.config.input_quantization_config[1].state = QuantizationStates.FP32


class ComputingFusionPass(QuantizationOptimizationPass):
    def __init__(self, optimize_count: int = 2) -> None:
        super().__init__("XSlim Computing Ops Fusion")
        self.optimize_count = optimize_count

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs,
    ) -> None:
        search_engine = SearchableGraph(graph=graph)
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
                len(graph.get_downstream_operations(computing_op)) != 1
                or len(graph.get_upstream_operations(mul_op)) != 1
            ):
                logger.warning(
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
                    b = alpha * b

                elif computing_op.type == "Gemm":
                    # calculate new weight and bias
                    scale = alpha
                    if computing_op.attributes.get("transB", 0):
                        w = w * scale.reshape([-1, 1])
                    else:
                        w = w * scale.reshape([1, -1])
                    b = alpha * b

                elif computing_op.type == "ConvTranspose":
                    scale = alpha
                    group = computing_op.attributes.get("group", 1)
                    scale = scale.reshape([group, 1, -1] + [1] * (w.ndim - 2))
                    w = w.reshape([group, -1] + list(w.shape[1:])) * scale
                    w = w.reshape([w.shape[0] * w.shape[1]] + list(w.shape[2:]))
                    b = alpha * b
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
            graph.remove_operation(computing_op)
            graph.remove_operation(mul_op)

            # insert new
            graph.append_operation(merged_op)
            merged_op.inputs.extend([input_var, weight_var, bias_var])
            merged_op.outputs.extend([output_var])

            graph.append_variable(weight_var)
            graph.append_variable(bias_var)

        self.optimize_count -= 1
        if self.optimize_count > 0:
            self.optimize(graph, dataloader, executor, **kwargs)
