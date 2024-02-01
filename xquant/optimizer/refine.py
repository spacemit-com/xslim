#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Iterable, List, Set, Union, Dict, Callable, Tuple, Sequence
import torch
import math
from ..ppq_decorator import (
    QuantizationProperty,
    QuantizationStates,
    empty_ppq_cache,
    QuantizationVisibility,
    convert_any_to_torch_tensor,
    ppq_common,
    BaseGraph,
    Operation,
    QuantableOperation,
    Variable,
    QuantizationOptimizationPass,
    SearchableGraph,
    minmax_to_scale_offset,
)
from ..defs import (
    XQUANT_CONFIG,
    PASSIVE_OPERATIONS,
    COMPUTING_OP,
    OBSERVER_MIN_SCALE_THRESHOLD,
    OBSERVER_SIGMOID_MAX_VALUE,
)


class PassiveParameterBakingPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name="XQuant PassiveParameterBakingPass Pass")

    @staticmethod
    def passive_parameters_quant(op: QuantableOperation):
        def check_state(state: QuantizationStates):
            return state in {
                QuantizationStates.PASSIVE,
                QuantizationStates.ACTIVATED,
                QuantizationStates.BAKED,
                QuantizationStates.OVERLAPPED,
            }

        if not isinstance(op, QuantableOperation):
            return

        if op.type in {"Clip"}:
            # inputs are [input value, min[optional], max[optional]]
            i_cfg = op.config.input_quantization_config[0]
            if not check_state(i_cfg.state):
                return
            for config in op.config.input_quantization_config[1:]:
                config.master_by = i_cfg
                config.visibility = QuantizationVisibility.INTERNAL
        elif op.type in {"Pad"}:
            # inputs are [input value, pad[shape-related], pad value[optional]]
            if op.num_of_input != 3:
                return
            i_cfg = op.config.input_quantization_config[0]
            if not check_state(i_cfg.state):
                return
            if len(op.config.input_quantization_config) > 1:
                pad_config = op.config.input_quantization_config[-1]
                pad_config.master_by = i_cfg
                pad_config.visibility = QuantizationVisibility.INTERNAL
        elif op.type not in COMPUTING_OP:
            for in_config, in_var in op.config_with_variable:
                if (
                    in_var.is_parameter
                    and in_config.state == QuantizationStates.INITIAL
                    and in_config.policy.has_property(QuantizationProperty.ASYMMETRICAL)
                    and in_config.policy.has_property(QuantizationProperty.PER_TENSOR)
                ):
                    max_range_val = float(in_var.value.max())
                    min_range_val = float(in_var.value.min())
                    if torch.all(in_var.value.to(torch.int8) - in_var.value == 0):
                        scale = torch.tensor(1.0, dtype=torch.float32, device=in_var.value.device)
                        offset = torch.tensor(
                            math.ceil((in_config.quant_max - in_config.quant_min) / 2),
                            dtype=torch.float32,
                            device=in_var.value.device,
                        )
                    elif torch.all(in_var.value.to(torch.uint8) - in_var.value == 0):
                        scale = torch.tensor(1.0, dtype=torch.float32, device=in_var.value.device)
                        offset = torch.tensor(
                            in_config.quant_min,
                            dtype=torch.float32,
                            device=in_var.value.device,
                        )
                    elif min_range_val != max_range_val:
                        scale, offset = minmax_to_scale_offset(
                            min_range_val, max_range_val, in_config, OBSERVER_MIN_SCALE_THRESHOLD
                        )
                    elif max_range_val > 0:
                        scale, offset = minmax_to_scale_offset(
                            0, max_range_val, in_config, OBSERVER_MIN_SCALE_THRESHOLD
                        )
                    elif max_range_val < 0:
                        scale, offset = minmax_to_scale_offset(
                            max_range_val, 0, in_config, OBSERVER_MIN_SCALE_THRESHOLD
                        )
                    else:
                        continue
                    in_config.scale = convert_any_to_torch_tensor(scale)
                    in_config.offset = convert_any_to_torch_tensor(offset)
                    in_config.state = QuantizationStates.PASSIVE

    @staticmethod
    def passive_bias_quant(operation: QuantableOperation):
        if not isinstance(operation, QuantableOperation):
            return
        if operation.type not in {"Conv", "ConvTranspose", "Gemm"}:
            return
        if operation.num_of_input == 3:
            i_cfg, w_cfg, b_cfg = operation.config.input_quantization_config
            o_cfg = operation.config.output_quantization_config[0]
            if b_cfg.state not in {QuantizationStates.PASSIVE_INIT}:
                return
            bias = operation.inputs[-1].value
            if bias is None:
                raise ValueError(
                    f"Bias Varaible {operation.inputs[-1].name} must be a constant. " "Please check it again."
                )
            assert bias.numel() == bias.shape[-1], (
                f"For op {operation.name}, expect Bias shape to be {[bias.numel()]}, " f"however {bias.shape} was given"
            )
            operation.inputs[-1].value = bias.squeeze()

            if operation.inputs[-1].value.ndim == 0 and operation.inputs[-1].value.numel() == 1:
                operation.inputs[-1].value = operation.inputs[-1].value.unsqueeze(0)
            if w_cfg.scale is None or i_cfg.scale is None:
                return

            if w_cfg.scale.numel() > 1 and operation.type in {"Conv"}:
                zero_channel = torch.where(w_cfg.scale < OBSERVER_MIN_SCALE_THRESHOLD)[0]
                if zero_channel.numel() > 0:
                    operation.inputs[1].value[zero_channel] = 0
                    store_state = w_cfg.state
                    w_cfg.state = QuantizationStates.INITIAL
                    w_cfg.scale[zero_channel] = 1.0
                    w_cfg.state = store_state

            _b_scale = w_cfg.scale * i_cfg.scale
            _i_bias = bias.to(torch.float64) / _b_scale.to(torch.float64)
            if torch.all(torch.abs(_i_bias) < 2 ** (b_cfg.num_of_bits - 1)) or operation.type not in {"Conv"}:
                b_cfg.scale = _b_scale
            elif o_cfg.scale is not None:
                # in frac + w frac无法表示就使用 out frac
                b_cfg.scale = o_cfg.scale
                operation.attributes["quant_bias_apply"] = 1
            else:
                return
            b_cfg.state = QuantizationStates.PASSIVE
            b_cfg.offset = torch.zeros_like(b_cfg.scale)

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for _, operation in graph.operations.items():
            PassiveParameterBakingPass.passive_parameters_quant(operation)
            PassiveParameterBakingPass.passive_bias_quant(operation)


class AsymmetricaUnsignlAlignSign(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name="XQuant AsymmetricalAlignS8 Pass")

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue
            for config, var in [_ for _ in operation.config_with_variable]:
                if config.policy.has_property(QuantizationProperty.ASYMMETRICAL) and config.quant_min == 0:
                    config.quant_min = -(2 ** (config.num_of_bits - 1))
                    config.quant_max = 2 ** (config.num_of_bits - 1) - 1
                    if config.dominated_by == config and config.offset is not None:
                        store_state = config.state
                        config.state = QuantizationStates.INITIAL
                        config.offset = config.offset + config.quant_min
                        config.state = store_state


class QuantizeFusionPass(QuantizationOptimizationPass):
    def __init__(
        self,
        fuse_relu_clip: bool = True,
        fuse_passive_op: bool = True,
    ) -> None:
        self.fuse_relu_clip = fuse_relu_clip
        self.fuse_passive_op = fuse_passive_op
        super().__init__(name="XQuant Quantization Fusion Pass")

    def is_same_platform(self, operations: List[Operation]):
        platforms = [operation.platform for operation in operations]
        return all([platform == platforms[0] for platform in platforms])

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        processor = SearchableGraph(graph)

        if self.fuse_passive_op:
            # all passive operations should never changes quantization configuration of its input
            # so to say their input and output share a same scale.
            for op in graph.operations.values():
                if op.type not in PASSIVE_OPERATIONS:
                    continue
                source_op = op.inputs[0].source_op
                if source_op is None:
                    continue  # beginning op, can not merge.
                if isinstance(op, QuantableOperation) and self.is_same_platform([op, source_op]):
                    TQC = op.config.input_quantization_config[0]
                    for output_cfg in op.config.output_quantization_config:
                        output_cfg.dominated_by = TQC

        if self.fuse_relu_clip:
            patterns = processor.pattern_matching(
                patterns=[lambda x: True, lambda x: x.type in {"Relu", "Clip"}], edges=[[0, 1]], exclusive=True
            )
            for computing_op, act_op in patterns:
                if not isinstance(act_op, QuantableOperation) or not isinstance(computing_op, QuantableOperation):
                    continue

                if (
                    len(graph.get_downstream_operations(computing_op)) == 1
                    and len(graph.get_upstream_operations(act_op)) == 1
                ):
                    computing_op.config.output_quantization_config[
                        0
                    ].dominated_by = act_op.config.output_quantization_config[0]
                    act_op.config.input_quantization_config[0].dominated_by = act_op.config.output_quantization_config[
                        0
                    ]


class ActivationClipRefine(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name="XQuant ActivationClipRefine Pass")
        self.act_op_set = {"HardSigmoid", "Sigmoid"}

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue

            for tqc, var in operation.config_with_variable:
                if tqc.state in {QuantizationStates.INITIAL}:
                    if any([dest_op.type in self.act_op_set for dest_op in var.dest_ops]):
                        force_range_min = None
                        force_range_max = None
                        check_hardsigmoid = all([dest_op.type in {"HardSigmoid"} for dest_op in var.dest_ops])
                        check_hardswish = all([dest_op.type in {"HardSigmoid", "Mul"} for dest_op in var.dest_ops])

                        check_sigmoid = all([dest_op.type in {"Sigmoid"} for dest_op in var.dest_ops])
                        check_swish = all([dest_op.type in {"Sigmoid", "Mul"} for dest_op in var.dest_ops])

                        if check_hardsigmoid:
                            alpha = var.dest_ops[0].attributes.get("alpha", 0.1666666716337204)
                            beta = var.dest_ops[0].attributes.get("beta", 0.5)
                            force_range_min = -beta / alpha
                            force_range_max = (1.0 - beta) / alpha

                        elif check_sigmoid or (check_swish and len(var.dest_ops) == 2):
                            force_range_min = -OBSERVER_SIGMOID_MAX_VALUE
                            if check_sigmoid:
                                force_range_max = OBSERVER_SIGMOID_MAX_VALUE

                        elif check_hardswish and len(var.dest_ops) == 2:
                            dest_op_type = [dest_op.type for dest_op in var.dest_ops]
                            if "HardSigmoid" in dest_op_type and "Mul" in dest_op_type:
                                act_index = dest_op_type.index("HardSigmoid")
                                mul_index = dest_op_type.index("Mul")
                                if var.dest_ops[act_index].outputs[0] in var.dest_ops[mul_index].inputs:
                                    alpha = var.dest_ops[act_index].attributes.get("alpha", 0.1666666716337204)
                                    beta = var.dest_ops[act_index].attributes.get("beta", 0.5)
                                    force_range_min = -beta / alpha

                        if force_range_min is not None or force_range_max is not None:
                            operation._detail["output_force_range"] = {
                                var.name: {"min": force_range_min, "max": force_range_max}
                            }
                            operation._detail["output_force_range"].update({var.name: {"min": force_range_min}})


class QuantizeConfigRefinePass(QuantizationOptimizationPass):
    def __init__(
        self, precision_level: int = 0, custom_setting: Sequence["CustomQuantizationParameterSetting"] = None
    ) -> None:
        super().__init__(name="XQuant QuantizeConfigRefine Pass")
        self._precision_level = precision_level
        self._max_bits = XQUANT_CONFIG.max_bits
        self._quant_max = 2**self._max_bits - 1
        self._quant_min = 0
        self._custom_setting = custom_setting

    def precesion_level_2(self, operation: QuantableOperation):
        if operation.type in {"Conv"}:
            in_var = operation.inputs[0]
            in_tqc = operation.input_quant_config[0]
            out_var = operation.outputs[0]
            out_tqc = operation.output_quant_config[0]
            if in_tqc.dominated_by is in_tqc:
                in_tqc.num_of_bits = self._max_bits
                in_tqc._quant_min, in_tqc._quant_max = self._quant_min, self._quant_max
            else:
                in_tqc.num_of_bits = self._max_bits
                in_tqc._quant_min, in_tqc._quant_max = self._quant_min, self._quant_max
                in_tqc.dominated_by.num_of_bits = self._max_bits
                in_tqc.dominated_by._quant_min, in_tqc.dominated_by._quant_max = (
                    self._quant_min,
                    self._quant_max,
                )
                in_tqc.dominated_by.state = QuantizationStates.INITIAL
            if out_tqc.dominated_by is out_tqc:
                out_tqc.num_of_bits = self._max_bits
                out_tqc._quant_min, out_tqc._quant_max = self._quant_min, self._quant_max
            else:
                out_tqc.num_of_bits = self._max_bits
                out_tqc._quant_min, out_tqc._quant_max = self._quant_min, self._quant_max
                out_tqc.dominated_by.num_of_bits = self._max_bits
                out_tqc.dominated_by._quant_min, out_tqc.dominated_by._quant_max = (
                    self._quant_min,
                    self._quant_max,
                )
        else:
            for tqc, var in operation.config_with_variable:
                if tqc.dominated_by is tqc and tqc.num_of_bits != self._max_bits:
                    tqc.state = QuantizationStates.FP32

    def custom_tqc_set(self, graph: BaseGraph, custom_tqc: "CustomQuantizationParameterSetting"):
        var_dict = graph.variables
        input_names = custom_tqc.input_names
        output_names = custom_tqc.output_names
        precision_level = custom_tqc.precision_level
        max_percentile = custom_tqc.max_percentile
        calibration_type = custom_tqc.calibration_type

        input_tensors = []
        for var_name in input_names:
            var = graph.variables.get(var_name, None)
            if var is None:
                raise ValueError("var name {} not in the graph.".format(var_name))
            if not var.is_parameter:
                input_tensors.append(var)

        output_tensors = []
        for var_name in output_names:
            var = graph.variables.get(var_name, None)
            if var is None:
                raise ValueError("var name {} not in the graph.".format(var_name))
            output_tensors.append(var)

        if len(input_tensors) < 1 or len(output_tensors) < 1:
            raise ValueError("input_names and output_names should be set.")

        visited_ops = []
        output_tensors_set = set(output_tensors)

        def get_op_blocks(in_vars: Sequence[Variable]):
            for var in in_vars:
                if var in output_tensors_set:
                    return
                for op in var.dest_ops:
                    visited_ops.append(op)
                    get_op_blocks(op.outputs)

        get_op_blocks(input_tensors)

        for op in visited_ops:
            if not isinstance(op, QuantableOperation):
                continue
            if isinstance(precision_level, int) and precision_level >= 2:
                self.precesion_level_2(op)
            tqc = op.config

            for in_var, in_tqc in zip(op.inputs, tqc.input_quantization_config):
                if not in_var.is_parameter:
                    if isinstance(max_percentile, float):
                        in_tqc.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE] = max_percentile
                    if isinstance(calibration_type, str):
                        in_tqc.observer_algorithm = calibration_type
            for out_var, out_tqc in zip(op.outputs, tqc.output_quantization_config):
                if not out_var.is_parameter:
                    if isinstance(max_percentile, float):
                        out_tqc.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE] = max_percentile
                    if isinstance(calibration_type, str):
                        out_tqc.observer_algorithm = calibration_type

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        if self._precision_level >= 2:
            sorted_ops = graph.topological_sort()
            for operation in sorted_ops:
                if not isinstance(operation, QuantableOperation):
                    continue
                self.precesion_level_2(operation)

        if isinstance(self._custom_setting, Sequence):
            for tqc_setting in self._custom_setting:
                self.custom_tqc_set(graph, tqc_setting)
