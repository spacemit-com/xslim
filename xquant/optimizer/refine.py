from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
from ppq.core import (
    OBSERVER_MSE_HIST_BINS,
    PASSIVE_OPERATIONS,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
    empty_ppq_cache,
    ppq_warning,
    QuantizationVisibility,
    convert_any_to_torch_tensor,
)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.quantization.optim import (
    QuantizationOptimizationPipeline,
    QuantizationOptimizationPass,
    RuntimeCalibrationPass,
)
from ppq.IR.search import SearchableGraph
from ppq.executor import BaseGraphExecutor
from ppq.quantization.qfunction import PPQuantFunction
from ppq.quantization.observer import range as ppq_range


class PassiveParameterBakingPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name="XQuant PassiveParameterBakingPass Pass")
        self._quantize_function = PPQuantFunction

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue
            if operation.type not in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
                for in_config, in_var in operation.config_with_variable:
                    if in_var.is_parameter and in_config.state == QuantizationStates.INITIAL:
                        if in_config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                            raise NotImplementedError("only Computing Ops have perchannel parameters")

                        max_range_val = float(in_var.value.max())
                        min_range_val = float(in_var.value.min())

                        if torch.all(in_var.value.to(torch.int32) - in_var.value == 0):
                            scale = torch.tensor(1, dtype=torch.float32, device=in_var.value.device)
                            offset = torch.tensor(0, dtype=torch.float32, device=in_var.value.device)
                        elif min_range_val != max_range_val:
                            scale, offset = ppq_range.minmax_to_scale_offset(min_range_val, max_range_val, in_config)
                        elif max_range_val > 0:
                            scale, offset = ppq_range.minmax_to_scale_offset(0, max_range_val, in_config)
                        elif max_range_val < 0:
                            scale, offset = ppq_range.minmax_to_scale_offset(max_range_val, 0, in_config)
                        else:
                            continue

                        in_config.scale = convert_any_to_torch_tensor(scale)
                        in_config.offset = convert_any_to_torch_tensor(offset)
                        in_config.state = QuantizationStates.PASSIVE


class BiasParameterBakingPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name="XQuant BiasParameterBaking Pass")
        self._quantize_function = PPQuantFunction

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

        if op.type in {"Conv", "ConvTranspose", "Gemm"}:
            # inputs are [input value, weight, bias(optional)]
            if op.num_of_input == 3:
                i_cfg, w_cfg, b_cfg = op.config.input_quantization_config
                if b_cfg.state not in {QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT}:
                    return

                # PATCH 2022.07.29 有的时候 bias 是个多维的东西，此时要求前面的维度都是1
                bias = op.inputs[-1].value
                if bias is None:
                    raise ValueError(
                        f"Bias Varaible {op.inputs[-1].name} must be a constant. " "Please check it again."
                    )

                assert bias.numel() == bias.shape[-1], (
                    f"For op {op.name}, expect Bias shape to be {[bias.numel()]}, " f"however {bias.shape} was given"
                )
                op.inputs[-1].value = bias.squeeze()
                # PATCH 2022.08.02 只有一个数的 bias 经过 squeeze 会变成零维的, 再给它多加一维补回来
                if op.inputs[-1].value.ndim == 0 and op.inputs[-1].value.numel() == 1:
                    op.inputs[-1].value = op.inputs[-1].value.unsqueeze(0)

                if not check_state(i_cfg.state):
                    raise PermissionError(
                        f"Can not quantize bias of layer {op.name}, " "cause input has not been correctly quantized."
                    )

                b_cfg.scale = w_cfg.scale * i_cfg.scale
                b_cfg.state = QuantizationStates.PASSIVE
                b_cfg.offset = torch.zeros_like(b_cfg.scale)
                assert not b_cfg.policy.has_property(
                    QuantizationProperty.ASYMMETRICAL
                ), "Passive parameter does not support ASYMMETRICAL quantization"

            if op.type in {"Clip"}:
                # inputs are [input value, min[optional], max[optional]]
                i_cfg = op.config.input_quantization_config[0]

                if not check_state(i_cfg.state):
                    raise PermissionError(
                        f"Can not quantize clip value of layer {op.name}, "
                        "cause input has not been correctly quantized."
                    )

                for config in op.config.input_quantization_config[1:]:
                    config.master_by = i_cfg
                    config.visibility = QuantizationVisibility.INTERNAL

            if op.type in {"Pad"}:
                # inputs are [input value, pad[shape-related], pad value[optional]]
                if op.num_of_input != 3:
                    return
                i_cfg = op.config.input_quantization_config[0]

                if not check_state(i_cfg.state):
                    raise PermissionError(
                        f"Can not quantize pad value of layer {op.name}, "
                        "cause input has not been correctly quantized."
                    )

                if len(op.config.input_quantization_config) > 1:
                    pad_config = op.config.input_quantization_config[-1]
                    pad_config.master_by = i_cfg
                    pad_config.visibility = QuantizationVisibility.INTERNAL

    @staticmethod
    def passive_bias_quant(operation: QuantableOperation):
        if not isinstance(operation, QuantableOperation):
            return
        if operation.type not in {"Conv", "ConvTranspose", "Gemm"}:
            return
        if operation.num_of_input == 3:
            i_cfg, w_cfg, b_cfg = operation.config.input_quantization_config
            o_cfg = operation.config.output_quantization_config[0]
            if b_cfg.state not in {QuantizationStates.FP32}:
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
            _b_scale = w_cfg.scale * i_cfg.scale
            _i_bias = bias.to(torch.float64) / _b_scale.to(torch.float64)
            if torch.all(torch.abs(_i_bias) < 2 ** (b_cfg.num_of_bits - 1)):
                b_cfg.scale = _b_scale
            elif o_cfg.scale is not None:
                # in frac + w frac无法表示就使用 out frac
                b_cfg.scale = o_cfg.scale
            else:
                return
            b_cfg.state = QuantizationStates.PASSIVE
            b_cfg.offset = torch.zeros_like(b_cfg.scale)

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for _, operation in graph.operations.items():
            BiasParameterBakingPass.passive_bias_quant(operation)


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
    ) -> None:
        self.fuse_relu_clip = fuse_relu_clip
        super().__init__(name="PPQ Quantization Fusion Pass")

    def is_same_platform(self, operations: List[Operation]):
        platforms = [operation.platform for operation in operations]
        return all([platform == platforms[0] for platform in platforms])

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        processor = SearchableGraph(graph)

        if self.fuse_relu_clip:
            patterns = processor.pattern_matching(
                patterns=[lambda x: True, lambda x: x.type in {"Relu", "Clip"}], edges=[[0, 1]], exclusive=True
            )
            for computing_op, act_op in patterns:
                if not isinstance(act_op, QuantableOperation):
                    continue
                if not isinstance(computing_op, QuantableOperation):
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
                            force_range_min = -10.0
                            if check_sigmoid:
                                force_range_max = 10.0

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
