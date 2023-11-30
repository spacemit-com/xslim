from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
import numpy as np
import ppq
import functools
from ppq.core import (
    PPQ_CONFIG,
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
    common as ppq_common,
)
from ppq.quantization.quantizer.base import BaseQuantizer
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.api.setting import QuantizationSetting
from ppq.executor import BaseGraphExecutor
from ppq.quantization.optim import (
    QuantizationOptimizationPipeline,
    RuntimeCalibrationPass,
    QuantizeFusionPass as PPQQuantizeFusionPass,
)
from ..optimizer import (
    HardSwishFusionPass,
    SwishFusionPass,
    QuantizeFusionPass,
    BiasParameterBakingPass,
    AsymmetricaUnsignlAlignSign,
    RuntimePerlayerCalibrationPass,
)


def _get_quant_min_max(num_of_bits: int, signed: bool = True):
    if signed:
        return -(2 ** (num_of_bits - 1)), 2 ** (num_of_bits - 1) - 1
    else:
        return 0, 2**num_of_bits - 1


def _get_version_number(ver_str):
    return functools.reduce(
        lambda x, y: x + y, [pow(100, i) * int(ver) for i, ver in enumerate(reversed(ver_str.split(".")))]
    )


class XQuantizer(BaseQuantizer):
    def __init__(self, graph: BaseGraph) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._precision_level = 0
        self._num_of_bits = 8
        self._quant_min, self._quant_max = _get_quant_min_max(self._num_of_bits, False)
        perchannel_policy = QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL + QuantizationProperty.PER_CHANNEL + QuantizationProperty.LINEAR
        )
        pertensor_policy = QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL
            + QuantizationProperty.PER_TENSOR
            + QuantizationProperty.LINEAR
            + QuantizationProperty.POWER_OF_2
        )
        self._op_type_to_policy = {
            "Conv": perchannel_policy,
            "ConvTranspose": pertensor_policy,
            "Gemm": pertensor_policy,
            "MatMul": pertensor_policy,
        }

    def quantize(
        self,
        inputs: Union[torch.Tensor, list, dict],
        calib_dataloader: Iterable,
        executor: BaseGraphExecutor,
        setting: QuantizationSetting,
        **kwargs,
    ) -> None:
        self._setting = setting
        self._set_asymmetrical = getattr(self._setting.quantize_activation_setting, "asymmetrical", True)
        self._set_percentile = getattr(self._setting.quantize_activation_setting, "percentile", None)
        self._gemm_bits = 8

        self._precision_level = getattr(self._setting, "precision_level", 0)
        if self._precision_level == 2:
            self._gemm_bits = 12

        if not self._set_asymmetrical:
            self._quant_min, self._quant_max = _get_quant_min_max(self._num_of_bits)

        return super().quantize(inputs, calib_dataloader, executor, setting, **kwargs)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy,
            rounding=self.rounding_policy,
            op=operation,
            num_of_bits=self._num_of_bits,
            exponent_bits=0,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
            observer_algorithm="percentile",
        )
        if self._set_percentile is not None:
            for in_var, in_tqc in zip(operation.inputs, base_quant_config.input_quantization_config):
                if not in_var.is_parameter:
                    in_tqc.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE] = self._set_percentile
            for out_var, out_tqc in zip(operation.outputs, base_quant_config.output_quantization_config):
                if not out_var.is_parameter:
                    out_tqc.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE] = self._set_percentile

        if operation.type in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, "Seems you got a Conv layer with no parameters."

            # set act tqc
            for in_var, in_tqc in zip(operation.inputs, base_quant_config.input_quantization_config):
                if not in_var.is_parameter:
                    in_tqc.num_of_bits = self._gemm_bits
                    in_tqc._quant_min, in_tqc._quant_max = _get_quant_min_max(in_tqc.num_of_bits, False)
                    break

            # set weight tqc
            for in_var, in_tqc in zip(operation.inputs, base_quant_config.input_quantization_config):
                if in_var.is_parameter:
                    in_tqc.num_of_bits = 8
                    in_tqc._quant_min, in_tqc._quant_max = _get_quant_min_max(in_tqc.num_of_bits)
                    in_tqc.policy = self._op_type_to_policy[operation.type]
                    in_tqc.observer_algorithm = "minmax"
                    if in_tqc.policy.has_property(QuantizationProperty.PER_CHANNEL):
                        in_tqc.channel_axis = 1 if operation.type == "ConvTranspose" else 0
                    break

            # if operation has bias
            if operation.num_of_input > 2:
                in_tqc = base_quant_config.input_quantization_config[-1]
                in_tqc.policy = self._op_type_to_policy[operation.type]
                in_tqc.num_of_bits = 32
                in_tqc._quant_min, in_tqc._quant_max = _get_quant_min_max(in_tqc.num_of_bits)
                in_tqc.state = QuantizationStates.FP32
                if in_tqc.policy.has_property(QuantizationProperty.PER_CHANNEL):
                    in_tqc.channel_axis = 0

        return base_quant_config

    @property
    def quant_operation_types(self) -> set:
        QUANTTYPE = {
            "Conv",
            "ConvTranspose",
            "Gemm",
            "Relu",
            "PRelu",
            "Clip",
            "Pad",
            "Resize",
            "MaxPool",
            "AveragePool",
            "GlobalMaxPool",
            "GlobalAveragePool",
            "Mul",
            "Add",
            "Max",
            "Sub",
            "Div",
            "Reshape",
            "LeakyRelu",
            "Concat",
            "Sigmoid",
            "ReduceMean",
            "Transpose",
            "Slice",
            "Flatten",
            "HardSwish",
            "HardSigmoid",
            "MatMul",
            "Gelu",
        }
        QUANTTYPE.update(PASSIVE_OPERATIONS)

        if self._precision_level == 2:
            return {
                "Conv",
                "ConvTranspose",
                "Gemm",
            }
        return QUANTTYPE

    @property
    def quantize_policy(self) -> QuantizationPolicy:
        if self._set_asymmetrical:
            return QuantizationPolicy(
                QuantizationProperty.ASYMMETRICAL
                + QuantizationProperty.POWER_OF_2
                + QuantizationProperty.PER_TENSOR
                + QuantizationProperty.LINEAR
            )
        else:
            return QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL
                + QuantizationProperty.POWER_OF_2
                + QuantizationProperty.PER_TENSOR
                + QuantizationProperty.LINEAR
            )

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @property
    def activation_fusion_types(self) -> set:
        return {"Relu", "Clip"}

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        quant_pipeline = super().build_quant_pipeline(setting)

        # for idx, quant_opt in enumerate(quant_pipeline):
        #    if isinstance(quant_opt, RuntimeCalibrationPass):
        #        quant_pipeline._pipeline[idx] = RuntimePerlayerCalibrationPass(
        #            quant_opt._method, quant_opt._override, quant_opt._calib_steps
        #        )
        #        break
        ppq_ver = _get_version_number(PPQ_CONFIG.VERSION)
        if ppq_ver <= _get_version_number("0.6.6"):
            for idx, quant_opt in enumerate(quant_pipeline):
                if isinstance(quant_opt, PPQQuantizeFusionPass):
                    quant_pipeline._pipeline[idx] = QuantizeFusionPass(
                        quant_opt.activation_types,
                        quant_opt.fuse_activation,
                        quant_opt.fuse_passive_op,
                        True,
                    )
                    break

        # quant_pipeline.append_optimization_to_pipeline(HardSwishFusionPass(), True)
        # quant_pipeline.append_optimization_to_pipeline(SwishFusionPass(), True)
        quant_pipeline.append_optimization_to_pipeline(BiasParameterBakingPass())
        quant_pipeline.append_optimization_to_pipeline(AsymmetricaUnsignlAlignSign())
        return quant_pipeline
