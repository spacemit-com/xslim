#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
from collections import OrderedDict
from enum import Enum
import torch
import numpy as np
import functools
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.core import (
    PPQ_CONFIG,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    common as ppq_common,
)
from ppq import QuantizationSettingFactory
from ppq.quantization.quantizer import BaseQuantizer
from ppq.api.setting import QuantizationSetting
from ppq.executor import BaseGraphExecutor
from ppq.quantization.optim import (
    HorizontalLayerSplitPass,
    ChannelwiseSplitPass,
    QuantizationOptimizationPipeline,
    SSDEqualizationPass,
    QuantizeFusionPass as PPQQuantizeFusionPass,
    QuantizeSimplifyPass,
    ParameterQuantizePass,
    QuantAlignmentPass,
    PassiveParameterQuantizePass,
    ParameterBakingPass,
)
from ..optimizer import (
    ActivationClipRefine,
    HardSwishFusionPass,
    SwishFusionPass,
    QuantizeFusionPass,
    PassiveParameterBakingPass,
    AsymmetricaUnsignlAlignSign,
    RuntimeBlockWiseCalibrationPass,
    ComputingFusionPass,
    PassiveParameterBakingPass,
    CustomLayerwiseEqualizationPass,
    QuantizeConfigRefinePass,
)
from ..defs import XQUANT_CONFIG, AutoFinetuneLevel, PrecisionLevel, xquant_info, xquant_warning, PASSIVE_OPERATIONS
from ..xquant_setting import XQuantSetting


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
        self._precision_level = PrecisionLevel.BIT_8
        self._num_of_bits = 8
        self._auto_finetune_level = AutoFinetuneLevel.DO_NOTHING
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
        self._observer_mapping = {
            "default": "xquant",
            "kl": "kl",
            "mse": "mse",
            "minmax": "minmax",
            "percentile": "percentile",
        }
        self._verbose = False

    def quantize(
        self,
        inputs: Union[torch.Tensor, list, dict],
        calib_dataloader: Iterable,
        executor: BaseGraphExecutor,
        xquant_setting: XQuantSetting,
        **kwargs,
    ) -> None:
        self._xquant_setting = xquant_setting
        self._set_max_percentile = xquant_setting.quantization_parameters.max_percentile
        self._calibration_type = xquant_setting.calibration_parameters.calibration_type
        self._auto_finetune_level = xquant_setting.quantization_parameters.finetune_level
        self._precision_level = xquant_setting.quantization_parameters.precision_level

        quant_setting = QuantizationSettingFactory.default_setting()
        quant_setting.fusion_setting.align_quantization = False

        return super().quantize(inputs, calib_dataloader, executor, quant_setting, **kwargs)

    def report(self):
        pass

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        observer = "default" if self._calibration_type is None else self._calibration_type
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy,
            rounding=self.rounding_policy,
            op=operation,
            num_of_bits=self._num_of_bits,
            exponent_bits=0,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
            observer_algorithm=self._observer_mapping.get(observer),
        )
        # 对常量输入永远采用minmax
        for in_var, in_tqc in zip(operation.inputs, base_quant_config.input_quantization_config):
            if in_var.is_parameter:
                in_tqc.observer_algorithm = "minmax"

        if isinstance(self._set_max_percentile, float):
            for in_var, in_tqc in zip(operation.inputs, base_quant_config.input_quantization_config):
                if not in_var.is_parameter:
                    in_tqc.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE] = self._set_max_percentile
            for out_var, out_tqc in zip(operation.outputs, base_quant_config.output_quantization_config):
                if not out_var.is_parameter:
                    out_tqc.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE] = self._set_max_percentile

        if operation.type in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, "Seems you got a Conv layer with no parameters."
            # set weight tqc
            if operation.inputs[1].is_parameter:
                in_tqc = base_quant_config.input_quantization_config[1]
                in_tqc.num_of_bits = 8
                in_tqc._quant_min, in_tqc._quant_max = _get_quant_min_max(in_tqc.num_of_bits)
                in_tqc.policy = self._op_type_to_policy[operation.type]
                in_tqc.observer_algorithm = "minmax"
                if in_tqc.policy.has_property(QuantizationProperty.PER_CHANNEL):
                    in_tqc.channel_axis = 1 if operation.type == "ConvTranspose" else 0
                    if operation.type == "Gemm" and operation.attributes.get("transB", 0) == 0:
                        in_tqc.channel_axis = 1

            # if operation has bias
            if operation.num_of_input > 2 and operation.inputs[-1].is_parameter:
                in_tqc = base_quant_config.input_quantization_config[-1]
                in_tqc.policy = self._op_type_to_policy[operation.type]
                in_tqc.num_of_bits = 32
                in_tqc._quant_min, in_tqc._quant_max = _get_quant_min_max(in_tqc.num_of_bits)
                in_tqc.state = QuantizationStates.PASSIVE_INIT
                if in_tqc.policy.has_property(QuantizationProperty.PER_CHANNEL):
                    in_tqc.channel_axis = 0

        elif operation.type in {"LayerNormalization", "InstanceNormalization", "BatchNormalization"}:
            for in_tqc in base_quant_config.input_quantization_config[1:]:
                in_tqc.state = QuantizationStates.FP32

        return base_quant_config

    @property
    def quant_operation_types(self) -> set:
        QUANTTYPE = {
            "Conv",
            "ConvTranspose",
            "Gemm",
            "MatMul",
            #
            "Relu",
            "PRelu",
            "LeakyRelu",
            "Sigmoid",
            "HardSwish",
            "HardSigmoid",
            "Gelu",
            "LRN",
            "Clip",
            #
            "Pad",
            #
            "Resize",
            #
            "MaxPool",
            "AveragePool",
            "GlobalMaxPool",
            "GlobalAveragePool",
            "ReduceMean",
            #
            "Add",
            "Sub",
            "Mul",
            # "Div",
            # "Max",
            #
            "LayerNormalization",
            "InstanceNormalization",
            #
            "Gather",
            "Reshape",
            "Concat",
            "Split",
            "Transpose",
            "Slice",
            "Flatten",
        }
        QUANTTYPE.update(PASSIVE_OPERATIONS)
        return QUANTTYPE

    @property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.ASYMMETRICAL + QuantizationProperty.PER_TENSOR + QuantizationProperty.LINEAR
        )

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @property
    def activation_fusion_types(self) -> set:
        return {"Relu", "Clip"}

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        assert isinstance(setting, QuantizationSetting), (
            f"PPQ needs a OptimSetting instance to initialize optimization pipeline,"
            f" however {type(setting)} was given."
        )
        ppq_ver = _get_version_number(PPQ_CONFIG.VERSION)

        list_of_passes = []

        list_of_passes.append(PassiveParameterBakingPass())

        list_of_passes.append(
            PPQQuantizeFusionPass(
                fuse_activation=True,
                fuse_passive_op=True,
                activation_type=self.activation_fusion_types,
            )
        )

        if ppq_ver <= _get_version_number("0.6.6"):
            list_of_passes.append(QuantizeFusionPass(True))

        list_of_passes.append(QuantizeSimplifyPass())

        list_of_passes.append(ActivationClipRefine())

        param_setting = setting.quantize_parameter_setting
        list_of_passes.append(ParameterQuantizePass(method=param_setting.calib_algorithm))

        list_of_passes.append(
            QuantizeConfigRefinePass(
                self._precision_level.value, self._xquant_setting.quantization_parameters.custom_setting
            )
        )

        if setting.quantize_activation:
            act_setting = setting.quantize_activation_setting
            list_of_passes.append(
                RuntimeBlockWiseCalibrationPass(
                    act_setting.calib_algorithm,
                    calib_block_size=XQUANT_CONFIG.default_block_size,
                    block_wise=True,
                    fintune_epoch=XQUANT_CONFIG.fine_tune_epoch,
                    auto_finetune_level=self._auto_finetune_level.value,
                )
            )

        list_of_passes.append(ParameterBakingPass())
        list_of_passes.append(PassiveParameterBakingPass())
        list_of_passes.append(AsymmetricaUnsignlAlignSign())
        return QuantizationOptimizationPipeline(passes=list_of_passes)

    def build_prequant_pipeline(
        self, setting: QuantizationSetting, executor: BaseGraphExecutor
    ) -> QuantizationOptimizationPipeline:
        assert isinstance(setting, QuantizationSetting), (
            f"PPQ needs a OptimSetting instance to initialize optimization pipeline,"
            f" however {type(setting)} was given."
        )

        list_of_passes = []

        if setting.weight_split:
            weight_split_setting = setting.weight_split_setting
            list_of_passes.append(
                HorizontalLayerSplitPass(
                    interested_layers=weight_split_setting.interested_layers,
                    method=weight_split_setting.method,
                    value_threshold=weight_split_setting.value_threshold,
                )
            )

        if setting.channel_split:
            channel_split_setting = setting.channel_split_setting
            list_of_passes.append(
                ChannelwiseSplitPass(
                    optimize_level=channel_split_setting.opt_level,
                    iterations=channel_split_setting.iterations,
                    threshold=channel_split_setting.value_threshold,
                    including_bias=channel_split_setting.including_bias,
                    including_act=channel_split_setting.including_act,
                    bias_multiplier=channel_split_setting.bias_multiplier,
                    act_multiplier=channel_split_setting.act_multiplier,
                )
            )

        if self._auto_finetune_level.value >= AutoFinetuneLevel.DO_NOTHING.value:
            equalization_setting = setting.equalization_setting
            list_of_passes.append(
                CustomLayerwiseEqualizationPass(
                    optimize_level=1,
                    iterations=XQUANT_CONFIG.equalization_iterations,
                    weight_threshold=equalization_setting.value_threshold,
                    including_bias=True,
                    including_act=self._auto_finetune_level.value >= AutoFinetuneLevel.LEVEL_1.value,
                    bias_multiplier=equalization_setting.bias_multiplier,
                    act_multiplier=equalization_setting.act_multiplier,
                )
            )

        return QuantizationOptimizationPipeline(passes=list_of_passes)
