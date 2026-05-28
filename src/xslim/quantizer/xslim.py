#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import functools
import os
from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import torch
from xslim.logger import logger

from ..defs import (PASSIVE_OPERATIONS, XQUANT_CONFIG, AutoFinetuneLevel,
                    PrecisionLevel)
from ..optimizer import (ActivationClipRefine, AsymmetricaUnsignlAlignSign,
                         ComputingFusionPass, FlattenGemmFusionPass,
                         FormatBatchNormalizationPass, HardSwishFusionPass,
                         PassiveParameterBakingPass, QuantizeConfigRefinePass,
                         RuntimeBlockWiseCalibrationPass, SwishFusionPass,
                         XSlimLayerwiseEqualizationPass)
from ..ppq_decorator import (BaseGraph, BaseGraphExecutor, GraphReplacer,
                             Operation, OperationQuantizationConfig,
                             ParameterBakingPass, ParameterQuantizePass,
                             QuantableGraph, QuantableOperation,
                             QuantizationOptimizationPipeline,
                             QuantizationPolicy, QuantizationProperty,
                             QuantizationStates, QuantizeFusionPass,
                             QuantizeOperationCommand, QuantizeSimplifyPass,
                             RoundingPolicy, TargetPlatform,
                             TensorQuantizationConfig, Variable, ppq_common)
from ..xslim_setting import XSlimSetting


def _get_quant_min_max(num_of_bits: int, signed: bool = True):
    if signed:
        return -(2 ** (num_of_bits - 1)), 2 ** (num_of_bits - 1) - 1
    else:
        return 0, 2**num_of_bits - 1


class XSlimQuantizer:
    def __init__(self, graph: BaseGraph) -> Union[torch.Tensor, list, dict]:
        self._graph = graph
        self._processor = QuantableGraph(GraphReplacer(self._graph))
        self._precision_level = PrecisionLevel.LEVEL_0
        self._num_of_bits = 8
        self._quant_min, self._quant_max = _get_quant_min_max(self._num_of_bits)
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
            "BatchMatMul": pertensor_policy,
        }
        self._observer_mapping = {
            "default": "xslim",
            "kl": "kl",
            "mse": "mse",
            "minmax": "minmax",
            "percentile": "percentile",
        }
        self._verbose = False
        self._bias_to_fp32 = bool(int(os.environ.get("XQUANT_BIAS_TO_FP32", "1")) > 0)

        if self._bias_to_fp32:
            logger.info("Set Bias to Float32.")

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.XQUANT_INT8

    def quantize_operation(self, op_name: str, platform: TargetPlatform = None) -> QuantableOperation:
        if op_name not in self._graph.operations:
            raise KeyError(f"Can not find op {op_name} in your graph, chech operation name again.")
        converting_operation = self._graph.operations[op_name]
        if isinstance(converting_operation, QuantableOperation):
            logger.warning(f"Operation {op_name} has been quantized, can not to quantize it twice.")
            return converting_operation

        # override platform with calling parameter.
        if platform is not None:
            converting_operation.platform = platform
        else:
            platform = converting_operation.platform

        if platform in {TargetPlatform.FP32, TargetPlatform.SOI}:
            return self._graph.operations[op_name]

        # if platform == TargetPlatform.UNSPECIFIED we can skip its quantization when type is not supported.
        if platform == TargetPlatform.UNSPECIFIED and converting_operation.type not in self.quant_operation_types:
            return self._graph.operations[op_name]

        # create quantize config and convert operation.
        self._processor(
            QuantizeOperationCommand(
                op_name=op_name,
                target_platform=platform,
                config=self.init_quantize_config(operation=converting_operation),
            )
        )
        return self._graph.operations[op_name]

    def quantize(
        self,
        inputs: Union[torch.Tensor, list, dict],
        calib_dataloader: Iterable,
        executor: BaseGraphExecutor,
        xslim_setting: XSlimSetting,
        **kwargs,
    ) -> None:
        self._set_max_percentile = xslim_setting.quantization_parameters.max_percentile
        self._calibration_type = xslim_setting.calibration_parameters.calibration_type

        executor.load_graph(self._graph)
        executor.tracing_operation_meta(inputs=inputs)
        # step - 1, prequant pipeline:
        # prequant pipeline will change your network structure and float value.
        prequant_pipeline = self.build_prequant_pipeline(xslim_setting, executor=executor)
        prequant_pipeline.optimize(
            graph=self._graph, dataloader=calib_dataloader, executor=executor, verbose=self._verbose, **kwargs
        )

        # step - 2, quantize all operations
        executor.load_graph(self._graph)
        executor.tracing_operation_meta(inputs=inputs)

        for op_name, operation in self._graph.operations.items():
            if operation.platform == TargetPlatform.UNSPECIFIED:
                if operation.type in self.quant_operation_types:
                    operation.platform = self.target_platform
                else:
                    operation.platform = TargetPlatform.FP32

            if operation.platform not in {TargetPlatform.FP32, TargetPlatform.SOI}:
                self.quantize_operation(op_name)

        # quantize operation will modify network structure
        # it is necessary calling self._executor before further execution
        # step - 3, calling graph optimization pipeline
        executor.load_graph(self._graph)
        quant_pipeline = self.build_quant_pipeline(xslim_setting)

        quant_pipeline.optimize(
            graph=self._graph, dataloader=calib_dataloader, executor=executor, verbose=self._verbose, **kwargs
        )

    @staticmethod
    def create_default_quant_config(
        op: Operation,
        num_of_bits: int = 8,
        quant_min: Union[int, float] = -127,
        quant_max: Union[int, float] = 128,
        observer_algorithm: str = "percentile",
        policy: QuantizationPolicy = QuantizationPolicy(
            QuantizationProperty.PER_TENSOR + QuantizationProperty.LINEAR + QuantizationProperty.SYMMETRICAL
        ),
        rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
        exponent_bits: int = 0,
    ) -> OperationQuantizationConfig:
        assert isinstance(op, Operation), f"Can only initialize OQC for PPQ.IR.Operation, however {type(op)} was given."
        assert isinstance(
            policy, QuantizationPolicy
        ), f"Can not create quantization config - Quantization Policy Type Error."
        assert isinstance(rounding, RoundingPolicy), f"Can not create quantization config - Rounding Policy Type Error."

        socket = op.socket
        input_cfgs, output_cfgs = [], []
        for index in range(op.num_of_input):
            state = QuantizationStates.INITIAL
            # for those unexpected inputs and outputs
            # ppq just initilize them as normal variable.
            if index < len(socket.in_plat):
                target_plat = socket.in_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            input_cfgs.append(
                TensorQuantizationConfig(
                    policy=policy,
                    rounding=rounding,
                    num_of_bits=num_of_bits,
                    scale=None,
                    offset=None,
                    exponent_bits=exponent_bits,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    observer_algorithm=observer_algorithm,
                    state=state,
                )
            )

        for index in range(op.num_of_output):
            state = QuantizationStates.INITIAL
            # for those unexpected inputs and outputs
            # ppq just initilize them as normal variable.
            if index < len(socket.out_plat):
                target_plat = socket.out_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            output_cfgs.append(
                TensorQuantizationConfig(
                    policy=policy,
                    rounding=rounding,
                    num_of_bits=num_of_bits,
                    scale=None,
                    offset=None,
                    exponent_bits=exponent_bits,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    observer_algorithm=observer_algorithm,
                    state=state,
                )
            )

        return OperationQuantizationConfig(input_cfgs, output_cfgs)

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
                if self._bias_to_fp32:
                    in_tqc.state = QuantizationStates.FP32
                else:
                    in_tqc.policy = self._op_type_to_policy[operation.type]
                    in_tqc.num_of_bits = 32
                    in_tqc._quant_min, in_tqc._quant_max = _get_quant_min_max(in_tqc.num_of_bits)
                    in_tqc.state = QuantizationStates.PASSIVE_INIT
                    if in_tqc.policy.has_property(QuantizationProperty.PER_CHANNEL):
                        in_tqc.channel_axis = 0

        elif operation.type in {"BatchMatMul"}:
            for in_tqc in base_quant_config.input_quantization_config[2:]:
                in_tqc.state = QuantizationStates.FP32
            if operation.inputs[1].is_parameter:
                in_tqc = base_quant_config.input_quantization_config[1]
                in_tqc.num_of_bits = 8
                in_tqc._quant_min, in_tqc._quant_max = _get_quant_min_max(in_tqc.num_of_bits)
                in_tqc.policy = self._op_type_to_policy[operation.type]
                in_tqc.observer_algorithm = "minmax"

        elif operation.type in {"LayerNormalization", "LayerNorm", "InstanceNormalization", "BatchNormalization"}:
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
            "BatchMatMul",
            #
            "Relu",
            "PRelu",
            "LeakyRelu",
            "Sigmoid",
            "HardSwish",
            "HardSigmoid",
            "Gelu",
            "GELU",
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
            "LayerNorm",
            "InstanceNormalization",
            "GroupNormalization",
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

    def build_quant_pipeline(self, setting: XSlimSetting) -> QuantizationOptimizationPipeline:
        list_of_passes = []

        list_of_passes.append(
            QuantizeConfigRefinePass(
                setting.quantization_parameters.precision_level.value,
                setting.quantization_parameters.custom_setting,
            )
        )

        list_of_passes.append(
            QuantizeFusionPass(
                fuse_activation=True,
                fuse_passive_op=True,
                fuse_relu_clip=True,
                activation_type=self.activation_fusion_types,
            )
        )
        list_of_passes.append(QuantizeSimplifyPass())

        list_of_passes.append(ActivationClipRefine())

        list_of_passes.append(PassiveParameterBakingPass())

        list_of_passes.append(ParameterQuantizePass(method="minmax"))

        list_of_passes.append(
            RuntimeBlockWiseCalibrationPass(
                self._observer_mapping.get(setting.calibration_parameters.calibration_type, "xslim"),
                block_wise=True,
                fintune_epoch=XQUANT_CONFIG.fine_tune_epoch,
                auto_finetune_level=setting.quantization_parameters.finetune_level.value,
            )
        )

        list_of_passes.append(ParameterBakingPass())
        list_of_passes.append(PassiveParameterBakingPass())
        return QuantizationOptimizationPipeline(passes=list_of_passes)

    def build_prequant_pipeline(
        self, setting: XSlimSetting, executor: BaseGraphExecutor
    ) -> QuantizationOptimizationPipeline:
        list_of_passes = []

        finetune_level = setting.quantization_parameters.finetune_level.value

        list_of_passes.append(FlattenGemmFusionPass())
        list_of_passes.append(FormatBatchNormalizationPass())

        if finetune_level >= AutoFinetuneLevel.DO_NOTHING.value:
            list_of_passes.append(
                XSlimLayerwiseEqualizationPass(
                    optimize_level=2 if finetune_level > AutoFinetuneLevel.DO_NOTHING.value else 1,
                    including_act=finetune_level > AutoFinetuneLevel.DO_NOTHING.value,
                )
            )

        return QuantizationOptimizationPipeline(passes=list_of_passes)
