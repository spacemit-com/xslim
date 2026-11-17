#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import copy
import json
import os
from enum import Enum
from typing import Dict, Literal, Optional, Pattern, Sequence, Union

import onnx

from xslim.logger import logger

from .defs import XQUANT_CONFIG, AutoFinetuneLevel, PrecisionLevel
from .ppq_decorator import BaseGraph


class SettingSerialize:
    def check(self, qsetting):
        pass

    def from_json(self, obj_setting: dict, qsetting):
        for key, value in obj_setting.items():
            if key in self.__dict__:
                if "builtin" in self.__dict__[key].__class__.__module__:
                    if isinstance(value, list) and hasattr(self, "from_list"):
                        self.__dict__[key] = getattr(
                            self, "from_list")(key, value, qsetting)
                    else:
                        self.__dict__[key] = copy.deepcopy(value)
                elif isinstance(self.__dict__[key], Enum):
                    self.__dict__[key] = self.__dict__[key].__class__(value)
                else:
                    assert isinstance(value, dict)
                    if isinstance(self.__dict__[key], SettingSerialize):
                        self.__dict__[key].from_json(value, qsetting)

        self.check(qsetting)

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False)

    def to_dict(self) -> str:
        return {k: v for k, v in self.__dict__.items()}


class ModelParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.onnx_model: str = None
        self.output_prefix: str = None
        self.working_dir: str = None
        self.skip_onnxsim: bool = False
        self.opset: Optional[int] = None

    def check(self, qsetting):
        # if not os.path.exists(self.onnx_model):
        #    raise FileExistsError(self.onnx_model)

        if self.opset is not None and not isinstance(self.opset, int):
            raise TypeError("opset type error, {} .vs int".format(type(self.opset)))

        if self.opset is not None and self.opset < 1:
            raise ValueError("opset should be a positive integer.")

        if self.working_dir is None:
            if isinstance(self.onnx_model, str) and os.path.exists(self.onnx_model):
                self.working_dir = os.path.dirname(self.onnx_model)
            else:
                self.working_dir = os.path.join(os.curdir, "temp")
            logger.info(
                "Not set working_dir, deatults to {}.".format(self.working_dir))

        if self.output_prefix is None:
            if isinstance(self.onnx_model, str) and os.path.exists(self.onnx_model):
                model_name = os.path.splitext(
                    os.path.basename(self.onnx_model))[0]
                self.output_prefix = "{}.q".format(model_name)
            else:
                self.output_prefix = "xslim.q"
            logger.info("Not set output_prefix, deatults to {}.".format(
                self.output_prefix))


class InputParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.input_name: str = None
        self.input_shape: Sequence[int] = None
        self.file_type: Literal["img", "npy", "raw"] = "img"
        self.color_format: Literal["rgb", "bgr"] = "bgr"
        self.mean_value: Sequence[float] = None
        self.std_value: Sequence[float] = None
        self.preprocess_file: str = None
        self.data_list_path: str = None
        self.dtype: str = "float32"

    def check(self, qsetting):
        if self.preprocess_file is not None and not isinstance(self.preprocess_file, str):
            raise TypeError(
                "preprocess_file type error, {} .vs str".format(self.preprocess_file))

        if self.data_list_path is not None and not isinstance(self.data_list_path, str):
            raise TypeError(
                "data_list_path type error, {} .vs str".format(self.data_list_path))

        if isinstance(self.data_list_path, str) and not os.path.exists(self.data_list_path):
            raise FileExistsError(self.data_list_path)

        if self.mean_value is not None and not isinstance(self.mean_value, list):
            raise TypeError(
                "mean_value type error, {} .vs str".format(self.mean_value))

        if self.std_value is not None and not isinstance(self.std_value, list):
            raise TypeError(
                "std_value type error, {} .vs str".format(self.std_value))

        if self.file_type not in {"img", "npy", "raw"}:
            raise NotImplementedError(
                "file_type {} not implemented yet.".format(self.file_type))


class CustomQuantizationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.input_names: Sequence[str] = None
        self.output_names: Sequence[str] = None
        self.max_percentile: float = None
        self.precision_level: int = None
        self.calibration_type: str = None


class OperatorQuantizationSetting(SettingSerialize):
    def __init__(self) -> None:
        self.pattern: str = None
        self.precision: str = None
        self.calibration_type: str = None

    def check(self, qsetting):
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("operator_quantization pattern must be a non-empty string.")
        if self.precision not in {"int8", "fp32"}:
            raise ValueError("operator_quantization precision must be int8 or fp32.")
        if self.calibration_type is not None:
            if self.precision != "int8" or self.calibration_type not in {"default", "minmax", "percentile", "kl", "mse", "auto"}:
                raise ValueError("operator calibration_type requires int8 and a supported calibration method.")


class QuantizationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.precision_level: PrecisionLevel = PrecisionLevel.LEVEL_0
        self.max_percentile: float = None
        self.finetune_level: AutoFinetuneLevel = AutoFinetuneLevel.LEVEL_1
        self.custom_setting: Sequence[CustomQuantizationParameterSetting] = None
        self.analysis_enable: bool = True
        self.truncate_var_names: Sequence[str] = []
        self.ignore_op_types: Sequence[str] = []
        self.ignore_op_names: Sequence[str] = []
        self.operator_quantization: Sequence[OperatorQuantizationSetting] = []

    def from_list(self, value_name, obj_setting, qsetting):
        if value_name == "operator_quantization":
            rules = []
            for item in obj_setting:
                if not isinstance(item, dict):
                    raise ValueError("operator_quantization entries must be objects.")
                rule = OperatorQuantizationSetting()
                rule.from_json(item, qsetting)
                rules.append(rule)
            return rules
        if value_name == "custom_setting":
            custom_setting = []
            for item_dict in obj_setting:
                setting_item = CustomQuantizationParameterSetting()
                setting_item.from_json(item_dict, qsetting)
                custom_setting.append(setting_item)
            return custom_setting
        else:
            return obj_setting

    def check(self, qsetting):
        if not isinstance(self.operator_quantization, list):
            raise ValueError("operator_quantization must be a list.")
        for rule in self.operator_quantization:
            if not isinstance(rule, OperatorQuantizationSetting):
                raise ValueError("operator_quantization entries must be OperatorQuantizationSetting objects.")
            rule.check(qsetting)
        if self.operator_quantization and self.precision_level.value not in {0, 1, 2}:
            raise ValueError("operator_quantization is supported only for static INT8 precision levels 0, 1 and 2.")
        if self.precision_level.value > PrecisionLevel.LEVEL_0.value:
            logger.info("set higher precision level.")
        if self.finetune_level.value > AutoFinetuneLevel.LEVEL_1.value:
            logger.info("set higher finetune level.")


class CalibrationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.calibration_batch_size: int = 1
        self.calibration_step: int = 500
        self.calibration_device: str = "cuda" if XQUANT_CONFIG.cuda_support else "cpu"
        self.calibration_type: str = "default"
        self.auto_calibration_methods: Sequence[str] = ["minmax", "percentile", "mse", "kl"]
        self.auto_calibration_samples: int = 8192
        self.input_parameters: Sequence[InputParameterSetting] = None

    def from_list(self, value_name: str, obj_setting: Sequence, qsetting):
        if value_name == "input_parameters" or value_name == "input_parametres":
            input_parameters = []
            for item_dict in obj_setting:
                setting_item = InputParameterSetting()
                setting_item.from_json(item_dict, qsetting)
                input_parameters.append(setting_item)
            return input_parameters
        else:
            return obj_setting

    def check(self, qsetting):
        methods = self.auto_calibration_methods
        if (not isinstance(methods, list) or not methods
                or any(not isinstance(method, str) or method not in {"minmax", "percentile", "mse", "kl"}
                       for method in methods) or len(set(methods)) != len(methods)):
            raise ValueError("auto_calibration_methods must be a non-empty list of distinct minmax/percentile/mse/kl.")
        if type(self.auto_calibration_samples) is not int or self.auto_calibration_samples < 1:
            raise ValueError("auto_calibration_samples must be a positive integer.")
        if self.calibration_device == "cuda" and not XQUANT_CONFIG.cuda_support:
            logger.warning(
                "Specifies that cuda is used but not detected. Turn to cpu.")
            self.calibration_device = "cpu"

        if self.calibration_step > 1000 or self.calibration_step < 10:
            logger.warning(
                "Specifies that calibration_step is too large or too small, it should be in [10, 1000].")

        assert len(
            self.input_parameters) > 0, "Calibration input_parameters setting not detected."

        if self.calibration_type not in {"default", "minmax", "percentile", "kl", "mse", "auto"}:
            raise NotImplementedError(
                "calibration_type {} not implemented yet.".format(self.calibration_type))

        if self.calibration_type != "default":
            logger.info("set calibration_type {}.".format(
                self.calibration_type))

    def check_input_parameters(self, ppq_ir: BaseGraph):
        assert len(self.input_parameters) == len(
            ppq_ir.inputs
        ), "Calibration input_parameters size should <= model inputs size."

        pb_input = ppq_ir._detail.get("pb_input", [])
        for in_type, calib_parameter in zip(pb_input, self.input_parameters):
            input_shape = [
                i.dim_value if isinstance(
                    i.dim_value, int) and i.dim_value > 0 else None
                for i in in_type.type.tensor_type.shape.dim
            ]
            if isinstance(in_type.type.tensor_type.elem_type, int):
                input_dtype = onnx.helper.tensor_dtype_to_np_dtype(
                    in_type.type.tensor_type.elem_type
                ).name
            else:
                input_dtype = None
            if input_dtype is not None:
                calib_parameter.dtype = input_dtype
            if calib_parameter.input_name is None:
                calib_parameter.input_name = in_type.name
            if calib_parameter.input_shape is None and len(input_shape) > 0:
                input_shape[0] = input_shape[0] if isinstance(
                    input_shape[0], int) else 1
                calib_parameter.input_shape = input_shape


class StrategyParameterSetting(SettingSerialize):
    def __init__(self):
        self.mode = "standard"
        self.allowed_precisions = ["int8", "fp32"]
        self.default_precision = None
        self.node_precisions = {}
        self.op_type_precisions = {}
        self.rules = []

    @property
    def active(self):
        return (self.mode != "standard" or self.allowed_precisions != ["int8", "fp32"]
                or self.default_precision is not None
                or bool(self.node_precisions or self.op_type_precisions or self.rules))

    def check(self, qsetting):
        if self.mode != "standard":
            raise NotImplementedError(f"Strategy mode {self.mode!r} is not supported.")
        allowed = self.allowed_precisions
        if (not isinstance(allowed, list) or not allowed
                or any(not isinstance(p, str) or p not in {"int8", "fp32", "fp16"} for p in allowed)
                or len(set(allowed)) != len(allowed)):
            raise ValueError("allowed_precisions must be a non-empty distinct list of int8, fp32 or fp16.")
        if self.default_precision is not None and self.default_precision not in allowed:
            raise ValueError("default_precision must be in allowed_precisions.")
        for field_name in ("node_precisions", "op_type_precisions"):
            values = getattr(self, field_name)
            if (not isinstance(values, dict)
                    or any(not isinstance(k, str) or not k or not isinstance(v, str) or v not in allowed
                           for k, v in values.items())):
                raise ValueError(f"{field_name} must map non-empty names to allowed precisions.")
        if not isinstance(self.rules, list):
            raise ValueError("strategy rules must be a list.")
        for rule in self.rules:
            if not isinstance(rule, dict) or rule.get("precision") not in allowed:
                raise ValueError("Each strategy rule must be an object with an allowed precision.")
            selector = rule.get("selector")
            fields = {"node": ["names"], "type": ["op_types"],
                      "subgraph": ["start_nodes", "end_nodes"]}.get(selector)
            if fields is None or set(rule) != {"selector", "precision", *fields}:
                raise ValueError("Strategy rules require selector-specific fields (node/type/subgraph).")
            for name in fields:
                values = rule[name]
                if (not isinstance(values, list) or not values
                        or any(not isinstance(v, str) or not v for v in values)):
                    raise ValueError(f"Strategy rule {name} must be a non-empty list of names.")


class XSlimSetting(SettingSerialize):
    def __init__(self) -> None:
        self.model_parameters = ModelParameterSetting()
        self.calibration_parameters = CalibrationParameterSetting()
        self.quantization_parameters = QuantizationParameterSetting()
        self.strategy_parameters = StrategyParameterSetting()


    def check(self, qsetting):
        self.strategy_parameters.check(qsetting)
        if (self.strategy_parameters.active
                and self.quantization_parameters.precision_level.value not in {0, 1, 2}):
            raise ValueError("Precision strategies are supported only for static INT8 precision levels 0, 1 and 2.")
        if (self.calibration_parameters.calibration_type == "auto"
                and self.quantization_parameters.precision_level.value not in {0, 1, 2}):
            raise ValueError("auto calibration is supported only for static INT8 precision levels 0, 1 and 2.")


class XSlimSettingFactory:
    def from_json(json_obj: Union[str, dict]) -> XSlimSetting:
        if isinstance(json_obj, str):
            setting_dict = json.loads(json_obj)
        else:
            setting_dict = json_obj

        setting = XSlimSetting()
        setting.from_json(setting_dict, setting)

        return setting
