#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Union, Sequence, Dict, Literal, Pattern
from enum import Enum
import json
import copy
import os
import onnx
from ppq.core import common as ppq_common
from .defs import XQUANT_CONFIG, AutoFinetuneLevel, PrecisionLevel, xquant_warning, xquant_info


class SettingSerialize:
    def check(self, qsetting):
        pass

    def from_json(self, obj_setting: dict, qsetting):
        for key, value in obj_setting.items():
            if key in self.__dict__:
                if "builtin" in self.__dict__[key].__class__.__module__:
                    if isinstance(value, list) and hasattr(self, "from_list"):
                        self.__dict__[key] = getattr(self, "from_list")(key, value, qsetting)
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

    def check(self, qsetting):
        # if not os.path.exists(self.onnx_model):
        #    raise FileExistsError(self.onnx_model)

        if self.working_dir is None:
            if isinstance(self.onnx_model, str) and os.path.exists(self.onnx_model):
                self.working_dir = os.path.dirname(self.onnx_model)
            else:
                self.working_dir = os.path.join(os.curdir, "temp")
            xquant_info("Not set working_dir, deatults to {}.".format(self.working_dir))

        if self.output_prefix is None:
            if isinstance(self.onnx_model, str) and os.path.exists(self.onnx_model):
                model_name = os.path.splitext(os.path.basename(self.onnx_model))[0]
                self.output_prefix = "{}.q".format(model_name)
            else:
                self.output_prefix = "xquant.q"
            xquant_info("Not set output_prefix, deatults to {}.".format(self.output_prefix))


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
            raise TypeError("preprocess_file type error, {} .vs str".format(self.preprocess_file))

        if self.data_list_path is not None and not isinstance(self.data_list_path, str):
            raise TypeError("data_list_path type error, {} .vs str".format(self.data_list_path))

        if isinstance(self.data_list_path, str) and not os.path.exists(self.data_list_path):
            raise FileExistsError(self.data_list_path)

        if self.mean_value is not None and not isinstance(self.mean_value, list):
            raise TypeError("mean_value type error, {} .vs str".format(self.mean_value))

        if self.std_value is not None and not isinstance(self.std_value, list):
            raise TypeError("std_value type error, {} .vs str".format(self.std_value))

        if self.file_type not in {"img", "npy", "raw"}:
            raise NotImplementedError("file_type {} not implemented yet.".format(self.file_type))


class CustomQuantizationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.input_names: Sequence[str] = None
        self.output_names: Sequence[str] = None
        self.max_percentile: float = None
        self.precision_level: int = None
        self.calibration_type: str = None


class QuantizationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.precision_level: PrecisionLevel = PrecisionLevel.BIT_8
        self.max_percentile: float = None
        self.finetune_level: AutoFinetuneLevel = AutoFinetuneLevel.LEVEL_1
        self.custom_setting: Sequence[CustomQuantizationParameterSetting] = None
        self.analysis_enable: bool = True
        self.truncate_var_names: Sequence[str] = []

    def from_list(self, value_name, obj_setting, qsetting):
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
        if self.precision_level.value > PrecisionLevel.BIT_8.value:
            xquant_info("set higher precision level.")
        if self.finetune_level.value > AutoFinetuneLevel.LEVEL_1.value:
            xquant_info("set higher finetune level.")


class CalibrationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.calibration_step: int = 500
        self.calibration_device: str = "cuda" if XQUANT_CONFIG.cuda_support else "cpu"
        self.calibration_type: str = "default"
        self.input_parametres: Sequence[InputParameterSetting] = None

    def from_list(self, value_name: str, obj_setting: Sequence, qsetting):
        if value_name == "input_parametres":
            input_parametres = []
            for item_dict in obj_setting:
                setting_item = InputParameterSetting()
                setting_item.from_json(item_dict, qsetting)
                input_parametres.append(setting_item)
            return input_parametres
        else:
            return obj_setting

    def check(self, qsetting):
        if self.calibration_device == "cuda" and not XQUANT_CONFIG.cuda_support:
            xquant_warning("Specifies that cuda is used but not detected. Turn to cpu.")
            self.calibration_device = "cpu"

        if self.calibration_step > 1000 or self.calibration_step < 10:
            xquant_warning("Specifies that calibration_step is too large or too small, it should be in [10, 1000].")

        assert len(self.input_parametres) > 0, "Calibration input_parametres setting not detected."

        if self.calibration_type not in {"default", "minmax", "percentile", "kl", "mse"}:
            raise NotImplementedError("calibration_type {} not implemented yet.".format(self.calibration_type))

        if self.calibration_type != "default":
            xquant_info("set calibration_type {}.".format(self.calibration_type))

        if isinstance(qsetting, XQuantSetting):
            onnx_model_path = qsetting.model_parameters.onnx_model
            onnx_model = onnx.load(onnx_model_path)

            assert len(self.input_parametres) == len(
                onnx_model.graph.input
            ), "Calibration input_parametres size should equal to model inputs size."

            for input_idx, in_var in enumerate(onnx_model.graph.input):
                calib_parameter = self.input_parametres[input_idx]
                input_shape = [
                    i.dim_value if isinstance(i.dim_value, int) and i.dim_value > 0 else None
                    for i in in_var.type.tensor_type.shape.dim
                ]
                if isinstance(in_var.type.tensor_type.elem_type, int):
                    input_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[in_var.type.tensor_type.elem_type].name
                else:
                    input_dtype = None

                if input_dtype is not None:
                    calib_parameter.dtype = input_dtype

                if calib_parameter.input_name is None:
                    calib_parameter.input_name = in_var.name
                if calib_parameter.input_shape is None:
                    input_shape[0] = input_shape[0] if isinstance(input_shape[0], int) else 1
                    if all([isinstance(i, int) for i in input_shape[1:]]):
                        pass
                    else:
                        raise RuntimeError(
                            "Calibration input_parametres.shape or Model input shape should be vaild for var {}".format(
                                calib_parameter.input_name
                            )
                        )
                    calib_parameter.input_shape = input_shape


class XQuantSetting(SettingSerialize):
    def __init__(self) -> None:
        self.model_parameters = ModelParameterSetting()
        self.calibration_parameters = CalibrationParameterSetting()
        self.quantization_parameters = QuantizationParameterSetting()


class XQuantSettingFactory:
    def from_json(json_obj: Union[str, dict]) -> XQuantSetting:
        if isinstance(json_obj, str):
            setting_dict = json.loads(json_obj)
        else:
            setting_dict = json_obj

        setting = XQuantSetting()
        setting.from_json(setting_dict, setting)

        return setting
