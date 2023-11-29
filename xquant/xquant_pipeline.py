from typing import Union, Dict, Sequence
from collections import OrderedDict
import json
import torch
import os
from .calibration_helper import XQuantDataset, CalibrationCollect
from torch.utils.data import Dataset, DataLoader
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_onnx_model as ppq_quantize_onnx_model


def parse_xquant_config(file_or_dict: Union[str, dict]):
    config_dict = file_or_dict
    if isinstance(file_or_dict, str):
        with open(file_or_dict, "r") as fp:
            config_dict = json.loads(fp.read())

    if not isinstance(config_dict, dict):
        raise TypeError

    model_parameters = config_dict.get("model_parameters")
    calibration_parameters = config_dict.get("calibration_parameters")
    return config_dict


def quantize_onnx_model(path_or_config: Union[str, dict]):
    config_setting = parse_xquant_config(path_or_config)
    data_set = XQuantDataset(config_setting["calibration_parameters"])
    quant_setting = QuantizationSettingFactory.default_setting()
    quant_setting.fusion_setting.align_quantization = False
    quant_setting.equalization = True

    model_path = config_setting["model_parameters"].get("onnx_model")
    output_model_file_prefix = config_setting["model_parameters"].get("output_model_file_prefix")
    working_dir = config_setting["model_parameters"].get("working_dir")
    calibration_step = config_setting["calibration_parameters"].get("calibration_step")
    calibration_device = config_setting["calibration_parameters"].get("calibration_device", "cuda")
    input_parametres = config_setting["calibration_parameters"].get("input_parametres")

    calibration_type = config_setting["calibration_parameters"].get("calibration_type", "default")
    if calibration_type != "default":
        quant_setting.quantize_activation_setting.calib_algorithm = calibration_type
    if "quantization_parameters" in config_setting:
        precision_level = config_setting["quantization_parameters"].get("precision_level", 0)
        quant_setting.precision_level = precision_level

    inputs_list = []
    for input_item in input_parametres:
        input_shape = input_item.get("input_shape")
        inputs_list.append(torch.zeros(size=input_shape, device=calibration_device, dtype=torch.float32))
    calib_loader = torch.utils.data.DataLoader(data_set)
    quantized_graph = ppq_quantize_onnx_model(
        onnx_import_file=model_path,
        calib_dataloader=calib_loader,
        calib_steps=calibration_step,
        input_shape=None,
        inputs=inputs_list,
        setting=quant_setting,
        collate_fn=CalibrationCollect(input_parametres),
        platform=TargetPlatform.ONNXRUNTIME,
        device=calibration_device,
        verbose=0,
    )

    export_ppq_graph(
        graph=quantized_graph,
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=os.path.join(working_dir, "{}.onnx".format(output_model_file_prefix)),
        config_save_to=os.path.join(working_dir, "{}.json".format(output_model_file_prefix)),
    )

    return quantized_graph
