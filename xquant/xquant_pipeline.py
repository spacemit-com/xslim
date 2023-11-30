from typing import Union, Dict, Sequence
from collections import OrderedDict
import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import load_onnx_graph, dispatch_graph, export_ppq_graph
from ppq.executor import TorchExecutor
import ppq.lib as PFL
from .calibration_helper import XQuantDataset, CalibrationCollect
from .quantizer import XQUANT_CONFIG, AutoFinetuneLevel
from .optimizer import GraphLegalized


def parse_xquant_config(file_or_dict: Union[str, dict]):
    config_dict = file_or_dict
    if isinstance(file_or_dict, str):
        with open(file_or_dict, "r") as fp:
            config_dict = json.loads(fp.read())

    if not isinstance(config_dict, dict):
        raise TypeError("config type error {}".format(type(config_dict)))

    model_parameters = config_dict.get("model_parameters")
    calibration_parameters = config_dict.get("calibration_parameters")
    return config_dict


def quantize_onnx_model(path_or_config: Union[str, dict]):
    config_setting = parse_xquant_config(path_or_config)
    data_set = XQuantDataset(config_setting["calibration_parameters"])
    quant_setting = QuantizationSettingFactory.default_setting()

    model_path = config_setting["model_parameters"].get("onnx_model")
    output_prefix = config_setting["model_parameters"].get("output_prefix")
    working_dir = config_setting["model_parameters"].get("working_dir")
    calibration_step = config_setting["calibration_parameters"].get("calibration_step")
    calibration_device = config_setting["calibration_parameters"].get("calibration_device", "cuda")
    if calibration_device == "cuda" and not XQUANT_CONFIG.cuda_support:
        calibration_device = "cpu"

    input_parametres = config_setting["calibration_parameters"].get("input_parametres")

    calibration_type = config_setting["calibration_parameters"].get("calibration_type", "default")
    if calibration_type not in {"default", "minmax", "percentile"}:
        raise NotImplementedError("calibration_type {} not implemented yet.")
    quant_setting.quantize_activation_setting.calib_algorithm = calibration_type

    if "quantization_parameters" in config_setting:

        def set_dict_attr(obj, attr_name, default_value, type):
            value = config_setting["quantization_parameters"].get(attr_name, default_value)
            assert isinstance(value, type) or value is None, "value is {}".format(type)
            if value is not None:
                setattr(obj, attr_name, value)

        set_dict_attr(quant_setting.quantize_activation_setting, "max_percentile", None, float)
        set_dict_attr(quant_setting, "precision_level", 0, int)
        set_dict_attr(quant_setting, "auto_finetune_level", AutoFinetuneLevel.LEVEL_1.value, int)

    quant_setting.fusion_setting.align_quantization = False

    inputs_list = []
    for input_item in input_parametres:
        input_shape = input_item.get("input_shape")
        inputs_list.append(torch.zeros(size=input_shape, device=calibration_device, dtype=torch.float32))

    calib_dataloader = torch.utils.data.DataLoader(data_set)

    platform = TargetPlatform.ONNXRUNTIME
    ppq_ir = load_onnx_graph(onnx_import_file=model_path)

    GraphLegalized(ppq_ir)()

    ppq_ir = dispatch_graph(
        graph=ppq_ir,
        platform=platform,
        dispatcher=quant_setting.dispatcher,
        dispatching_table=quant_setting.dispatching_table,
    )

    dummy_input = inputs_list

    quantizer = PFL.Quantizer(platform, ppq_ir)
    executor = TorchExecutor(graph=quantizer._graph, device=calibration_device)

    quantizer.quantize(
        inputs=dummy_input,
        calib_dataloader=calib_dataloader,
        executor=executor,
        setting=quant_setting,
        calib_steps=calibration_step,
        collate_fn=CalibrationCollect(input_parametres, calibration_device),
    )
    quantizer.report()

    export_ppq_graph(
        graph=quantizer._graph,
        platform=platform,
        graph_save_to=os.path.join(working_dir, "{}.onnx".format(output_prefix)),
        config_save_to=os.path.join(working_dir, "{}.json".format(output_prefix)),
    )

    return quantizer._graph
