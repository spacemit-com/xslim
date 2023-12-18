#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT
from typing import Union, Dict, Sequence
from collections import OrderedDict
import json
import torch
import os
from ppq import TargetPlatform, BaseQuantizer
from ppq.api import load_onnx_graph, dispatch_graph, export_ppq_graph
from ppq.executor import TorchExecutor
import ppq.lib as PFL
from .calibration_helper import XQuantDataset, CalibrationCollect
from .optimizer import GraphLegalized
from .xquant_setting import XQuantSettingFactory, XQuantSetting


def parse_xquant_config(file_or_dict: Union[str, dict]) -> XQuantSetting:
    config_dict = file_or_dict
    if isinstance(file_or_dict, str):
        with open(file_or_dict, "r") as fp:
            config_dict = json.load(fp)

    return XQuantSettingFactory.from_json(config_dict)


def quantize_onnx_model(path_or_config: Union[str, dict]):
    config_setting = parse_xquant_config(path_or_config)
    data_set = XQuantDataset(config_setting.calibration_parameters)

    model_path = config_setting.model_parameters.onnx_model
    output_prefix = config_setting.model_parameters.output_prefix
    working_dir = config_setting.model_parameters.working_dir
    calibration_step = config_setting.calibration_parameters.calibration_step
    calibration_device = config_setting.calibration_parameters.calibration_device
    input_parametres = config_setting.calibration_parameters.input_parametres

    inputs_list = []
    for input_item in input_parametres:
        input_shape = input_item.input_shape
        inputs_list.append(torch.zeros(size=input_shape, device=calibration_device, dtype=torch.float32))

    calib_dataloader = torch.utils.data.DataLoader(data_set)

    platform = TargetPlatform.ONNXRUNTIME
    ppq_ir = load_onnx_graph(onnx_import_file=model_path)

    GraphLegalized(ppq_ir)()

    ppq_ir = dispatch_graph(
        graph=ppq_ir,
        platform=platform,
        dispatcher="conservative",
        dispatching_table=None,
    )

    dummy_input = inputs_list

    quantizer = PFL.Quantizer(platform, ppq_ir)
    executor = TorchExecutor(graph=quantizer._graph, device=calibration_device)

    quantizer.quantize(
        inputs=dummy_input,
        calib_dataloader=calib_dataloader,
        executor=executor,
        xquant_setting=config_setting,
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
