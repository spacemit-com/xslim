#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Union, Dict, Sequence
from collections import OrderedDict
import json
import torch
import os
from pandas import DataFrame
from ppq import TargetPlatform, BaseQuantizer
from ppq.api import load_onnx_graph, dispatch_graph, export_ppq_graph
from ppq.executor import TorchExecutor
from ppq.quantization.analyse import graphwise_error_analyse, statistical_analyse
import ppq.lib as PFL
from .defs import xquant_info, xquant_warning
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

    if not os.path.exists(working_dir):
        xquant_info("{} not existed and make new one.".format(working_dir))
        os.makedirs(working_dir)

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

    if config_setting.quantization_parameters.analysis_enable:
        test_dataloader = torch.utils.data.DataLoader(data_set, shuffle=True)
        graphwise_analyse_results = statistical_analyse(
            quantizer._graph,
            calibration_device,
            test_dataloader,
            CalibrationCollect(input_parametres, calibration_device),
            steps=16,
        )
        result_keys = [
            "Op name",
            "Variable name",
            "Noise:Signal Power Ratio",
            "Quantized Max",
            "Quantized Min",
            "Float Max",
            "Float Min",
        ]
        variable_reports = []
        for report_info in graphwise_analyse_results:
            if not report_info["Is parameter"]:
                variable_reports.append({k: report_info[k] for k in result_keys})
                variable_reports[-1]["Float Hist"] = ",".join([str(int(i)) for i in report_info["Float Hist"]])
        sort_variable_reports = sorted(variable_reports, key=lambda x: x["Noise:Signal Power Ratio"], reverse=True)

        snr_topk = [x["Noise:Signal Power Ratio"] for x in sort_variable_reports[:10]]
        if sum(snr_topk) / len(snr_topk) > 2.0:
            xquant_warning("Noise check error, quantization may be failed.")
        for report_info in sort_variable_reports:
            snr_value = report_info["Noise:Signal Power Ratio"]
            if snr_value > 2.0:
                report_info["Noise:Signal Power Ratio"] = "=={}==".format(snr_value)
        sort_variable_reports_df = DataFrame(sort_variable_reports)
        report_path = os.path.join(working_dir, "{}_report.md".format(output_prefix))
        xquant_info("export quantization statistical results file to {}".format(report_path))
        reports_md = sort_variable_reports_df.to_markdown(report_path)

    export_ppq_graph(
        graph=quantizer._graph,
        platform=platform,
        graph_save_to=os.path.join(working_dir, "{}.onnx".format(output_prefix)),
        # config_save_to=os.path.join(working_dir, "{}.json".format(output_prefix)),
    )

    return quantizer._graph
