#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import copy
import json
import os
import time
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnxslim
import torch
from xslim.logger import logger

from . import CalibrationCollect, XSlimDataset
from .analyse import statistical_analyse
from .defs import XQUANT_CONFIG
from .onnx_graph_helper import (format_onnx_model, merge_onnx_model,
                                truncate_onnx_model)
from .optimizer import GraphLegalized
from .ppq_decorator import (DISPATCHER_TABLE, BaseGraph, GraphDispatcher,
                            OnnxParser, ONNXRUNTIMExporter, TargetPlatform,
                            TorchExecutor)
from .quantizer import (XSlimQuantizer, convert_to_fp16_onnx_model,
                        dynamic_quantize_onnx_model)
from .xslim_setting import XSlimSetting, XSlimSettingFactory


def dispatch_graph(graph: BaseGraph, dispatcher: Union[str, GraphDispatcher] = "conservative") -> BaseGraph:
    quantizer = XSlimQuantizer(graph)
    if isinstance(dispatcher, str):
        dispatcher = dispatcher.lower()
        if dispatcher not in DISPATCHER_TABLE:
            raise ValueError(f'Can not found dispatcher type "{dispatcher}", check your input again.')
        dispatcher = DISPATCHER_TABLE[dispatcher](graph)
    else:
        if not isinstance(dispatcher, GraphDispatcher):
            raise TypeError(
                'Parameter "dispachter" of function ppq.api.dispatch_graph must be String or GraphDispatcher, '
                f"however {type(dispatcher)} was given."
            )
        dispatcher = dispatcher

    assert isinstance(dispatcher, GraphDispatcher)
    quant_types = quantizer.quant_operation_types
    dispatching_table = dispatcher.dispatch(
        graph=graph,
        quant_types=quant_types,
        quant_platform=TargetPlatform.UNSPECIFIED,
        fp32_platform=TargetPlatform.FP32,
        SOI_platform=TargetPlatform.SOI,
    )

    for operation in graph.operations.values():
        assert (
            operation.name in dispatching_table
        ), f"Internal Error, Can not find operation {operation.name} in dispatching table."
        operation.platform = dispatching_table[operation.name]
    return graph


def xslim_load_onnx_graph(
    file_or_model: Union[str, onnx.ModelProto], sim_en: bool = True, truncate_var_name: Sequence[str] = []
):
    if isinstance(file_or_model, onnx.ModelProto):
        onnx_model = file_or_model
    elif isinstance(file_or_model, str):
        onnx_model = onnx.load(file_or_model)
    else:
        raise TypeError("type of file_or_model error, {} .vs str or modelproto".format(type(file_or_model)))

    onnx_model = format_onnx_model(onnx_model, sim_en)
    onnx_model, truncate_left_graph, truncate_vars = truncate_onnx_model(onnx_model, truncate_var_name)

    graph = OnnxParser().build(onnx_model)
    return graph, truncate_left_graph, truncate_vars


def parse_xslim_config(file_or_dict: Union[str, dict]) -> XSlimSetting:
    config_dict = file_or_dict
    if isinstance(file_or_dict, str):
        with open(file_or_dict, "r") as fp:
            config_dict = json.load(fp)

    return XSlimSettingFactory.from_json(config_dict)


def quantize_onnx_model(
    path_or_config: Union[str, dict],
    input_onnx_model_or_path: Optional[Union[str, onnx.ModelProto]] = None,
    output_path: Optional[str] = None,
) -> onnx.ModelProto:
    """
    xslim model quantize api

    Args:
        path_or_config (Union[str, dict]): xslim config json file or config dict
        input_onnx_model_or_path (Optional[Union[str, onnx.ModelProto]], optional): input onnx model proto or path. Defaults to None.
        output_path (Optional[str], optional): output model path or output model dir. Defaults to None.

    Raises:
        RuntimeError: maybe set input_onnx_model_or_path error

    Returns:
        onnx.ModelProto: output model onnx proto
    """
    time_start = time.time()
    config_setting = parse_xslim_config(path_or_config)

    model_path = config_setting.model_parameters.onnx_model

    if input_onnx_model_or_path is not None:
        logger.info("using api input onnx model {}.".format(input_onnx_model_or_path))
        if isinstance(input_onnx_model_or_path, onnx.ModelProto):
            model_path = input_onnx_model_or_path
        elif os.path.exists(input_onnx_model_or_path):
            model_path = input_onnx_model_or_path
            config_setting.model_parameters.output_prefix = os.path.splitext(
                os.path.basename(input_onnx_model_or_path)
            )[0]
        else:
            raise RuntimeError("input_onnx_model_or_path set error.")

    if isinstance(output_path, str):
        logger.info("using api output path {}.".format(output_path))
        if os.path.isdir(output_path):
            config_setting.model_parameters.working_dir = output_path
        elif output_path[-5:] == ".onnx":
            config_setting.model_parameters.working_dir = os.path.dirname(output_path)
            config_setting.model_parameters.output_prefix = os.path.splitext(os.path.basename(output_path))[0]
        else:
            config_setting.model_parameters.working_dir = output_path

    config_setting.model_parameters.working_dir = os.path.realpath(config_setting.model_parameters.working_dir)

    output_prefix = config_setting.model_parameters.output_prefix
    working_dir = config_setting.model_parameters.working_dir
    calibration_step = config_setting.calibration_parameters.calibration_step
    calibration_device = config_setting.calibration_parameters.calibration_device

    if not os.path.exists(working_dir):
        logger.info("{} not existed and make new one.".format(working_dir))
        os.makedirs(working_dir)

    if isinstance(model_path, onnx.ModelProto):
        ori_onnx_model = model_path
    elif isinstance(model_path, str):
        ori_onnx_model = onnx.load(model_path)
    else:
        raise TypeError("type of file_or_model error, {} .vs str or modelproto".format(type(file_or_model)))

    if config_setting.quantization_parameters.precision_level.value >= 100:
        quant_onnx_model = format_onnx_model(ori_onnx_model, not config_setting.model_parameters.skip_onnxsim)
    elif config_setting.quantization_parameters.precision_level.value >= 4:
        if len(config_setting.quantization_parameters.ignore_op_types) > 0:
            logger.info(f"Ignoring op types: {config_setting.quantization_parameters.ignore_op_types}")
        if len(config_setting.quantization_parameters.ignore_op_names) > 0:
            logger.info(f"Ignoring op names: {config_setting.quantization_parameters.ignore_op_names}")
        quant_onnx_model = convert_to_fp16_onnx_model(
            model_path,
            config_setting.quantization_parameters.ignore_op_types,
            config_setting.quantization_parameters.ignore_op_names,
            not config_setting.model_parameters.skip_onnxsim,
        )
    elif config_setting.quantization_parameters.precision_level.value == 3:
        if len(config_setting.quantization_parameters.ignore_op_types) > 0:
            logger.info(f"Ignoring op types: {config_setting.quantization_parameters.ignore_op_types}")
        if len(config_setting.quantization_parameters.ignore_op_names) > 0:
            logger.info(f"Ignoring op names: {config_setting.quantization_parameters.ignore_op_names}")
        quant_onnx_model = dynamic_quantize_onnx_model(
            model_path,
            config_setting.quantization_parameters.ignore_op_types,
            config_setting.quantization_parameters.ignore_op_names,
            not config_setting.model_parameters.skip_onnxsim,
        )
    else:
        ppq_ir, truncate_left_graph, truncate_vars = xslim_load_onnx_graph(
            model_path,
            not config_setting.model_parameters.skip_onnxsim,
            config_setting.quantization_parameters.truncate_var_names,
        )

        GraphLegalized(ppq_ir)()

        ppq_ir = dispatch_graph(graph=ppq_ir, dispatcher="conservative")

        config_setting.calibration_parameters.check_input_parametres(ppq_ir)
        input_parametres = config_setting.calibration_parameters.input_parametres

        data_set = XSlimDataset(config_setting.calibration_parameters)
        calib_dataloader = torch.utils.data.DataLoader(data_set, batch_size=data_set.auto_batch_size)
        quantizer = XSlimQuantizer(ppq_ir)
        executor = TorchExecutor(graph=quantizer._graph, device=calibration_device)

        collate_fn = CalibrationCollect(input_parametres, calibration_device)

        single_graph_input_name = None
        dummy_input = None
        for k, v in ppq_ir.inputs.items():
            single_graph_input_name = k
        for data in calib_dataloader:
            data = collate_fn(data)
            if isinstance(data, torch.Tensor) and len(ppq_ir.inputs) == 1:
                if collate_fn is not None:
                    dummy_input = {single_graph_input_name: data}
            elif isinstance(data, dict):
                dummy_input = data
            else:
                raise TypeError(type(data))

        quantizer.quantize(
            inputs=dummy_input,
            calib_dataloader=calib_dataloader,
            executor=executor,
            xslim_setting=config_setting,
            calib_steps=calibration_step,
            collate_fn=collate_fn,
        )

        if config_setting.quantization_parameters.analysis_enable:
            try:
                test_dataloader = torch.utils.data.DataLoader(data_set, shuffle=True)
                report_path = os.path.join(working_dir, "{}_report.md".format(output_prefix))
                graphwise_analyse_results = statistical_analyse(
                    quantizer._graph,
                    calibration_device,
                    test_dataloader,
                    collate_fn,
                    steps=XQUANT_CONFIG.analyse_steps,
                    report_path=report_path,
                )
            except:
                logger.warning("quantize analysis failed and skiped.")

        quant_onnx_model = ONNXRUNTIMExporter().export(
            quantizer._graph,
        )

        quant_onnx_model = merge_onnx_model(quant_onnx_model, truncate_left_graph, truncate_vars)

    quant_onnx_model.metadata_props.extend(ori_onnx_model.metadata_props)
    quant_onnx_model.ir_version = max(9, ori_onnx_model.ir_version)
    onnx.save(quant_onnx_model, os.path.join(working_dir, "{}.onnx".format(output_prefix)))

    logger.info("quantization eplased time {:.2f} s".format(time.time() - time_start))
    return quant_onnx_model
