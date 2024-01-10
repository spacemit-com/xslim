#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Union, Dict, Sequence
from collections import OrderedDict
import json
import torch
import os
import onnx
import time
import copy
import onnxsim
import onnx_graphsurgeon as osg
from pandas import DataFrame
from ppq import TargetPlatform, BaseQuantizer, BaseGraph
from ppq.executor import TorchExecutor
from ppq.api.interface import format_graph as ppq_format_graph
from ppq.scheduler import DISPATCHER_TABLE, GraphDispatcher
from .defs import xquant_info, xquant_warning
from .calibration_helper import XQuantDataset, CalibrationCollect
from .optimizer import GraphLegalized
from .analyse import statistical_analyse
from .ppq_decorator.onnxruntime_exporter import ONNXRUNTIMExporter
from .ppq_decorator.onnx_parser import OnnxParserDecorator
from .xquant_setting import XQuantSettingFactory, XQuantSetting
from .quantizer import XQuantizer


def dispatch_graph(graph: BaseGraph, dispatcher: Union[str, GraphDispatcher] = "conservative") -> BaseGraph:
    quantizer = XQuantizer(graph)
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
    assert isinstance(quantizer, BaseQuantizer)
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


def get_onnx_opset(onnx_model: onnx.ModelProto) -> Dict[str, int]:
    opset_dict = {}
    for opset in onnx_model.opset_import:
        _domain = opset.domain
        _domain = "ai.onnx" if _domain == "" else _domain
        opset_dict[_domain] = opset.version

    return opset_dict


def xquant_load_onnx_graph(file: str, sim_en: bool = True, truncate_var_name: Sequence[str] = []):
    onnx_model = onnx.load(file)
    opset_dict = get_onnx_opset(onnx_model)
    ai_onnx_version = opset_dict.get("ai.onnx", 13)
    if ai_onnx_version < 13:
        xquant_warning("convert ai.onnx version {} to 13...".format(ai_onnx_version))
        onnx_model = onnx.version_converter.convert_version(onnx_model, 13)
    if sim_en:
        xquant_info("simplify onnx model...")
        onnx_model, _ = onnxsim.simplify(onnx_model, mutable_initializer=True)

    osg_graph = osg.import_onnx(onnx_model)
    truncate_left_graph = None
    truncate_vars = []
    if len(truncate_var_name) > 0:
        tensors = osg_graph.tensors()
        for k, v in tensors.items():
            if k in set(truncate_var_name):
                truncate_vars.append(v)

        valid_node_names = set()
        invalid_node_names = set()
        valid_nodes = []
        invalid_nodes = []

        def _truncate_graph_upstream(out_vars: Sequence[osg.Tensor]):
            for o_var in out_vars:
                for source_op in o_var.inputs:
                    if source_op.name in valid_node_names:
                        continue
                    valid_nodes.append(source_op)
                    valid_node_names.add(source_op.name)
                    _truncate_graph_upstream(source_op.inputs)

        def _truncate_graph_downstream(out_vars: Sequence[osg.Tensor]):
            for o_var in out_vars:
                for dest_op in o_var.outputs:
                    if dest_op.name in invalid_node_names:
                        continue
                    invalid_nodes.append(dest_op)
                    invalid_node_names.add(dest_op.name)
                    _truncate_graph_downstream(dest_op.outputs)

        _truncate_graph_upstream(truncate_vars)
        _truncate_graph_downstream(truncate_vars)

        if len(valid_nodes) + len(invalid_nodes) != len(osg_graph.nodes):
            raise RuntimeError("truncate graph failed.")

        truncate_graph = osg.Graph(
            nodes=valid_nodes,
            inputs=osg_graph.inputs,
            outputs=truncate_vars,
            name=copy.copy(osg_graph.name),
            doc_string=copy.copy(osg_graph.doc_string),
            opset=copy.copy(osg_graph.opset),
            import_domains=osg_graph.import_domains,
        )

        truncate_left_graph = osg.Graph(
            nodes=invalid_nodes,
            inputs=[],
            outputs=osg_graph.outputs,
            name=copy.copy(osg_graph.name),
            doc_string=copy.copy(osg_graph.doc_string),
            opset=copy.copy(osg_graph.opset),
            import_domains=osg_graph.import_domains,
        )

        truncate_onnx_model = osg.export_onnx(truncate_graph)
        for var in truncate_vars:
            var.inputs.clear()
    else:
        truncate_onnx_model = onnx_model

    graph = OnnxParserDecorator().build(truncate_onnx_model)
    graph = ppq_format_graph(graph)
    return graph, truncate_left_graph, truncate_vars


def parse_xquant_config(file_or_dict: Union[str, dict]) -> XQuantSetting:
    config_dict = file_or_dict
    if isinstance(file_or_dict, str):
        with open(file_or_dict, "r") as fp:
            config_dict = json.load(fp)

    return XQuantSettingFactory.from_json(config_dict)


def quantize_onnx_model(path_or_config: Union[str, dict]):
    time_start = time.time()
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

    ppq_ir, truncate_left_graph, truncate_vars = xquant_load_onnx_graph(
        model_path,
        not config_setting.model_parameters.skip_onnxsim,
        config_setting.quantization_parameters.truncate_var_names,
    )

    GraphLegalized(ppq_ir)()

    ppq_ir = dispatch_graph(graph=ppq_ir, dispatcher="conservative")

    dummy_input = inputs_list

    quantizer = XQuantizer(ppq_ir)
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
        variable_reports = []
        for report_info in graphwise_analyse_results:
            if not report_info["Is parameter"]:
                variable_reports.append(report_info)
                variable_reports[-1]["F.Hist"] = ",".join([str(int(i)) for i in report_info["F.Hist"]])
        sort_variable_reports = sorted(variable_reports, key=lambda x: x["SNR"], reverse=True)

        def md_red_float(value):
            return '<font color="red">{:.4f}</font>'.format(value)

        for report_info in sort_variable_reports:
            snr_value = report_info["SNR"]
            if snr_value > 0.1:
                report_info["SNR"] = md_red_float(snr_value)
            else:
                report_info["SNR"] = "{:.4f}".format(snr_value)

            report_info["MSE"] = "{:.4f}".format(report_info["MSE"])

            cos_value = report_info["Cosine"]
            if cos_value < 0.99:
                report_info["Cosine"] = md_red_float(cos_value)
            else:
                report_info["Cosine"] = "{:.4f}".format(cos_value)

        sort_variable_reports_df = DataFrame(sort_variable_reports)
        report_path = os.path.join(working_dir, "{}_report.md".format(output_prefix))
        xquant_info("export quantization statistical results file to {}".format(report_path))
        sort_variable_reports_df.to_markdown(report_path)

    quant_onnx_model = ONNXRUNTIMExporter().export(
        quantizer._graph,
    )

    if isinstance(truncate_left_graph, osg.Graph):
        osg_graph = osg.import_onnx(quant_onnx_model)
        for idx, o_var in enumerate(osg_graph.outputs):
            o_idx = o_var.inputs[0].outputs.index(o_var)
            o_var.inputs[0].outputs[o_idx] = truncate_vars[idx]

        new_osg_graph = osg.Graph(
            nodes=osg_graph.nodes + truncate_left_graph.nodes,
            inputs=osg_graph.inputs,
            outputs=truncate_left_graph.outputs,
            name=copy.copy(osg_graph.name),
            doc_string=copy.copy(osg_graph.doc_string),
            opset=copy.copy(osg_graph.opset),
            import_domains=osg_graph.import_domains,
        )
        quant_onnx_model = osg.export_onnx(new_osg_graph)

    onnx.save(quant_onnx_model, os.path.join(working_dir, "{}.onnx".format(output_prefix)))

    xquant_info("quantization eplased time {:.2f} s".format(time.time() - time_start))
    return quantizer._graph
