#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from collections import OrderedDict
from typing import Callable, Dict, Iterator, List, Optional

import torch
from pandas import DataFrame
from tqdm import tqdm
from xslim.logger import logger

from ..defs import PASSIVE_OPERATIONS
from ..ppq_decorator.ppq.executor import RuntimeHook, TorchExecutor
from ..ppq_decorator.ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ..ppq_decorator.ppq.quantization.measure import torch_cosine_similarity, torch_mean_square_error, torch_snr_error
from ..ppq_decorator.ppq.utils.fetch import batch_random_fetch, generate_torch_indexer, tensor_random_fetch


class DetailedRecorder(RuntimeHook):
    def __init__(self, operation: Operation, fetchs: int = 1024) -> None:
        self.fetchs = fetchs
        self.i_storage = [[] for _ in range(operation.num_of_input)]
        self.o_storage = [[] for _ in range(operation.num_of_output)]
        self.i_indexer = [None for _ in range(operation.num_of_input)]
        self.o_indexer = [None for _ in range(operation.num_of_output)]
        super().__init__(operation)

    def pre_forward_hook(self, inputs: List[torch.Tensor], **kwargs) -> list:
        for idx, input in enumerate(inputs):
            if isinstance(input, torch.Tensor) and input.numel() > 0:
                if self.i_indexer[idx] is None:
                    self.i_indexer[idx] = generate_torch_indexer(self.fetchs, input.numel())
                self.i_storage[idx].append(input.flatten()[self.i_indexer[idx]])
            else:
                self.i_storage[idx].append(torch.ones([1]))
        return super().pre_forward_hook(inputs, **kwargs)

    def post_forward_hook(self, outputs: List[torch.Tensor], **kwargs) -> list:
        for idx, output in enumerate(outputs):
            if isinstance(output, torch.Tensor) and output.numel() > 0:
                if self.o_indexer[idx] is None:
                    self.o_indexer[idx] = generate_torch_indexer(self.fetchs, output.numel())
                self.o_storage[idx].append(output.flatten()[self.o_indexer[idx]])
            else:
                self.o_storage[idx].append(torch.ones([1]))
        return super().post_forward_hook(outputs, **kwargs)

    def clear(self):
        self.i_storage = [[] for _ in range(self._hook_to.num_of_input)]
        self.o_storage = [[] for _ in range(self._hook_to.num_of_output)]


def statistical_analyse(
    graph: BaseGraph,
    running_device: str,
    dataloader: Iterator,
    collate_fn: Callable = None,
    steps: int = 8,
    report_path: Optional[str] = None,
) -> List[dict]:
    class StatisticalErrorAnalyser:
        def __init__(self, x_fp: List[torch.Tensor], x_qt: List[torch.Tensor], op: Operation, var: Variable) -> None:
            self.x_qt = torch.cat(x_qt, dim=0)
            self.x_fp = torch.cat(x_fp, dim=0)
            self.x_er = self.x_qt - self.x_fp
            self.op = op
            self.var = var

            self.num_of_samples = self.x_fp.shape[0]

        def stat(self) -> dict:
            x_er, x_fp, x_qt = self.x_er, self.x_fp, self.x_qt
            er_mean = x_er.mean().item()
            er_std = x_er.std().item()

            qt_mean = x_qt.mean().item()
            qt_std = x_qt.std().item()
            qt_min = x_qt.min().item()
            qt_max = x_qt.max().item()

            fp_mean = x_fp.mean().item()
            fp_std = x_fp.std().item()
            fp_min = x_fp.min().item()
            fp_max = x_fp.max().item()
            fp_hist = torch.histc(x_fp, bins=32, min=x_fp.min(), max=x_fp.max()).cpu().tolist()

            snr = torch_snr_error(x_qt, x_fp).item()
            cosine = torch_cosine_similarity(x_qt, x_fp).item()
            mse = torch_mean_square_error(x_qt, x_fp).item()
            return {
                "SNR": snr,
                "MSE": mse,
                "Cosine": cosine,
                "Q.MinMax": "{:.3f}, {:.3f}".format(qt_min, qt_max),
                "F.MinMax": "{:.3f}, {:.3f}".format(fp_min, fp_max),
                "F.Hist": ",".join([str(int(i)) for i in fp_hist]),
                "is_parameter": self.var.is_parameter,
            }

    executor = TorchExecutor(graph=graph, device=running_device)
    # find all quantable operations.
    interested_op = []
    operation_list = graph.topological_sort()
    for operation in operation_list:
        if isinstance(operation, QuantableOperation) and operation.type not in PASSIVE_OPERATIONS:
            interested_op.append(operation)
    if len(interested_op) == 0:
        logger.warning("No analyzable operators were found.")

    # set up all hooks.
    hooks, caches = OrderedDict(), OrderedDict()
    for operation in interested_op:
        if isinstance(operation, QuantableOperation):
            hooks[operation.name] = DetailedRecorder(operation=operation, fetchs=2048)
            caches[operation.name] = {
                "Quantized Input": [],
                "Quantized Output": [],
                "Dequantized Input": [],
                "Dequantized Output": [],
            }

    # dequantize all
    for operation in operation_list:
        if isinstance(operation, QuantableOperation):
            operation.dequantize()

    # run for each quantable operations:
    analyse_data_list = []
    for idx, batch in enumerate(dataloader):
        if collate_fn is not None:
            batch = collate_fn(batch)
        analyse_data_list.append(batch)
        if idx >= steps:
            break

    for batch in tqdm(analyse_data_list, desc="Analysing Dequantized"):
        executor.forward(inputs=batch, hooks=hooks)

    for operation in interested_op:
        hook = hooks[operation.name]
        assert isinstance(hook, DetailedRecorder)
        caches[operation.name]["Dequantized Input"] = hook.i_storage.copy()
        caches[operation.name]["Dequantized Output"] = hook.o_storage.copy()
        hook.clear()

    # restore all
    for operation in operation_list:
        if isinstance(operation, QuantableOperation):
            operation.restore_quantize_state()

    # run for each quantable operations:
    for batch in tqdm(analyse_data_list, desc="Analysing Quantized"):
        executor.forward(inputs=batch, hooks=hooks)

    for operation in interested_op:
        hook = hooks[operation.name]
        assert isinstance(hook, DetailedRecorder)
        caches[operation.name]["Quantized Input"] = hook.i_storage.copy()
        caches[operation.name]["Quantized Output"] = hook.o_storage.copy()
        hook.clear()

    # analysing cache
    records = []
    visited_var = set()
    for name, record in caches.items():
        op_records = {}
        operation = graph.operations[name]
        if not isinstance(operation, Operation):
            continue
        op_records["Op"] = "{}[{}]".format(operation.name, operation.type)
        op_records["Vars"] = {}
        for idx, input_var in enumerate(operation.inputs):
            if input_var in visited_var:
                continue
            visited_var.add(input_var)
            x_qt = record["Quantized Input"][idx]
            x_fp = record["Dequantized Input"][idx]
            if x_fp[0].dtype not in {torch.float32, torch.float64, torch.float16}:
                continue
            op_records["Vars"][input_var.name] = StatisticalErrorAnalyser(
                x_fp=x_fp, x_qt=x_qt, op=operation, var=input_var
            ).stat()

        max_snr = 0
        for idx, output_var in enumerate(operation.outputs):
            if output_var in visited_var:
                continue
            visited_var.add(output_var)
            x_qt = record["Quantized Output"][idx]
            x_fp = record["Dequantized Output"][idx]
            if x_fp[0].dtype not in {torch.float32, torch.float64, torch.float16}:
                continue
            _detail = StatisticalErrorAnalyser(x_fp=x_fp, x_qt=x_qt, op=operation, var=output_var).stat()
            if _detail["SNR"] > max_snr:
                max_snr = _detail["SNR"]
            op_records["Vars"][output_var.name] = _detail

        op_records["MAX_SNR"] = max_snr
        records.append(op_records)

    def md_red_float(value):
        return '<font color="red">{:.4f}</font>'.format(value)

    sort_variable_reports = sorted(records, key=lambda x: x["MAX_SNR"], reverse=True)

    report_info_list = []
    for report_info in sort_variable_reports:
        for var_name, var_info in report_info["Vars"].items():
            report_info_list.append({"Op": report_info["Op"], "Var": var_name, **var_info})
            if "is_parameter" in report_info_list[-1]:
                is_parameter = report_info_list[-1].pop("is_parameter")
                if is_parameter:
                    report_info_list[-1]["Var"] = "{}[Constant]".format(report_info_list[-1]["Var"])
            snr_value = report_info_list[-1]["SNR"]
            cos_value = report_info_list[-1]["Cosine"]
            if snr_value > 0.1:
                report_info_list[-1]["SNR"] = md_red_float(snr_value)
            else:
                report_info_list[-1]["SNR"] = "{:.4f}".format(snr_value)

            if cos_value < 0.99:
                report_info_list[-1]["Cosine"] = md_red_float(cos_value)
            else:
                report_info_list[-1]["Cosine"] = "{:.4f}".format(cos_value)

            report_info_list[-1]["MSE"] = "{:.4f}".format(report_info_list[-1]["MSE"])

    report_index = ["Op", "Var", "SNR", "MSE", "Cosine", "Q.MinMax", "F.MinMax", "F.Hist"]
    report_info_df = DataFrame(report_info_list, columns=report_index)

    if isinstance(report_path, str):
        logger.info("export quantization statistical results file to {}".format(report_path))
        report_info_df.to_markdown(report_path)
    return report_info_list
