from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
from tqdm import tqdm
from ppq.core import (
    OBSERVER_MSE_HIST_BINS,
    PASSIVE_OPERATIONS,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
    empty_ppq_cache,
    ppq_warning,
)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.quantization.optim import (
    QuantizationOptimizationPipeline,
    QuantizationOptimizationPass,
    RuntimeCalibrationPass,
)
from ppq.IR.search import SearchableGraph
from ppq.executor import BaseGraphExecutor
from ppq.quantization.qfunction import PPQuantFunction
from ppq.quantization.observer import (
    CalibrationHook,
    OperationObserver,
    TensorObserverFactroy,
    TorchHistObserver,
    TorchMinMaxObserver,
    TorchMSEObserver,
    TorchPercentileObserver,
    OBSERVER_TABLE,
    range as ppq_range,
)


class RuntimePerlayerCalibrationPass(RuntimeCalibrationPass):
    def __init__(
        self,
        method: str = None,
        override: bool = False,
        calib_steps: int = 32,
        calib_block_size: int = 4,
    ) -> None:
        super().__init__(method, override, calib_steps)
        self.name = "XQuant Runtime Calibration Pass(Per Layer)"
        self._calib_block_size = calib_block_size

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        calib_steps: int = 32,
        collate_fn: Callable = None,
        **kwargs,
    ) -> None:
        self._collate_fn = collate_fn
        self._calib_steps = calib_steps
        assert calib_steps >= 8, (
            "Insufficient Calibration Detected, to better quantize your network, "
            "more calibration steps is demonded, we strongly recommend you to prepare more calibration data "
            "and more calibration steps is perferred here. (at least 8)"
        )

        assert calib_steps <= 512, (
            "Calibration steps is too large, ppq is capable for quantizing your network within 32-128 "
            "calibration steps. More calibraiton steps will greatly delay ppq's calibration procedure. "
            "Reset your calib_steps parameter please."
        )

        # -------------------------------------------------
        # Override existing quantization configurations
        # -------------------------------------------------
        if self._override:
            for operation in graph.operations.values():
                if not isinstance(operation, QuantableOperation):
                    continue

                for config, var in operation.config_with_variable:
                    if (
                        not var.is_parameter
                        and config.state == QuantizationStates.ACTIVATED
                        and config.dominated_by == config
                    ):
                        config.state = QuantizationStates.INITIAL

        single_graph_input_name = None
        dataloader_cache = {}
        for k, v in graph.inputs.items():
            dataloader_cache[k] = []
            single_graph_input_name = k

        calib_step = 0
        for data in dataloader:
            data = self._collate_fn(data)
            if isinstance(data, torch.Tensor) and len(graph.inputs) == 1:
                if self._collate_fn is not None:
                    dataloader_cache[single_graph_input_name].append(data)
            elif isinstance(data, dict):
                for k, v in data.items():
                    dataloader_cache[k].append(v)
            else:
                raise TypeError(type(data))

            calib_step += 1
            if calib_step >= self._calib_steps:
                break

        var_to_operations = dict()
        for var_name, var in graph.variables.items():
            if not var.is_parameter:
                var_to_operations[var.name] = set([_ for _ in var.dest_ops])

        operation_visited = set()
        operation_cache = list()
        operation_observer_cache = list()
        topo_sort_ops = graph.topological_sort()

        def _calib_part_operations():
            var_input_names = set()
            var_output_names = set()
            complex_vars = set()

            for op in operation_cache:
                for var in op.outputs:
                    if not var.is_parameter:
                        var_output_names.add(var.name)
                        if len(var.dest_ops) > 1:
                            complex_vars.add(var.name)

            for op in operation_cache:
                for var in op.inputs:
                    if not var.is_parameter:
                        var_input_names.add(var.name)

            block_input_names = var_input_names - var_output_names
            block_output_names = var_output_names - var_input_names
            block_output_names.update(complex_vars)

            hooks = {ob._operation.name: ob.hook for ob in operation_observer_cache}
            output_names = [var_name for var_name in block_output_names]

            for idx in range(calib_step):
                inputs_feed = {var_name: dataloader_cache[var_name][idx] for var_name in block_input_names}
                outputs = executor._TorchExecutor__forward(
                    inputs_feed, operation_cache, output_names=output_names, hooks=hooks
                )
                for o_var, o_name in zip(outputs, output_names):
                    if o_name not in dataloader_cache:
                        dataloader_cache[o_name] = []
                    dataloader_cache[o_name].append(o_var)

            not_has_hist_ob = True
            for ob in operation_observer_cache:
                ob.render_quantization_config()
                ob.report()

                if not_has_hist_ob:
                    not_has_hist_ob = all(
                        [
                            not isinstance(var_observer, (TorchHistObserver, TorchMSEObserver))
                            for var_observer in ob._hook._observer_table.values()
                        ]
                    )
            if not_has_hist_ob:
                pass
            else:
                for idx in range(calib_step):
                    inputs_feed = {var_name: dataloader_cache[var_name][idx] for var_name in block_input_names}
                    outputs = executor._TorchExecutor__forward(
                        inputs_feed,
                        operation_cache,
                        output_names=output_names,
                        hooks=hooks,
                    )
                    # for o_var, o_name in zip(outputs, output_names):
                    #    dataloader_cache[o_name][idx] = o_var

                for ob in operation_observer_cache:
                    ob.render_quantization_config()
                    ob.report()

            for idx in range(calib_step):
                inputs_feed = {var_name: dataloader_cache[var_name][idx] for var_name in block_input_names}
                outputs = executor._TorchExecutor__forward(
                    inputs_feed,
                    operation_cache,
                    output_names=output_names,
                    hooks=hooks,
                )
                for o_var, o_name in zip(outputs, output_names):
                    dataloader_cache[o_name][idx] = o_var

            remove_opset = set(operation_cache)
            remove_ovarnames = []
            for o_name, _ in dataloader_cache.items():
                if o_name in var_to_operations:
                    var_to_operations[o_name] -= remove_opset
                    if len(var_to_operations[o_name]) == 0:
                        remove_ovarnames.append(o_name)
            for o_name in remove_ovarnames:
                dataloader_cache.pop(o_name)
            operation_visited.update(operation_cache)
            operation_cache.clear()
            operation_observer_cache.clear()

        for operation in tqdm(topo_sort_ops, desc="Runtime Calibration(Per Layer)"):
            operation_cache.append(operation)
            if not isinstance(operation, QuantableOperation):
                continue

            # override algorithm setting if necessary
            for config, var in operation.config_with_variable:
                if not var.is_parameter and self._method is not None:
                    config.observer_algorithm = self._method

            operation_observer = OperationObserver(operation=operation, monitor_parameter=False)
            operation_observer_cache.append(operation_observer)

            ob_table_num = sum([len(ob.hook._observer_table) for ob in operation_observer_cache])
            if ob_table_num == 0:
                continue

            if len(operation_observer_cache) < self._calib_block_size:
                continue

            _calib_part_operations()

        if len(operation_observer_cache) > 0:
            _calib_part_operations()
