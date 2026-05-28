#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import functools
from collections import deque
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, Union

import torch
from tqdm import tqdm

from ..defs import BIAS_CORRECTION_INTERST_TYPE, COMPUTING_OP, OBSERVER_MAX_BIAS_VAL, PASSIVE_OPERATIONS, XQUANT_CONFIG
from ..ppq_decorator import (
    BaseGraph,
    Operation,
    OperationObserver,
    QuantableOperation,
    QuantizationOptimizationPass,
    QuantizationStates,
    TorchExecutor,
    Variable,
    ppq_common,
    torch_cosine_similarity,
    torch_mean_square_error,
    torch_snr_error,
)
from .observer import TorchXSlimObserver
from .refine import PassiveParameterBakingPass
from .training import LearnedStepSizePassDecorator, XSlimBlockBuilder, XSlimTrainableBlock


class BlockWiseCalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, dataloader_cache: Sequence[dict]):
        # torch.save(dataloader_cache, "calib_dataloader_cache")
        self.__data_cache = dataloader_cache
        # self.__data_cache = torch.load("calib_dataloader_cache", mmap=True, map_location=map_location)

    def __len__(self):
        return len(self.__data_cache)

    def __getitem__(self, idx):
        return self.__data_cache[idx]

    def update_data(self, idx, name, data):
        if data.device.type == "cuda":
            mem_free, mem_all = torch.cuda.mem_get_info()
            mem_free_ratio = mem_free / mem_all
            if mem_free_ratio < 0.2:
                data = data.to("cpu")
        self.__data_cache[idx].update({name: data})


class RuntimeBlockWiseCalibrationPass(QuantizationOptimizationPass):
    def __init__(
        self,
        method: str = None,
        override: bool = False,
        calib_steps: int = 32,
        block_wise: bool = True,
        fintune_epoch: int = 2,
        auto_finetune_level: int = 1,
    ) -> None:
        self.name = "XSlim Runtime Calibration Pass(BlockWise)"
        self._method = method
        self._collate_fn = None
        self._calib_steps = calib_steps
        self._override = override
        self._block_wise = block_wise
        self._fintune_epoch = fintune_epoch
        self._block_wise_loss = []
        self._auto_finetune_level = auto_finetune_level

    def split_graph_into_blocks(
        self,
        graph: BaseGraph,
        executing_order: List[Operation],
    ) -> List[XSlimTrainableBlock]:
        block_builder = XSlimBlockBuilder(graph=graph, topo_order=executing_order)
        return block_builder.build()

    def forward_trainable_block(
        self,
        block: XSlimTrainableBlock,
        dataset: BlockWiseCalibrationDataset,
        executor: TorchExecutor,
        executor_hook: dict,
        calib_steps: int,
        extern_output_var_hooks: Dict[str, Callable] = None,
        collect_dataloader_cache: bool = False,
    ):
        operation_cache = block.rps
        if extern_output_var_hooks is None:
            extern_output_var_hooks = {}

        block_input_names = list(block.in_var_names)

        single_data = dataset[0]
        for var_name in block_input_names:
            if var_name not in single_data:
                raise RuntimeError("missing input variable {} for block".format(var_name))

        for idx, batch_data in tqdm(enumerate(dataset), desc="Runtime Calibration Single Block Collect", disable=True):
            inputs_feed = {
                var_name: batch_data[var_name].to(executor._executing_context.executing_device)
                for var_name in block_input_names
            }
            output_names = list(block.out_var_names | set(extern_output_var_hooks.keys()))
            outputs = executor._TorchExecutor__forward(
                inputs_feed,
                operation_cache,
                output_names=output_names,
                hooks=executor_hook,
            )
            for o_var, o_name in zip(outputs, output_names):
                if o_name in block.out_var_names and collect_dataloader_cache:
                    dataset.update_data(idx, o_name, o_var)
                    batch_data[o_name] = o_var
                if o_name in extern_output_var_hooks:
                    extern_output_var_hooks[o_name](o_name, o_var, batch_data)

    def clean_dataloader_cache(
        self,
        block: XSlimTrainableBlock,
        dataset: BlockWiseCalibrationDataset,
        var_to_operations: dict,
    ):
        remove_op_set = set(block.rps)
        remove_ovarnames = []
        for o_name, o_var in dataset[0].items():
            if o_name in var_to_operations:
                var_to_operations[o_name] -= remove_op_set
                if len(var_to_operations[o_name]) == 0:
                    remove_ovarnames.append(o_name)
        for data in dataset:
            for o_name in remove_ovarnames:
                data.pop(o_name)

    def calib_single_block(
        self,
        block: XSlimTrainableBlock,
        dataset: BlockWiseCalibrationDataset,
        var_to_operations: dict,
        executor: TorchExecutor,
        calib_steps: int,
    ):
        bias_op_var_names = dict()
        bias_fp_cache = dict()
        operation_observer_cache = []
        extern_output_var_hooks = {}

        def __collect_bias(o_name: str, o_var: torch.Tensor, batch_data: Dict[str, torch.Tensor]):
            reduce_bias = self.collect_bias(o_var, bias_op_var_names[o_name])
            if o_name not in bias_fp_cache:
                bias_fp_cache[o_name] = []
            bias_fp_cache[o_name].append(reduce_bias)

        for operation in block.rps:
            if isinstance(operation, QuantableOperation):
                operation_observer = OperationObserver(operation=operation, monitor_parameter=False)
                operation_observer_cache.append(operation_observer)
                if operation.type in BIAS_CORRECTION_INTERST_TYPE and operation.num_of_parameter == 2:
                    bias_op_var_names[operation.outputs[0].name] = operation
                    extern_output_var_hooks[operation.outputs[0].name] = __collect_bias

        ob_table_num = sum([len(ob.hook._observer_table) for ob in operation_observer_cache])
        hooks = {ob._operation.name: ob.hook for ob in operation_observer_cache}
        if ob_table_num == 0:
            hooks = {}

        not_has_hist_ob = True
        for ob in operation_observer_cache:
            if not_has_hist_ob:
                not_has_hist_ob = all(
                    [
                        not isinstance(var_observer, TorchXSlimObserver)
                        for var_observer in ob._hook._observer_table.values()
                    ]
                )

        with torch.no_grad():
            self.forward_trainable_block(block, dataset, executor, hooks, calib_steps, extern_output_var_hooks, True)

        if len(hooks) == 0:
            return

        for ob in operation_observer_cache:
            ob.render_quantization_config()
            ob.report()

        if not_has_hist_ob:
            pass
        else:
            with torch.no_grad():
                self.forward_trainable_block(block, dataset, executor, hooks, calib_steps)
            for ob in operation_observer_cache:
                ob.render_quantization_config()
                ob.report()

        if self._auto_finetune_level >= 1:
            self.block_bias_correct(
                block,
                dataset,
                executor,
                calib_steps,
                bias_op_var_names,
                bias_fp_cache,
            )

        loss_dict = self.compute_block_loss(block, dataset, executor, calib_steps)
        self._block_wise_loss.append(
            self.create_block_loss_info(
                block, loss_dict["mse"], loss_dict["snr"], loss_dict["mean"], loss_dict["cosine"]
            )
        )
        self._block_wise_loss[-1]["block_idx"] = len(self._block_wise_loss)
        self.clean_dataloader_cache(block, dataset, var_to_operations)

    def format_tqc(self, operaion: Operation):
        if not isinstance(operaion, QuantableOperation):
            return

        for tqc, var in operaion.config_with_variable:
            if tqc.dominated_by is tqc and tqc.scale is not None:
                if torch.any(tqc.scale < 0):
                    raise RuntimeError("tqc.scale for {} < 0, you can change fintune_level <= 1.".format(operaion.name))

        PassiveParameterBakingPass.passive_parameters_quant(operaion)
        PassiveParameterBakingPass.passive_bias_quant(operaion)

    def compute_block_loss(
        self,
        block: XSlimTrainableBlock,
        dataset: BlockWiseCalibrationDataset,
        executor: TorchExecutor,
        calib_steps: int,
    ) -> dict:
        block_output_names = list(block.out_var_names)
        mse_losses = {o_name: 0 for o_name in block_output_names}
        snr_losses = {o_name: 0 for o_name in block_output_names}
        cosine = {o_name: 0 for o_name in block_output_names}
        means = {o_name: 0 for o_name in block_output_names}

        def __collect_loss(o_name: str, o_var: torch.Tensor, batch_data: Dict[str, torch.Tensor]):
            exe_device = executor._executing_context.executing_device
            ref_var = batch_data[o_name]
            ref_var = ref_var.to(exe_device)
            o_var = o_var.to(exe_device)
            if ref_var.dtype in {torch.float32, torch.float64, torch.float16}:
                batch_mse_loss = torch_mean_square_error(o_var, ref_var)
                batch_snr_loss = torch_snr_error(o_var, ref_var)
                batch_cosine = torch_cosine_similarity(o_var, ref_var)
                means[o_name] += ref_var.abs().mean().detach().item()
            else:
                batch_mse_loss = torch.tensor([0], dtype=torch.float32, device=exe_device)
                batch_snr_loss = torch.tensor([0], dtype=torch.float32, device=exe_device)
                batch_cosine = torch.tensor([1], dtype=torch.float32, device=exe_device)
                means[o_name] += torch.tensor([0], dtype=torch.float32, device=exe_device)
            mse_losses[o_name] += batch_mse_loss.detach().item()
            snr_losses[o_name] += batch_snr_loss.detach().item()
            cosine[o_name] += batch_cosine.detach().item()

        extern_output_var_hooks = {o_name: __collect_loss for o_name in block_output_names}

        with torch.no_grad():
            self.forward_trainable_block(block, dataset, executor, {}, calib_steps, extern_output_var_hooks)

        for o_name in mse_losses:
            mse_losses[o_name] /= calib_steps
            snr_losses[o_name] /= calib_steps
            means[o_name] /= calib_steps
            cosine[o_name] /= calib_steps

        return {"mse": mse_losses, "snr": snr_losses, "cosine": cosine, "mean": means}

    def collect_bias(self, output: torch.Tensor, op: Operation) -> torch.Tensor:
        if output.ndim < 1:
            raise ValueError("Forward value has an unexpected dimension.")
        op_type = op.type
        if op_type in {"Conv", "ConvTranspose", "InstanceNormalization", "GroupNormalization"}:
            # for convolution layer, bias always been added on axis 1
            reduce_dims = [i for i in range(output.ndim) if i != 1]
            return torch.mean(output, dim=reduce_dims).unsqueeze(0)
        elif op_type in {"Gemm"}:
            reduce_dims = [i for i in range(output.ndim) if i != (output.ndim - 1)]
            return torch.mean(output, dim=(0,)).unsqueeze(0)
        elif op_type in {"LayerNormalization"}:
            axis = op.attributes.get("axis", -1)
            if axis < 0:
                axis = output.ndim + axis
            reduce_dims = [i for i in range(axis)]
            return torch.mean(output, dim=reduce_dims).unsqueeze(0)
        else:
            raise TypeError(f"Unsupported Operation type: {op_type}")

    def block_bias_correct(
        self,
        block: XSlimTrainableBlock,
        dataset: BlockWiseCalibrationDataset,
        executor: TorchExecutor,
        calib_steps: int,
        bias_op_var_names: dict,
        bias_fp_cache: dict,
    ) -> None:
        if len(bias_op_var_names) == 0:
            return
        operation_cache = [operation for operation in block.rps]
        bias_quant_cache = {}
        output_names = list(bias_op_var_names.keys())
        for batch_data in tqdm(dataset, desc="Runtime Calibration Single Block Finetune", disable=True):
            inputs_feed = {
                var_name: batch_data[var_name].to(executor._executing_context.executing_device)
                for var_name in block.in_var_names
            }
            outputs = executor.partial_graph_forward(
                operations=operation_cache, feed_dict=inputs_feed, output_names=list(bias_op_var_names.keys())
            )
            for o_var, o_name in zip(outputs, output_names):
                reduce_bias = self.collect_bias(o_var, bias_op_var_names[o_name])
                if o_name not in bias_quant_cache:
                    bias_quant_cache[o_name] = [reduce_bias]
                else:
                    bias_quant_cache[o_name].append(reduce_bias)

        def get_bias_valid_mean(value: torch.Tensor):
            target_device = value.device
            percentile = 0.999
            value = value.cpu()
            batch, channel = value.size()
            value = value.permute(1, 0)
            new_bias = []

            min_idx, max_idx = int(batch * (1 - percentile)), int(batch * (percentile))
            min_idx = max(0, min_idx) + 1
            max_idx = min(max_idx, batch - 1) + 1

            for i in range(channel):
                temp_val = value[i].flatten()
                _min = torch.kthvalue(temp_val, k=min_idx, dim=0)[0].item()
                _max = torch.kthvalue(temp_val, k=max_idx, dim=0)[0].item()
                temp_val = temp_val[torch.where(temp_val >= _min)]
                temp_val = temp_val[torch.where(temp_val <= _max)]
                new_bias.append(temp_val.mean().item())
            return torch.tensor(new_bias, dtype=value.dtype, device=target_device)

        for o_name in bias_op_var_names:
            DC_term_fp = bias_fp_cache[o_name]
            DC_term_qt = bias_quant_cache[o_name]

            if len(DC_term_fp) == 0 or len(DC_term_qt) == 0:
                continue
            DC_term_fp = get_bias_valid_mean(torch.cat(DC_term_fp, axis=0))
            DC_term_qt = get_bias_valid_mean(torch.cat(DC_term_qt, axis=0))
            bias_error = DC_term_fp - DC_term_qt
            bias_op_var_names[o_name].inputs[-1].value += bias_error

    def create_block_loss_info(
        self, block: XSlimTrainableBlock, mse_losses: dict, snr_losses: dict, means: dict, cosines: dict
    ):
        return {
            "start_op": "[{}]{}".format(block.sp.type, block.sp.name),
            "end_op": "[{}]{}".format(block.ep.type, block.ep.name),
            "snr": max([v for k, v in snr_losses.items()]),
            "mse": max([v for k, v in mse_losses.items()]),
            "cosine": min([v for k, v in cosines.items()]),
            "block": block,
        }

    def finetune(
        self,
        graph: BaseGraph,
        blocks: Sequence[XSlimTrainableBlock],
        dataloader: Iterable,
        executor: TorchExecutor,
        calib_steps: int = 32,
        collate_fn: Callable = None,
    ):
        lsq_optimizer = LearnedStepSizePassDecorator(is_scale_trainable=True, lr=0.0001)
        for block in tqdm(blocks, desc="Runtime Calibration Single Block Finetune"):
            qt_inputs, fp_outputs = lsq_optimizer.collect(
                graph=graph,
                block=block,
                executor=executor,
                dataloader=dataloader,
                collate_fn=collate_fn,
                collecting_device=executor._executing_context.executing_device,
            )

            pre_loss, post_loss = lsq_optimizer.finetune(
                steps=calib_steps * self._fintune_epoch,
                learning_rate=lsq_optimizer.lr,
                block=block,
                qt_inputs=qt_inputs,
                fp_outputs=fp_outputs,
                executor=executor,
            )

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: TorchExecutor,
        calib_steps: int = 32,
        collate_fn: Callable = None,
        **kwargs,
    ) -> None:
        self._collate_fn = collate_fn
        self._calib_steps = calib_steps
        self._block_wise_loss.clear()
        assert calib_steps >= 10, (
            "Insufficient Calibration Detected, to better quantize your network, "
            "more calibration steps is demonded, we strongly recommend you to prepare more calibration data "
            "and more calibration steps is perferred here. (at least 10)"
        )
        assert calib_steps <= 1000, (
            "Calibration steps is too large, xslim is capable for quantizing your network within 10-1000 "
            "calibration steps. More calibraiton steps will greatly delay ppq's calibration procedure. "
            "Reset your calib_steps parameter please."
        )
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
        dataloader_cache = []
        for k, v in graph.inputs.items():
            single_graph_input_name = k

        calib_step = 0
        for data in dataloader:
            data = self._collate_fn(data)
            if isinstance(data, torch.Tensor) and len(graph.inputs) == 1:
                if self._collate_fn is not None:
                    dataloader_cache.append({single_graph_input_name: data})
            elif isinstance(data, dict):
                dataloader_cache.append(data)
            else:
                raise TypeError(type(data))

            calib_step += 1
            if calib_step >= self._calib_steps:
                break

        var_to_operations = dict()
        for var_name, var in graph.variables.items():
            if not var.is_parameter:
                var_to_operations[var.name] = set([_ for _ in var.dest_ops])

        topo_sort_ops = graph.topological_sort()

        blockwise_dataset = BlockWiseCalibrationDataset(dataloader_cache)

        if self._block_wise:
            op_blocks = self.split_graph_into_blocks(graph, topo_sort_ops)
            for block in tqdm(op_blocks, desc="Runtime Calibration(BlockWise)"):
                self.calib_single_block(block, blockwise_dataset, var_to_operations, executor, calib_step)

            self._block_wise_loss = sorted(self._block_wise_loss, key=lambda x: x["mse"], reverse=True)
            if self._auto_finetune_level >= 2:
                _auto_finetune_blocks = 10 if self._auto_finetune_level >= 3 else 5
                finetune_blocks_info = [loss_info for loss_info in self._block_wise_loss[:_auto_finetune_blocks]]
                finetune_blocks_info = sorted(finetune_blocks_info, key=lambda x: x["block_idx"])
                self.finetune(
                    graph,
                    [loss_info["block"] for loss_info in finetune_blocks_info],
                    dataloader,
                    executor,
                    calib_step,
                    collate_fn,
                )
        else:
            raise NotImplementedError("block wise only.")

        for op in topo_sort_ops:
            self.format_tqc(op)
