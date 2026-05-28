#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import functools
import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from xslim.logger import logger

from ..defs import XQUANT_CONFIG
from ..ppq_decorator import (
    BaseGraph,
    BaseGraphExecutor,
    LearnedStepSizePass,
    LSQDelegator,
    Operation,
    PPQLinearQuantFunction,
    QuantableGraph,
    QuantableOperation,
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
    TorchExecutor,
    TrainableBlock,
    Variable,
    torch_mean_square_error,
    torch_snr_error,
)


class XSlimTrainableBlock(TrainableBlock):
    OP_DEPTH_DICT = {
        "Conv": 4,
        "ConvTranspose": 4,
        "Gemm": 4,
        "MatMul": 4,
        "LayerNormalization": 2,
        "InstanceNormalization": 2,
        "GroupNormalization": 2,
        "MaxPool": 2,
        "AveragePool": 2,
        "GlobalMaxPool": 2,
        "GlobalAveragePool": 2,
        "ReduceMean": 2,
        "Reshape": 0.5,
        "Transpose": 0.5,
        "Slice": 0.5,
        "Squeeze": 0.5,
        "Unsqueeze": 0.5,
        "Gather": 0.5,
        "Flatten": 0.5,
        "Split": 0.5,
    }

    def __init__(self, sp: Operation, ep: Operation, rps: List[Operation], graph: BaseGraph):
        super().__init__(sp, ep, rps)
        self.graph: BaseGraph = graph
        self.in_var_names = set()
        self.out_var_names = set()
        self.depth = 0
        self.update_block_io_var_names()
        self.get_depth()

    @staticmethod
    def get_sequence_block_depth(rps: List[Operation]):
        depth = 0
        for op in rps:
            depth += XSlimTrainableBlock.OP_DEPTH_DICT.get(op.type, 1)
        return depth

    def get_depth(self):
        block_ops = set(self.rps)
        toposort_ops = []
        visited_vars = set()

        def visit_ops(in_vars: List[Variable], depth):
            next_vars = []
            max_depth = 0
            for var in in_vars:
                for dest_op in var.dest_ops:
                    if (
                        dest_op in block_ops
                        and set([var for var in dest_op.inputs if not var.is_parameter and var.source_op is not None])
                        <= visited_vars
                    ):
                        toposort_ops.append(dest_op)
                        block_ops.remove(dest_op)
                        next_vars.extend(dest_op.outputs)
                        visited_vars.update([var for var in dest_op.outputs])
                        max_depth = max(XSlimTrainableBlock.OP_DEPTH_DICT.get(dest_op.type, 1), max_depth)
            depth = max_depth + depth
            if len(next_vars) > 0:
                return visit_ops(next_vars, depth)
            return depth

        in_vars = [self.graph.variables[name] for name in self.in_var_names]
        visited_vars.update(in_vars)
        depth = visit_ops(in_vars, 0)
        if len(block_ops) > 0:
            raise RuntimeError(
                "get block depth and toposort error, {} no visited".format([op.name for op in block_ops])
            )
        self.depth = depth
        self.rps = toposort_ops
        return depth

    def update_block_io_var_names(self) -> Tuple[Set[str], Set[str]]:
        block_op_set = set(self.rps)
        block_output_names_set = set()
        block_input_names_set = set()
        for operation in self.rps:
            for o_var in operation.outputs:
                if isinstance(o_var.dest_ops, Sequence) and len(o_var.dest_ops) > 0:
                    for dest_op in o_var.dest_ops:
                        if dest_op not in block_op_set:
                            block_output_names_set.add(o_var.name)
                            break
                else:
                    block_output_names_set.add(o_var.name)
            for i_var in operation.inputs:
                if i_var.source_op is not None:
                    if i_var.source_op not in block_op_set:
                        block_input_names_set.add(i_var.name)
                elif not i_var.is_parameter and i_var.name != "":
                    block_input_names_set.add(i_var.name)
        self.in_var_names = block_input_names_set
        self.out_var_names = block_output_names_set
        return block_input_names_set, block_output_names_set


class XSlimBlockBuilder:
    def __init__(self, graph: BaseGraph, topo_order: List[Operation]) -> None:
        self.graph = graph
        self.op_orders = topo_order
        self.sequence_blocks: List[XSlimTrainableBlock] = []
        self.var_to_dest_blocks = dict()
        self.var_to_src_blocks = dict()
        self.sequence_init()
        self.sequence_io_init()

    def sequence_init(self):
        """

        初始化一个单序列的块列表
        """
        visited_ops = set()

        def _find_coherent_ops(s_op: Operation, rps: List[Operation]) -> Operation:
            rps.append(s_op)
            downstrem_ops = self.graph.get_downstream_operations(s_op)
            upstrem_ops = self.graph.get_upstream_operations(s_op)
            if (
                len(downstrem_ops) == 1
                and len(upstrem_ops) <= 1
                and len(self.graph.get_upstream_operations(downstrem_ops[0])) == 1
                and XSlimTrainableBlock.get_sequence_block_depth(rps) < XQUANT_CONFIG.min_block_size
            ):
                return _find_coherent_ops(downstrem_ops[0], rps)
            else:
                return s_op

        for op in self.op_orders:
            if op in visited_ops:
                continue
            s_op = op
            rps = []
            e_op = _find_coherent_ops(s_op, rps)
            for _op in rps:
                visited_ops.add(_op)
            self.sequence_blocks.append(XSlimTrainableBlock(s_op, e_op, rps, self.graph))

    def sequence_io_init(self):
        self.var_to_dest_blocks.clear()
        self.var_to_src_blocks.clear()
        for block in self.sequence_blocks:
            for var_name in block.in_var_names:
                if var_name not in self.var_to_dest_blocks:
                    self.var_to_dest_blocks[var_name] = []
                self.var_to_dest_blocks[var_name].append(block)

            for var_name in block.out_var_names:
                if var_name not in self.var_to_src_blocks:
                    self.var_to_src_blocks[var_name] = []
                self.var_to_src_blocks[var_name].append(block)

    def get_downstream_blocks(self, block: XSlimTrainableBlock):
        downstream_blocks = set()
        for var_name in block.out_var_names:
            downstream_blocks.update(self.var_to_dest_blocks.get(var_name, []))
        return list(downstream_blocks)

    def get_upstream_blocks(self, block: XSlimTrainableBlock):
        upstream_blocks = set()
        for var_name in block.in_var_names:
            upstream_blocks.update(self.var_to_src_blocks.get(var_name, []))
        return list(upstream_blocks)

    def build(self):
        def _find_same_input_block(block: XSlimTrainableBlock):
            visited_var_names = block.in_var_names | block.out_var_names
            merge_blocks = set()
            merge_block_list = []
            down_blocks = self.get_downstream_blocks(block)

            for _ in range(len(down_blocks)):
                # 因为一个块的合并，则另一个块也可能可以合并，但这个顺序不一定
                for down_block in down_blocks:
                    if down_block.in_var_names <= visited_var_names and down_block not in merge_blocks:
                        merge_blocks.add(down_block)
                        merge_block_list.append(down_block)
                        visited_var_names.update(down_block.out_var_names)
            return merge_block_list

        def _merge_block(
            start_block: XSlimTrainableBlock, end_block: XSlimTrainableBlock, rp_blocks: List[XSlimTrainableBlock]
        ):
            rp_blocks.append(start_block)
            if end_block is start_block:
                return
            for down_block in self.get_downstream_blocks(start_block):
                if down_block not in rp_blocks:
                    _merge_block(down_block, end_block, rp_blocks)

        for merge_step in range(XQUANT_CONFIG.merge_block_step):
            valid_blocks = []
            visited_block = set()
            for block in self.sequence_blocks:
                if block in visited_block:
                    continue
                visited_block.add(block)
                merge_blocks = []
                if block.depth < XQUANT_CONFIG.max_block_size:
                    merge_blocks = _find_same_input_block(block)

                update_block = block
                if len(merge_blocks) > 0:
                    max_add_depth = 0
                    for rp_block in merge_blocks:
                        max_add_depth = max(rp_block.depth, max_add_depth)

                    if (max_add_depth + block.depth) < XQUANT_CONFIG.max_block_size:
                        for rp_block in merge_blocks:
                            visited_block.add(rp_block)
                            update_block.rps.extend(rp_block.rps)
                        update_block.ep = merge_blocks[-1].ep
                        update_block.update_block_io_var_names()
                        update_block.get_depth()
                valid_blocks.append(update_block)

            self.sequence_blocks = valid_blocks
            self.sequence_io_init()

        for block in self.sequence_blocks:
            block.get_depth()

        return self.sequence_blocks


class LSQDelegatorDecorator(LSQDelegator):
    def finalize(self) -> None:
        with torch.no_grad():
            if isinstance(self.config.scale, torch.Tensor):
                if torch.any(self.config.scale < 0):
                    self.withdraw()

        self.scale_backup = None
        self.offset_backup = None
        self.param_backup = None

    def __call__(self, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
        scale, offset = config.scale, config.offset

        def grad_scale(t, factor):
            return (t - (t * factor)).detach() + (t * factor)

        grad_factor = 1 / (tensor.numel() * config.quant_max) ** 0.5
        offset = (offset.round() - offset).detach() + offset
        if self.is_scale_trainable:
            scale = scale.abs()
            scale = grad_scale(scale, grad_factor)

        if self.is_offset_trainable:
            offset = grad_scale(offset, grad_factor)

        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
            scale = scale.view(shape)
            offset = offset.view(shape)

        quantized = tensor / scale + offset
        quantized = (quantized.round() - quantized).detach() + quantized
        quantized = torch.clamp(quantized, config.quant_min, config.quant_max)
        return (quantized - offset) * scale


class LearnedStepSizePassDecorator(LearnedStepSizePass):
    def collect(
        self,
        graph: BaseGraph,
        block: XSlimTrainableBlock,
        executor: TorchExecutor,
        dataloader: Iterable,
        collate_fn: Callable,
        collecting_device: str,
        steps: int = None,
        expire_device: str = "cpu",
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        def cache_fn(data: torch.Tensor):
            if collecting_device == "cuda":
                data = data.cuda()
                mem_free, mem_all = torch.cuda.mem_get_info()
                mem_free_ratio = mem_free / mem_all
                if mem_free_ratio < 0.2:
                    data = data.cpu()
            else:
                data = data.cpu()
            return data

        block_output_names = list(block.out_var_names)
        block_input_names = list(block.in_var_names)
        with torch.no_grad():
            quant_graph = QuantableGraph(graph)  # helper class
            fp_outputs, qt_inputs = [], []

            cur_iter = 0
            # dequantize graph, collect fp32 outputs
            quant_graph.dequantize_graph(expire_device=expire_device)
            for data in dataloader:
                if collate_fn is not None:
                    data = collate_fn(data)

                fp_output = executor.forward(data, block_output_names)
                fp_output = {var_name: cache_fn(value) for var_name, value in zip(block_output_names, fp_output)}
                fp_outputs.append(fp_output)
                cur_iter += 1
                if steps is not None and cur_iter > steps:
                    break

            cur_iter = 0
            # restore quantization state, collect quant inputs
            quant_graph.restore_quantize_state(expire_device=expire_device)
            for data in dataloader:
                if collate_fn is not None:
                    data = collate_fn(data)
                qt_input = executor.forward(data, block_input_names)
                qt_input = {var_name: cache_fn(value) for var_name, value in zip(block_input_names, qt_input)}
                qt_inputs.append(qt_input)
                cur_iter += 1
                if steps is not None and cur_iter > steps:
                    break

        return qt_inputs, fp_outputs

    def compute_block_loss(
        self,
        block: XSlimTrainableBlock,
        qt_inputs: List[Dict[str, torch.Tensor]],
        fp_outputs: List[Dict[str, torch.Tensor]],
        executor: TorchExecutor,
        loss_fn: Callable = torch_mean_square_error,
    ) -> float:
        with torch.no_grad():
            block_output_names = list(block.out_var_names)
            block_input_names = list(block.in_var_names)
            losses = {var_name: 0.0 for var_name in block_output_names}
            for qt_input, fp_output in zip(qt_inputs, fp_outputs):
                feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}

                qt_output = executor.partial_graph_forward(
                    operations=block.rps, feed_dict=feed_dict, output_names=block_output_names
                )

                for name, quant_output in zip(block_output_names, qt_output):
                    batch_loss = loss_fn(quant_output.to(executor._device), fp_output[name].to(executor._device))
                    losses[name] += batch_loss.detach().item()

            for name in losses:
                losses[name] /= len(qt_inputs)
        return sum([v for v in losses.values()])

    def finetune(
        self,
        steps: int,
        learning_rate: float,
        block: XSlimTrainableBlock,
        executor: TorchExecutor,
        qt_inputs: List[Dict[str, torch.Tensor]],
        fp_outputs: List[Dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer = None,
    ) -> Tuple[float, float]:
        # step - 1: enable gradient for training.
        self.enable_block_gradient(block)

        # record pre training loss.
        pre_loss = self.compute_block_loss(
            block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs, executor=executor, loss_fn=self.loss_fn
        )

        # collect trainable params
        delegators = {}
        trainable_scales = []
        for op in block.rps:
            if not isinstance(op, QuantableOperation):
                continue

            # register quant delegator
            for cfg, var in op.config_with_variable:
                if cfg.detail.get("NONE_VALUE", False):
                    continue
                if cfg.state in {QuantizationStates.ACTIVATED}:
                    offset_trainable = (
                        cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL) and self.is_scale_trainable
                    )
                    delegator = LSQDelegatorDecorator(
                        config=cfg,
                        var=var,
                        is_scale_trainable=self.is_scale_trainable,
                        is_offset_trainable=offset_trainable,
                    )
                    trainable_scales.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator
                elif cfg.state in {QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT} and var.is_parameter:
                    delegator = LSQDelegatorDecorator(
                        config=cfg,
                        var=var,
                        is_scale_trainable=False,
                        is_offset_trainable=False,
                    )
                    trainable_scales.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator

        # check if empty.
        tensors = [tensor for tensor in trainable_scales if tensor.requires_grad]
        tensors = set(tensors)  # remove duplicated tensor
        if len(tensors) == 0:
            for cfg, delegator in delegators.items():
                executor.remove_quantize_delegate(config=cfg)
            return 0, 0

        # initilize optimizer.
        if self.optimizer is None:
            optimizer = torch.optim.Adam(tensors, lr=learning_rate)
        else:
            optimizer = self.optimizer(tensors, lr=learning_rate)

        dataset_length = len(qt_inputs)
        if dataset_length == 0:
            raise ValueError("Dataset is empty.")

        range_steps = [i for i in range(steps)]
        random.shuffle(range_steps)
        for idx in tqdm(range_steps, desc="Block Tuning"):
            qt_input, fp_output = qt_inputs[idx % dataset_length], fp_outputs[idx % dataset_length]

            # forward
            optimizer.zero_grad()
            feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}
            output_names = [name for name in fp_output]

            qt_output = executor.partial_graph_forward(
                operations=block.rps, feed_dict=feed_dict, output_names=output_names
            )

            # compute loss
            loss = 0.0
            for name, o_var in zip(output_names, qt_output):
                loss += self.loss_fn(o_var, fp_output[name].to(executor._device))

            for op in block.rps:
                if self.gamma == 0:
                    continue
                if isinstance(op, QuantableOperation) and op.is_computing_op:
                    weight = op.inputs[1].value
                    wconfig = op.config.input_quantization_config[1]
                    loss += torch_mean_square_error(weight, PPQLinearQuantFunction(weight, wconfig)) * self.gamma

            # backward from loss
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            optimizer.step()

        # step - 3: record post training loss
        post_loss = self.compute_block_loss(
            block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs, executor=executor, loss_fn=self.loss_fn
        )
        # check and withdraw
        if post_loss > pre_loss:
            for cfg, delegator in delegators.items():
                delegator.withdraw()
            logger.info(f"Tuning Finished: loss no change and withdraw.\n")
        else:
            logger.info(f"Tuning Finished: ({pre_loss:.5f} -> {min(pre_loss, post_loss):.5f})\n")

        for cfg, delegator in delegators.items():
            delegator.finalize()
            executor.remove_quantize_delegate(config=cfg)

        # disable gradient for evaluation.
        self.disable_block_gradient(block)

        # clear cache
        torch.cuda.empty_cache()

        return pre_loss, post_loss
