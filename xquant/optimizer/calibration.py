#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Iterable, List, Set, Union, Dict, Callable, Tuple, Sequence
import functools
import torch
from tqdm import tqdm
from ppq.core import (
    QuantizationStates,
    common as ppq_common,
)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.quantization.optim import (
    RuntimeCalibrationPass,
)
from ppq.executor import TorchExecutor
from ppq.quantization.observer import (
    OperationObserver,
    TorchHistObserver,
    TorchMSEObserver,
)
from ppq.quantization.algorithm.training import TrainableBlock, BlockBuilder, PriorityQueue
from ppq.quantization.measure import (
    torch_mean_square_error,
    torch_snr_error,
    torch_cosine_similarity,
    torch_cosine_similarity_as_loss,
)
from ppq.quantization.optim import LearnedStepSizePass, AdaroundPass
from ppq.utils.round import ppq_round_to_power_of_2
from ..optimizer import PassiveParameterBakingPass
from ..defs import COMPUTING_OP, PASSIVE_OPERATIONS, BIAS_CORRECTION_INTERST_TYPE, XQUANT_CONFIG


class CalibrationBlock:
    def __init__(self, s_vars: Set[Variable], e_vars: Set[Variable], rps: Sequence[Operation]) -> None:
        self.s_vars = s_vars  # 起始边
        self.e_vars = e_vars  # 终止边
        self.rps = rps  # 中继节点

    def __str__(self) -> str:
        s_var_names = ", ".join([i.name for i in self.s_vars])
        e_var_names = ", ".join([i.name for i in self.e_vars])
        return "[Graph Block from [{}] to [{}]]".format(s_var_names, e_var_names)

    @staticmethod
    def convert_from_trainableblock(block: TrainableBlock, graph: BaseGraph) -> "CalibrationBlock":
        in_var_names, out_var_names = get_block_io_var_names(block)
        return CalibrationBlock(
            set([graph.variables[i] for i in in_var_names]),
            set([graph.variables[i] for i in out_var_names]),
            block.rps,
        )


class CustomBlockBuilder(BlockBuilder):
    def build(self, op: Operation, max_limit: int, min_limit: int) -> TrainableBlock:
        def _find_multi_input_ep(op: Operation):
            # 如果当前节点后继节点存在多个，层序遍历寻找阻断节点
            least_first_queue = PriorityQueue()
            least_first_queue.push(self.depth[op], op)
            least_first_queue.pop()

            for down_op in self.graph.get_downstream_operations(op):
                least_first_queue.push(self.depth[down_op], down_op)

            while not least_first_queue.empty():
                iter_operation = least_first_queue.pop()[-1]
                if least_first_queue.empty():
                    upstream_ops = self.graph.get_upstream_operations(iter_operation)
                    if all([op in least_first_queue._ops for op in upstream_ops]) and len(upstream_ops) > 1:
                        return iter_operation
                for down_op in self.graph.get_downstream_operations(iter_operation):
                    least_first_queue.push(self.depth[down_op], down_op)

            # if least_first_queue is empty, it means we can not find an blocking ep from given sp.
            return None

        def _find_coherent_ep(op: Operation):
            # 如果当前节点后继节点只有一个，向下寻找直系节点
            # 但如果直系后继节点有多于一个输入，算法立即停机
            ops = self.graph.get_downstream_operations(op)
            if len(ops) == 1:
                following_op = ops[0]
                # PATCH 20220811，get_upstream_operations 不足以判断算子是否只有一个输入
                # 因为算子可以直接与图的 input 相连...
                non_parameter_input = following_op.num_of_input - following_op.num_of_parameter
                upstream_ops = len(self.graph.get_upstream_operations(following_op))
                if non_parameter_input == 1 and upstream_ops == 1:
                    return ops[0]
            return None

        def _find_computing_ops_in_route(sp: Operation, ep: Operation):
            if sp == ep:
                return 1 if sp.type in COMPUTING_OP else 0
            rps = self.search_engine.opset_matching(
                sp_expr=lambda x: x == sp, rp_expr=lambda x, y: True, ep_expr=lambda x: x == ep, direction="down"
            )
            rps = [(self.op_orders.index(op), op) for op in rps if isinstance(op, QuantableOperation)]
            return len(rps)

        sp, ep, future_ep = op, op, op
        while future_ep is not None:
            if len(self.graph.get_downstream_operations(ep)) <= 1:
                future_ep = _find_coherent_ep(ep)
            else:
                future_ep = _find_multi_input_ep(ep)

            if future_ep is None:
                return self.create_block(sp, ep)

            current_depth = self.depth[ep] - self.depth[sp]
            future_depth = self.depth[future_ep] - self.depth[sp]

            if future_ep is not None and len(self.graph.get_downstream_operations(future_ep)) > 1:
                # 如果下个节点产生分叉，分叉后产生超过限制的深度就不合并
                next_future_ep = _find_multi_input_ep(future_ep)
                next_future_depth = self.depth[next_future_ep] - self.depth[sp]
                if next_future_depth > max_limit and current_depth > min_limit:
                    return self.create_block(sp, ep)

            if future_depth > max_limit:
                quantable_op_num = _find_computing_ops_in_route(sp, ep)
                if quantable_op_num > 0:
                    return self.create_block(sp, ep)
            ep = future_ep
        return self.create_block(sp=sp, ep=ep)


@functools.lru_cache(maxsize=None)
def get_block_io_var_names(block: Union[TrainableBlock, CalibrationBlock]):
    block_op_set = set(block.rps)
    block_output_names_set = set()
    block_input_names_set = set()
    for operation in block.rps:
        for o_var in operation.outputs:
            if isinstance(o_var.dest_ops, Sequence) and len(o_var.dest_ops) > 0:
                for dest_op in o_var.dest_ops:
                    if dest_op not in block_op_set:
                        block_output_names_set.add(o_var.name)
                        break
            else:
                block_output_names_set.add(o_var.name)
        # block内所有op输入的source_op不属于当前block
        for i_var in operation.inputs:
            if i_var.source_op is not None:
                if i_var.source_op not in block_op_set:
                    block_input_names_set.add(i_var.name)
            elif not i_var.is_parameter and i_var.name != "":
                block_input_names_set.add(i_var.name)
    return block_input_names_set, block_output_names_set


class RuntimeBlockWiseCalibrationPass(RuntimeCalibrationPass):
    """
    逐块执行量化，

    Args:
        RuntimeCalibrationPass (_type_): _description_
    """

    def __init__(
        self,
        method: str = None,
        override: bool = False,
        calib_steps: int = 32,
        block_wise: bool = True,
        fintune_epoch: int = 2,
        auto_finetune_level: int = 1,
    ) -> None:
        super().__init__(method, override, calib_steps)
        self.name = "XQuant Runtime Calibration Pass(BlockWise)"
        self._block_wise = block_wise
        self._fintune_epoch = fintune_epoch
        self._block_wise_loss = []
        self._auto_finetune_level = auto_finetune_level

    def split_graph_into_blocks(
        self,
        graph: BaseGraph,
        executing_order: List[Operation],
    ) -> List[TrainableBlock]:
        visited_ops, blocks = set(), []
        block_builder = CustomBlockBuilder(graph=graph, topo_order=executing_order)

        for op in executing_order:
            # start from computing op
            if op in visited_ops:
                continue
            block = block_builder.build(op, XQUANT_CONFIG.max_block_size, XQUANT_CONFIG.min_block_size)
            # by default blocks are exclusive from each other
            for op in block.rps:
                visited_ops.add(op)
            blocks.append(block)

        return blocks

    def forward_trainable_block(
        self,
        block: TrainableBlock,
        dataloader_cache: dict,
        executor: TorchExecutor,
        executor_hook: dict,
        calib_steps: int,
        extern_output_var_hooks: Dict[str, Callable] = None,
        collect_dataloader_cache: bool = False,
    ):
        operation_cache = block.rps
        if extern_output_var_hooks is None:
            extern_output_var_hooks = {}

        block_input_names_set, block_output_names_set = get_block_io_var_names(block)

        block_input_names = list(block_input_names_set)

        for var_name in block_input_names:
            if var_name not in dataloader_cache:
                raise RuntimeError("missing input variable {} for block".format(var_name))

        for idx in tqdm(range(calib_steps), desc="Runtime Calibration Single Block Collect", disable=True):
            inputs_feed = {
                var_name: dataloader_cache[var_name][idx].to(executor._executing_context.executing_device)
                for var_name in block_input_names
            }
            output_names = list(block_output_names_set | set(extern_output_var_hooks.keys()))
            outputs = executor._TorchExecutor__forward(
                inputs_feed,
                operation_cache,
                output_names=output_names,
                hooks=executor_hook,
            )
            for o_var, o_name in zip(outputs, output_names):
                if o_name in block_output_names_set and collect_dataloader_cache:
                    if o_name not in dataloader_cache:
                        dataloader_cache[o_name] = []
                    if o_var.device.type == "cuda":
                        mem_free, mem_all = torch.cuda.mem_get_info()
                        mem_free_ratio = mem_free / mem_all
                        if mem_free_ratio < 0.2:
                            dataloader_cache[o_name].append(o_var.to("cpu"))
                        else:
                            dataloader_cache[o_name].append(o_var)
                    else:
                        dataloader_cache[o_name].append(o_var)
                if o_name in extern_output_var_hooks:
                    extern_output_var_hooks[o_name](o_name, o_var, idx)

    def clean_dataloader_cache(
        self,
        block: TrainableBlock,
        dataloader_cache: dict,
        var_to_operations: dict,
    ):
        remove_op_set = set(block.rps)
        remove_ovarnames = []
        for o_name, _ in dataloader_cache.items():
            if o_name in var_to_operations:
                var_to_operations[o_name] -= remove_op_set
                if len(var_to_operations[o_name]) == 0:
                    remove_ovarnames.append(o_name)
        for o_name in remove_ovarnames:
            dataloader_cache.pop(o_name)

    def calib_single_block(
        self,
        block: TrainableBlock,
        dataloader_cache: dict,
        var_to_operations: dict,
        executor: TorchExecutor,
        calib_steps: int,
    ):
        bias_op_var_names = dict()
        bias_fp_cache = dict()
        operation_observer_cache = []
        extern_output_var_hooks = {}

        def __collect_bias(o_name: str, o_var: torch.Tensor, idx: int):
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
                        not isinstance(var_observer, (TorchHistObserver, TorchMSEObserver))
                        for var_observer in ob._hook._observer_table.values()
                    ]
                )

        with torch.no_grad():
            self.forward_trainable_block(
                block, dataloader_cache, executor, hooks, calib_steps, extern_output_var_hooks, True
            )

        if len(hooks) == 0:
            return

        for ob in operation_observer_cache:
            ob.render_quantization_config()
            ob.report()

        if not_has_hist_ob:
            pass
        else:
            with torch.no_grad():
                self.forward_trainable_block(block, dataloader_cache, executor, hooks, calib_steps)
            for ob in operation_observer_cache:
                ob.render_quantization_config()
                ob.report()

        if self._auto_finetune_level >= 1:
            self.block_bias_correct(
                block,
                dataloader_cache,
                executor,
                calib_steps,
                bias_op_var_names,
                bias_fp_cache,
            )

        loss_dict = self.compute_block_loss(block, dataloader_cache, executor, calib_steps)
        self._block_wise_loss.append(
            self.create_block_loss_info(
                block, loss_dict["mse"], loss_dict["snr"], loss_dict["mean"], loss_dict["cosine"]
            )
        )
        self._block_wise_loss[-1]["block_idx"] = len(self._block_wise_loss)
        self.clean_dataloader_cache(block, dataloader_cache, var_to_operations)

    def format_tqc(self, operaion: Operation):
        if not isinstance(operaion, QuantableOperation):
            return

        for tqc, var in operaion.config_with_variable:
            if tqc.dominated_by is tqc and tqc.scale is not None:
                if torch.any(tqc.scale < 0):
                    raise RuntimeError("tqc.scale for {} < 0".format(operaion.name))

        PassiveParameterBakingPass.passive_parameters_quant(operaion)
        PassiveParameterBakingPass.passive_bias_quant(operaion)

    def compute_block_loss(
        self,
        block: TrainableBlock,
        dataloader_cache: dict,
        executor: TorchExecutor,
        calib_steps: int,
    ) -> dict:
        _, block_output_names_set = get_block_io_var_names(block)
        block_output_names = list(block_output_names_set)
        mse_losses = {o_name: 0 for o_name in block_output_names}
        snr_losses = {o_name: 0 for o_name in block_output_names}
        cosine = {o_name: 0 for o_name in block_output_names}
        means = {o_name: 0 for o_name in block_output_names}

        def __collect_loss(o_name: str, o_var: torch.Tensor, idx: int):
            ref_var = dataloader_cache[o_name][idx]
            ref_var = ref_var.to(executor._executing_context.executing_device)
            batch_mse_loss = torch_mean_square_error(o_var, ref_var)
            batch_snr_loss = torch_snr_error(o_var, ref_var)
            batch_cosine = torch_cosine_similarity(o_var, ref_var)
            means[o_name] += ref_var.abs().mean().detach().item()
            mse_losses[o_name] += batch_mse_loss.detach().item()
            snr_losses[o_name] += batch_snr_loss.detach().item()
            cosine[o_name] += batch_cosine.detach().item()

        extern_output_var_hooks = {o_name: __collect_loss for o_name in block_output_names}

        with torch.no_grad():
            self.forward_trainable_block(block, dataloader_cache, executor, {}, calib_steps, extern_output_var_hooks)

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
        block: TrainableBlock,
        dataloader_cache: dict,
        executor: TorchExecutor,
        calib_steps: int,
        bias_op_var_names: dict,
        bias_fp_cache: dict,
    ) -> None:
        if len(bias_op_var_names) == 0:
            return
        block_input_names_set, block_output_names_set = get_block_io_var_names(block)
        operation_cache = [operation for operation in block.rps]
        bias_quant_cache = {}
        output_names = list(bias_op_var_names.keys())
        for idx in tqdm(range(calib_steps), desc="Runtime Calibration Single Block Finetune", disable=True):
            inputs_feed = {
                var_name: dataloader_cache[var_name][idx].to(executor._executing_context.executing_device)
                for var_name in block_input_names_set
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
        self, block: TrainableBlock, mse_losses: dict, snr_losses: dict, means: dict, cosines: dict
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
        blocks: Sequence[TrainableBlock],
        dataloader: Iterable,
        executor: TorchExecutor,
        calib_steps: int = 32,
        collate_fn: Callable = None,
    ):
        lsq_optimizer = LearnedStepSizePass(is_scale_trainable=True, lr=0.0001)
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
            "Calibration steps is too large, xquant is capable for quantizing your network within 10-1000 "
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

        topo_sort_ops = graph.topological_sort()

        if self._block_wise:
            op_blocks = self.split_graph_into_blocks(graph, topo_sort_ops)
            for block in tqdm(op_blocks, desc="Runtime Calibration(BlockWise)"):
                self.calib_single_block(block, dataloader_cache, var_to_operations, executor, calib_step)

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
