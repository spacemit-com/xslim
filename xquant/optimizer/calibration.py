from typing import Iterable, List, Set, Union, Dict, Callable, Tuple, Sequence
import torch
from tqdm import tqdm
from ppq.core import (
    COMPUTING_OP,
    OBSERVER_MSE_HIST_BINS,
    PASSIVE_OPERATIONS,
    BIAS_CORRECTION_INTERST_TYPE,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    common as ppq_common,
    convert_any_to_torch_tensor,
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
    OBSERVER_TABLE,
)
from ppq.quantization.algorithm.training import TrainableBlock, BlockBuilder, PriorityQueue
from ppq.quantization.measure import torch_mean_square_error, torch_snr_error
from ppq.quantization.optim import LearnedStepSizePass, AdaroundPass
from ppq.utils.round import ppq_round_to_power_of_2
from ..optimizer import BiasParameterBakingPass


class CustomBlockBuilder(BlockBuilder):
    def build(self, op: Operation, limit: int) -> TrainableBlock:
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

            quantable_op_num = _find_computing_ops_in_route(sp, ep)
            if future_ep is None or ((self.depth[future_ep] - self.depth[sp] > limit) and quantable_op_num > 0):
                return self.create_block(sp, ep)
            ep = future_ep
        return self.create_block(sp=sp, ep=ep)


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
        calib_block_size: int = 8,
        block_wise: bool = True,
        fintune_epoch: int = 2,
        auto_finetune_level: int = 1,
    ) -> None:
        super().__init__(method, override, calib_steps)
        self.name = "XQuant Runtime Calibration Pass(BlockWise)"
        self._calib_block_size = calib_block_size
        self._block_wise = block_wise
        self._fintune_epoch = fintune_epoch
        self._block_wise_loss = []
        self._auto_finetune_level = auto_finetune_level

    def split_graph_into_blocks(
        self,
        graph: BaseGraph,
        executing_order: List[Operation],
        blocksize: int = ppq_common.OPTIM_ADVOPT_GRAPH_MAXDEPTH,
    ) -> List[TrainableBlock]:
        visited_ops, blocks = set(), []
        block_builder = CustomBlockBuilder(graph=graph, topo_order=executing_order)

        for op in executing_order:
            # start from computing op
            if op in visited_ops:
                continue
            block = block_builder.build(op, blocksize)
            # by default blocks are exclusive from each other
            for op in block.rps:
                visited_ops.add(op)
            blocks.append(block)
        return blocks

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

        block_op_set = set(block.rps)
        block_output_names_set = set()
        block_input_names_set = set()
        block_input_names = []

        operation_cache = []
        operation_observer_cache = []
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

            operation_cache.append(operation)
            if isinstance(operation, QuantableOperation):
                operation_observer = OperationObserver(operation=operation, monitor_parameter=False)
                operation_observer_cache.append(operation_observer)
                if operation.type in BIAS_CORRECTION_INTERST_TYPE and operation.num_of_parameter == 2:
                    bias_op_var_names[operation.outputs[0].name] = operation

        ob_table_num = sum([len(ob.hook._observer_table) for ob in operation_observer_cache])
        hooks = {ob._operation.name: ob.hook for ob in operation_observer_cache}
        if ob_table_num == 0:
            hooks = {}

        block_input_names = list(block_input_names_set)
        block_output_names = list(block_output_names_set)

        for var_name in block_input_names:
            if var_name not in dataloader_cache:
                raise RuntimeError("missing input {} for block".format(var_name))

        with torch.no_grad():
            for idx in tqdm(range(calib_steps), desc="Runtime Calibration Single Block Collect"):
                inputs_feed = {
                    var_name: dataloader_cache[var_name][idx].to(executor._executing_context.executing_device)
                    for var_name in block_input_names
                }
                output_names = list(block_output_names_set | set(bias_op_var_names.keys()))
                outputs = executor._TorchExecutor__forward(
                    inputs_feed,
                    operation_cache,
                    output_names=output_names,
                    hooks=hooks,
                )
                for o_var, o_name in zip(outputs, output_names):
                    if o_name in block_output_names_set:
                        if o_name not in dataloader_cache:
                            dataloader_cache[o_name] = []
                        mem_free, mem_all = torch.cuda.mem_get_info()
                        mem_free_ratio = mem_free / mem_all
                        if mem_free_ratio < 0.1:
                            dataloader_cache[o_name].append(o_var.to("cpu"))
                        else:
                            dataloader_cache[o_name].append(o_var)
                    if o_name in bias_op_var_names:
                        reduce_bias = self.collect_bias(o_var, bias_op_var_names[o_name].type)
                        if o_name not in bias_fp_cache:
                            bias_fp_cache[o_name] = []
                        bias_fp_cache[o_name].append(reduce_bias)

        if len(hooks) == 0:
            return

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
            with torch.no_grad():
                for idx in tqdm(range(calib_steps), desc="Runtime Calibration Single Block Collect"):
                    inputs_feed = {
                        var_name: dataloader_cache[var_name][idx].to(executor._executing_context.executing_device)
                        for var_name in block_input_names
                    }
                    outputs = executor._TorchExecutor__forward(
                        inputs_feed,
                        operation_cache,
                        output_names=block_output_names,
                        hooks=hooks,
                    )
            for ob in operation_observer_cache:
                ob.render_quantization_config()
                ob.report()

        self.format_parameter_tqc(operation_cache)
        if self._auto_finetune_level >= 1:
            self.block_bias_correct(
                block, dataloader_cache, executor, calib_steps, bias_op_var_names, bias_fp_cache, block_input_names_set
            )
            self.format_parameter_tqc(operation_cache)

        loss_dict = self.compute_block_loss(
            block, dataloader_cache, executor, calib_steps, block_input_names_set, block_output_names_set
        )
        self._block_wise_loss.append(
            self.create_block_loss_info(block, loss_dict["mse"], loss_dict["snr"], loss_dict["mean"])
        )
        self._block_wise_loss[-1]["block_idx"] = len(self._block_wise_loss)

        remove_opset = set(operation_cache)
        remove_ovarnames = []
        for o_name, _ in dataloader_cache.items():
            if o_name in var_to_operations:
                var_to_operations[o_name] -= remove_opset
                if len(var_to_operations[o_name]) == 0:
                    remove_ovarnames.append(o_name)
        for o_name in remove_ovarnames:
            dataloader_cache.pop(o_name)

    def format_activation_tqc(self, op_list: Sequence[Operation]):
        for op in op_list:
            if isinstance(op, QuantableOperation):
                for tqc, var in op.config_with_variable:
                    if (
                        not var.is_parameter
                        and tqc.policy.has_property(QuantizationProperty.PER_TENSOR)
                        and tqc.dominated_by == tqc
                        and tqc.scale is not None
                    ):
                        _scale = ppq_round_to_power_of_2(float(tqc.scale), policy=RoundingPolicy.ROUND_UP)
                        tqc.scale = convert_any_to_torch_tensor(_scale, dtype=tqc.scale.dtype, device=tqc.scale.device)

    def format_parameter_tqc(self, op_list: Sequence[Operation]):
        for op in op_list:
            if isinstance(op, QuantableOperation):
                for tqc, var in op.config_with_variable:
                    if (
                        var.is_parameter
                        and tqc.policy.has_property(QuantizationProperty.PER_CHANNEL)
                        and isinstance(tqc.scale, torch.Tensor)
                    ):
                        if not torch.all(tqc.scale >= 0):
                            negative_idx = torch.where(tqc.scale < 0)[0]
                            scale_shape = var.value.ndim * [1]
                            scale_shape[tqc.channel_axis] = -1
                            var_scale = torch.ones(
                                [var.value.size()[tqc.channel_axis]],
                                dtype=var.value.dtype,
                                device=var.value.device,
                            )
                            var_scale[negative_idx] = -1
                            tqc.scale[negative_idx] = -tqc.scale[negative_idx]
                            var_scale = var_scale.reshape(scale_shape)
                            var.value *= var_scale
                BiasParameterBakingPass.passive_parameters_quant(op)
                BiasParameterBakingPass.passive_bias_quant(op)

    def compute_block_loss(
        self,
        block: TrainableBlock,
        dataloader_cache: dict,
        executor: TorchExecutor,
        calib_steps: int,
        block_input_names_set: Set[str],
        block_output_names_set: Set[str],
    ) -> dict:
        block_output_names = list(block_output_names_set)
        operation_cache = [operation for operation in block.rps]
        mse_losses = {o_name: 0 for o_name in block_output_names}
        snr_losses = {o_name: 0 for o_name in block_output_names}
        means = {o_name: 0 for o_name in block_output_names}

        with torch.no_grad():
            for idx in tqdm(range(calib_steps), desc="Runtime Calibration Single Block Quant Check"):
                inputs_feed = {
                    var_name: dataloader_cache[var_name][idx].to(executor._executing_context.executing_device)
                    for var_name in block_input_names_set
                }
                outputs = executor.partial_graph_forward(
                    operations=operation_cache, feed_dict=inputs_feed, output_names=block_output_names
                )
                for o_var, o_name in zip(outputs, block_output_names):
                    ref_var = dataloader_cache[o_name][idx]
                    ref_var = ref_var.to(executor._executing_context.executing_device)
                    batch_mse_loss = torch_mean_square_error(o_var, ref_var)
                    batch_snr_loss = torch_snr_error(o_var, ref_var)
                    means[o_name] += ref_var.abs().mean().detach().item()
                    mse_losses[o_name] += batch_mse_loss.detach().item()
                    snr_losses[o_name] += batch_snr_loss.detach().item()

        for o_name in mse_losses:
            mse_losses[o_name] /= calib_steps
            snr_losses[o_name] /= calib_steps
            means[o_name] /= calib_steps

        return {"mse": mse_losses, "snr": snr_losses, "mean": means}

    def collect_bias(self, output: torch.Tensor, op_type: str) -> torch.Tensor:
        if output.ndim < 1:
            raise ValueError("Forward value has an unexpected dimension.")
        if op_type in {"Conv", "ConvTranspose"}:
            # for convolution layer, bias always been added on axis 1
            reduce_dims = [i for i in range(output.ndim) if i != 1]
            return torch.mean(output, dim=reduce_dims).unsqueeze(0)
        elif op_type in {"Gemm"}:
            # for convolution layer, bias always been added on axis -1
            reduce_dims = [i for i in range(output.ndim) if i != (output.ndim - 1)]
            return torch.mean(output, dim=(0,)).unsqueeze(0)
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
        block_input_names_set: Set[str],
    ) -> None:
        if len(bias_op_var_names) == 0:
            return
        operation_cache = [operation for operation in block.rps]
        bias_quant_cache = {}
        output_names = list(bias_op_var_names.keys())
        for idx in tqdm(range(calib_steps), desc="Runtime Calibration Single Block Finetune"):
            inputs_feed = {
                var_name: dataloader_cache[var_name][idx].to(executor._executing_context.executing_device)
                for var_name in block_input_names_set
            }
            outputs = executor.partial_graph_forward(
                operations=operation_cache, feed_dict=inputs_feed, output_names=list(bias_op_var_names.keys())
            )
            for o_var, o_name in zip(outputs, output_names):
                reduce_bias = self.collect_bias(o_var, bias_op_var_names[o_name].type)
                if o_name not in bias_quant_cache:
                    bias_quant_cache[o_name] = [reduce_bias]
                else:
                    bias_quant_cache[o_name].append(reduce_bias)

        for o_name in bias_op_var_names:
            DC_term_fp = bias_fp_cache[o_name]
            DC_term_qt = bias_quant_cache[o_name]

            if len(DC_term_fp) == 0 or len(DC_term_qt) == 0:
                continue
            DC_term_fp = torch.mean(torch.cat(DC_term_fp, axis=0), dim=0)
            DC_term_qt = torch.mean(torch.cat(DC_term_qt, axis=0), dim=0)
            bias_error = DC_term_fp - DC_term_qt
            bias_op_var_names[o_name].inputs[-1].value += bias_error

    def create_block_loss_info(self, block: TrainableBlock, mse_losses: dict, snr_losses: dict, means: dict):
        return {
            "start_op": "[{}]{}".format(block.sp.type, block.sp.name),
            "end_op": "[{}]{}".format(block.ep.type, block.ep.name),
            "snr": max([v for k, v in snr_losses.items()]),
            "mse": max([v for k, v in mse_losses.items()]),
            "block": block,
        }

    def report_block_loss(self, block_losses: Sequence[dict], top_k=5):
        for loss_info in block_losses[:top_k]:
            loss_str = "{} -> {}: mse = {:.4f}, snr = {:.4f}\n".format(
                loss_info["start_op"], loss_info["end_op"], loss_info["snr"], loss_info["mse"]
            )
            print(loss_str)

    def finetune(
        self,
        graph: BaseGraph,
        blocks: Sequence[TrainableBlock],
        dataloader: Iterable,
        executor: TorchExecutor,
        calib_steps: int = 32,
        collate_fn: Callable = None,
    ):
        lsq_optimizer = LearnedStepSizePass()
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

            print(f"# Tuning Finished  : ({pre_loss:.4f} -> {min(pre_loss, post_loss):.4f}) [Block Loss]")
            print("")  # blank line

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
            "Calibration steps is too large, ppq is capable for quantizing your network within 10-1000 "
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

        topo_sort_ops = graph.topological_sort()

        if self._block_wise:
            op_blocks = self.split_graph_into_blocks(graph, topo_sort_ops, self._calib_block_size)
            for block in tqdm(op_blocks, desc="Runtime Calibration(BlockWise)"):
                self.calib_single_block(block, dataloader_cache, var_to_operations, executor, calib_step)

            self._block_wise_loss = sorted(self._block_wise_loss, key=lambda x: x["mse"], reverse=True)

            if self._auto_finetune_level >= 2:
                _auto_finetune_blocks = 10 if self._auto_finetune_level >= 3 else 3
                finetune_blocks_info = [loss_info for loss_info in self._block_wise_loss[:_auto_finetune_blocks]]
                finetune_blocks_info = sorted(finetune_blocks_info, key=lambda x: x["block_idx"], reverse=True)
                self.finetune(
                    graph,
                    [loss_info["block"] for loss_info in finetune_blocks_info],
                    dataloader,
                    executor,
                    calib_step,
                    collate_fn,
                )
            self.report_block_loss(self._block_wise_loss)
        else:
            raise NotImplementedError
