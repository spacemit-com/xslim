from collections import defaultdict
from typing import Callable, Dict, Iterable, List

import torch
from ppq.core import *
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import BaseGraph, BaseGraph, Operation, QuantableOperation
from ppq.IR.quantize import QuantableGraph
from ppq.quantization.algorithm.training import *
from ppq.quantization.measure import torch_mean_square_error, torch_snr_error
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction
from tqdm import tqdm
from ppq.quantization.optim.base import QuantizationOptimizationPass


class CustomTrainingBasedPass(QuantizationOptimizationPass):
    """Training Based Pass is a basic class that provides necessary function
    for all training optimizition passes. Optimization will be more stable and
    accurate with functions provided by this pass. (Might be a little slower).

    This pass will collect result of interested outputs after optimization and
        check if the optimized result has a lower SNR. If so, the optimization will be
        accepted, layer weight will be updated, otherwise optimization will be rejected and
        takes no effects.

    Choose interested_outputs carefully, cause we compare loss only with those output variables.
        If interested_outputs is None, all graph output variables will be chosen.

    YOUR SHOULD NOTICE THAT SNR REFERS TO: POWER OF NOISE / POWER OF SIGNAL IN PPQ.

    Args:
        QuantizationOptimizationPass ([type]): [description]
    """

    def __init__(
        self, name: str = "Default Quanzation Optim", interested_outputs: List[str] = None, verbose: bool = True
    ) -> None:
        self._loss_fn = torch_snr_error
        self._interested_outputs = interested_outputs
        self._checkpoints = {}
        self._verbose = verbose
        self._quant_state_recorder = {}
        super().__init__(name=name)

    @empty_ppq_cache
    def initialize_checkpoints(
        self, graph: BaseGraph, executor: BaseGraphExecutor, dataloader: Iterable, collate_fn: Callable
    ):
        """
        Establish a series of network checkpoints with your network.
            Checkpoint is a data structure that helps us compare quant results and fp32 results.
        Args:
            graph (BaseGraph): [description]
            executor (BaseGraphExecutor): [description]
            dataloader (Iterable): [description]
            collate_fn (Callable): [description]

        Raises:
            PermissionError: [description]
        """
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
                        raise PermissionError(
                            "Can not initialize checkpoints when weight value is baked. "
                            f"Variable {var.name} has a baked value."
                        )

        if self._interested_outputs is None or len(self._interested_outputs) == 0:
            self._interested_outputs = [name for name in graph.outputs]

        for name in self._interested_outputs:
            self._checkpoints[name] = FinetuneCheckPoint(variable=name)

        # dequantize graph, collect references
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.dequantize()

        for data in tqdm(dataloader, desc="Collecting Referecens"):
            if collate_fn is not None:
                data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=self._interested_outputs)
            for name, output in zip(self._interested_outputs, outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.push(tensor=output, is_reference=True)

        # restore quantization state:
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.restore_quantize_state()

        # update state
        verbose, self._verbose = self._verbose, False
        self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn)
        self._verbose = verbose

    def check(self, executor: BaseGraphExecutor, dataloader: Iterable, collate_fn: Callable):
        """Check quantization error with a given dataloader with current
        checkpoints. Return whether quantization error is lower than before.

        Args:
            executor (BaseGraphExecutor): [description]
            dataloader (Iterable): [description]
            collate_fn (Callable): [description]

        Returns:
            [type]: [description]
        """

        # step - 1, collecting data
        for data in dataloader:
            if collate_fn is not None:
                data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=self._interested_outputs)
            for name, output in zip(self._interested_outputs, outputs):
                self._checkpoints[name].push(tensor=output, is_reference=False)

        # step - 2, calculating loss
        losses = []
        for name in self._interested_outputs:
            ckpt = self._checkpoints[name]
            assert isinstance(ckpt, FinetuneCheckPoint)
            qt_out, fp_out = ckpt.pop()
            qt_out = torch.cat([tensor for tensor in qt_out])
            fp_out = torch.cat([tensor for tensor in fp_out])
            losses.append(self._loss_fn(y_pred=qt_out, y_real=fp_out).item())
            ckpt.clear()

        # step - 3, comparing loss
        loss_now, loss_old = sum(losses), sum([ckpt.best_loss for ckpt in self._checkpoints.values()])
        loss_now, loss_old = loss_now / len(losses), loss_old / len(losses)
        if self._verbose:
            print(f"NOISE-SIGNAL RATIO: {loss_old * 100 :.4f}% -> {loss_now * 100:.4f}%.")

        # if there is a loss drop, update all losses.
        if loss_old > (loss_now * CHECKPOINT_TOLERANCE):
            for idx, name in enumerate(self._interested_outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.best_loss = losses[idx]
            return True
        return False

    @empty_ppq_cache
    def enable_block_gradient(self, block: TrainableBlock):
        """
        Make all tensors inside a given block to be trainable(requres_grad = True)
        Both quantization scale and weight itself are going to be trained in training procedure

        Args:
            block (TrainableBlock): _description_
        """
        for op in block.rps:
            for var in op.inputs + op.outputs:
                if var.is_parameter and isinstance(var.value, torch.Tensor):
                    # PATCH 2022 08 01 Clip op can not be train
                    if op.type == "Clip":
                        continue
                    if var.value.dtype == torch.float:
                        var.value.requires_grad = True
            if isinstance(op, QuantableOperation):
                for cfg, _ in op.config_with_variable:
                    if isinstance(cfg.scale, torch.Tensor):
                        cfg.scale.requires_grad = True

    @empty_ppq_cache
    def disable_block_gradient(self, block: TrainableBlock):
        for op in block.rps:
            for var in op.inputs + op.outputs:
                if var.is_parameter and isinstance(var.value, torch.Tensor):
                    if var.value.is_leaf:
                        var.value.requires_grad = False
                        var.value._grad = None
            if isinstance(op, QuantableOperation):
                for cfg, _ in op.config_with_variable:
                    if isinstance(cfg.scale, torch.Tensor):
                        if cfg.scale.is_leaf:
                            cfg.scale.requires_grad = False
                            cfg.scale._grad = None

    def split_graph_into_blocks(
        self,
        graph: BaseGraph,
        executing_order: List[Operation],
        blocksize: int = None,
        overlap: bool = False,
        interested_layers: List[str] = None,
    ) -> List[TrainableBlock]:
        """block construction function for training-based algorithms, if
        `block_limit` is not specified, block grandularity will be controlled by
        the default value OPTIM_ADVOPT_GRAPH_MAXSIZE specified in ppq.core.common.

        Args:
            graph (BaseGraph): ppq ir graph
            executing_order (List[Operation]): topo search order
            block_limit (int, optional): controls maximum depth of a block. Defaults to None.

        Returns:
            List[TrainableBlock]: list of all partitioned blocks
        """
        if blocksize is None:
            blocksize = OPTIM_ADVOPT_GRAPH_MAXDEPTH
        visited_ops, blocks = set(), []
        block_builder = BlockBuilder(graph=graph, topo_order=executing_order)

        for op in graph.operations.values():
            # start from computing op
            if op in visited_ops and overlap is False:
                continue
            if isinstance(op, QuantableOperation) and op.is_computing_op:
                block = block_builder.build(op, blocksize)
                # by default blocks are exclusive from each other
                for op in block.rps:
                    visited_ops.add(op)
                blocks.append(block)

        ret = []
        if interested_layers is None or len(interested_layers) == 0:
            ret = blocks  # if no interested_layers, finetune all.
        else:
            for candidate in blocks:
                assert isinstance(candidate, TrainableBlock)
                if any([op.name in interested_layers for op in candidate.rps]):
                    ret.append(candidate)
        return ret

    def collect(
        self,
        graph: BaseGraph,
        block: TrainableBlock,
        executor: TorchExecutor,
        dataloader: Iterable,
        collate_fn: Callable,
        collecting_device: str,
        steps: int = None,
        expire_device: str = "cpu",
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Collect training data for given block.
        This function will collect fp32 output and quantized input data by
            executing your graph twice.
        For collecting fp32 output, all related operations will be dequantized.
        For collecting quantized input, all related operations' quantization state will be restored.

        collecting device declares where cache to be stored:
            executor - store cache to executor device.(default)
            cpu      - store cache to system memory.
            cuda     - store cache to gpu memory.(2x speed up)
            disk     - not implemented.

        Args:
            block (TrainableBlock): _description_
            executor (TorchExecutor): _description_
            dataloader (Iterable): _description_
            collate_fn (Callable): _description_
            collecting_device (str): _description_

        Returns:
            _type_: _description_
        """

        def cache_fn(data: torch.Tensor):
            # TODO move this function to ppq.core.IO
            if not isinstance(data, torch.Tensor):
                raise TypeError(
                    "Unexpected Type of value, Except network output to be torch.Tensor, "
                    f"however {type(data)} was given."
                )
            if collecting_device == "cpu":
                data = data.cpu()
            if collecting_device == "cuda":
                data = data.cuda()
            # TODO restrict collecting device.
            return data

        with torch.no_grad():
            try:
                if len(dataloader) > 1024:
                    ppq_warning(
                        "Large finetuning dataset detected(>1024). "
                        "You are suppose to prepare a smaller dataset for finetuning. "
                        "Large dataset might cause system out of memory, "
                        "cause all data are cache in memory."
                    )
            except Exception as e:
                pass  # dataloader has no __len__

            quant_graph = QuantableGraph(graph)  # helper class
            fp_outputs, qt_inputs = [], []

            cur_iter = 0
            # dequantize graph, collect fp32 outputs
            quant_graph.dequantize_graph(expire_device=expire_device)
            for data in dataloader:
                if collate_fn is not None:
                    data = collate_fn(data)
                fp_output = executor.forward(data, [var.name for var in block.ep.outputs])
                fp_output = {var.name: cache_fn(data) for data, var in zip(fp_output, block.ep.outputs)}
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
                # PATCH 20220829, 有些 computing op 权重并非定值
                non_constant_input = [var for var in block.sp.inputs if not var.is_parameter]
                qt_input = executor.forward(data, [var.name for var in non_constant_input])
                qt_input = {var.name: cache_fn(value) for var, value in zip(non_constant_input, qt_input)}
                qt_inputs.append(qt_input)
                cur_iter += 1
                if steps is not None and cur_iter > steps:
                    break

        return qt_inputs, fp_outputs

    def compute_block_loss(
        self,
        block: TrainableBlock,
        qt_inputs: List[Dict[str, torch.Tensor]],
        fp_outputs: List[Dict[str, torch.Tensor]],
        executor: TorchExecutor,
        loss_fn: Callable = torch_mean_square_error,
    ) -> float:
        """
        loss computing for fp32 and quantized graph outputs, used
        in multiple training-based algorithms below

        Args:
            output_names (List[str]): output variable names
            graph (BaseGraph): ppq ir graph
            dataloader (Iterable): calibration dataloader
            collate_fn (Callable): batch collate func
            executor (TorchExecutor): ppq torch executor
            loss_fn (Callable, optional): loss computing func. Defaults to torch_mean_square_error.
        Returns:
            Dict[str, float]: loss dict for variables specified in `output_names`
        """
        with torch.no_grad():
            losses = {var.name: 0.0 for var in block.ep.outputs}
            output_names = [var.name for var in block.ep.outputs]

            for qt_input, fp_output in zip(qt_inputs, fp_outputs):
                feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}

                qt_output = executor.partial_graph_forward(
                    operations=block.rps, feed_dict=feed_dict, output_names=output_names
                )

                for name, quant_output in zip(output_names, qt_output):
                    batch_loss = loss_fn(quant_output, fp_output[name].to(executor._device))
                    losses[name] += batch_loss.detach().item()

            for name in losses:
                losses[name] /= len(qt_inputs)
        return sum([v for v in losses.values()])
