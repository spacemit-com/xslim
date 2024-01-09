from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple
from tqdm import tqdm
import random
import numpy as np
import torch
from ppq.core import *
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import BaseGraph, BaseGraph, Operation, QuantableOperation
from ppq.IR.quantize import QuantableGraph
from ppq.quantization.algorithm.training import LSQDelegator, CuLSQ_LC, CuLSQ_LT, TrainableBlock
from ppq.quantization.measure import torch_mean_square_error, torch_snr_error
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction
from ppq.quantization.qfunction import PPQuantFunction
from ppq.utils.fetch import batch_random_fetch
from ppq.utils.round import ppq_tensor_round
from ppq.quantization.optim.base import QuantizationOptimizationPass
from ppq.quantization.optim import LearnedStepSizePass


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
    def finetune(
        self,
        steps: int,
        learning_rate: float,
        block: TrainableBlock,
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
                if cfg.state in {QuantizationStates.ACTIVATED}:
                    if var.is_parameter:
                        delegator = LSQDelegator(
                            config=cfg,
                            var=var,
                            is_scale_trainable=True,
                            is_offset_trainable=False,
                        )
                    else:
                        delegator = LSQDelegator(
                            config=cfg,
                            var=var,
                            is_scale_trainable=self.is_scale_trainable,
                            is_offset_trainable=self.is_scale_trainable,
                        )
                    trainable_scales.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator
                elif cfg.state in {QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT} and var.is_parameter:
                    delegator = LSQDelegator(
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
        for idx in tqdm(range_steps, desc="Block Finetune Tuning"):
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

        for cfg, delegator in delegators.items():
            delegator.finalize()
            executor.remove_quantize_delegate(config=cfg)

        # disable gradient for evaluation.
        self.disable_block_gradient(block)

        # clear cache
        torch.cuda.empty_cache()

        print(f"Tuning Finished  : ({pre_loss:.5f} -> {min(pre_loss, post_loss):.5f}) [Block Loss]\n")
        return pre_loss, post_loss
