#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from collections import defaultdict
from typing import Callable, Dict, Iterable, List
from enum import Enum
import torch
import numpy as np
from tqdm import tqdm
from ppq.core import empty_ppq_cache
from ppq.executor.torch import TorchExecutor
from ppq.IR import BaseGraph, Operation, QuantableOperation
from ppq.executor import BaseGraphExecutor
from ppq.quantization.algorithm import equalization as equalization_alg
from ppq.quantization.optim.equalization import (
    LayerwiseEqualizationPass,
    EQUALIZATION_OPERATION_TYPE,
    OPTIMIZATION_LAYERTYPE_CONFIG,
)


class CustomLayerwiseEqualizationPass(LayerwiseEqualizationPass):
    def collect_activations(
        self,
        graph: BaseGraph,
        executor: TorchExecutor,
        dataloader: Iterable,
        collate_fn: Callable,
        operations: List[Operation],
        steps: int = 32,
    ) -> Dict[str, torch.Tensor]:
        def aggregate(op: Operation, tensor: torch.Tensor):
            if op.type in {"Conv", "ConvTranspose"}:  # Conv result: [n,c,h,w]
                num_of_channel = tensor.shape[1]
                tensor = tensor.transpose(0, 1)
                tensor = tensor.reshape(shape=[num_of_channel, -1])
                tensor = torch.max(tensor.abs(), dim=-1, keepdim=False)[0]
            elif op.type in {"MatMul", "Gemm"}:  # Gemm result: [n, c]
                tensor = tensor.transpose(0, 1)
                tensor = torch.max(tensor.abs(), dim=-1, keepdim=False)[0]
            return tensor

        output_names = []
        for operation in operations:
            assert operation.num_of_output == 1, f"Num of output of layer {operation.name} is supposed to be 1"
            output_names.append(operation.outputs[0].name)

        output_collector = defaultdict(list)

        steps = min(steps, len(dataloader))
        loader_step_index = set(np.random.randint(0, len(dataloader), [steps]).tolist())
        for idx, batch in enumerate(dataloader):
            if idx not in loader_step_index:
                continue
            if collate_fn is not None:
                data = collate_fn(batch)
            outputs = executor.forward(data, output_names=output_names)
            for name, output in zip(output_names, outputs):
                op = graph.variables[name].source_op
                output_collector[name].append(aggregate(op, output).unsqueeze(-1))

        result = {}
        for name, output in zip(output_names, outputs):
            result[name] = torch.cat(output_collector[name], dim=-1)
        return result

    @empty_ppq_cache
    def optimize(
        self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, collate_fn: Callable, **kwargs
    ) -> None:
        interested_operations = []

        if self.interested_layers is None:
            for operation in graph.operations.values():
                if operation.type in EQUALIZATION_OPERATION_TYPE:
                    interested_operations.append(operation)
        else:
            for name in self.interested_layers:
                if name in graph.operations:
                    interested_operations.append(graph.operations[name])

        pairs = self.find_equalization_pair(graph=graph, interested_operations=interested_operations)

        print(f"{len(pairs)} equalization pair(s) was found, ready to run optimization.")
        for iter_times in tqdm(range(self.iterations), desc="Layerwise Equalization", total=self.iterations):
            if self.including_act:
                activations = self.collect_activations(
                    graph=graph,
                    executor=executor,
                    dataloader=dataloader,
                    collate_fn=collate_fn,
                    operations=interested_operations,
                    steps=50,
                )

                for name, act in activations.items():
                    graph.variables[name].value = act  # 将激活值写回网络

            for equalization_pair in pairs:
                equalization_pair.equalize(
                    value_threshold=self.value_threshold,
                    including_bias=self.including_bias,
                    including_act=self.including_act,
                    bias_multiplier=self.bias_multiplier,
                    act_multiplier=self.act_multiplier,
                )

        # equalization progress directly changes fp32 value of weight,
        # store it for following procedure.
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.store_parameter_value()
