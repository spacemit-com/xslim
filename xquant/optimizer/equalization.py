from collections import defaultdict
from typing import Callable, Dict, Iterable, List
from enum import Enum
import torch
from ppq.executor.torch import TorchExecutor
from ppq.IR import BaseGraph, Operation
from ppq.IR.base.graph import BaseGraph
from tqdm import tqdm
from ppq.quantization.optim import LayerwiseEqualizationPass


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

        loader_step = int(len(dataloader) / steps)
        if steps >= len(dataloader):
            loader_step = 1
        for idx, batch in tqdm(
            enumerate(dataloader), desc="Equalization Data Collecting.", total=min(len(dataloader), steps)
        ):
            if idx % loader_step != 0:
                continue
            data = batch
            if collate_fn is not None:
                data = collate_fn(batch)
            outputs = executor.forward(data, output_names=output_names)
            for name, output in zip(output_names, outputs):
                op = graph.variables[name].source_op
                output_collector[name].append(aggregate(op, output).unsqueeze(-1))
            if idx > steps:
                break

        result = {}
        for name, output in zip(output_names, outputs):
            result[name] = torch.cat(output_collector[name], dim=-1)
        return result
