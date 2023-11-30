from typing import Any, Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
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
from ppq.IR import GraphMerger
from ppq.IR.search import SearchableGraph
from ppq.executor import BaseGraphExecutor


class GraphLegalized:
    def __init__(self, graph) -> None:
        self._graph = graph
        self._merger = GraphMerger(self._graph)

    def __call__(self) -> Any:
        self._merger.fuse_layernorm()
        self._merger.fuse_gelu()
        self._merger.fuse_bias_add()
        self.remove_dropout()
        # self._merger.fuse_matmul_add()

    def remove_dropout(self):
        removing_ops = []
        for op in self._graph.operations.values():
            if op.type == "Dropout":
                removing_ops.append(op)

        for op in removing_ops:
            self._graph.remove_operation(op, keep_coherence=True)
