from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
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


class HardSwishFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__("HardSwish Fusion")

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs,
    ) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(
            patterns=[lambda x: x.is_computing_op, "HardSigmoid", "Mul"],
            edges=[[0, 1], [1, 2], [0, 2]],
            exclusive=True,
        )

        for pattern in patterns:
            if any([not isinstance(op, QuantableOperation) for op in pattern]):
                ppq_warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            if any([op.platform != pattern[0].platform for op in pattern]):
                ppq_warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            computing, sigmoid, mul = pattern

            assert isinstance(computing, QuantableOperation)
            assert isinstance(sigmoid, QuantableOperation)
            assert isinstance(mul, QuantableOperation)

            computing_config = computing.config.output_quantization_config[0]
            sigmoid.config.input_quantization_config[0].dominated_by = computing_config
            sigmoid.config.output_quantization_config[0].state = QuantizationStates.FP32
            mul.config.input_quantization_config[0].dominated_by = computing_config
            mul.config.input_quantization_config[1].state = QuantizationStates.FP32


class SwishFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__("HardSwish Fusion")

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs,
    ) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(
            patterns=[lambda x: x.is_computing_op, "Sigmoid", "Mul"],
            edges=[[0, 1], [1, 2], [0, 2]],
            exclusive=True,
        )

        for pattern in patterns:
            if any([not isinstance(op, QuantableOperation) for op in pattern]):
                ppq_warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            if any([op.platform != pattern[0].platform for op in pattern]):
                ppq_warning(
                    f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                    "however part of your swish activation is not quantable, "
                    "so that graph fusion can not merge their quantization configuration."
                )
                continue
            computing, sigmoid, mul = pattern

            assert isinstance(computing, QuantableOperation)
            assert isinstance(sigmoid, QuantableOperation)
            assert isinstance(mul, QuantableOperation)

            computing_config = computing.config.output_quantization_config[0]
            sigmoid.config.input_quantization_config[0].dominated_by = computing_config
            sigmoid.config.output_quantization_config[0].state = QuantizationStates.FP32
            mul.config.input_quantization_config[0].dominated_by = computing_config
            mul.config.input_quantization_config[1].state = QuantizationStates.FP32
