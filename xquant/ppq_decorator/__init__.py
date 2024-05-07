from . import ppq
from .ppq import (
    DataType,
    empty_ppq_cache,
    QuantizationVisibility,
    convert_any_to_torch_tensor,
    PPQ_CONFIG,
    TargetPlatform,
    TensorQuantizationConfig,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    common as ppq_common,
    ppq_quant_param_computing_function,
)
from .ppq.IR import BaseGraph, Operation, QuantableOperation, Variable, QuantableGraph, GraphReplacer
from .ppq.IR.search import SearchableGraph
from .ppq.IR.base.opdef import Opset
from .ppq.IR import GraphMerger, GraphFormatter
from .ppq.executor import TorchExecutor, BaseGraphExecutor
from .ppq.scheduler import DISPATCHER_TABLE, GraphDispatcher
from .ppq.IR.base.command import QuantizeOperationCommand
from .ppq.quantization.optim import (
    QuantizationOptimizationPass,
    QuantizationOptimizationPipeline,
    QuantizeFusionPass,
    QuantizeSimplifyPass,
    ParameterQuantizePass,
    ParameterBakingPass,
    LearnedStepSizePass,
    LayerwiseEqualizationPass,
)
from .ppq.quantization.algorithm.training import LSQDelegator, TrainableBlock
from .ppq.parser import OnnxParser
from .ppq.parser import ONNXRUNTIMExporter
from .ppq.quantization.measure import torch_mean_square_error, torch_snr_error, torch_cosine_similarity
from .ppq.quantization.qfunction.linear import PPQLinearQuantFunction, PPQLinearQuant_toInt
from .ppq.quantization import observer as ppq_observer
from .ppq.quantization.observer import BaseTensorObserver, TorchMinMaxObserver, OperationObserver
from .ppq.quantization.observer.range import minmax_to_scale_offset
from .ppq.quantization.measure import torch_KL_divergence


def load_onnx_graph(file: str):
    return OnnxParser().build(file)
