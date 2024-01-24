import ppq
from ppq.quantization.observer import (
    OBSERVER_TABLE,
)
from ppq.quantization import optim as ppq_optim
from ppq.quantization.algorithm import training as ppq_algorithm_training
from ppq.parser import onnx_parser as ppq_onnx_parser
from .onnxruntime_exporter import ONNXRUNTIMExporter
from ppq.core import TargetPlatform, defs as ppq_defs
import ppq.core as ppq_core
from ..quantizer import XQuantizer
from ..optimizer import (
    TorchXQuantObserver,
    LearnedStepSizePassDecorator,
    LSQDelegatorDecorator,
    TorchXQuantKLObserver,
    TorchXQuantMSEObserver,
    TorchPercentileObserverDecorator,
    TorchMinMaxObserverObserverDecorator,
)
from .onnx_parser import OnnxParserDecorator
from ..defs import PASSIVE_OPERATIONS, COMPUTING_OP

ppq_optim.training.LearnedStepSizePass.finetune = LearnedStepSizePassDecorator.finetune
ppq_algorithm_training.LSQDelegator.__call__ = LSQDelegatorDecorator.__call__
ppq_algorithm_training.LSQDelegator.finalize = LSQDelegatorDecorator.finalize
ppq_onnx_parser.OnnxParser.build = OnnxParserDecorator.build
ppq_core.PASSIVE_OPERATIONS = PASSIVE_OPERATIONS
ppq_core.COMPUTING_OP = COMPUTING_OP
OBSERVER_TABLE["xquant"] = TorchXQuantObserver
OBSERVER_TABLE["kl"] = TorchXQuantKLObserver
OBSERVER_TABLE["mse"] = TorchXQuantMSEObserver
OBSERVER_TABLE["percentile"] = TorchPercentileObserverDecorator
OBSERVER_TABLE["minmax"] = TorchMinMaxObserverObserverDecorator
