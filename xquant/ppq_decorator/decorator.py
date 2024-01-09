import ppq
from ppq.lib import (
    register_network_quantizer,
    register_network_exporter,
    register_operation_handler,
    register_calibration_observer,
)
from ppq.quantization.observer import (
    OBSERVER_TABLE,
)
from ppq.quantization import optim as ppq_optim
from ppq.quantization.algorithm import training as ppq_algorithm_training
from ppq.parser import onnx_parser as ppq_onnx_parser
from .onnxruntime_exporter import ONNXRUNTIMExporter
from ppq.core import TargetPlatform, defs as ppq_defs
from ..quantizer import XQuantizer
from ..optimizer import (
    TorchXQuantObserver,
    LearnedStepSizePassDecorator,
    LSQDelegatorDecorator,
    TorchXQuantKLObserver,
    TorchXQuantMSEObserver,
)
from .onnx_parser import OnnxParserDecorator

ppq_optim.training.LearnedStepSizePass.finetune = LearnedStepSizePassDecorator.finetune
ppq_algorithm_training.LSQDelegator.__call__ = LSQDelegatorDecorator.__call__
ppq_algorithm_training.LSQDelegator.finalize = LSQDelegatorDecorator.finalize
ppq_onnx_parser.OnnxParser.build = OnnxParserDecorator.build
OBSERVER_TABLE["xquant"] = TorchXQuantObserver
OBSERVER_TABLE["kl"] = TorchXQuantKLObserver
OBSERVER_TABLE["mse"] = TorchXQuantMSEObserver
register_network_exporter(ONNXRUNTIMExporter, TargetPlatform.ONNXRUNTIME)
register_network_quantizer(XQuantizer, TargetPlatform.ONNXRUNTIME)
