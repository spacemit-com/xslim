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
from .onnxruntime_exporter import ONNXRUNTIMExporter
from ppq.core import TargetPlatform, defs as ppq_defs
from ..quantizer import XQuantizer
from ..optimizer import TorchXQuantObserver, CustomTrainingBasedPass, TorchXQuantKLObserver, TorchXQuantMSEObserver


ppq_optim.training.TrainingBasedPass.initialize_checkpoints = CustomTrainingBasedPass.initialize_checkpoints
ppq_optim.training.TrainingBasedPass.collect = CustomTrainingBasedPass.collect
ppq_optim.training.TrainingBasedPass.compute_block_loss = CustomTrainingBasedPass.compute_block_loss
OBSERVER_TABLE["xquant"] = TorchXQuantObserver
OBSERVER_TABLE["kl"] = TorchXQuantKLObserver
OBSERVER_TABLE["mse"] = TorchXQuantMSEObserver
register_network_exporter(ONNXRUNTIMExporter, TargetPlatform.ONNXRUNTIME)
register_network_quantizer(XQuantizer, TargetPlatform.ONNXRUNTIME)
