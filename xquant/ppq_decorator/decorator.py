from ppq.lib import (
    register_network_quantizer,
    register_network_exporter,
    register_operation_handler,
    register_calibration_observer,
)
from .onnxruntime_exporter import ONNXRUNTIMExporter
from ppq.core import TargetPlatform
from ..quantizer import XQuantizer

register_network_exporter(ONNXRUNTIMExporter, TargetPlatform.ONNXRUNTIME)
register_network_quantizer(XQuantizer, TargetPlatform.ONNXRUNTIME)
