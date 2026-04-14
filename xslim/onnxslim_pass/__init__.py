import onnx
import onnxslim

from xslim.logger import logger

from .batchnorm import *
from .dynamic_q_matmul import *
from .gelu import *
from .layernorm import *
from .padpool import *

# from .swish import *


def infer_onnx_model(onnx_model):
    opset_import = onnx_model.opset_import
    try:
        onnx_model = onnxslim.core.shape_infer(onnx_model)
    except Exception as e:
        logger.warning(
            f"onnx shape_inference error and skip. {e}")
    if onnx_model.opset_import is None or len(onnx_model.opset_import) == 0:
        onnx_model.opset_import.extend(opset_import)
    return onnx_model


def optimize_onnx_model(onnx_model):
    onnx_model = onnxslim.slim(onnx_model, skip_fusion_patterns=["FusionGemm"])
    return onnx_model
