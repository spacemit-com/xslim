import onnx
import onnxslim

from xslim.logger import logger

from .dynamic_q_matmul import *
from .gelu import *
from .layernorm import *
from .batchnorm import *
# from .swish import *


def infer_onnx_model(onnx_model):
    opset_import = onnx_model.opset_import
    try:
        onnx_model = onnx.shape_inference.infer_shapes(
            onnx_model, data_prop=True, strict_mode=True, check_type=True)
    except Exception as e:
        logger.warning(
            f"onnx shape_inference error after onnxslim and skip. {e}")
    if onnx_model.opset_import is None or len(onnx_model.opset_import) == 0:
        onnx_model.opset_import.extend(opset_import)
    return onnx_model


def optimize_onnx_model(onnx_model):
    onnx_model = infer_onnx_model(onnx_model)
    onnx_model = onnxslim.slim(onnx_model)
    return onnx_model
