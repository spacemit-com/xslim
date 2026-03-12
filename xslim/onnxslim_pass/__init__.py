import onnx
import onnxslim

from xslim.logger import logger

from .dynamic_q_matmul import *
from .gelu import *
from .layernorm import *
from .batchnorm import *
# from .swish import *

def optimize_onnx_model(onnx_model):
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
    except Exception as e:
        logger.warning(f"onnx shape_inference error before onnxslim and skip. {e}")
    onnx_model = onnxslim.slim(onnx_model)
    return onnx_model
