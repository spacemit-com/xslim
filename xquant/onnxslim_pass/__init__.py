import onnxslim

from .dynamic_q_matmul import *
from .gelu import *
from .layernorm import *


def optimize_onnx_model(onnx_model):
    onnx_model = onnxslim.slim(onnx_model)
    return onnx_model
