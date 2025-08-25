#!/usr/bin/env python3
import onnx
from onnxconverter_common import float16

model = onnx.load("resnet50.onnx")
model_fp16 = float16.convert_float_to_float16(model)
# print("fp32 model to fp16 pass!")
onnx.save(model_fp16, "resnet50_fp16.onnx")