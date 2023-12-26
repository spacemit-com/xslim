import xquant

# common
xquant.quantize_onnx_model("resnet18.json")

# custom quantize config
xquant.quantize_onnx_model("mobilenet_v3_small.json")

# custom preprocess
xquant.quantize_onnx_model("inception_v1.json")
