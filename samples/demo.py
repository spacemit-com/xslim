import xquant

# common
xquant.quantize_onnx_model("resnet18.json")

# xquant.quantize_onnx_model("resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx")

# xquant.quantize_onnx_model(
#    "resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx", "resnet18_output.onnx"
# )

# import onnx
# onnx_model = onnx.load("/home/share/modelzoo/classification/resnet18/resnet18.onnx")
# quantized_onnx_model = xquant.quantize_onnx_model("resnet18.json", onnx_model)

# custom quantize config
xquant.quantize_onnx_model("mobilenet_v3_small.json")

# custom preprocess
xquant.quantize_onnx_model("inception_v1.json")

# xquant.quantize_onnx_model("bertsquad.json")
