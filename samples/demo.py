import xslim

# common
# xslim.quantize_onnx_model("resnet18.json")

# xslim.quantize_onnx_model("resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx")

# xslim.quantize_onnx_model(
#    "resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx", "resnet18_output.onnx"
# )

# import onnx
# onnx_model = onnx.load("/home/share/modelzoo/classification/resnet18/resnet18.onnx")
# quantized_onnx_model = xslim.quantize_onnx_model("resnet18.json", onnx_model)

# custom quantize config
# xslim.quantize_onnx_model("mobilenet_v3_small.json")
#
## custom preprocess
# xslim.quantize_onnx_model("resnet18_custom_preprocess.json")
#
## bert
# xslim.quantize_onnx_model("bertsquad.json")

## dynamic quantize
# xslim.quantize_onnx_model("mobilenet_v3_small_dyn_quantize.json")
