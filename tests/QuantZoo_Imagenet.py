# Test Quantization System Performace on Image Classification Models with ILSVRC2012 Dataset
#
#   1. How to use:
#      Run this script with python directly.
OUTPUT_DIR = "/home/huangjinghui/1_workspace/2_ppq/Output/imagenet"
# 开启浮点模型精度评估
EVAL_FP_EN = False
# 开启量化模型精度评估
EVAL_QUANT_EN = True
EVAL_VERBOSE = False
EVAL_ERROR_PERFORMANCE = False
CALIB_STEPS = 100
# calibration & test batchsize
BATCHSIZE = 1

# Should contains model file(.onnx)
MODEL_DIR = "QuantZoo/Model/Imagenet"

# Should contains Calib & Test Img Folder
CALIB_DIR = "QuantZoo/Data/Imagenet/Calib"
TEST_DIR = "QuantZoo/Data/Imagenet/Test"

# write report to here
REPORT_DIR = "QuantZoo/Reports"

CONFIGS = [
    {"Model": "resnet18", "Output": ["/layer4/layer4.1/relu_1/Relu_output_0"]},
    # {"Model": "resnet50", "Output": ["/layer4/layer4.2/relu_2/Relu_output_0"]},
    # {
    #    "Model": "mobilenet_v2",
    #    "Output": ["/features/features.18/features.18.2/Clip_output_0"],
    # },
    # {
    #    "Model": "mobilenet_v3_large",
    #    "Output": ["/classifier/classifier.1/Mul_output_0"],
    # },
    # {
    #    "Model": "mobilenet_v3_small",
    #    "Output": ["/classifier/classifier.1/Mul_output_0"],
    # },
    # {
    #    "Model": "efficientnet_v1_b0",
    #    "Output": ["/features/features.8/features.8.2/Mul_output_0"],
    # },
    # {
    #    "Model": "efficientnet_v1_b1",
    #    "Output": ["/features/features.8/features.8.2/Mul_output_0"],
    # },
    # {"Model": "efficientnet_v2_s", "Output": ["/features/features.7/features.7.2/Mul_output_0"]},
    # {"Model": "mnasnet0_5", "Output": ["/layers/layers.16/Relu_output_0"]},
    # {"Model": "mnasnet1_0", "Output": ["/layers/layers.16/Relu_output_0"]},
    # {"Model": "repvgg", "Output": ["input.172"]},
    # {"Model": "v100_gpu64@5ms_top1@71.6_finetune@25", "Output": ["471"]},
    # {"Model": "v100_gpu64@6ms_top1@73.0_finetune@25", "Output": ["471"]},
    # {"Model": "shufflenet_v2_x1_0", "Output": ["978"]},
    # {"Model": "lcnet_050", "Output": ["/act2/Mul_output_0"]},
    # {"Model": "lcnet_100", "Output": ["/act2/Mul_output_0"]},
    # {"Model": "inception_v3", "Output": ["output"]},
    # {"Model": "seresnet50", "Output": ["output"]},
    # {"Model": "vgg16", "Output": ["output"]},
    # {
    #    # vit_b_16 requires BATCHSIZE = 1!
    #    "Model": "vit_b_16",
    #    "Output": ["onnx::Gather_1703"],
    # },
]

import os
import torch
from ppq.core import TargetPlatform
from ppq.api import load_onnx_graph
import xquant
from ppq import (
    BaseGraph,
    QuantizationSettingFactory,
    TargetPlatform,
    layerwise_error_analyse,
    graphwise_error_analyse,
)
from ppq.api import export_ppq_graph, quantize_onnx_model
from QuantZoo.Data.Imagenet.Eval import (
    evaluate_ppq_module_with_imagenet,
    load_imagenet_from_directory,
)
from QuantZoo.Util import error_analyze
import torchvision.transforms as transforms
from ppq.core import common as ppq_common

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for config in CONFIGS:
    model = config["Model"]
    monitoring_vars = config["Output"]
    print(f"Ready to run quant benchmark on {model}")

    input_model_path = os.path.join(MODEL_DIR, model + ".onnx")
    opt_model_path = input_model_path
    float_graph = load_onnx_graph(onnx_import_file=input_model_path)

    custom_transforms = None
    input_shape = [BATCHSIZE, 3, 224, 224]
    quant_setting = QuantizationSettingFactory.default_setting()
    quant_setting.fusion_setting.align_quantization = False
    quant_setting.equalization = True

    if model == "inception_v3":
        input_shape = [BATCHSIZE, 3, 299, 299]
    #
    # if model in {"mnasnet1_0"}:
    #    pass
    #
    # if model in {"efficientnet_v1_b0"}:
    #    pass
    #    # quant_setting.quantize_activation = False
    #    # quant_setting.fusion_setting.align_quantization = False
    #
    # if model in {"lcnet_050", "lcnet_010"}:
    #    quant_setting.quantize_activation = False
    #    quant_setting.fusion_setting.align_quantization = False

    if model in {"vit_b_16"}:
        opt_model_path = os.path.join(OUTPUT_DIR, model + "_opt.onnx")
        from ppq.IR import GraphMerger

        processor = GraphMerger(float_graph)
        processor.fuse_layernorm()
        processor.fuse_gelu()
        export_ppq_graph(
            graph=float_graph,
            platform=TargetPlatform.ONNXRUNTIME,
            graph_save_to=opt_model_path,
        )

    calib_loader = load_imagenet_from_directory(
        directory=CALIB_DIR,
        batchsize=BATCHSIZE,
        shuffle=True,
        require_label=False,
        num_of_workers=0,
        custom_transforms=custom_transforms,
    )

    test_loader = load_imagenet_from_directory(
        directory=TEST_DIR,
        batchsize=BATCHSIZE,
        shuffle=False,
        require_label=True,
        num_of_workers=0,
        custom_transforms=custom_transforms,
    )

    error_test_loader = load_imagenet_from_directory(
        directory=TEST_DIR,
        subset=50,
        batchsize=BATCHSIZE,
        shuffle=False,
        require_label=True,
        num_of_workers=0,
        custom_transforms=custom_transforms,
    )

    quantized_graph = xquant.quantize_onnx_model("/home/huangjinghui/1_workspace/2_ppq/xquant/demo_setting.json")

    if EVAL_FP_EN:
        print(f"Evaluate float Model Accurarcy....")
        # evaluation
        acc = evaluate_ppq_module_with_imagenet(
            model=float_graph,
            imagenet_validation_loader=test_loader,
            batchsize=BATCHSIZE,
            device="cuda",
            verbose=EVAL_VERBOSE,
        )
        print(f"Model Classify Accurarcy = {acc: .4f}%")

    output_model_path = os.path.join(OUTPUT_DIR, "{}.q.onnx".format(model))
    export_ppq_graph(
        graph=quantized_graph,
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=output_model_path,
        config_save_to=os.path.join(OUTPUT_DIR, "{}.json".format(model)),
    )

    if EVAL_QUANT_EN:
        # 需要使用导出后的ONNX模型推理 保持与ORT的算子一致
        quantized_export_graph = load_onnx_graph(onnx_import_file=output_model_path)

        print(f"Evaluate quantized Model Accurarcy....")
        # evaluation
        acc = evaluate_ppq_module_with_imagenet(
            model=quantized_export_graph,
            imagenet_validation_loader=test_loader,
            batchsize=BATCHSIZE,
            device="cuda",
            verbose=EVAL_VERBOSE,
        )
        print(f"Model Classify Accurarcy = {acc: .4f}%")
