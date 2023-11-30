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
    # {"Model": "resnet18", "Output": ["/layer4/layer4.1/relu_1/Relu_output_0"]},
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
    {"Model": "inception_v3", "Output": ["output"]},
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
from ppq.api import export_ppq_graph, quantize_onnx_model
from QuantZoo.Data.Imagenet.Eval import (
    evaluate_ppq_module_with_imagenet,
    load_imagenet_from_directory,
)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

demo_json = {
    "model_parameters": {
        "onnx_model": "/home/huangjinghui/1_workspace/2_ppq/PPQQuantZoo/QuantZoo/Model/Imagenet/resnet18.onnx",
        "output_model_file_prefix": "resnet18.q",
        "working_dir": "/home/huangjinghui/1_workspace/2_ppq/temp_output",
    },
    "calibration_parameters": {
        "calibration_step": 200,
        "calibration_device": "cuda",
        "calibration_type": "default",
        "input_parametres": [
            {
                "input_name": "input.1",
                "input_shape": [1, 3, 224, 224],
                "file_type": "img",
                "mean_value": [123.675, 116.28, 103.53],
                "std_value": [58.395, 57.12, 57.375],
                "preprocess_file": "IMAGENET",
                "data_list_path": "/home/huangjinghui/1_workspace/2_ppq/quant_temp/img_list.txt",
            }
        ],
    },
    "quantization_parameters": {"precision_level": 1},
}

for config in CONFIGS:
    model = config["Model"]
    monitoring_vars = config["Output"]
    print(f"Ready to run quant benchmark on {model}")

    input_model_path = os.path.join(MODEL_DIR, model + ".onnx")
    opt_model_path = input_model_path
    float_graph = load_onnx_graph(onnx_import_file=input_model_path)

    custom_transforms = None
    mean_value = [123.675, 116.28, 103.53]
    std_value = [58.395, 57.12, 57.375]
    input_shape = [1, 3, 224, 224]
    if model == "inception_v3":
        input_shape = [1, 3, 299, 299]
        mean_value = [127.5, 127.5, 127.5]
        std_value = [127.5, 127.5, 127.5]

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

    test_loader = load_imagenet_from_directory(
        directory=TEST_DIR,
        batchsize=BATCHSIZE,
        shuffle=False,
        require_label=True,
        num_of_workers=0,
        custom_transforms=custom_transforms,
    )

    demo_json["model_parameters"]["onnx_model"] = opt_model_path
    demo_json["model_parameters"]["output_model_file_prefix"] = "{}.q".format(model)
    demo_json["model_parameters"]["working_dir"] = OUTPUT_DIR
    demo_json["calibration_parameters"]["input_parametres"][0]["input_shape"] = input_shape
    demo_json["calibration_parameters"]["input_parametres"][0]["mean_value"] = mean_value
    demo_json["calibration_parameters"]["input_parametres"][0]["std_value"] = std_value
    demo_json["calibration_parameters"]["input_parametres"][0]["input_name"] = list(float_graph.inputs.keys())[0]

    quantized_graph = xquant.quantize_onnx_model(demo_json)

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

    output_model_path = os.path.join(
        OUTPUT_DIR, "{}.onnx".format(demo_json["model_parameters"]["output_model_file_prefix"])
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
