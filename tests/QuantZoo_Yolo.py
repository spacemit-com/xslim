# Test Quantization System Performance on Detection Models with Coco Dataset
OUTPUT_DIR = "/home/huangjinghui/1_workspace/2_ppq/Output/detection"
# 开启浮点模型精度评估
EVAL_FP_EN = False
# 开启量化模型精度评估
EVAL_QUANT_EN = False
EVAL_VERBOSE = False
CALIB_STEPS = 100
# calibration & test batchsize
BATCHSIZE = 1

# Should contains model file(.onnx)
MODEL_DIR = "QuantZoo/Model/yolo"

# Should contains Calib & Test Img Folder
CALIB_DIR = "QuantZoo/Data/Coco/Calib"
TEST_DIR = "QuantZoo/Data/Coco/Test"
CALIB_ANN_FILE = "QuantZoo/Data/Coco/Calib/DetectionAnnotation.json"
TEST_ANN_FILE = "QuantZoo/Data/Coco/Test/DetectionAnnotation.json"
PRED_ANN_FILE = "QuantZoo/Data/Coco/Test/DetectionPrediction.json"
EVAL_MODE = False  # only for evaluation, it will slow down the system.

# write report to here
REPORT_DIR = "QuantZoo/Reports"

CONFIGS = [
    {
        "Model": "yolov6p5_n",
        "Output": ["/Concat_5_output_0", "/Concat_4_output_0"],
        "collate_fn": lambda x: x[0],  # img preprocessing function
    },
    {
        "Model": "yolov6p5_t",
        "Output": ["/Concat_5_output_0", "/Concat_4_output_0"],
        "collate_fn": lambda x: x[0],  # img preprocessing function
    },
    {
        "Model": "yolov5s6_n",
        "Output": [
            "/baseModel/head_module/convs_pred.1/Conv_output_0",
            "/baseModel/head_module/convs_pred.2/Conv_output_0",
            "/baseModel/head_module/convs_pred.0/Conv_output_0",
        ],
        "collate_fn": lambda x: x[0],  # img preprocessing function
    },
    {
        "Model": "yolov5s6_s",
        "Output": [
            "/baseModel/head_module/convs_pred.1/Conv_output_0",
            "/baseModel/head_module/convs_pred.2/Conv_output_0",
            "/baseModel/head_module/convs_pred.0/Conv_output_0",
        ],
        "collate_fn": lambda x: x[0],  # img preprocessing function
    },
    {
        "Model": "yolov7p5_tiny",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
        "collate_fn": lambda x: x[0],  # img preprocessing function
    },
    {
        "Model": "yolov7p5_l",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
        "collate_fn": lambda x: x[0],  # img preprocessing function
    },
    {
        "Model": "yolox_s",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
        "collate_fn": lambda x: x[0] * 255,  # img preprocessing function
    },
    {
        "Model": "yolox_tiny",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
        "collate_fn": lambda x: x[0] * 255,  # img preprocessing function
    },
    {
        "Model": "ppyoloe_m",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
        "collate_fn": lambda x: (x[0] * 255 - torch.tensor([103.53, 116.28, 123.675]).reshape([1, 3, 1, 1]))
        / 255,  # img preprocessing function
    },
    {
        "Model": "ppyoloe_s",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
        "collate_fn": lambda x: (x[0] * 255 - torch.tensor([103.53, 116.28, 123.675]).reshape([1, 3, 1, 1]))
        / 255,  # img preprocessing function
    },
]

import os

import torch
from ppq.api import export_ppq_graph, quantize_onnx_model
import ppq.lib as PFL
from ppq.api import ENABLE_CUDA_KERNEL, load_onnx_graph, export_ppq_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
import SpacemiTQuant
from ppq.IR import GraphFormatter
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from QuantZoo.Data.Coco.Data import load_coco_detection_dataset
from QuantZoo.Data.Coco.Eval import evaluate_ppq_module_with_coco
from QuantZoo.Util import error_analyze


calib_loader = load_coco_detection_dataset(data_dir=CALIB_DIR, batchsize=BATCHSIZE)

test_loader = load_coco_detection_dataset(data_dir=TEST_DIR, batchsize=BATCHSIZE)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

quant_setting = QuantizationSettingFactory.default_setting()
quant_setting.fusion_setting.align_quantization = False
quant_setting.equalization = True

# with ENABLE_CUDA_KERNEL():
for config in CONFIGS:
    model = config["Model"]
    if model not in {"yolov6p5_n", "yolov6p5_t"}:
        continue
    monitoring_vars = config["Output"]
    collate_fn = config["collate_fn"]
    print(f"Ready to run quant benchmark on {model}")

    input_model_path = os.path.join(MODEL_DIR, model + ".onnx")
    truncate_model_path = os.path.join(OUTPUT_DIR, model + "_truncate.onnx")

    float_graph = load_onnx_graph(onnx_import_file=input_model_path)
    # truncate graph
    float_graph.outputs.clear()
    editor = GraphFormatter(float_graph)
    for var in monitoring_vars:
        float_graph.mark_variable_as_graph_output(float_graph.variables[var])
    editor.delete_isolated()

    export_ppq_graph(
        graph=float_graph,
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=truncate_model_path,
    )

    # if model in {"rtmdet_s", "rtmdet_tiny"}:
    #    # 别问我为什么，只是因为大家的输出都叫 '/Split_output_1'，但是 rtmdet 的不叫这个
    #    # rename variable 'onnx::Shape_1182' to '/Split_output_1'
    #    float_graph.variables["/Split_output_1"] = float_graph.variables["onnx::Shape_1182"]
    #    float_graph.variables["/Split_output_1"]._name = "/Split_output_1"
    # editor = GraphFormatter(float_graph)
    # float_graph.outputs.pop("scores")
    # float_graph.outputs.pop("num_dets")
    # float_graph.mark_variable_as_graph_output(float_graph.variables["/Split_output_1"])
    # editor.delete_isolated()

    quantized_graph = quantize_onnx_model(
        onnx_import_file=truncate_model_path,
        calib_dataloader=calib_loader,
        calib_steps=CALIB_STEPS,
        input_shape=[BATCHSIZE, 3, 640, 640],
        setting=quant_setting,
        collate_fn=collate_fn,
        platform=TargetPlatform.ONNXRUNTIME,
        device="cpu",
        verbose=0,
    )

    # if EVAL_FP_EN:
    #    # call pipeline.
    #    executor = TorchExecutor(graph=float_graph)
    #    executor.tracing_operation_meta(torch.zeros(size=[BATCHSIZE, 3, 640, 640]))
    #    # evaluation 好像 batchsize != 1 会错
    #    evaluate_ppq_module_with_coco(
    #        ann_file=TEST_ANN_FILE,
    #        output_file=PRED_ANN_FILE,
    #        executor=executor,
    #        dataloader=test_loader,
    #        collate_fn=collate_fn,
    #    )

    if EVAL_QUANT_EN:
        # error analyze
        performance = error_analyze(
            graph=quantized_graph, outputs=monitoring_vars, dataloader=test_loader, collate_fn=collate_fn, verbose=True
        )

    # export quantized graph.

    export_ppq_graph(
        graph=quantized_graph,
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=os.path.join(OUTPUT_DIR, "{}.q.onnx".format(model)),
        config_save_to=os.path.join(OUTPUT_DIR, "{}.json".format(model)),
    )
