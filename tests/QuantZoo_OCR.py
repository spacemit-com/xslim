# Test Quantization System Performace on OCR Models with IC15 Dataset
OUTPUT_DIR = "/home/huangjinghui/1_workspace/2_ppq/Output/ocr"
# 开启浮点模型精度评估
EVAL_FP_EN = False
# 开启量化模型精度评估
EVAL_QUANT_EN = False
EVAL_VERBOSE = False
CALIB_STEPS = 100
# calibration & test batchsize
BATCHSIZE = 1

# Should contains model file(.onnx)
MODEL_DIR = "QuantZoo/Model/ocr"

# Should contains Calib & Test Img Folder
CALIB_DIR = "QuantZoo/Data/IC15"
CALIB_LABEL = "QuantZoo/Data/IC15/rec_gt_train.txt"
TEST_DIR = "QuantZoo/Data/IC15"
TEST_LABEL = "QuantZoo/Data/IC15/rec_gt_test.txt"
CHAR_DIR = "QuantZoo/Data/IC15/ic15_dict.txt"

# write report to here
REPORT_DIR = "QuantZoo/Reports"

CONFIGS = [
    {
        "Model": "en_PP-OCRv3_rec_infer",
        "Output": ["swish_13.tmp_0"],
        "Dictionary": "en_dict.txt",
        "Reshape": [3, 48, 320],
        "Language": "en",
    },
    {
        "Model": "en_number_mobile_v2.0_rec_infer",
        "Output": ["save_infer_model/scale_0.tmp_1"],
        "Dictionary": "en_dict.txt",
        "Reshape": [3, 32, 320],
        "Language": "en",
    },
    {
        "Model": "ch_PP-OCRv2_rec_infer",
        "Output": ["p2o.LSTM.5"],
        "Dictionary": "ppocr_keys_v1.txt",
        "Reshape": [3, 32, 320],
        "Language": "ch",
    },
    {
        "Model": "ch_PP-OCRv3_rec_infer",
        "Output": ["swish_27.tmp_0"],
        "Dictionary": "ppocr_keys_v1.txt",
        "Reshape": [3, 48, 320],
        "Language": "ch",
    },
    {
        "Model": "ch_ppocr_mobile_v2.0_rec_infer",
        "Output": ["p2o.LSTM.5"],
        "Dictionary": "ppocr_keys_v1.txt",
        "Reshape": [3, 32, 320],
        "Language": "ch",
    },
    {
        "Model": "ch_ppocr_server_v2.0_rec_infer",
        "Output": ["p2o.LSTM.5"],
        "Dictionary": "ppocr_keys_v1.txt",
        "Reshape": [3, 32, 320],
        "Language": "ch",
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
from QuantZoo.Data.IC15.Data import IC15_PaddleOCR
from QuantZoo.Data.IC15.Eval import evaluate_ppq_module_with_ic15
from QuantZoo.Quantizers import MyFP8Quantizer, MyInt8Quantizer
from QuantZoo.Util import error_analyze, report

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

quant_setting = QuantizationSettingFactory.default_setting()
quant_setting.fusion_setting.align_quantization = False
quant_setting.equalization = True

# with ENABLE_CUDA_KERNEL():
for config in CONFIGS:
    model = config["Model"]
    monitoring_vars = config["Output"]
    dictionary = config["Dictionary"]
    shape = config["Reshape"]
    chinese = config["Language"] == "ch"
    input_model_path = os.path.join(MODEL_DIR, model + ".onnx")
    truncate_model_path = os.path.join(OUTPUT_DIR, model + "_truncate.onnx")
    calib_loader = IC15_PaddleOCR(
        images_path=CALIB_DIR, label_path=CALIB_LABEL, input_shape=shape, is_chinese_version=chinese
    ).dataloader(batchsize=BATCHSIZE, shuffle=False)
    test_loader = IC15_PaddleOCR(
        images_path=TEST_DIR, label_path=TEST_LABEL, input_shape=shape, is_chinese_version=chinese
    ).dataloader(batchsize=BATCHSIZE, shuffle=False)
    print(f"Ready to run quant benchmark on {model}")

    quantized_graph = quantize_onnx_model(
        onnx_import_file=input_model_path,
        calib_dataloader=calib_loader,
        calib_steps=CALIB_STEPS,
        input_shape=[BATCHSIZE, 3, 32, 100],
        setting=quant_setting,
        collate_fn=lambda x: x[0],
        platform=TargetPlatform.ONNXRUNTIME,
        device="cpu",
        verbose=0,
    )

    export_ppq_graph(
        graph=quantized_graph,
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=os.path.join(OUTPUT_DIR, "{}.q.onnx".format(model)),
        config_save_to=os.path.join(OUTPUT_DIR, "{}.json".format(model)),
    )
