# Test Quantization System Performance on Detection Models with Coco Dataset
CONFIGS = [
    {
        "Model": "yolov7_tiny_dyn",
        "truncate_var_names": [
            "onnx::Shape_540",
            "onnx::Shape_597",
            "onnx::Shape_654",
        ],
    },
    {
        "Model": "yolov6_n_dyn",
        "truncate_var_names": [
            "onnx::Shape_287",
            "onnx::Transpose_288",
            "onnx::Shape_301",
            "onnx::Transpose_302",
            "onnx::Shape_315",
            "onnx::Transpose_316",
        ],
    },
    {
        "Model": "yolov6_s_dyn",
        "truncate_var_names": [
            "onnx::Shape_287",
            "onnx::Transpose_288",
            "onnx::Shape_301",
            "onnx::Transpose_302",
            "onnx::Shape_315",
            "onnx::Transpose_316",
        ],
    },
    {
        "Model": "yolov5_n_dyn",
        "truncate_var_names": [
            "onnx::Shape_610",
            "onnx::Shape_665",
            "onnx::Shape_720",
        ],
    },
    {
        "Model": "yolov5_s_dyn",
        "truncate_var_names": [
            "onnx::Shape_610",
            "onnx::Shape_665",
            "onnx::Shape_720",
        ],
    },
    {
        "Model": "yolov3_mobilenetv2",
        "truncate_var_names": [
            "868",
            "978",
            "1088",
        ],
        "mean_value": [123.675, 116.28, 103.53],
        "std_value": [58.395, 57.12, 57.375],
    },
    {
        "Model": "yolov3_darknet53_dyn",
        "truncate_var_names": [
            "684",
            "688",
            "692",
        ],
    },
    {
        "Model": "yolox_s",
        "mean_value": [0, 0, 0],
        "std_value": [1, 1, 1],
        "input_shape": [1, 3, 640, 640],
        "truncate_var_names": [
            "807",
            "808",
            "809",
            "826",
            "827",
            "828",
            "845",
            "846",
            "847",
        ],
    },
    {
        "Model": "ppyoloe_m_dyn",
        "truncate_var_names": ["pred_bboxes", "y", "flatten_priors"],
        "skip_onnxsim": True,
    }
    # {
    #    "Model": "ppyoloe_m",
    #    "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
    #    "mean_value": [123.675, 116.28, 103.53],
    #    "std_value": [255, 255, 255],
    # },
    # {
    #    "Model": "ppyoloe_s",
    #    "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
    #    "mean_value": [123.675, 116.28, 103.53],
    #    "std_value": [255, 255, 255],
    # },
    # {
    #    "Model": "ppyoloe_s",
    #    "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
    #    "mean_value": [123.675, 116.28, 103.53],
    #    "std_value": [255, 255, 255],
    # },
    # {
    #    "Model": "ssd-mobilenet-300x300",
    #    "input_shape": [1, 3, 300, 300],
    #    "mean_value": [123.675, 116.28, 103.53],
    #    "std_value": [255, 255, 255],
    # },
    # {
    #    "Model": "ssd-resnet34-1200x1200",
    #    "Output": ["Concat_470", "Concat_471"],
    #    "input_shape": [1, 3, 1200, 1200],
    #    "mean_value": [123.675, 116.28, 103.53],
    #    "std_value": [255, 255, 255],
    # },
    # {
    #    "Model": "retinanet-800x800",
    #    "input_shape": [1, 3, 800, 800],
    #    "mean_value": [123.675, 116.28, 103.53],
    #    "std_value": [255, 255, 255],
    # },
]

from typing import Callable, Optional, Sequence
import os
import json
import torch
import numpy as np
import cv2
from ppq.api import load_onnx_graph
import xquant
from xquant import xquant_info
import argparse
from tqdm import tqdm
import coco_eval_helper
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_dir",
    required=False,
    default="/home/share/modelzoo",
    help="Path to the QuantZoo Base directory.",
)
parser.add_argument(
    "--output_dir",
    required=False,
    default="/home/huangjinghui/1_workspace/2_ppq/Output/detection",
    help="Path to the Output directory.",
)
parser.add_argument("--filter", required=False, default="ppyoloe_m_dyn", help="model name filter.")
parser.add_argument("--batch_size", required=False, default=1, help="batch_size.")
parser.add_argument("--device", required=False, default="cuda", help="device.")
parser.add_argument("--quant_disable", action="store_true", help="quant_disable.")
parser.add_argument("--eval_fp", action="store_true", help="eval_fp.")
parser.add_argument("--eval_quant", action="store_true", help="eval_quant.")


def eval_yolo_results(onnx_model_path, test_loader, test_dir, mean_value, std_value, pred_file, ann_file):
    with open(ann_file, mode="r", encoding="utf-8") as file:
        anno_obj = json.load(file)
    filename_to_id = {}
    for img in anno_obj["images"]:
        file_name = img["file_name"]
        filename_to_id[file_name] = img["id"]
    onnx_model = onnx.load(onnx_model_path)
    test_session = ort.InferenceSession(onnx_model.SerializeToString())
    eval_resuluts = []
    for img_batch, paths, resize_ratios, padding in tqdm(test_loader, desc="Evaluation"):
        ori_img = cv2.imread(os.path.join(test_dir, paths[0]))
        img_batch = (img_batch - torch.tensor(mean_value, dtype=torch.float32).reshape(-1, 1, 1)) / torch.tensor(
            std_value, dtype=torch.float32
        ).reshape(-1, 1, 1)
        outputs = test_session.run(None, {test_session.get_inputs()[0].name: img_batch.numpy()})
        boxes, labels = outputs
        if labels.size < 1:
            continue
        boxes = boxes.reshape(-1, boxes.shape[-1])
        labels = labels.reshape(labels.shape[-1])
        valid_top = padding[0]
        valid_left = padding[1]
        valid_bottom = input_shape[-2] - padding[2]
        valid_right = input_shape[-1] - padding[3]
        scores = boxes[:, -1:]
        boxes = boxes[:, :-1]
        boxes[:, 0] = np.minimum(np.maximum(boxes[:, 0], valid_left), valid_right)
        boxes[:, 1] = np.minimum(np.maximum(boxes[:, 1], valid_top), valid_bottom)
        boxes[:, 2] = np.minimum(np.maximum(boxes[:, 2], valid_left), valid_right)
        boxes[:, 3] = np.minimum(np.maximum(boxes[:, 3], valid_top), valid_bottom)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes /= resize_ratios

        for box, conf, id in zip(boxes, scores, labels):
            id = int(id)
            conf = float(conf)
            x, y, w, h = [int(i) for i in box[:4]]
            cls_label = coco_eval_helper.coco_class_names[id]
            encoded = {}
            encoded["image_id"] = filename_to_id[paths[0]]
            encoded["bbox"] = [x, y, w, h]
            encoded["score"] = conf
            encoded["category_id"] = cls_label["id"]
            eval_resuluts.append(encoded)

    with open(file=pred_file, mode="w", encoding="utf-8") as file:
        json.dump(eval_resuluts, file)

    coco_eval_helper.coco_eval(pred_file=pred_file, ann_file=ann_file)


if __name__ == "__main__":
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_dir = os.path.join(args.base_dir, "dataset", "Coco", "Test")

    TEST_ANN_FILE = os.path.join(args.base_dir, "dataset", "Coco", "Test", "DetectionAnnotation.json")

    test_batch_size = args.batch_size
    collect_device = args.device

    model_filter = args.filter
    if model_filter is not None:
        MODEL_FILTER = set([i for i in model_filter.strip().split(";") if len(i) > 0])
    else:
        MODEL_FILTER = set()

    for config in CONFIGS:
        model = config["Model"]

        if len(MODEL_FILTER) > 0:
            if model not in MODEL_FILTER:
                continue
        print(f"Ready to run quant benchmark on {model}")

        mean_value = config.get("mean_value", [0, 0, 0])
        std_value = config.get("std_value", [255, 255, 255])
        input_shape = config.get("input_shape", [1, 3, 320, 320])
        preprocess_file = config.get("preprocess_file", "PT_IMAGENET")
        calibration_type = config.get("calibration_type", "default")
        finetune_level = config.get("finetune_level", 1)
        color_format = config.get("color_format", "rgb")
        truncate_var_names = config.get("truncate_var_names", [])
        input_model_path = os.path.join(args.base_dir, "detection", model, "{}.onnx".format(model))

        test_loader = coco_eval_helper.load_coco_detection_dataset(
            data_dir=test_dir,
            batchsize=1,
            rgb_format=color_format == "rgb",
            except_h=input_shape[-2],
            except_w=input_shape[-1],
        )

        demo_json = {
            "model_parameters": {
                "onnx_model": input_model_path,
                "output_prefix": "{}.q".format(model),
                "working_dir": output_dir,
                "skip_onnxsim": config.get("skip_onnxsim", False),
            },
            "calibration_parameters": {
                "calibration_step": 500,
                "calibration_device": "cuda",
                "calibration_type": "default",
                "input_parametres": [
                    {
                        "input_shape": input_shape,
                        "file_type": "img",
                        "mean_value": mean_value,
                        "std_value": std_value,
                        "data_list_path": os.path.join(args.base_dir, "dataset", "Coco", "calib_img_list.txt"),
                    }
                ],
            },
            "quantization_parameters": {
                "precision_level": 0,
                "finetune_level": finetune_level,
                "truncate_var_names": truncate_var_names,
            },
        }

        if args.eval_fp:
            xquant_info(f"Evaluate float Model Accurarcy....")
            eval_yolo_results(
                input_model_path,
                test_loader,
                test_dir,
                mean_value,
                std_value,
                os.path.join(output_dir, "{}_pred_fp.json".format(model)),
                TEST_ANN_FILE,
            )

        if not args.quant_disable:
            quantized_graph = xquant.quantize_onnx_model(demo_json)

        output_model_path = os.path.join(output_dir, "{}.onnx".format(demo_json["model_parameters"]["output_prefix"]))

        if args.eval_quant:
            xquant_info(f"Evaluate quantized Model Accurarcy....")
            eval_yolo_results(
                output_model_path,
                test_loader,
                test_dir,
                mean_value,
                std_value,
                os.path.join(output_dir, "{}_pred_quant.json".format(model)),
                TEST_ANN_FILE,
            )
