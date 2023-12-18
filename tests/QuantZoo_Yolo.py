# Test Quantization System Performance on Detection Models with Coco Dataset
CONFIGS = [
    {
        "Model": "yolov6p5_n",
        "Output": ["/Concat_5_output_0", "/Concat_4_output_0"],
    },
    {
        "Model": "yolov6p5_t",
        "Output": ["/Concat_5_output_0", "/Concat_4_output_0"],
    },
    {
        "Model": "yolov5s6_n",
        "Output": [
            "/baseModel/head_module/convs_pred.1/Conv_output_0",
            "/baseModel/head_module/convs_pred.2/Conv_output_0",
            "/baseModel/head_module/convs_pred.0/Conv_output_0",
        ],
    },
    {
        "Model": "yolov5s6_s",
        "Output": [
            "/baseModel/head_module/convs_pred.1/Conv_output_0",
            "/baseModel/head_module/convs_pred.2/Conv_output_0",
            "/baseModel/head_module/convs_pred.0/Conv_output_0",
        ],
    },
    {
        "Model": "yolov7p5_tiny",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
    },
    {
        "Model": "yolov7p5_l",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
    },
    {
        "Model": "yolox_s",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
        "std_value": [1, 1, 1],
    },
    {
        "Model": "yolox_tiny",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0", "/Concat_6_output_0"],
        "std_value": [1, 1, 1],
    },
    {
        "Model": "ppyoloe_m",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
        "mean_value": [123.675, 116.28, 103.53],
        "std_value": [255, 255, 255],
    },
    {
        "Model": "ppyoloe_s",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
        "mean_value": [123.675, 116.28, 103.53],
        "std_value": [255, 255, 255],
    },
    {
        "Model": "ppyoloe_s",
        "Output": ["/Concat_4_output_0", "/Concat_5_output_0"],
        "mean_value": [123.675, 116.28, 103.53],
        "std_value": [255, 255, 255],
    },
    {
        "Model": "ssd-mobilenet-300x300",
        "input_shape": [1, 3, 300, 300],
        "mean_value": [123.675, 116.28, 103.53],
        "std_value": [255, 255, 255],
    },
    {
        "Model": "ssd-resnet34-1200x1200",
        "Output": ["Concat_470", "Concat_471"],
        "input_shape": [1, 3, 1200, 1200],
        "mean_value": [123.675, 116.28, 103.53],
        "std_value": [255, 255, 255],
    },
    {
        "Model": "retinanet-800x800",
        "input_shape": [1, 3, 800, 800],
        "mean_value": [123.675, 116.28, 103.53],
        "std_value": [255, 255, 255],
    },
]

from typing import Callable, Optional, Sequence
import os
import json
import torch
import numpy as np
import cv2
from ppq.api import export_ppq_graph
import ppq.lib as PFL
from ppq.api import ENABLE_CUDA_KERNEL, load_onnx_graph, export_ppq_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
import xquant
from ppq.IR import GraphFormatter
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import onnx_tool
import onnx

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_dir",
    required=False,
    default="/home/huangjinghui/1_workspace/2_ppq/PPQQuantZoo/QuantZoo",
    help="Path to the QuantZoo Base directory.",
)
parser.add_argument(
    "--output_dir",
    required=False,
    default="/home/huangjinghui/1_workspace/2_ppq/Output/detection",
    help="Path to the Output directory.",
)
parser.add_argument("--filter", required=False, default="retinanet-800x800", help="model name filter.")
parser.add_argument("--batch_size", required=False, default=1, help="batch_size.")
parser.add_argument("--device", required=False, default="cuda", help="device.")
parser.add_argument("--quant_disable", action="store_true", help="quant_disable.")
parser.add_argument("--eval_fp", action="store_true", help="eval_fp.")
parser.add_argument("--eval_quant", action="store_true", help="eval_quant.")


def resize_and_pad(image: np.ndarray, allow_scale_up: bool = False):
    """
    Resize single image to 640*640 with pad on side.
    """
    EXCEPT_W = 640
    EXCEPT_H = 640
    image_shape = image.shape[:2]  # height, width

    # Scale ratio (new / old)
    ratio = min(EXCEPT_H / image_shape[0], EXCEPT_W / image_shape[1])

    # only scale down, do not scale up (for better test mAP)
    if not allow_scale_up:
        ratio = min(ratio, 1.0)

    ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

    # compute the best size of the image
    no_pad_shape = (int(round(image_shape[0] * ratio[0])), int(round(image_shape[1] * ratio[1])))

    # padding height & width
    padding_h, padding_w = [EXCEPT_H - no_pad_shape[0], EXCEPT_W - no_pad_shape[1]]

    if image_shape != no_pad_shape:
        # compare with no resize and padding size
        image = cv2.resize(image, (no_pad_shape[1], no_pad_shape[0]))

    # padding
    top_padding = 0
    left_padding = 0
    bottom_padding = padding_h
    right_padding = padding_w

    if top_padding != 0 or bottom_padding != 0 or left_padding != 0 or right_padding != 0:
        image = np.pad(
            image, [(top_padding, bottom_padding), (left_padding, right_padding), (0, 0)], "constant", constant_values=0
        )

    return image, ratio[0], [top_padding, left_padding, bottom_padding, right_padding]


class CocoDetectionDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        self._imgs = [img for img in os.listdir(root) if img.endswith(".jpg")]
        print(f"{len(self._imgs)} imgs has been loaded.")
        super().__init__(root, transforms, transform, target_transform)

    def _load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index: int):
        """
        return a coco data sample:
        [img: torch.Tensor, path: str, scale_factor: float, padding: int4]

        img: [batch, 3, 640, 640]
        path: img path
        scale_factor: resizing factor
        padding: padding size
        """
        path = self._imgs[index]
        img = self._load_image(path)

        img, scale_factor, padding = resize_and_pad(img)
        img = torch.from_numpy(img).permute((2, 0, 1)).to(torch.float32)
        return [img, path, scale_factor, padding]

    def __len__(self) -> int:
        return len(self._imgs)


def load_coco_detection_dataset(data_dir: str, batchsize: int = 1, shuffle: bool = False) -> DataLoader:
    # Define your dataset
    data_dir = data_dir
    dataset = CocoDetectionDataset(root=data_dir)

    # Create a dataloader
    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0)

    return data_loader


def xyxy2xywh(bbox: Sequence[int], x_ratio: float, y_ratio: float):
    return [
        bbox[0] / x_ratio,
        bbox[1] / y_ratio,
        (bbox[2] - bbox[0]) / x_ratio,
        (bbox[3] - bbox[1]) / y_ratio,
    ]


def coco80_to_coco91_class():
    """用在test.py中   从80类映射到91类的coco索引 取得对应的class id
    将80个类的coco索引换成91类的coco索引
    :return x: 为80类的每一类在91类中的位置
    """
    x = [i for i in range(1, 90)]
    return x


def encode_result(result: dict, img_id: str, x_resize_ratio: float, y_resize_ratio: float) -> dict:
    """Convert detection results to COCO json style.

    Param result should contains 3 elements, namely:
        *. bbox
        *. score
        *. label

    """
    data = dict()
    data["image_id"] = img_id
    data["bbox"] = xyxy2xywh(result["bbox"], x_resize_ratio, y_resize_ratio)
    data["score"] = float(result["score"])
    data["category_id"] = coco80_to_coco91_class()[int(result["label"])]
    return data


def coco_eval(pred_file: str, ann_file: str):
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(pred_file)  # 自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def evaluate_ppq_module_with_coco(
    ann_file: str,
    output_file: str,
    executor: TorchExecutor,
    dataloader: DataLoader,
    collate_fn,
) -> float:
    with open(ann_file, mode="r", encoding="utf-8") as file:
        j_obj = json.load(file)

    filename_to_id, results = {}, []
    images = j_obj["images"]  # a list of image descriptions

    for img in images:
        file_name = img["file_name"]
        filename_to_id[file_name] = img["id"]

    for img_batch, paths, resize_ratios, padding in tqdm(dataloader, desc="Evaluation"):
        bboxes, labels, scores = executor.forward(
            collate_fn([img_batch]), output_names=["boxes", "labels", "/Split_output_1"]
        )
        bboxes = bboxes.cpu()
        labels = labels.cpu()
        scores = scores.cpu()

        for sample_bbox, sample_label, sample_score, sample_resize, filename in zip(
            bboxes, labels, scores, resize_ratios, paths
        ):
            for bbox, label, score in zip(sample_bbox, sample_label, sample_score):
                if label == -1:
                    continue
                encoded = encode_result(
                    {"bbox": bbox.tolist(), "score": score.item(), "label": label.item()},
                    img_id=filename_to_id[filename],
                    x_resize_ratio=sample_resize.item(),
                    y_resize_ratio=sample_resize.item(),
                )

                results.append(encoded)

    with open(file=output_file, mode="w", encoding="utf-8") as file:
        json.dump(results, file)

    coco_eval(pred_file=output_file, ann_file=ann_file)


if __name__ == "__main__":
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_dir = os.path.join(args.base_dir, "Model", "detection")
    calib_dir = os.path.join(args.base_dir, "Data", "Coco", "Calib")
    test_dir = os.path.join(args.base_dir, "Data", "Coco", "Test")

    TEST_ANN_FILE = os.path.join(args.base_dir, "Data/Coco/Test/DetectionAnnotation.json")
    PRED_ANN_FILE = os.path.join(args.base_dir, "Data/Coco/Test/DetectionPrediction.json")

    test_batch_size = args.batch_size
    collect_device = args.device

    model_filter = args.filter
    if model_filter is not None:
        MODEL_FILTER = set([i for i in model_filter.strip().split(";") if len(i) > 0])
    else:
        MODEL_FILTER = set()

    demo_json = {
        "model_parameters": {
            "onnx_model": "yolov6p5_n.onnx",
            "output_prefix": "yolov6p5_n.q",
            "working_dir": "temp_output",
        },
        "calibration_parameters": {
            "calibration_step": 100,
            "calibration_device": "cuda",
            "calibration_type": "default",
            "input_parametres": [
                {
                    "input_name": "input.1",
                    "input_shape": [1, 3, 640, 640],
                    "file_type": "img",
                    "mean_value": [0, 0, 0],
                    "std_value": [255, 255, 255],
                    "data_list_path": os.path.join(args.base_dir, "Data", "Coco", "calib_img_list.txt"),
                }
            ],
        },
        "quantization_parameters": {"precision_level": 0},
    }

    for config in CONFIGS:
        model = config["Model"]

        if len(MODEL_FILTER) > 0:
            if model not in MODEL_FILTER:
                continue
        print(f"Ready to run quant benchmark on {model}")

        change_output_vars = config.get("Output", [])
        change_input_vars = config.get("Input", [])

        input_model_path = os.path.join(model_dir, model + ".onnx")
        opt_model_path = input_model_path
        truncate_model_path = os.path.join(model_dir, model + "_truncate.onnx")

        custom_transforms = None
        custom_loader = None
        mean_value = config.get("mean_value", [0, 0, 0])
        std_value = config.get("std_value", [255, 255, 255])
        input_shape = config.get("input_shape", [1, 3, 640, 640])
        preprocess_file = config.get("preprocess_file", "PT_IMAGENET")
        calibration_type = config.get("calibration_type", "default")
        auto_finetune_level = config.get("auto_finetune_level", None)

        float_graph = load_onnx_graph(onnx_import_file=input_model_path)

        xquant.GraphLegalized(float_graph)()

        if len(change_output_vars) > 0:
            # truncate graph
            float_graph.outputs.clear()
            editor = GraphFormatter(float_graph)
            for var in change_output_vars:
                float_graph.mark_variable_as_graph_output(float_graph.variables[var])
            editor.delete_isolated()

            export_ppq_graph(
                graph=float_graph,
                platform=TargetPlatform.ONNXRUNTIME,
                graph_save_to=truncate_model_path,
            )

            opt_model_path = truncate_model_path

        # if "ssd-mobilenet-300x300" == model:
        #    onnx_model = onnx.load(opt_model_path)
        #    onnx_graph = onnx_tool.loadmodel(onnx_model)
        #    onnx_graph.graph.tensormap[onnx_graph.graph.input[0]].shape = ["N", 3, 640, 640]
        #    onnx_graph.mproto.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        #    onnx_graph.save_model(opt_model_path)
        #    float_graph = load_onnx_graph(onnx_import_file=opt_model_path)
        #    executor = TorchExecutor(graph=float_graph)
        #    executor.tracing_operation_meta(torch.zeros(size=input_shape))
        #    export_ppq_graph(
        #        graph=float_graph,
        #        platform=TargetPlatform.ONNXRUNTIME,
        #        graph_save_to=opt_model_path,
        #    )

        test_loader = load_coco_detection_dataset(data_dir=test_dir, batchsize=test_batch_size)

        demo_json["model_parameters"]["onnx_model"] = opt_model_path
        demo_json["model_parameters"]["output_prefix"] = "{}.q".format(model)
        demo_json["model_parameters"]["working_dir"] = output_dir
        demo_json["calibration_parameters"]["calibration_type"] = calibration_type
        demo_json["calibration_parameters"]["calibration_device"] = collect_device
        demo_json["calibration_parameters"]["input_parametres"][0]["input_shape"] = input_shape
        demo_json["calibration_parameters"]["input_parametres"][0]["mean_value"] = mean_value
        demo_json["calibration_parameters"]["input_parametres"][0]["std_value"] = std_value
        demo_json["calibration_parameters"]["input_parametres"][0]["input_name"] = list(float_graph.inputs.keys())[0]
        demo_json["calibration_parameters"]["input_parametres"][0]["preprocess_file"] = preprocess_file
        if isinstance(auto_finetune_level, int):
            demo_json["quantization_parameters"]["auto_finetune_level"] = auto_finetune_level

        if not args.quant_disable:
            quantized_graph = xquant.quantize_onnx_model(demo_json)

        def coco_eval_fn(x):
            img = x[0]
            img = (img - torch.tensor(mean_value).to(torch.float32).reshape(-1, 1, 1)) / torch.tensor(std_value).to(
                torch.float32
            ).reshape(-1, 1, 1)
            return img

        if args.eval_fp:
            executor = TorchExecutor(graph=float_graph)
            executor.tracing_operation_meta(torch.zeros(size=input_shape))
            evaluate_ppq_module_with_coco(
                ann_file=TEST_ANN_FILE,
                output_file=PRED_ANN_FILE,
                executor=executor,
                dataloader=test_loader,
                collate_fn=coco_eval_fn,
            )

        output_model_path = os.path.join(output_dir, "{}.onnx".format(demo_json["model_parameters"]["output_prefix"]))

        if args.eval_quant:
            # 需要使用导出后的ONNX模型推理 保持与ORT的算子一致
            quantized_export_graph = load_onnx_graph(onnx_import_file=output_model_path)
