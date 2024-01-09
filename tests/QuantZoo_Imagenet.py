# Test Quantization System Performace on Image Classification Models with ILSVRC2012 Dataset
CONFIGS = [
    {
        "Model": "resnet18",
        "finetune_level": 2,
    },
    {
        "Model": "resnet50",
        "finetune_level": 2,
    },
    {
        "Model": "resnet50-v1.5",
        "mean_value": [123.68, 116.78, 103.94],
        "std_value": [1, 1, 1],
        "preprocess_file": "IMAGENET",
        "finetune_level": 2,
    },
    {
        "Model": "resnext50",
    },
    {
        "Model": "seresnet50",
        "finetune_level": 2,
    },
    {
        "Model": "mobilenet_v1",
        "mean_value": [127.5, 127.5, 127.5],
        "std_value": [127.5, 127.5, 127.5],
        "preprocess_file": "IMAGENET",
        "finetune_level": 2,
    },
    {
        "Model": "mobilenet_v2",
        "finetune_level": 2,
    },
    {
        "Model": "mobilenet_v3_large",
        "finetune_level": 2,
    },
    {"Model": "mobilenet_v3_small"},
    {
        "Model": "efficientnet_v1_b0",
        "finetune_level": 2,
    },
    {
        "Model": "efficientnet_v1_b1",
        "finetune_level": 2,
    },
    {
        "Model": "efficientnet_v2_s",
        "finetune_level": 2,
    },
    {
        "Model": "mnasnet0_5",
        "finetune_level": 2,
    },
    {
        "Model": "mnasnet1_0",
        "finetune_level": 2,
    },
    {
        "Model": "repvgg",
    },
    {
        "Model": "v100_gpu64@5ms_top1@71.6_finetune@25",
        "finetune_level": 2,
    },
    {
        "Model": "v100_gpu64@6ms_top1@73.0_finetune@25",
        "finetune_level": 2,
    },
    {
        "Model": "shufflenet_v2_x1_0",
        "finetune_level": 2,
    },
    {"Model": "lcnet_050"},
    {"Model": "lcnet_100"},
    {
        "Model": "inception_v1",
        "mean_value": [104, 117, 123],
        "std_value": [1.0, 1.0, 1.0],
        "preprocess_file": "IMAGENET",
        "color_format": "bgr",
        "finetune_level": 2,
    },
    {
        "Model": "inception_resnet_v2",
        "input_shape": [1, 3, 299, 299],
        "mean_value": [127.5, 127.5, 127.5],
        "std_value": [127.5, 127.5, 127.5],
        "finetune_level": 2,
    },
    {
        "Model": "inception_v3",
        "input_shape": [1, 3, 299, 299],
        "mean_value": [127.5, 127.5, 127.5],
        "std_value": [127.5, 127.5, 127.5],
        "finetune_level": 2,
    },
    {
        "Model": "squeezenet1.1",
        "finetune_level": 2,
    },
    # {
    #    "Model": "vit_b_16",
    #    "calibration_type": "percentile",
    # },
    # {"Model": "vgg16"},
    # {
    #    "Model": "swin_small_patch4_window7_224",
    #    "calibration_type": "percentile",
    # },
    # {
    #    "Model": "swinv2_small_window8_256",
    #    "calibration_type": "percentile",
    # },
]

from typing import Callable, Optional
import os
import torch
import cv2
import numpy as np
import time
import argparse
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
from ppq.api import load_onnx_graph
import xquant
from xquant.calibration_helper import PTImagenetPreprocess, ImagenetPreprocess
from ppq.api import export_ppq_graph
from ppq.executor.torch import TorchExecutor
from torch.utils.data.dataset import Subset
from ppq.IR import BaseGraph
import torchvision.transforms as transforms

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
    default="output",
    help="Path to the Output directory.",
)
parser.add_argument("--filter", required=False, default="mobilenet_v2", help="model name filter.")
parser.add_argument("--batch_size", required=False, default=1, help="batch_size.")
parser.add_argument("--device", required=False, default="cuda", help="device.")
parser.add_argument("--quant_disable", action="store_true", help="quant_disable.")
parser.add_argument("--eval_fp", action="store_true", help="eval_fp.")
parser.add_argument("--eval_quant", action="store_true", help="eval_quant.")


def imagenet_preprocess(input_shape, mean_value, std_value):
    return transforms.Compose(
        [
            ImagenetPreprocess(input_shape[-2], input_shape[-1], mean_value, std_value),
        ]
    )


def pytorch_imagenet_preprocess(input_shape, mean_value, std_value):
    return transforms.Compose([PTImagenetPreprocess(input_shape[-2], input_shape[-1], mean_value, std_value)])


def load_imagenet_from_directory(
    directory: str,
    subset: int = None,
    batchsize: int = 32,
    shuffle: bool = False,
    require_label: bool = True,
    num_of_workers: int = 12,
    custom_transforms: Callable = None,
    custom_loader: Callable = None,
) -> torch.utils.data.DataLoader:
    """
    一套十分标准的 Imagenet 数据加载流程，
    directory: 数据加载的位置
    subset: 如果设置为非空，则从数据集中抽取subset大小的子集
    require_label: 是否需要标签
    shuffle: 是否打乱数据集
    """
    dataset = datasets.ImageFolder(
        directory,
        custom_transforms
        if custom_transforms
        else transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        loader=custom_loader if custom_loader else None,
    )

    if subset:
        dataset = Subset(dataset, indices=[_ for _ in range(0, subset)])
    if require_label:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_of_workers,
            pin_memory=False,
            drop_last=True,  # onnx 模型不支持动态 batchsize，最后一个批次的数据尺寸可能不对齐，因此丢掉最后一个批次的数据
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_of_workers,
            pin_memory=False,
            collate_fn=lambda x: torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0),
            drop_last=False,  # 不需要标签的数据为 calib 数据，无需 drop
        )


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_ppq_module_with_imagenet(
    model: BaseGraph,
    imagenet_validation_dir: str = None,
    batchsize: int = 32,
    device: str = "cuda",
    imagenet_validation_loader: DataLoader = None,
    verbose: bool = True,
):
    """
    一套用来测试 ppq 模块的逻辑，
    直接送入 ppq.IR.BaseGraph 就好了
    """

    executor = TorchExecutor(graph=model, device=device)
    model_forward_function = lambda input_tensor: executor(*[input_tensor])

    return _evaluate_any_module_with_imagenet(
        model_forward_function=model_forward_function,
        batchsize=batchsize,
        device=device,
        imagenet_validation_dir=imagenet_validation_dir,
        imagenet_validation_loader=imagenet_validation_loader,
        verbose=verbose,
    )


def _evaluate_any_module_with_imagenet(
    model_forward_function: Callable,
    imagenet_validation_dir: str,
    batchsize: int = 32,
    device: str = "cuda",
    imagenet_validation_loader: DataLoader = None,
    verbose: bool = True,
) -> float:
    """
    一套十分标准的imagenet测试逻辑
    """

    recorder = {"loss": [], "top1_accuracy": [], "top5_accuracy": [], "batch_time": []}

    if imagenet_validation_loader is None:
        imagenet_validation_loader = load_imagenet_from_directory(
            imagenet_validation_dir, batchsize=batchsize, shuffle=False
        )

    loss_fn = torch.nn.CrossEntropyLoss().to("cpu")

    eval_range = tqdm(
        enumerate(imagenet_validation_loader),
        desc="Evaluating Model...",
        total=len(imagenet_validation_loader),
    )

    for batch_idx, (batch_input, batch_label) in eval_range:
        batch_input = batch_input.to(device)
        batch_label = batch_label.to(device)
        batch_time_mark_point = time.time()

        batch_pred = model_forward_function(batch_input)

        batch_pred_conf = batch_pred
        batch_pred_index = None
        if isinstance(batch_pred, list):
            if len(batch_pred) == 2:
                if not isinstance(batch_pred[0], torch.Tensor):
                    batch_pred = [torch.from_numpy(item).to(device) for item in batch_pred]
                dtypes = [item.dtype for item in batch_pred]
                argmax_idx = dtypes.index(torch.int64)
                batch_pred_conf = batch_pred_conf
                batch_pred_conf = batch_pred[1 - argmax_idx]
                batch_pred_index = batch_pred[argmax_idx]
            elif len(batch_pred) == 1:
                batch_pred_conf = batch_pred[0].reshape(1, -1)

        _, cls_num = batch_pred_conf.size()
        batch_pred_conf = batch_pred_conf[:, cls_num - 1000 :]
        recorder["batch_time"].append(time.time() - batch_time_mark_point)
        recorder["loss"].append(loss_fn(batch_pred_conf.to("cpu"), batch_label.to("cpu")))
        prec1, prec5 = accuracy(torch.tensor(batch_pred_conf).to("cpu"), batch_label.to("cpu"), topk=(1, 5))
        if batch_pred_index is not None:
            prec1 = 100 * torch.sum(batch_pred_index == batch_label + 1) / batch_label.numel()
        recorder["top1_accuracy"].append(prec1.item())
        recorder["top5_accuracy"].append(prec5.item())

        eval_range.desc = "Evaluating Model Prec@1 {:.2f}".format(
            sum(recorder["top1_accuracy"]) / len(recorder["top1_accuracy"])
        )
        if batch_idx % 100 == 0 and verbose:
            print(
                "Test: [{0} / {1}]\t"
                "Prec@1 {top1:.3f} ({top1:.3f})\t"
                "Prec@5 {top5:.3f} ({top5:.3f})".format(
                    batch_idx,
                    len(imagenet_validation_loader),
                    top1=sum(recorder["top1_accuracy"]) / len(recorder["top1_accuracy"]),
                    top5=sum(recorder["top5_accuracy"]) / len(recorder["top5_accuracy"]),
                )
            )

    if verbose:
        print(
            " * Prec@1 {top1:.3f} Prec@5 {top5:.3f}".format(
                top1=sum(recorder["top1_accuracy"]) / len(recorder["top1_accuracy"]),
                top5=sum(recorder["top5_accuracy"]) / len(recorder["top5_accuracy"]),
            )
        )

    return sum(recorder["top1_accuracy"]) / len(recorder["top1_accuracy"])


if __name__ == "__main__":
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_dir = os.path.join(args.base_dir, "classification")
    calib_dir = os.path.join(args.base_dir, "dataset", "Imagenet", "Calib")
    test_dir = os.path.join(args.base_dir, "dataset", "Imagenet", "Test")

    test_batch_size = args.batch_size
    collect_device = args.device

    model_filter = args.filter

    MODEL_FILTER = set(model_filter.strip().split(";")) if model_filter is not None else set()
    MODEL_FILTER = {i for i in MODEL_FILTER if i != ""}

    for config in CONFIGS:
        demo_json = {
            "model_parameters": {
                "onnx_model": "resnet18.onnx",
                "output_prefix": "resnet18.q",
                "working_dir": "temp_output",
            },
            "calibration_parameters": {
                "calibration_step": 500,
                "calibration_device": "cuda",
                "calibration_type": "default",
                "input_parametres": [
                    {
                        "input_name": "input_0",
                        "input_shape": [1, 3, 224, 224],
                        "file_type": "img",
                        "color_format": "rgb",
                        "mean_value": [123.675, 116.28, 103.53],
                        "std_value": [58.395, 57.12, 57.375],
                        "preprocess_file": "PT_IMAGENET",
                        "data_list_path": os.path.join(args.base_dir, "dataset", "Imagenet", "calib_img_list.txt"),
                    }
                ],
            },
            "quantization_parameters": {"precision_level": 0},
        }

        model = config["Model"]

        if len(MODEL_FILTER) > 0:
            if model not in MODEL_FILTER:
                continue
        print(f"Ready to run quant benchmark on {model}")

        input_model_path = os.path.join(model_dir, model, model + ".onnx")

        opt_model_path = input_model_path
        float_graph = load_onnx_graph(onnx_import_file=input_model_path)

        xquant.GraphLegalized(float_graph)()
        mean_value = config.get("mean_value", [123.675, 116.28, 103.53])
        std_value = config.get("std_value", [58.395, 57.12, 57.375])
        input_shape = config.get("input_shape", [1, 3, 224, 224])
        preprocess_file = config.get("preprocess_file", "PT_IMAGENET")
        calibration_type = config.get("calibration_type", "default")
        color_format = config.get("color_format", "rgb")
        finetune_level = config.get("finetune_level", None)
        max_percentile = config.get("max_percentile", None)

        # onnx_model = osg.import_onnx(onnx.load(input_model_path))
        # for idx, in_var in enumerate(onnx_model.inputs):
        #    in_var.name = "input_{}".format(idx)
        #    in_var.shape = input_shape
        #    in_var.dtype = np.float32
        # new_onnx_model = osg.export_onnx(onnx_model)
        # onnx.save_model(new_onnx_model, input_model_path)
        # continue

        custom_transforms = pytorch_imagenet_preprocess(input_shape, mean_value, std_value)
        if preprocess_file == "IMAGENET":
            custom_transforms = imagenet_preprocess(input_shape, mean_value, std_value)

        def custom_loader(x):
            img = cv2.imread(x)
            if color_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        test_loader = load_imagenet_from_directory(
            directory=test_dir,
            batchsize=test_batch_size,
            shuffle=False,
            require_label=True,
            num_of_workers=0,
            custom_transforms=custom_transforms,
            custom_loader=custom_loader,
        )

        demo_json["model_parameters"]["onnx_model"] = opt_model_path
        demo_json["model_parameters"]["output_prefix"] = "{}.q".format(model)
        demo_json["model_parameters"]["working_dir"] = output_dir
        demo_json["calibration_parameters"]["calibration_type"] = calibration_type
        demo_json["calibration_parameters"]["calibration_device"] = collect_device
        demo_json["calibration_parameters"]["input_parametres"][0]["input_shape"] = input_shape
        demo_json["calibration_parameters"]["input_parametres"][0]["mean_value"] = mean_value
        demo_json["calibration_parameters"]["input_parametres"][0]["std_value"] = std_value
        demo_json["calibration_parameters"]["input_parametres"][0]["color_format"] = color_format
        demo_json["calibration_parameters"]["input_parametres"][0]["input_name"] = list(float_graph.inputs.keys())[0]
        demo_json["calibration_parameters"]["input_parametres"][0]["preprocess_file"] = preprocess_file

        precision_level = config.get("precision_level", None)
        if isinstance(precision_level, int):
            demo_json["quantization_parameters"]["precision_level"] = precision_level
        if isinstance(finetune_level, int):
            demo_json["quantization_parameters"]["finetune_level"] = finetune_level
        if isinstance(max_percentile, float):
            demo_json["quantization_parameters"]["max_percentile"] = max_percentile
        if not args.quant_disable:
            quantized_graph = xquant.quantize_onnx_model(demo_json)

        if args.eval_fp:
            print(f"Evaluate float Model Accurarcy....")
            # evaluation
            acc = evaluate_ppq_module_with_imagenet(
                model=float_graph,
                imagenet_validation_loader=test_loader,
                batchsize=test_batch_size,
                device=collect_device,
                verbose=False,
            )
            print(f"Model Classify Accurarcy = {acc: .4f}%")

        output_model_path = os.path.join(output_dir, "{}.onnx".format(demo_json["model_parameters"]["output_prefix"]))

        if args.eval_quant:
            quantized_export_graph = load_onnx_graph(onnx_import_file=output_model_path)

            print(f"Evaluate quantized Model Accurarcy....")
            # evaluation
            acc = evaluate_ppq_module_with_imagenet(
                model=quantized_export_graph,
                imagenet_validation_loader=test_loader,
                batchsize=test_batch_size,
                device=collect_device,
                verbose=False,
            )
            print(f"Model Classify Accurarcy = {acc: .4f}%")
