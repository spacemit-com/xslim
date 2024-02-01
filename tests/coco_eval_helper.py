from typing import Callable, Optional, Sequence
import os
import json
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

coco_class_names = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 10, "name": "traffic light"},
    {"id": 11, "name": "fire hydrant"},
    {"id": 13, "name": "stop sign"},
    {"id": 14, "name": "parking meter"},
    {"id": 15, "name": "bench"},
    {"id": 16, "name": "bird"},
    {"id": 17, "name": "cat"},
    {"id": 18, "name": "dog"},
    {"id": 19, "name": "horse"},
    {"id": 20, "name": "sheep"},
    {"id": 21, "name": "cow"},
    {"id": 22, "name": "elephant"},
    {"id": 23, "name": "bear"},
    {"id": 24, "name": "zebra"},
    {"id": 25, "name": "giraffe"},
    {"id": 27, "name": "backpack"},
    {"id": 28, "name": "umbrella"},
    {"id": 31, "name": "handbag"},
    {"id": 32, "name": "tie"},
    {"id": 33, "name": "suitcase"},
    {"id": 34, "name": "frisbee"},
    {"id": 35, "name": "skis"},
    {"id": 36, "name": "snowboard"},
    {"id": 37, "name": "sports ball"},
    {"id": 38, "name": "kite"},
    {"id": 39, "name": "baseball bat"},
    {"id": 40, "name": "baseball glove"},
    {"id": 41, "name": "skateboard"},
    {"id": 42, "name": "surfboard"},
    {"id": 43, "name": "tennis racket"},
    {"id": 44, "name": "bottle"},
    {"id": 46, "name": "wine glass"},
    {"id": 47, "name": "cup"},
    {"id": 48, "name": "fork"},
    {"id": 49, "name": "knife"},
    {"id": 50, "name": "spoon"},
    {"id": 51, "name": "bowl"},
    {"id": 52, "name": "banana"},
    {"id": 53, "name": "apple"},
    {"id": 54, "name": "sandwich"},
    {"id": 55, "name": "orange"},
    {"id": 56, "name": "broccoli"},
    {"id": 57, "name": "carrot"},
    {"id": 58, "name": "hot dog"},
    {"id": 59, "name": "pizza"},
    {"id": 60, "name": "donut"},
    {"id": 61, "name": "cake"},
    {"id": 62, "name": "chair"},
    {"id": 63, "name": "couch"},
    {"id": 64, "name": "potted plant"},
    {"id": 65, "name": "bed"},
    {"id": 67, "name": "dining table"},
    {"id": 70, "name": "toilet"},
    {"id": 72, "name": "tv"},
    {"id": 73, "name": "laptop"},
    {"id": 74, "name": "mouse"},
    {"id": 75, "name": "remote"},
    {"id": 76, "name": "keyboard"},
    {"id": 77, "name": "cell phone"},
    {"id": 78, "name": "microwave"},
    {"id": 79, "name": "oven"},
    {"id": 80, "name": "toaster"},
    {"id": 81, "name": "sink"},
    {"id": 82, "name": "refrigerator"},
    {"id": 84, "name": "book"},
    {"id": 85, "name": "clock"},
    {"id": 86, "name": "vase"},
    {"id": 87, "name": "scissors"},
    {"id": 88, "name": "teddy bear"},
    {"id": 89, "name": "hair drier"},
    {"id": 90, "name": "toothbrush"},
]


def coco80_to_coco91_class():
    """用在test.py中   从80类映射到91类的coco索引 取得对应的class id
    将80个类的coco索引换成91类的coco索引
    :return x: 为80类的每一类在91类中的位置
    """
    x = [i for i in range(1, 90)]
    return x


def coco_eval(pred_file: str, ann_file: str):
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(pred_file)  # 自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def resize_and_pad(
    image: np.ndarray, allow_scale_up: bool = False, except_h=640, except_w=640
):
    image_shape = image.shape[:2]  # height, width

    # Scale ratio (new / old)
    ratio = min(except_h / image_shape[0], except_w / image_shape[1])

    # only scale down, do not scale up (for better test mAP)
    if not allow_scale_up:
        ratio = min(ratio, 1.0)

    ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

    # compute the best size of the image
    no_pad_shape = (
        int(round(image_shape[0] * ratio[0])),
        int(round(image_shape[1] * ratio[1])),
    )

    # padding height & width
    padding_h, padding_w = [except_h - no_pad_shape[0], except_w - no_pad_shape[1]]

    if image_shape != no_pad_shape:
        # compare with no resize and padding size
        image = cv2.resize(image, (no_pad_shape[1], no_pad_shape[0]))

    # padding
    top_padding = 0
    left_padding = 0
    bottom_padding = padding_h
    right_padding = padding_w

    if (
        top_padding != 0
        or bottom_padding != 0
        or left_padding != 0
        or right_padding != 0
    ):
        image = np.pad(
            image,
            [(top_padding, bottom_padding), (left_padding, right_padding), (0, 0)],
            "constant",
            constant_values=0,
        )
    return image, ratio[0], [top_padding, left_padding, bottom_padding, right_padding]


class CocoDetectionDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        rgb_format: bool = False,
        except_h: int = 640,
        except_w: int = 640,
    ) -> None:
        self._imgs = [img for img in os.listdir(root) if img.endswith(".jpg")]
        print(f"{len(self._imgs)} imgs has been loaded.")
        self._except_h = except_h
        self._except_w = except_w
        self._rgb_format = rgb_format
        super().__init__(root, transforms, transform, target_transform)

    def _load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(os.path.join(self.root, path))
        if self._rgb_format:
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

        img, scale_factor, padding = resize_and_pad(
            img, except_h=self._except_h, except_w=self._except_w
        )
        img = torch.from_numpy(img).permute((2, 0, 1)).to(torch.float32)
        return [img, path, scale_factor, padding]

    def __len__(self) -> int:
        return len(self._imgs)


def load_coco_detection_dataset(
    data_dir: str,
    batchsize: int = 1,
    shuffle: bool = False,
    rgb_format: bool = False,
    except_h: int = 640,
    except_w: int = 640,
) -> DataLoader:
    dataset = CocoDetectionDataset(
        root=data_dir, rgb_format=rgb_format, except_h=except_h, except_w=except_w
    )
    data_loader = DataLoader(
        dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0
    )
    return data_loader
