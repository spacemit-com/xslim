#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import functools
import importlib
import os
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .xslim_setting import CalibrationParameterSetting, InputParameterSetting


class XSlimDataset(Dataset):
    def __init__(self, calibration_parameters: CalibrationParameterSetting):
        self.calibration_parameters = calibration_parameters
        input_parametres = self.calibration_parameters.input_parametres

        self._data_list = []
        self._data_dict = OrderedDict()
        min_size = 100000
        for input_item in input_parametres:
            data_list_path = input_item.data_list_path
            input_name = input_item.input_name
            if isinstance(data_list_path, str):
                relate_dir = os.path.dirname(data_list_path)
                with open(data_list_path, "r") as fp:
                    data_list_item = fp.readlines()
                    self._data_dict[input_name] = [
                        os.path.join(relate_dir, item.strip()) for item in data_list_item if len(item.strip()) > 0
                    ]
            else:
                self._data_dict[input_name] = [torch.zeros(input_item.input_shape).to(getattr(torch, input_item.dtype))]

            assert len(self._data_dict[input_name]), "Calibration input {} finds 0 items.".format(input_name)

            # 超过1个输入可以广播
            if len(self._data_dict[input_name]) == 1 and len(input_parametres) > 1:
                pass
            else:
                min_size = min(len(self._data_dict[input_name]), min_size)

        self.auto_batch_size = (
            input_parametres[0].input_shape[0]
            if len(input_parametres) == 1 and input_parametres[0].file_type == "img"
            else 1
        )

        self.auto_batch_size = (
            calibration_parameters.calibration_batch_size
            if calibration_parameters.calibration_batch_size > 1
            else self.auto_batch_size
        )

        min_size = (min_size // self.auto_batch_size) * self.auto_batch_size

        for i in range(min_size):
            data_iter = OrderedDict()
            for k, v in self._data_dict.items():
                data_iter[k] = v[0] if len(v) == 1 else v[i]
            self._data_list.append(data_iter)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]


class PTImagenetPreprocess:
    def __init__(self, out_height, out_width, mean_value, std_value):
        self.out_height = out_height
        self.out_width = out_width
        self.mean_value = mean_value
        self.std_value = std_value

    def __call__(self, img):
        import torchvision.transforms.functional as F

        img = Image.fromarray(img)
        img = F.resize(img, int(self.out_height / 0.875), Image.BILINEAR)
        img = F.center_crop(img, self.out_height)
        img = F.pil_to_tensor(img).to(torch.float32)
        img = F.normalize(img, mean=self.mean_value, std=self.std_value, inplace=False)
        return img


class ImagenetPreprocess:
    def resize_with_aspectratio(
        self, img: np.ndarray, out_height: int, out_width: int, scale: float = 87.5, inter_pol=cv2.INTER_LINEAR
    ):
        height, width, _ = img.shape
        new_height = int(100.0 * out_height / scale)
        new_width = int(100.0 * out_width / scale)
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation=inter_pol)
        return img

    def center_crop(self, img: np.ndarray, out_height: int, out_width: int):
        height, width, _ = img.shape
        left = int((width - out_width) / 2)
        right = int((width + out_width) / 2)
        top = int((height - out_height) / 2)
        bottom = int((height + out_height) / 2)
        img = img[top:bottom, left:right]
        return img

    def __init__(self, out_height: int, out_width: int, mean_value, std_value):
        self.out_height = out_height
        self.out_width = out_width
        self.mean_value = mean_value
        self.std_value = std_value

    def __call__(self, img):
        cv2_interpol = cv2.INTER_AREA
        img = self.resize_with_aspectratio(img, self.out_height, self.out_width, inter_pol=cv2_interpol)
        img = self.center_crop(img, self.out_height, self.out_width)
        img = np.asarray(img, dtype="float32")
        means = np.array(self.mean_value, dtype=np.float32) if self.mean_value is not None else 0
        stds = np.array(self.std_value, dtype=np.float32) if self.std_value is not None else 1.0
        img = (img - means) / stds
        img = img.transpose([2, 0, 1])
        return torch.from_numpy(img)


class CalibrationCollect:
    def __init__(self, input_parametres: Sequence[InputParameterSetting], calibration_device: str = "cuda") -> None:
        self.input_parametres = input_parametres
        self.calibration_device = calibration_device

        self.input_info_dict: Dict[str, InputParameterSetting] = OrderedDict()
        self.imagenet_transforms: Dict[str, Callable] = OrderedDict()
        self.custom_transforms: Dict[str, Callable] = OrderedDict()

        for input_info in self.input_parametres:
            self.input_info_dict[input_info.input_name] = input_info

            input_shape = input_info.input_shape
            mean_value = input_info.mean_value
            std_value = input_info.std_value
            calib_dataset_type = input_info.preprocess_file
            self.custom_transforms[input_info.input_name] = None
            self.imagenet_transforms[input_info.input_name] = None
            if calib_dataset_type == "PT_IMAGENET":
                self.imagenet_transforms[input_info.input_name] = transforms.Compose(
                    [PTImagenetPreprocess(input_shape[-2], input_shape[-1], mean_value, std_value)]
                )
            elif calib_dataset_type == "IMAGENET":
                self.imagenet_transforms[input_info.input_name] = transforms.Compose(
                    [ImagenetPreprocess(input_shape[-2], input_shape[-1], mean_value, std_value)]
                )
            elif isinstance(calib_dataset_type, str):
                module_file, func_name = calib_dataset_type.split(":")
                if os.path.dirname(module_file) not in sys.path:
                    sys.path.append(os.path.dirname(module_file))
                module_name = os.path.splitext(os.path.basename(module_file))[0]

                custom_module = importlib.import_module(module_name)
                self.custom_transforms[input_info.input_name] = getattr(custom_module, func_name)

    def __call__(self, data_item) -> Any:
        @functools.lru_cache(maxsize=(1024 * 1024 * 256))
        def calibration_data_collect(
            collector: CalibrationCollect,
            file_names: Tuple[str],
            tensor_name: str,
        ) -> torch.Tensor:
            input_info = collector.input_info_dict[tensor_name]
            if isinstance(collector.custom_transforms[tensor_name], Callable):
                return collector.custom_transforms[tensor_name](file_names, input_info.to_dict()).to(
                    getattr(torch, input_info.dtype)
                )

            file_type = input_info.file_type
            file_type = input_info.file_type
            color_format = input_info.color_format
            input_shape = input_info.input_shape
            mean_value = np.array(input_info.mean_value) if input_info.mean_value is not None else 0
            std_value = np.array(input_info.std_value) if input_info.std_value is not None else 1
            batch_list = []
            for batch_item in file_names:
                if isinstance(batch_item, torch.Tensor):
                    batch_list.append(batch_item)
                    continue
                elif file_type == "img":
                    if collector.imagenet_transforms[k] is not None:
                        img = cv2.imread(batch_item)
                        if color_format == "rgb":
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = collector.imagenet_transforms[k](img)
                        img = torch.unsqueeze(img, 0)
                        batch_list.append(img)
                    else:
                        img = cv2.imread(batch_item)
                        if len(img.shape) < 3:
                            # 灰度图
                            img = np.expand_dims(img, -1)
                        elif color_format == "rgb":
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (input_shape[-1], input_shape[-2]), interpolation=cv2.INTER_AREA)
                        img = img.astype(np.float32)
                        img = (img - mean_value) / std_value
                        img = np.transpose(img, (2, 0, 1))
                        img = torch.unsqueeze(torch.from_numpy(img), 0)
                        batch_list.append(img)
                elif file_type == "npy":
                    img = np.load(batch_item)
                    batch_list.append(torch.from_numpy(img))
                elif file_type == "raw":
                    img = np.fromfile(batch_item, dtype=np.float32)
                    batch_list.append(torch.from_numpy(img))
                else:
                    raise NotImplementedError("Calibration file type {}".format(file_type))

            return torch.cat(batch_list, dim=0).to(getattr(torch, input_info.dtype))

        dst_data_item = {}
        for k, v in data_item.items():
            dst_data_item[k] = calibration_data_collect(self, tuple(v), k)
        return dst_data_item
