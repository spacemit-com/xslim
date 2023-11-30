from typing import Any, Union, Dict, Sequence
from collections import OrderedDict
import json
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from PIL import Image
import numpy as np


class XQuantDataset(Dataset):
    def __init__(self, calibration_parameters: dict):
        self.calibration_parameters = calibration_parameters
        input_parametres = self.calibration_parameters.get("input_parametres")
        if not isinstance(input_parametres, Sequence) or len(input_parametres) < 1:
            raise RuntimeError

        self._data_list = []
        self._data_dict = OrderedDict()
        min_size = 1000
        for input_item in input_parametres:
            if not isinstance(input_item, dict):
                raise TypeError
            data_list_path = input_item.get("data_list_path")
            relate_dir = os.path.dirname(data_list_path)
            input_name = input_item.get("input_name")
            with open(data_list_path, "r") as fp:
                data_list_item = fp.readlines()
                self._data_dict[input_name] = [
                    os.path.join(relate_dir, item.strip()) for item in data_list_item if len(item.strip()) > 0
                ]

            min_size = min(len(self._data_dict[input_name]), min_size)

        for i in range(min_size):
            data_iter = OrderedDict()
            for k, v in self._data_dict.items():
                data_iter[k] = v[i]
            self._data_list.append(data_iter)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]


class CalibrationCollect:
    def __init__(self, input_parametres: Sequence, calibration_device: str = "cuda") -> None:
        self.input_parametres = input_parametres
        self.calibration_device = calibration_device
        if not isinstance(self.input_parametres, Sequence) or len(self.input_parametres) < 1:
            raise RuntimeError

        self.input_info_dict = OrderedDict()
        for input_info in self.input_parametres:
            self.input_info_dict[input_info.get("input_name")] = input_info

        if len(self.input_parametres) == 1 and self.input_parametres[0].get("preprocess_file", None) == "IMAGENET":
            input_shape = self.input_parametres[0].get("input_shape")
            mean_value = self.input_parametres[0].get("mean_value")
            std_value = self.input_parametres[0].get("std_value")

            mean_value = np.array(mean_value) / 255
            std_value = np.array(std_value) / 255
            self.imagenet_transforms = transforms.Compose(
                [
                    transforms.Resize(int(input_shape[-1] / 0.875)),
                    transforms.CenterCrop(input_shape[-1]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_value.tolist(), std=std_value.tolist()),
                ]
            )
        else:
            self.imagenet_transforms = transforms.Compose([])

    def __call__(self, data_item) -> Any:
        dst_data_item = dict()
        for k, v in data_item.items():
            input_info = self.input_info_dict[k]
            file_type = input_info.get("file_type")
            mean_value = np.array(input_info.get("mean_value"))
            std_value = np.array(input_info.get("std_value"))
            batch_list = []
            for batch_item in v:
                if file_type == "img" and self.imagenet_transforms is not None:
                    img = Image.open(batch_item)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img = self.imagenet_transforms(img)
                    img = torch.unsqueeze(img, 0)
                    batch_list.append(img)
                else:
                    img = cv2.imread(batch_item)
                    img = np.transpose(img, (2, 0, 1))
                    img = (img - mean_value.reshape(-1, 1, 1)) / std_value.reshape(-1, 1, 1)
                    batch_list.append(img)

            dst_data_item[k] = torch.cat(batch_list, dim=0).to(torch.float32).to(self.calibration_device)
        return dst_data_item
