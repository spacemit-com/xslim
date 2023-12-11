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
            data_list_path: str = input_item.get("data_list_path")  # type noqua
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


class PTImagenetPreprocess:
    def __init__(self, out_height, out_width, mean_value, std_value):
        self.out_height = out_height
        self.out_width = out_width
        self.mean_value = mean_value
        self.std_value = std_value

    def __call__(self, img):
        from PIL import Image
        import torchvision.transforms.functional as F

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = F.resize(img, int(self.out_height / 0.875), Image.BILINEAR)
        img = F.center_crop(img, self.out_height)
        img = F.pil_to_tensor(img).to(torch.float32)
        img = F.normalize(img, mean=self.mean_value, std=self.std_value, inplace=False)
        return img


class ImagenetPreprocess:
    def resize_with_aspectratio(self, img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
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

    def center_crop(self, img, out_height, out_width):
        height, width, _ = img.shape
        left = int((width - out_width) / 2)
        right = int((width + out_width) / 2)
        top = int((height - out_height) / 2)
        bottom = int((height + out_height) / 2)
        img = img[top:bottom, left:right]
        return img

    def __init__(self, out_height, out_width, mean_value, std_value):
        self.out_height = out_height
        self.out_width = out_width
        self.mean_value = mean_value
        self.std_value = std_value

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2_interpol = cv2.INTER_AREA
        img = self.resize_with_aspectratio(img, self.out_height, self.out_width, inter_pol=cv2_interpol)
        img = self.center_crop(img, self.out_height, self.out_width)
        img = np.asarray(img, dtype="float32")
        means = np.array(self.mean_value, dtype=np.float32)
        stds = np.array(self.std_value, dtype=np.float32)
        img = (img - means) / stds
        img = img.transpose([2, 0, 1])
        return torch.from_numpy(img)


class CalibrationCollect:
    def __init__(self, input_parametres: Sequence, calibration_device: str = "cuda") -> None:
        self.input_parametres = input_parametres
        self.calibration_device = calibration_device
        if not isinstance(self.input_parametres, Sequence) or len(self.input_parametres) < 1:
            raise RuntimeError

        self.input_info_dict = OrderedDict()
        for input_info in self.input_parametres:
            self.input_info_dict[input_info.get("input_name")] = input_info

        self.imagenet_transforms = None
        if len(self.input_parametres) == 1:
            input_shape = self.input_parametres[0].get("input_shape")
            mean_value = self.input_parametres[0].get("mean_value")
            std_value = self.input_parametres[0].get("std_value")
            calib_dataset_type = self.input_parametres[0].get("preprocess_file", None)
            if calib_dataset_type == "PT_IMAGENET":
                self.imagenet_transforms = transforms.Compose(
                    [PTImagenetPreprocess(input_shape[-2], input_shape[-1], mean_value, std_value)]
                )
            elif calib_dataset_type == "IMAGENET":
                self.imagenet_transforms = transforms.Compose(
                    [ImagenetPreprocess(input_shape[-2], input_shape[-1], mean_value, std_value)]
                )

    def __call__(self, data_item) -> Any:
        dst_data_item = dict()
        for k, v in data_item.items():
            input_info = self.input_info_dict[k]
            file_type = input_info.get("file_type")
            mean_value = np.array(input_info.get("mean_value"))
            std_value = np.array(input_info.get("std_value"))
            batch_list = []
            for batch_item in v:
                if file_type == "img":
                    if self.imagenet_transforms is not None:
                        img = cv2.imread(batch_item)
                        img = self.imagenet_transforms(img)
                        img = torch.unsqueeze(img, 0)
                        batch_list.append(img)
                    else:
                        img = cv2.imread(batch_item)
                        if len(img.shape) < 3:
                            img = np.expand_dims(img, -1)
                        img = img.astype(np.float32)
                        img = (img - mean_value) / std_value
                        img = np.transpose(img, (2, 0, 1))
                        img = torch.unsqueeze(torch.from_numpy(img), 0)
                        batch_list.append(img)
                elif file_type == "npy":
                    img = np.load(batch_item)
                    batch_list.append(img)
                elif file_type == "raw":
                    img = np.fromfile(batch_item, dtype=np.float32)
                    batch_list.append(img)
                else:
                    raise NotImplementedError("{}".format(file_type))

            dst_data_item[k] = torch.cat(batch_list, dim=0).to(torch.float32).to(self.calibration_device)
        return dst_data_item
