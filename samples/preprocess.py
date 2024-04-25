from typing import Any, Union, Dict, Sequence
import torch
import cv2
import numpy as np


def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    读取path_list, 并依据input_parametr中的参数预处理, 返回一个torch.Tensor

    Args:
        path_list (Sequence[str]): 一个校准batch的文件列表
        input_parametr (dict): 等同于配置中的calibration_parameters.input_parametres[idx]

    Returns:
        torch.Tensor: 一个batch的校准数据
    """
    # print("Custom Preprocessing ...")
    batch_list = []
    mean_value = input_parametr["mean_value"]
    std_value = input_parametr["std_value"]
    input_shape = input_parametr["input_shape"]
    for file_path in path_list:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_shape[-1], input_shape[-2]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = (img - mean_value) / std_value
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        batch_list.append(img)
    return torch.cat(batch_list, dim=0)
