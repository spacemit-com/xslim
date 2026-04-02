# 量化模型精度调优指南

中文 | [English](accuracy_tuning.md)

本文介绍在使用 XSlim 进行模型量化时，如何系统地排查和提升量化后的模型精度。

---

## 目录

1. [量化方案选择](#1-量化方案选择)
2. [调优流程概览](#2-调优流程概览)
3. [第一步：检查输入预处理](#3-第一步检查输入预处理)
4. [第二步：调整校准数据集数量](#4-第二步调整校准数据集数量)
5. [第三步：调整 precision_level 与 finetune_level](#5-第三步调整-precision_level-与-finetune_level)
6. [第四步：解读 Graphwise Analysis 报告](#6-第四步解读-graphwise-analysis-报告)
7. [第五步：基于分析结果进行定向调优](#7-第五步基于分析结果进行定向调优)
8. [常见问题与速查表](#8-常见问题与速查表)

---

## 1. 量化方案选择

XSlim 支持以下量化方案，推荐按以下优先级依次尝试：

| 优先级 | 方案 | `precision_level` | 说明 |
|:---:|---|:---:|---|
| 1 | **静态 INT8 量化** | `0`–`2` | 权重与激活值均静态量化，压缩率最高，推理速度最快。**首选方案。** |
| 2 | **动态量化** | `3` | 权重静态量化，激活值在运行时量化。无需校准数据，适合激活值分布不稳定的模型（如某些 NLP 模型）。 |
| 3 | **FP16 转换** | `4` | 所有浮点算子转为 FP16，无需校准数据。精度损失最小，但压缩率和推理加速低于 INT8。 |

> **建议：** 优先充分调优静态 INT8 量化。仅当 INT8 精度无法满足需求时，再考虑动态量化或 FP16。

### 快速对比

| 方案 | 需要校准数据 | 精度损失风险 | 推理加速 | 典型适用场景 |
|---|:---:|:---:|:---:|---|
| 静态 INT8 | ✅ 需要 | 中 | ⭐⭐⭐⭐ | 图像分类、目标检测、大多数 CNN |
| 动态量化 | ❌ 不需要 | 低–中 | ⭐⭐⭐ | LSTM、部分 Transformer |
| FP16 | ❌ 不需要 | 极低 | ⭐⭐ | 精度敏感、无法提供校准数据 |

---

## 2. 调优流程概览

```
准备模型与校准数据
        │
        ▼
  [步骤 1] 检查输入预处理
        │ 预处理与推理一致？
        ▼
  [步骤 2] 校准数据集数量是否充足？
        │
        ▼
  [步骤 3] 调整 precision_level / finetune_level
        │
        ▼
  [步骤 4] 开启 analysis_enable，解读 Graphwise 报告
        │ 发现高误差层？
        ▼
  [步骤 5] 用 custom_setting 对问题层定向调优
        │
        ▼
      精度满足要求？ ──是──▶ 完成
              │
             否
              ▼
        尝试更高 precision_level 或切换方案
```

---

## 3. 第一步：检查输入预处理

**输入预处理不一致是量化精度损失最常见的原因之一。** 校准时的预处理必须与模型在实际推理时所接受的输入完全一致，否则校准出的量化参数将无法反映真实的激活值分布，导致精度大幅下降。

### 必查项

| 检查项 | 说明 |
|---|---|
| **数值范围** | 校准数据的像素值范围（如 `[0, 255]` 还是 `[0.0, 1.0]`）必须与模型训练时一致 |
| **归一化参数** | `mean_value` 和 `std_value` 必须与训练时的归一化参数完全匹配 |
| **颜色通道顺序** | `color_format` 必须与模型训练时一致（OpenCV 默认读取为 BGR；PyTorch 通常使用 RGB） |
| **图像缩放策略** | 缩放方式（如 `INTER_AREA` 还是 `INTER_LINEAR`）、裁剪位置（center crop vs. resize）必须与训练时一致 |
| **数据类型** | 通常为 `float32`；`int64` 等整型输入（NLP 模型的 token id 等）无需归一化 |

### 内置预处理预设

| `preprocess_file` 值 | 说明 |
|---|---|
| `"PT_IMAGENET"` | PyTorch 风格 ImageNet 预处理：resize(256) → center-crop(224) → normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| `"IMAGENET"` | 标准 ImageNet 预处理（BGR 通道，适用于 Caffe 等框架） |
| `"path/to/script.py:function_name"` | 用户自定义预处理函数（见[自定义预处理](configuration_zh.md#自定义预处理)） |

> **提示：** 若模型来自 PyTorch 并使用 `torchvision.transforms`，`"PT_IMAGENET"` 通常是正确选择。若来自其他框架，请使用自定义预处理函数并仔细核对参数。

### 常见错误示例

```json
// ❌ 错误：训练时用 RGB，校准时用 BGR（默认）
"color_format": "bgr"

// ✅ 正确：与训练保持一致
"color_format": "rgb"
```

```json
// ❌ 错误：均值写成了 0-1 范围，但图像已缩放到 0-255
"mean_value": [0.485, 0.456, 0.406],
"std_value": [0.229, 0.224, 0.225]

// ✅ 正确：均值与输入数值范围一致（0-255 像素）
"mean_value": [123.675, 116.28, 103.53],
"std_value": [58.395, 57.12, 57.375]
```

---

## 4. 第二步：调整校准数据集数量

校准数据集的数量直接影响激活值范围统计的质量，进而影响量化精度。

### 数量建议

| 场景 | 推荐数量（图片张数） | `calibration_step` 参考值（batch_size=1） |
|---|:---:|:---:|
| 快速验证 | 50–100 | 50–100 |
| 一般场景（CNN、分类） | 100–500 | 100–500 |
| 复杂模型（Transformer、检测） | 500–1000 | 500–1000 |
| 精度要求极高 | ≥ 1000 | ≥ 1000 |

> **注意：** `calibration_step` 为 dataloader 的 batch 迭代次数，实际样本数约为 `calibration_step × batch_size`。默认值为 `500`。

### 校准数据的质量要求

- **多样性**：校准数据应覆盖模型实际推理时的典型输入分布，而非只包含单一类别或场景。
- **代表性**：从验证集或测试集中随机采样，避免使用训练数据中的重复样本。
- **真实性**：校准数据应尽量接近真实部署场景的数据，例如实际采集的图片而非合成图片。

### 数量不足的典型症状

- 精度在不同运行之间波动较大（校准数据随机性导致）
- Graphwise 报告中部分层的 `F.Hist` 分布极度不均匀（仅有少数 bin 有值）

---

## 5. 第三步：调整 precision_level 与 finetune_level

确认预处理正确、校准数据充足后，通过调整这两个参数可以系统性地提升精度。

### `precision_level`（精度级别）

| 级别 | 量化策略 | 适用场景 |
|:---:|---|---|
| `0` | 全 INT8（默认） | 大多数 CNN（ResNet、MobileNet、EfficientNet 等） |
| `1` | 部分 INT8，少量敏感层保持高精度 | 一般 Transformer 模型、含 Attention 的模型 |
| `2` | 部分 INT8，更多层保持高精度 | 精度敏感模型、复杂 Transformer |
| `3` | 动态量化 | 激活值分布变化大的模型，无校准数据场景 |
| `4` | FP16 | 对精度极为敏感，或完全无法提供校准数据 |

**调优建议：**
1. 从 `precision_level: 0` 开始；
2. 若精度不足，尝试 `precision_level: 1`；
3. 仍不足，尝试 `precision_level: 2`；
4. INT8 方案不可接受时，考虑 `precision_level: 3`（动态）或 `4`（FP16）。

### `finetune_level`（微调级别）

| 级别 | 行为 | 耗时 | 适用场景 |
|:---:|---|:---:|---|
| `0` | 不进行量化参数调整 | 最短 | 快速验证、精度要求宽松 |
| `1` | 轻量级静态量化参数校准（默认） | 短 | 大多数场景的默认选择 |
| `2` | 逐块量化参数校准 | 中 | 精度敏感模型、Transformer |
| `3` | 更激进的逐块校准 | 长 | 对精度要求极高的场景 |

**调优建议：**
- 一般模型：使用默认的 `finetune_level: 1`；
- Transformer / NLP 模型：推荐 `finetune_level: 2`；
- 精度要求极高时：尝试 `finetune_level: 3`（注意耗时显著增加）。

### 推荐组合

| 模型类型 | 推荐配置 |
|---|---|
| CNN 图像分类（ResNet、MobileNet 等） | `precision_level: 0`, `finetune_level: 1` |
| 目标检测（YOLO、SSD 等） | `precision_level: 1`, `finetune_level: 2` |
| Transformer / BERT / ViT | `precision_level: 1` 或 `2`, `finetune_level: 2` |
| LSTM / RNN | `precision_level: 3`（动态量化） |
| 精度极敏感（无法接受任何精度损失） | `precision_level: 4`（FP16） |

---

## 6. 第四步：解读 Graphwise Analysis 报告

XSlim 量化完成后默认会自动运行 Graphwise Analysis（由 `analysis_enable: true` 控制），并将报告输出到工作目录，文件名为 `<output_prefix>_report.md`。

### 报告结构

报告为 Markdown 表格，**按量化误差从高到低排序**，每行对应一个算子的一个 Tensor（输入或输出）：

| 列名 | 含义 | 健康参考值 |
|---|---|:---:|
| `Op` | 算子名称与类型，格式为 `算子名[算子类型]` | — |
| `Var` | Tensor 名称（`[Constant]` 标注表示权重参数） | — |
| `SNR` | 信噪比（量化噪声 / 信号，越低越好）。**红色标注（> 0.1）表示误差偏高** | `< 0.1` |
| `MSE` | 均方误差（越低越好） | 越小越好 |
| `Cosine` | 余弦相似度（量化值与浮点值的方向一致性，越高越好）。**红色标注（< 0.99）表示相似度偏低** | `> 0.99` |
| `Q.MinMax` | 量化后 Tensor 的最小值、最大值 | — |
| `F.MinMax` | 浮点（原始）Tensor 的最小值、最大值 | — |
| `F.Hist` | 浮点值的分布直方图（32 个 bin，用逗号分隔的计数） | — |

### 关键指标解读

#### SNR（信噪比）

SNR = 量化误差的能量 / 原始信号的能量。**SNR 越高，该 Tensor 的量化误差越大。**

- `SNR < 0.01`：量化误差极小，无需关注；
- `0.01 ≤ SNR < 0.1`：误差在可接受范围内；
- `SNR ≥ 0.1`（报告中标红）：误差较高，该算子是重点调优对象。

#### Cosine 相似度

衡量量化值与浮点值在方向上的一致性。

- `Cosine > 0.999`：方向高度一致，量化质量好；
- `0.99 ≤ Cosine ≤ 0.999`：轻微偏差；
- `Cosine < 0.99`（报告中标红）：方向偏差明显，该算子需要重点关注。

#### F.MinMax 与 Q.MinMax

- 若两者相差悬殊（如浮点范围 `[-100, 100]`，量化范围 `[-5, 5]`），说明存在**范围截断**，可能导致精度损失；
- 可尝试调整 `calibration_type`（如换用 `minmax`）或调整 `max_percentile`（如从 `0.9999` 调高至 `0.99999`）。

#### F.Hist（分布直方图）

- 若直方图严重不均匀（大量数据集中在少数几个 bin），说明激活值分布存在大量异常值，建议使用 `percentile` 校准类型并适当调低 `max_percentile`；
- 若直方图过于稀疏（许多 bin 为 0），可能是校准数据不足，应增加 `calibration_step`。

### 报告示例解读

```markdown
| Op | Var | SNR | MSE | Cosine | Q.MinMax | F.MinMax | F.Hist |
|---|---|---|---|---|---|---|---|
| /model/layer4/conv[Conv] | /layer4/conv/output | <font color="red">0.1523</font> | 0.0312 | <font color="red">0.9743</font> | -3.21, 3.18 | -15.32, 12.87 | 0,0,2,8,45,... |
| /model/layer1/conv[Conv] | /layer1/conv/output | 0.0045 | 0.0003 | 0.9998 | -1.02, 1.05 | -1.05, 1.08 | 3,12,55,... |
```

上例说明：
- `layer4/conv` 的 SNR 为 `0.1523`（标红），Cosine 为 `0.9743`（标红），**是精度损失的主要来源**；
- `layer4/conv` 的浮点范围 `[-15.32, 12.87]` 远大于量化范围 `[-3.21, 3.18]`，说明存在严重的范围截断；
- `layer1/conv` 的各项指标均正常，无需处理。

---

## 7. 第五步：基于分析结果进行定向调优

通过 Graphwise 报告定位到高误差算子后，可使用以下方法进行定向调优。

### 方法一：提升局部精度（`custom_setting`）

对报告中 SNR/Cosine 异常的算子所在子图，提升其 `precision_level`：

```json
"quantization_parameters": {
    "precision_level": 0,
    "custom_setting": [
        {
            "input_names": ["problem_layer_input_tensor"],
            "output_names": ["problem_layer_output_tensor"],
            "precision_level": 2
        }
    ]
}
```

> **提示：** 使用 [Netron](https://netron.app) 打开 ONNX 模型，找到问题算子的输入/输出 Tensor 名称。

### 方法二：调整校准类型（`calibration_type` / `max_percentile`）

当报告显示 F.MinMax 与 Q.MinMax 差距大（范围截断）时：

- 若激活分布有异常值（F.Hist 呈长尾分布），使用 `percentile` 并适当降低 `max_percentile`：

```json
"calibration_parameters": {
    "calibration_type": "percentile"
},
"quantization_parameters": {
    "max_percentile": 0.9995
}
```

- 若激活分布接近均匀或正态，可尝试 `minmax`：

```json
"calibration_parameters": {
    "calibration_type": "minmax"
}
```

也可以通过 `custom_setting` 仅对问题子图使用不同的校准策略：

```json
"quantization_parameters": {
    "custom_setting": [
        {
            "input_names": ["problem_layer_input"],
            "output_names": ["problem_layer_output"],
            "calibration_type": "percentile",
            "max_percentile": 0.9995
        }
    ]
}
```

### 方法三：排除敏感算子（`ignore_op_types` / `ignore_op_names`）

当某类算子（如 `Softmax`、`LayerNormalization`）在报告中一致出现高误差，可将其排除在量化范围之外：

```json
"quantization_parameters": {
    "ignore_op_types": ["Softmax", "LayerNormalization"]
}
```

或排除特定名称的算子：

```json
"quantization_parameters": {
    "ignore_op_names": ["/model/encoder/layer.0/attention/MatMul"]
}
```

### 方法四：提升 finetune_level

当报告中多个层出现中等程度的误差（`0.01 < SNR < 0.1`）但没有特别突出的问题层时，提升 `finetune_level` 往往能整体改善精度：

```json
"quantization_parameters": {
    "finetune_level": 2
}
```

### 综合调优示例

以下配置结合了上述多种方法，适用于精度要求较高的 CNN 模型：

```json
{
    "model_parameters": {
        "onnx_model": "models/my_model.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "calibration_step": 500,
        "calibration_type": "percentile",
        "input_parametres": [
            {
                "mean_value": [103.94, 116.78, 123.68],
                "std_value": [57.0, 57.0, 57.0],
                "color_format": "bgr",
                "preprocess_file": "PT_IMAGENET",
                "data_list_path": "./calib_data/img_list.txt"
            }
        ]
    },
    "quantization_parameters": {
        "precision_level": 0,
        "finetune_level": 2,
        "max_percentile": 0.9999,
        "analysis_enable": true,
        "ignore_op_types": ["Softmax"],
        "custom_setting": [
            {
                "input_names": ["problem_layer_input"],
                "output_names": ["problem_layer_output"],
                "precision_level": 2,
                "calibration_type": "minmax"
            }
        ]
    }
}
```

---

## 8. 常见问题与速查表

### 精度问题速查

| 现象 | 可能原因 | 建议操作 |
|---|---|---|
| 量化后精度大幅下降（>5%） | 预处理不一致 | 仔细核查 `mean_value`、`std_value`、`color_format` 是否与训练一致 |
| 量化后精度小幅下降（1–5%） | 校准数据不足或分布不均 | 增加 `calibration_step`，确保数据多样性 |
| 量化后精度小幅下降 | `precision_level` 过低 | 尝试 `precision_level: 1` 或 `2` |
| 量化后精度小幅下降 | 校准范围不准 | 尝试 `calibration_type: "percentile"` 或 `"kl"` |
| Graphwise 报告中某层 SNR 极高 | 该层激活值范围异常 | 对该子图使用 `custom_setting`，尝试 `precision_level: 2` 或 `minmax` 校准 |
| Graphwise 报告中多层 Cosine < 0.99 | 全局量化误差较高 | 提升 `finetune_level` 至 `2` 或 `3` |
| INT8 精度无论如何都无法满足需求 | 模型对量化极为敏感 | 使用 `precision_level: 3`（动态量化）或 `4`（FP16） |

### 参数选择速查

| 目标 | 推荐设置 |
|---|---|
| 最大压缩率 | `precision_level: 0`, `finetune_level: 1` |
| 最佳 INT8 精度 | `precision_level: 2`, `finetune_level: 3`, `calibration_step: 1000` |
| Transformer 模型 INT8 | `precision_level: 1`, `finetune_level: 2`, `calibration_type: "default"` |
| 无校准数据 | `precision_level: 3`（动态）或 `precision_level: 4`（FP16） |
| 最快调优速度 | `precision_level: 0`, `finetune_level: 0`, `calibration_step: 100` |

---

## 参考文档

- [配置参考](configuration_zh.md) — 所有 JSON 配置字段的完整说明
- [使用示例](examples_zh.md) — 常见量化场景的分步示例
