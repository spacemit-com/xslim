# XSlim 精度调优 / Quantization Accuracy Tuning

[中文版](#中文版) | [English](#english)

---

## English

**Skill:** XSlim Quantization Accuracy Tuning  
**Level:** Intermediate  
**Duration:** ~30 minutes  
**Description:** Learn how to systematically diagnose and improve model accuracy after post-training quantization with XSlim.

### What you will learn

After completing this skill you will be able to:

- Choose the right quantization strategy for your model
- Verify and fix input preprocessing mismatches
- Size your calibration dataset correctly
- Tune `precision_level` and `finetune_level` systematically
- Interpret the XSlim Graphwise Analysis report
- Apply targeted tuning methods (custom precision, calibration type, operator exclusion)

### Steps

| Step | Topic | File |
|:---:|---|---|
| 1 | Choosing a Quantization Strategy | [step1.md](step1.md) |
| 2 | Check Input Preprocessing | [step2.md](step2.md) |
| 3 | Calibration Dataset Size | [step3.md](step3.md) |
| 4 | Adjust `precision_level` and `finetune_level` | [step4.md](step4.md) |
| 5 | Read the Graphwise Analysis Report | [step5.md](step5.md) |
| 6 | Targeted Tuning Based on Analysis Results | [step6.md](step6.md) |

### Prerequisites

- XSlim installed (`pip install xslim`)
- A target ONNX model
- A calibration dataset (images or feature vectors)

### Reference

Full documentation: [Accuracy Tuning Guide](../../../doc/accuracy_tuning.md)

---

## 中文版

**技能：** XSlim 量化模型精度调优  
**级别：** 中级  
**时长：** 约 30 分钟  
**描述：** 学习如何在使用 XSlim 进行离线量化后，系统地排查和提升量化后的模型精度。

### 学习目标

完成本技能后，你将能够：

- 为你的模型选择合适的量化方案
- 校验并修正输入预处理不一致问题
- 合理配置校准数据集的规模
- 系统地调整 `precision_level` 与 `finetune_level`
- 解读 XSlim 的 Graphwise Analysis 报告
- 使用定向调优方法（自定义精度、校准类型、算子排除等）

### 步骤

| 步骤 | 主题 | 文件 |
|:---:|---|---|
| 1 | 量化方案选择 | [step1.md](step1.md) |
| 2 | 检查输入预处理 | [step2.md](step2.md) |
| 3 | 校准数据集规模 | [step3.md](step3.md) |
| 4 | 调整 `precision_level` 与 `finetune_level` | [step4.md](step4.md) |
| 5 | 解读 Graphwise Analysis 报告 | [step5.md](step5.md) |
| 6 | 基于分析结果的定向调优 | [step6.md](step6.md) |

### 前置条件

- 已安装 XSlim（`pip install xslim`）
- 目标 ONNX 模型
- 校准数据集（图像或特征向量）

### 参考文档

完整文档：[精度调优指南](../../../doc/accuracy_tuning_zh.md)
