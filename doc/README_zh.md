# XSlim 文档

中文 | [English](README.md)

欢迎阅读 XSlim 文档。本目录包含 XSlim 的详细使用指南。

## 目录

- [配置参考](configuration_zh.md) — 所有 JSON 配置字段的完整说明
- [使用示例](examples_zh.md) — 常见量化场景的分步示例
- [量化精度调优指南](accuracy_tuning_zh.md) — 静态量化、动态量化与 FP16 的精度调优方法，含 Graphwise Analysis 解读

## 概述

XSlim 是一款离线量化（PTQ）工具，可将 ONNX 模型转换为针对 SpacemiT 硬件优化的量化形式。它支持 INT8、FP16 和动态量化，所有操作均通过简洁的 JSON 配置文件驱动。

当前文档也覆盖 2.1.0 行为，包括自动 YOLO decode 融合、ONNX Function 保留、更完整的 ONNX 算子覆盖、opset-24 `Pad`、标量规约 Tensor、axes 输入形式的规约算子，以及拒绝对已包含 `QuantizeLinear` 或 `DequantizeLinear` 的模型执行静态 INT8 重复量化的保护逻辑。

快速入门请参阅[项目 README](../README_zh.md)。
