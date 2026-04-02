# XSlim 文档

中文 | [English](README.md)

欢迎阅读 XSlim 文档。本目录包含 XSlim 的详细使用指南。

## 目录

- [配置参考](configuration_zh.md) — 所有 JSON 配置字段的完整说明
- [使用示例](examples_zh.md) — 常见量化场景的分步示例
- [量化精度调优指南](accuracy_tuning_zh.md) — 静态量化、动态量化与 FP16 的精度调优方法，含 Graphwise Analysis 解读

## 概述

XSlim 是一款离线量化（PTQ）工具，可将 ONNX 模型转换为针对 SpacemiT 硬件优化的量化形式。它支持 INT8、FP16 和动态量化，所有操作均通过简洁的 JSON 配置文件驱动。

快速入门请参阅[项目 README](../README_zh.md)。
