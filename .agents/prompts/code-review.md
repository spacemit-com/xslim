---
description: XSlim 代码评审：交叉核对公开接口兼容、图语义与导出契约、双语文档同步、测试意图，输出证据化 PASS/FAIL。适用「review」「评审」「检查改动」「看看这段改动」
argument-hint: "<改动范围或 diff>"
---

对「$@」执行 XSlim 代码评审。
委派机制见 `.agents/shared/delegation.md`；run 目录、产物命名与收尾见 `.agents/shared/run.md`。
本次 run 目录：`.agents/workspace/review-<slug>-<yyyymmdd>/`（`<slug>` 为改动的简短英文标识）。

## 流程
single 委派 `qa-inspector` 做证据化评审，task 包含改动范围（文件列表或 `git diff`）与以下核对清单：
1. 公开接口（Rule 15）：CLI flag、`xslim.quantize_onnx_model` 等导出名、配置键/默认值、
   输出模型的 domain/FunctionProto/外部数据布局是否兼容；破坏性变更是否有迁移说明与版本号
2. 图语义：融合/pass 是否保持图输出、拓扑序、常量形式、动态 shape 与幂等性
3. 实现约定：precision 分支、`resolve_operator_domain`、`xslim_setting` 校验、
   `xslim.logger` 与异常风格是否一致；有无模型/节点名/shape 特判（Rule 4）
4. 双语文档同步（Rule 3）：`doc/*.md` 与 `doc/*_zh.md`、README 对、samples 文档
5. 测试意图（Rule 9）：用例是否编码了「为什么」，是否会在目标逻辑变化时失败
6. 证据口径（Rule 5）：容差、数据集、模型标识是否声明；有无把转换成功说成精度通过

## 输出
`qa-inspector` 产出 `<run-dir>/review.md`：逐条 issue（`文件:行号` + 复现/证据）+ 总体 PASS/FAIL。
你汇总为评审报告，突出会阻断合入的问题与可后续跟进的问题。

## 错误处理
- 无 diff 信息时先取 `git diff --stat` 与 `git status` 确定范围，再委派。
- 评审只读：不修代码，除非用户明确要求「按评审修复」。
