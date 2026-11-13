---
description: XSlim 精度调优工作流：建立基线、解读 Graphwise Analysis、按证据做最小调参实验并在真实验证集上复核。适用「量化后精度掉点」「选 precision_level」「校准数据怎么配」「模型精度优化」
argument-hint: "<模型或精度问题>"
---

对「$@」执行 XSlim 精度调优。
委派机制见 `.agents/shared/delegation.md`；run 目录、产物命名与收尾见 `.agents/shared/run.md`。
本次 run 目录：`.agents/workspace/acc-<slug>-<yyyymmdd>/`（`<slug>` 为改动的简短英文标识）。

## Phase 0: 分级
- conversational（问策略、参数含义、报告怎么读）：直接回答，引用
  `.agents/skills/xslim-accuracy-tuning/SKILL.md` 与 `doc/accuracy_tuning.md` / `_zh`。
- standard（有模型和数据，需要诊断/调优）：执行 Phase 1–3。
- substantial（判断为代码缺陷或需要改量化实现）：Phase 3 结束后按「代码缺陷升级」处理。

## Phase 1: 基线与证据（先做，不可跳过）
single 委派 `accuracy-analyst`：
1. 读 `.agents/skills/xslim-accuracy-tuning/SKILL.md`，按需加载 `references/`。
2. 确认模型、精度变体、校准/验证数据集、样本数、预处理、指标、容差与运行命令。
3. 测量原模型与量化模型基线，写入 `<run-dir>/eval.md`。
**没有真实数据或无法测量时，明确声明并停止下结论，不用随机数据代替。**

## Phase 2: 诊断与最小实验
single 委派 `accuracy-analyst`：
- 先排查预处理/校准数据等全局原因，再解读 Graphwise Analysis 定位局部热点。
- 一次只改一类参数（precision_level → finetune_level → 校准/局部 custom_setting），
  每次记录 before/after、命令与原始输出。
- 每轮以「下一最小改动集」形式给出建议，不在同一轮堆叠无关改动。

## Phase 3: 复核
single 委派 `accuracy-analyst` 在声明的验证集上复核最终候选；报告指标、容差、样本数与未验证项。
只有复核通过才可声称精度改善（Rule 5）。

## 代码缺陷升级
若证据指向实现缺陷（融合语义错误、导出 dtype 错误、legalize 缺失等）：
- 停止继续调参，输出最小复现与证据；
- 另起 `quant` 或 `pass` prefix 的 run，交给 `.agents/prompts/quantizer-dev.md` 或
  `.agents/prompts/onnx-pass-dev.md` 处理；`accuracy-analyst` 不修改 `src/`。

## 收尾
按 `.agents/shared/run.md` 写 `handoff.md`：模型/数据标识、基线、实验结论、复核结果、未验证项。
本工作流不自动提交：调优配置是否落到 `samples/` 或文档由用户确认；以 HANDOFF 结束。
若用户明确要求提交且复核通过，按 Rule 10 在分支上提交配置/文档改动，并在 handoff 中记录命令与结果。
