---
description: XSlim ONNX 图 pass/fusion 开发工作流：从需求到模式匹配、注册、回归测试与双语文档同步。适用「新增/修改 fusion」「支持 XX 子图」「简化 pass 没生效」「补充 pass 测试」
argument-hint: "<pass 或融合需求>"
---

对「$@」执行 XSlim ONNX 图 pass 开发。
委派机制见 `.agents/shared/delegation.md`；run 目录、产物命名与收尾见 `.agents/shared/run.md`。
本次 run 目录：`.agents/workspace/pass-<slug>-<yyyymmdd>/`（`<slug>` 为改动的简短英文标识）。

## Phase 0: 复杂度分级（先做）
把请求分为 conversational / small / standard / substantial：
- conversational（问某 pass 原理、匹配条件）：不派活，直接回答并引用源码与 `doc/`。
- small（单文件小改，例如修正一个既有 pattern 的属性判断）：跳过 Phase 1，single 委派
  `onnx-pass-developer`；由其自跑该 pass 的定向测试，把命令与原始结果写进 `<run-dir>/impl.md`，
  不另派 `qa-inspector`。改动一旦触及自定义域/FunctionProto、导出格式或公开配置，升为 standard。
- standard（新增一个 fusion/pass，或改变既有融合语义）：执行 Phase 1–3。
- substantial（新增自定义算子域、改变导出契约、跨 pass 顺序调整）：增加 Phase 1.5 设计确认。

## Phase 1: 侦察（standard 起）
single 委派 `codebase-scout`：定位同类 pattern、注册点、`optimize_onnx_model` 接入顺序、
受影响的测试与文档，读取 `.agents/skills/onnx-pass-dev/SKILL.md`。产物 `<run-dir>/scout.md`。

## Phase 1.5: 设计（仅 substantial，HITL 检查点）
single 委派 `onnx-pass-developer` 产出 `<run-dir>/design.md`：匹配条件、图改写方式、
自定义域/FunctionProto 影响、opset 边界、文档同步项、风险。
**把设计摘要展示给用户并等待确认后再实现**。

## Phase 2: 实现
single 委派 `onnx-pass-developer`，task 附带 scout 结论、`design.md`（substantial）与产物路径：
按 `.agents/skills/onnx-pass-dev/SKILL.md` 实现匹配/改写、注册与测试，并同步 Rule 3 文档。
产物 `<run-dir>/impl.md`。

## Phase 3: 验证
single 委派 `qa-inspector`：按 `.agents/skills/build-verify/SKILL.md` 运行定向测试与 `ruff check .`；
核对图输出、拓扑序、幂等性、自定义域一致性与双语文档同步，判定 PASS/FAIL。产物 `<run-dir>/review.md`。
FAIL 时把 issue 作为 task 回交 `onnx-pass-developer` 修复（最多 2 轮，范围不扩大）。

## 收尾
按 `.agents/shared/run.md` 写 `handoff.md`：改动文件、注册点、测试命令与结果（PASS/FAIL）、未决项。
standard 路径验证 PASS 后按 Rule 10 在当前分支自动提交；substantial 路径以设计确认为检查点，
以 HANDOFF 结束，是否提交由用户决定；FAIL 或未验证的状态保持未提交并显式说明。
