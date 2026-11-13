---
description: XSlim 量化流程开发工作流：静态 INT8/动态量化/FP16、导出器、配置 schema 与 CLI 改动，含接口兼容评审、回归测试与双语文档同步。适用「支持 XX 量化行为」「FP16 转换失败」「新配置项」「改 CLI」
argument-hint: "<量化行为或接口需求>"
---

对「$@」执行 XSlim 量化流程开发。
委派机制见 `.agents/shared/delegation.md`；run 目录、产物命名与收尾见 `.agents/shared/run.md`。
本次 run 目录：`.agents/workspace/quant-<slug>-<yyyymmdd>/`（`<slug>` 为改动的简短英文标识）。

## Phase 0: 复杂度分级
- conversational（问配置/精度策略/转换原理）：直接回答，引用 `doc/` 与源码。
- small（局部实现修复，不改公开接口）：跳过 Phase 1，single 委派 `quantizer-developer` 直接改；
  由其自跑受影响的测试模块，把命令与原始结果写进 `<run-dir>/impl.md`，不另派 `qa-inspector`。
- standard（新增/改变一种量化行为）：执行 Phase 1–3。
- substantial（改公开配置键/默认值、CLI、导出格式、precision 语义）：执行 Phase 1 → Phase 1.5 → Phase 2–3。

## Phase 1: 侦察
single 委派 `codebase-scout`：定位 precision 分支、相关 quantizer/legalize/exporter、
配置 schema、测试与双语文档，读取 `.agents/skills/quantization-dev/SKILL.md`。产物 `<run-dir>/scout.md`。

## Phase 1.5: 设计（standard 起涉及公开接口时 / substantial，HITL 检查点）
single 委派 `quantizer-developer` 产出 `<run-dir>/design.md`：接口影响（Rule 15）、兼容/迁移方案、
数据流与导出契约、文档同步项、风险。
**把设计摘要展示给用户并等待确认后再实现**（公开接口与导出格式回退成本高）。

## Phase 2: 实现
single 委派 `quantizer-developer`，task 附带 scout 结论、`design.md`（如适用）与产物路径：
实现改动、补回归测试、同步双语文档（Rule 3）。产物 `<run-dir>/impl.md`。

## Phase 3: 验证
single 委派 `qa-inspector`：按 `.agents/skills/build-verify/SKILL.md` 运行定向测试、`ruff check .`，
必要时跑打包检查与一次端到端转换 smoke；核对公开接口兼容、导出契约与双语文档，判定 PASS/FAIL。
产物 `<run-dir>/review.md`。FAIL 时把 issue 作为 task 回交 `quantizer-developer` 修复（最多 2 轮）。

## 收尾
按 `.agents/shared/run.md` 写 `handoff.md`：改动文件、接口影响、验证命令与结果、文档同步项、未决项。
standard 且验证 PASS 后按 Rule 10 在当前分支自动提交；substantial 或涉及公开接口的变更以设计确认为检查点，
以 HANDOFF 结束，是否提交由用户决定；FAIL 或未验证的状态保持未提交并显式说明。
