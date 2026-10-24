# 结构化 ModelIR 与可验证跨题修复框架

## 目标

本框架服务于下面的研究命题：

> 在相同模型、相同 token/时间预算下，通过结构化 `ModelIR` 和经过
> 程序证据验证的跨题修复经验，降低约束修复的 token 消耗和误改率，同时
> 不降低 BWOR/IndustryOR 的准确率。

现有 `code/evaluation_tir_gurobi_robust.py` 和
`code/constraint_auditor_v3.py` 保持不动。新框架采用旁路接入，只在初版
候选未通过验证时介入；如果 IR 修复失败或超出预算，可以继续回退到原有
healing 流程。

## 设计边界

第一阶段覆盖：

- 线性约束；
- LP/MILP 变量域；
- 线性目标和二次目标；
- Gurobi 模型；
- 确定性的补丁操作；
- 跨题共享的修复配方，而不是跨题共享答案或整段代码。

第一阶段不覆盖：

- 非线性约束、锥约束和一般逻辑约束的自动改写；
- 把 benchmark 标签暴露给修复器；
- 仅由 LLM 判断通过的“已验证记忆”；
- 未做留出集测试的准确率优势声明。

## 模块结构

```text
ir_repair_agent/
  contracts.py   ModelIR、验证证据、结构化诊断、不可变 IRPatch
  budget.py      全流程共享 token/时间/求解预算账本
  memory.py      SQLite 持久化、准入验证、检索、成功率与隔离
  repair.py      先查记忆，再生成 IR 补丁，统一执行硬验证
  experiment.py  公平性配置、Bootstrap 配对比较和结果记录

tests/
  test_ir_repair_agent.py
```

数据流如下：

```text
初版代码（保持现有生成方式）
        ↓
ModelIR 提取适配器（无答案）
        ↓
程序验证得到结构化诊断
        ↓
查询已验证跨题修复配方
        ↓
配方可直接执行 ──通过──> 接受，记录一次 memory success
        │
        否
        ↓
LLM 只返回 IRPatch JSON
        ↓
本地应用最小补丁
        ↓
编译代码 + 独立执行/约束/变形验证
        ↓
通过且无回归 ──> 接受，并把可迁移配方写入记忆
```

## 核心约束

### 1. 记忆准入

`SqliteRepairMemory.store_verified()` 默认要求以下检查全部通过：

```text
model_build
constraint_integrity
execution
```

只有来源为 `program`、`solver` 或 `metamorphic` 的证据才算程序证据。
`llm` 和 `gold` 检查不能单独进入跨题记忆。

记忆库只保存：

- 问题类别；
- 结构化失败代码和特征；
- 可迁移的补丁配方；
- 验证检查集合；
- 成功/失败统计和隔离状态。

不保存标准答案、目标值或完整题目文本。

### 2. 补丁作用域

诊断中的 `affected_elements` 是补丁允许修改的元素。例如：

```python
FailureDiagnosis(
    failure_code="CONSTRAINT_RELATION_MISMATCH",
    affected_elements=("constraint:capacity_1",),
    features=("family:capacity", "relation:le"),
)
```

如果补丁同时改动目标函数，`EditAudit.extra_changes` 会包含 `objective`，
`wrong_edit_rate` 随之升高。默认策略要求误改率为 0，并且不能丢失任何
修复前已经通过的关键检查。

### 3. 跨题复用

记忆配方中的目标使用占位符，例如：

```json
{
  "op": "replace_relation",
  "target": "{{affected:0}}",
  "payload": {"relation": "ge"}
}
```

检索时绑定当前题目的诊断元素。默认排除相同语义模型指纹，避免把同题
重复运行误当作跨题收益。

### 4. 预算公平

同一道题的初版生成、审计和修复必须共用一个 `BudgetLedger`。治疗组不能
因为增加 IR 修复而获得额外调用次数。`ExperimentConfig` 会比较：

- benchmark 和数据集哈希；
- 模型名称、prompt 版本、temperature 和 seed；
- token、调用次数、wall time 和 solver time 上限。

任何一项不一致，`compare_runs()` 都会拒绝比较。

## 与现有代码接入

现有脚本不需要修改。治疗组需要在旧流程外层增加三个适配器：

```python
from ir_repair_agent import (
    BudgetLedger,
    RepairEngine,
    RepairRequest,
    SqliteRepairMemory,
)

memory = SqliteRepairMemory("results/repair_memory.sqlite")

engine = RepairEngine(
    memory=memory,
    compiler=compile_model_ir_to_gurobi,
    validator=validate_with_gurobi_and_or_ci,
    patch_generator=generate_ir_patch_with_llm,
)

outcome = engine.run(
    RepairRequest(
        model_ir=current_ir,
        current_code=current_code,
        baseline_validation=current_validation,
    ),
    ledger=shared_ledger,
)
```

三个适配器的契约：

```python
def compile_model_ir_to_gurobi(model_ir, previous_code, ledger) -> str:
    ...


def validate_with_gurobi_and_or_ci(model_ir, candidate_code, ledger):
    # 必须返回 ValidationResult，并至少产生 model_build、
    # constraint_integrity、execution 三项程序证据。
    ...


def generate_ir_patch_with_llm(request, memory_matches, ledger):
    # 提示词可使用 build_ir_patch_prompt()。
    # 返回 IRPatch 序列；不要在这里执行代码或读取标准答案。
    ...
```

ModelIR 提取应尽量使用确定性适配器，例如从已构建的 Gurobi 模型提取 IR。
如果使用 LLM 生成 IR，相关 token 必须记入同一个 `BudgetLedger`，否则实验
不再公平。

## 实验设计

建议至少设置三个消融臂：

| 实验臂 | 初版代码 | ModelIR 修复 | 跨题记忆 |
|---|---:|---:|---:|
| A：legacy | 现有 SIR | 否 | 否 |
| B：IR-only | 相同冻结初版 | 是 | 否 |
| C：IR+verified memory | 相同冻结初版 | 是 | 是 |

`C` 相对 `B` 的差异用于证明记忆是否有效；`B` 相对 `A` 的差异用于证明 IR
修复本身是否有效。

数据必须按题目划分：

- 只在 development 集生成记忆；
- test 集运行前冻结 prompt、模型和预算；
- test 题目不能通过语义指纹或原题 ID 命中 development 中的同题；
- 标准答案只在离线评测阶段计算 `correct` 和 `false_pass`。

主要指标：

- accuracy；
- total/repair token；
- solver time 和 wall time；
- wrong edit rate；
- hard-check regression 数；
- false pass 数；
- memory hit rate 和 memory hit acceptance rate。

误改率的当前定义为：

```text
wrong_edit_rate = unintended_changes / changed_elements
```

如果有 gold patch，可以改用：

```text
unintended_changes = changed_elements - gold_changed_elements
```

这样能区分“作用于正确约束但参数方向错误”和“修改了完全无关的元素”。

## 最小实现步骤

1. 保持现有 baseline 原样，先冻结 BWOR/IndustryOR 的初版代码。
2. 新增 Gurobi/OR-CI ModelIR 提取适配器。
3. 把现有约束审计结果映射为 `FailureDiagnosis`，无法确定的诊断保持
   `REVIEW`，不要强行生成补丁。
4. 实现受限的 `generate_ir_patch_with_llm`，只返回 IR 操作 JSON。
5. 用 development 集建立记忆，在 test 集做 B/C 消融。
6. 只有准确率不下降且 token 或误改率有显著改善时，才进入更大规模实验。

## 当前已验证内容

当前单元测试覆盖：

- IR 指纹的顺序稳定性；
- 二次目标规范化；
- 补丁作用域和误改率；
- 预算硬限制；
- LLM-only 证据禁止入库；
- 跨题配方绑定；
- 记忆负迁移隔离；
- memory hit 不调用 LLM；
- 公平性检查和配对 Bootstrap 比较。

运行测试：

```powershell
python -m pytest -q tests/test_ir_repair_agent.py
```
