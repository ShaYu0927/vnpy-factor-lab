# Alpha 因子研究与行情回放

基于 vn.py 改造的量化研究项目，围绕本地 Parquet 行情、因子表达式计算、因子自动搜索及事件驱动的行情回放展开。

本文记录项目现状和每日更新。设计方案与已实现功能分别标明，更新按日期倒序排列。

## 项目目标

```text
历史行情 → 数据准备与时间切分 → 自动生成表达式
        → 训练集评价与进化 → 验证集筛选 → 固定因子库
                                                ↓
历史行情逐根回放 → 计算已选因子 → FACTOR 事件 → 策略
```

模型训练是后续可选衔接环节。上述整体流程尚未全部实现。

## 当前状态

| 部分 | 状态 |
| --- | --- |
| 本地 Parquet 行情读取 | 已有代码 |
| Alpha 公式、表达式树、解析与计算 | 已有代码 |
| 未来收益标签与 IC / Rank IC 评价 | 已有代码，统一研究数据契约待实现 |
| 行情事件回放 | 已有代码，当前配置已选择回放分支 |
| 回放因子与策略配置 | 尚未配置 `alphas`，默认策略未激活 |
| 自动生成表达式、交叉变异和进化搜索 | 已完成设计，尚未实现 |
| 固定因子库导出与加载 | 已完成设计，尚未实现 |
| 模型训练 | 已有独立流程，尚未与自动搜索和回放完整衔接 |

“已有代码”表示仓库中存在相应实现，不代表已完成本机端到端运行验证。

## 代码与设计入口

| 入口 | 内容 |
| --- | --- |
| [vnpy/main.py](vnpy/main.py) | 当前程序入口，选择导入或行情回放 |
| [config/runtime.json](config/runtime.json) | 行情路径、时间范围、回放与 Alpha 配置 |
| [vnpy/alpha/alpha.py](vnpy/alpha/alpha.py) | Alpha 名称与公式定义 |
| [vnpy/alpha/engine.py](vnpy/alpha/engine.py) | 统一因子计算引擎 |
| [vnpy/alpha/expression/](vnpy/alpha/expression/) | 表达式树、算子、解析、分析与执行 |
| [vnpy/factor/realtime_service.py](vnpy/factor/realtime_service.py) | 行情缓存与回放因子计算 |
| [vnpy/quant/workflow.py](vnpy/quant/workflow.py) | 独立模型训练流程 |
| [examples/alpha_formula_pipeline.py](examples/alpha_formula_pipeline.py) | 因子公式示例 |
| [整个项目代码框架](docs/code_framework_design.md) | 目标目录、模块职责、核心对象与接口 |
| [Alpha 自动搜索设计](docs/alpha_mining_design.md) | 搜索空间、进化、数据边界、评价与实施阶段 |

## 当前回放配置

`config/runtime.json` 使用 `mode = "parquet"`，并设置 `parquet_import.enabled = false`，因此进入行情回放分支。

启动前根据本机情况检查行情目录、实际可用日期和股票范围。依赖准备好后，在本 README 所在目录运行：

```powershell
python -m vnpy.main
```

当前未配置 `alphas`，因此回放不会产生因子结果；默认策略也未激活。若使用横截面公式，现有回放还要求显式股票池及同一时点行情对齐。完整流程尚需单独运行验证。

## 每日更新

### 2026-09-22

**已完成**

- 梳理当前启动入口、Parquet 行情读取、因子计算与策略事件路径。
- 将 `config/runtime.json` 中的 `parquet_import.enabled` 改为 `false`，默认走行情回放。
- 明确因子表达式通过顶层 `alphas` 配置，并梳理 `BAR → RealtimeAlphaService → AlphaEngine → FACTOR → 策略` 调用链。
- 新增 [Alpha 自动搜索设计](docs/alpha_mining_design.md)，确定数据切分、未来收益标签、表达式生成、训练评价、交叉变异及验证筛选流程。
- 新增 [整个项目代码框架](docs/code_framework_design.md)，规划 `alpha/research`、`alpha/mining`、`alpha/library.py` 和 `replay` 的职责与接口。
- 新建本 README，作为项目入口和每日更新记录。
- 调整 `.gitignore`，允许 README 和上述两份设计文档纳入版本管理。

**验证与限制**

- 回放配置已通过 JSON 解析及分支值检查。
- 设计文档已检查文件存在、代码块配对和相关 Git 忽略规则。
- 尚未启动完整行情回放，也未执行自动搜索或模型训练。
- 本次完成的是配置调整与架构设计；规划中的新增业务模块尚未实现。

**下一步**

- 建立研究数据对象、标签起止时间和时间切分接口。
- 使用少量行情与固定示例公式，验证“行情 → 因子 → 标签 → 评价”。
- 再实现候选表达式生成、单批评价和进化搜索。

## 更新记录约定

- 后续每次完成项目改动，同步维护“当前状态”和当天的更新记录。
- 日期按北京时间，使用 `YYYY-MM-DD`；新日期放在“每日更新”最上方，同一天的改动补充到同一条目。
- 每条记录说明实际完成的改动、相关文件、验证结果及未完成事项；设计完成不能写成代码已实现。
- 只记录有依据的工作，不推测补写历史日期；没有改动的日期不需要空记录。
- 每日记录随项目开发维护，不表示已配置定时自动执行任务。

后续日期可使用以下模板：

```markdown
### YYYY-MM-DD

**已完成**
- 具体改动与相关文件。

**验证与限制**
- 实际执行的检查、结果，以及尚未验证的部分。

**下一步**
- 待处理事项。
```
