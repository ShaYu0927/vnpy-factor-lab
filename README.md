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
| 行情事件回放 | 已验证单股票真实行情读取、因子计算和策略模块接收事件 |
| 回放因子与策略配置 | 默认加载 6 个基础因子公式，默认策略未激活 |
| 自动生成表达式、交叉变异和进化搜索 | 已完成设计，尚未实现 |
| 固定因子库导出与加载 | 已完成设计，尚未实现 |
| 模型训练 | 已有独立流程，尚未与自动搜索和回放完整衔接 |

“已有代码”表示仓库中存在相应实现，不代表已完成本机端到端运行验证。

## 代码与设计入口

| 入口 | 内容 |
| --- | --- |
| [vnpy/main.py](vnpy/main.py) | 当前程序入口，选择导入或行情回放 |
| [config/runtime.json](config/runtime.json) | 保留的原始行情回放配置，当前入口不默认加载 |
| [config/runtime.basic_alphas.json](config/runtime.basic_alphas.json) | 当前默认配置：单股票、6 个基础公式的回放验证 |
| [vnpy/alpha/alpha.py](vnpy/alpha/alpha.py) | Alpha 名称与公式定义 |
| [vnpy/alpha/engine.py](vnpy/alpha/engine.py) | 统一因子计算引擎 |
| [vnpy/alpha/expression/](vnpy/alpha/expression/) | 表达式树、算子、解析、分析与执行 |
| [vnpy/factor/realtime_service.py](vnpy/factor/realtime_service.py) | 行情缓存与回放因子计算 |
| [vnpy/quant/workflow.py](vnpy/quant/workflow.py) | 独立模型训练流程 |
| [examples/alpha_formula_pipeline.py](examples/alpha_formula_pipeline.py) | 因子公式示例 |
| [整个项目代码框架](docs/code_framework_design.md) | 目标目录、模块职责、核心对象与接口 |
| [Alpha 自动搜索设计](docs/alpha_mining_design.md) | 搜索空间、进化、数据边界、评价与实施阶段 |

## 当前回放配置

当前默认加载 `config/runtime.basic_alphas.json`，使用 `mode = "parquet"`，并设置 `parquet_import.enabled = false`，因此进入行情回放分支。

启动前根据本机情况检查行情目录、实际可用日期和股票范围。依赖准备好后，使用项目 `.venv` 解释器，在 IDE 中直接运行 `vnpy/main.py`，无需命令行参数。

当前已配置 6 个基础因子，默认策略未激活。单股票的真实行情读取、因子计算和策略模块接收事件已验证；策略信号及下单成交尚未验证。若使用横截面公式，现有回放还要求显式股票池及同一时点行情对齐。

## 基础因子验证配置

直接运行 `vnpy/main.py`，自动加载 [config/runtime.basic_alphas.json](config/runtime.basic_alphas.json)，验证固定公式的解析、表达式树编译和行情计算。默认路径在 `vnpy/config/runtime_config.py` 的 `DEFAULT_RUNTIME_CONFIG` 中设置，按源码位置定位，不依赖 IDE 的工作目录。

新配置读取 `SHSE.600000` 在 2026 年第一季度的日线，沿用本地行情目录。启动前确认该股票和日期有数据；缺少年度文件时直接报错。基础公式如下：

| 名称 | 公式 | 含义 |
| --- | --- | --- |
| `return_1` | `close / Ref(close, 1) - 1` | 相邻两根 K 线的收盘收益率 |
| `momentum_5` | `close / Ref(close, 5) - 1` | 5 根 K 线间的收盘收益率 |
| `ma_bias_5` | `close / Mean(close, 5) - 1` | 收盘价相对 5 根 K 线均价的偏离 |
| `volatility_5` | `Std(close / Ref(close, 1) - 1, 5)` | 最近 5 个单期收益率的总体标准差，未年化 |
| `volume_ratio_5` | `volume / Mean(volume, 5)` | 当前成交量与最近 5 根 K 线平均成交量的比值 |
| `amplitude_1` | `(high - low) / Ref(close, 1)` | 当根高低价差与上一根收盘价的比值 |

滚动均值包含当前 K 线，窗口按各股票的 K 线数量计数。整组公式至少需要 6 根 K 线，回放服务会先等待历史窗口满足再输出结果。公式均不需要横截面对齐。默认策略未激活；当前日志会记录因子事件及特征数量，不逐项打印因子值，也不自动保存因子表。

## 因子公式如何计算

当前使用固定公式：在配置文件顶层的 `alphas` 中定义名称和表达式，程序负责解析、编译和计算。例如：

```json
"alphas": [
  {"name": "momentum_5", "formula": "close / Ref(close, 5) - 1"}
]
```

以上是配置片段，放在完整 JSON 对象中，与 `mode`、`parquet` 同级。数据与公式通过下面的流程结合：

```text
JSON 配置 → Alpha 对象 → 表达式树 → Polars 编译结果
                                            ↓
历史 K 线 → 行情表（datetime、vt_symbol、close 等）→ 执行 → 因子值
```

`ExpressionParser` 使用 `ast.parse()` 读取公式语法，再转换成项目自己的节点；解析过程不会执行公式中的 Python 代码。上面的公式对应：

```text
sub（减法）
├── div（除法）
│   ├── close（字段）
│   └── ts_delay（Ref 的内部名称）
│       ├── close（字段）
│       └── 5（窗口参数）
└── 1（常数）
```

`ExpressionAnalyzer` 推导依赖字段、最少历史长度和横截面要求；`PolarsCompiler` 将字段映射为行情表的列，将算子转换成 Polars 运算。`PolarsExecutor` 在行情表上执行这些运算。

以 `momentum_5` 为例，核心运算相当于以下代码。输入须先按股票和时间排序，实际编译器会将窗口运算拆成中间计算步骤：

```python
previous_close = pl.col("close").shift(5).over("vt_symbol")
factor = pl.col("close") / previous_close - 1
```

`over("vt_symbol")` 保证每只股票使用自己的历史数据。当前收盘价为 15、5 根 K 线前为 10 时，当前因子值为 `15 / 10 - 1 = 0.5`。表达式树决定运算结构，行情提供具体数值；回放中复用编译结果，无需每根 K 线重新解析公式。

### 窗口与输出频率

5 根 K 线是回看跨度，历史足够后每根新 K 线都能产生一个新值：

| 计算时点 | `momentum_5` 的计算 |
| --- | --- |
| 第 6 根 | 第 6 根收盘价 / 第 1 根收盘价 − 1 |
| 第 7 根 | 第 7 根收盘价 / 第 2 根收盘价 − 1 |
| 第 8 根 | 第 8 根收盘价 / 第 3 根收盘价 − 1 |

对 100 根有效日线，该公式通常有 95 个有效值；零分母、缺失数据等会影响有效数量。当前 6 个公式作为一组等待最大历史窗口满足，再按股票、日期输出 `AlphaSample`，其中 `features` 保存因子名称到因子值的映射。

### 公式与结果存在哪里

| 位置 | 内容 |
| --- | --- |
| `config.raw["alphas"]` | JSON 中的原始因子名称和公式 |
| `AlphaEngine.definitions` | 解析后的因子定义，包括名称、规范化表达式和历史窗口 |
| `AlphaEngine._compiled` | 以因子名称为键的 Polars 编译结果 |
| `RealtimeAlphaService.latest_samples` | 最近一次计算产生的样本 |
| `RealtimeAlphaService.sample_cache` | 按股票缓存的历史样本 |

这些运行时对象保存在内存中。当前回放不自动导出因子表。主入口会创建引擎提前校验公式；回放服务会根据原始配置再创建计算引擎，因此启动时存在两次解析编译，后续可优化为复用。

## 一根 K 线的回放链路

```text
main() → load_runtime_config() → run_from_config()
    ↓
run_parquet_replay()：启动 factor 和 strategy 模块
    ↓
ParquetDataFeed.iter_history()：读取、筛选并按时间输出 K 线
    ↓
post_bar() → factor 队列中的 BAR 事件
    ↓
RealtimeFactorModule.handle()
    ↓
RealtimeAlphaService.on_bar()
    ├── 更新 BarCache，检查历史窗口
    ├── AlphaEngine.from_bars()：整理行情表
    ├── calculate_latest() → calculate() → PolarsExecutor.run()
    └── 提取当前时点 AlphaSample，写入样本缓存
    ↓
RealtimeFactorModule._publish_sample()
    ↓
strategy 队列中的 FACTOR 事件
    ↓
StrategyEngineModule.handle() → StrategyEngine.on_event() → on_factor()
```

两个模块通过各自的队列和线程处理事件，因此 `post_bar()` 返回只代表事件投递结果，计算在因子模块线程中进行。行情读取结束后，入口先等待因子队列处理完成，再等待策略队列处理完成。

当前已验证到策略引擎接收因子事件。具体策略只有启用且订阅匹配时才会执行 `strategy.on_factor()`；默认策略未启用，主入口也未注册交易和订单模块，因此尚未形成信号、下单、成交的完整闭环。

## 后续：表达式树与遗传搜索

自动搜索尚未实现。规划使用表达式树表示候选因子，通过交叉、变异生成新公式，再复用现有计算引擎评价。

| 操作 | 示例 |
| --- | --- |
| 参数变异 | 把 `Ref(close, 5)` 的窗口改为 `20` |
| 字段变异 | 把某个字段节点 `close` 换成 `volume` |
| 算子变异 | 把 `Mean(close, 5)` 改成 `Std(close, 5)` |
| 子树交叉 | 在两个候选因子之间交换参数类型兼容的子树 |

这些操作修改树的节点、参数或结构。修改后需要校验算子参数类型、树深度、节点数和回看长度，并排除未来标签等不允许的字段。

```text
生成初始候选树
    → 在训练集计算因子值并评价
    → 选择、交叉、变异
    → 生成下一代并重复训练评价
    → 冻结候选集
    → 验证集筛选与去重
    → 冻结最终集合并生成测试报告
    → 保存因子库，供行情回放使用
```

按现有设计，进化循环只使用训练集得分，验证集和测试集不反馈给交叉变异。IC、Rank IC 等用于评价因子与未来收益的关系；相关性指标本身不等同于交易收益。研究标签和评价已有独立实现，尚未接入主入口的回放流程。详细方案见 [Alpha 自动搜索设计](docs/alpha_mining_design.md)。

## 每日更新

### 2026-09-24

**已完成**

- 补充固定公式从 JSON 配置、表达式树到 Polars 执行的完整说明，以及 5 根 K 线窗口逐日输出的例子。
- 说明公式、编译结果和因子样本的内存存储位置，并整理 BAR 与 FACTOR 的队列调用链。
- 补充遗传搜索中参数、字段、算子变异和子树交叉的概念，明确训练、验证、测试边界及尚未实现的部分。

**验证与限制**

- 文档已与当前配置、入口、表达式编译执行及事件处理代码核对，并检查本地链接和 Markdown 代码块。
- 本次仅更新 README；真实行情回放与数值核对结果沿用 2026-09-23 的验证记录，未新增运行验证。

**下一步**

- 验证多股票及横截面计算，完善研究数据与评价流程，再实现候选树生成和进化搜索。

### 2026-09-23

**已完成**

- 新建 `config/runtime.basic_alphas.json`，配置单股票、小时间范围和 6 个基础因子公式。
- 默认配置路径指向 `config/runtime.basic_alphas.json`，直接运行 `main.py` 即可加载，无需命令行参数。
- `main.py` 直接从定义文件导入 `Alpha` 和 `AlphaEngine`，便于 IDE 静态定位定义。
- 确认 Alpha 加载流程后，移除临时配置及因子加载日志，保留原有加载逻辑。
- 补充 IDE 启动方式、公式含义及历史窗口说明。

**验证与限制**

- 使用项目 `.venv` 完成新配置加载、表达式解析编译及 6 根合成 K 线计算，6 个因子结果均与独立算术计算一致，确认最少历史长度为 6。
- `tests/test_alpha_formula_pipeline.py` 的 49 项测试通过；默认配置入口检查通过，替换回放调用后验证 `main()` 无参数加载 6 个公式。
- 已验证直接导入与原包级动态导入得到相同的类对象；IDE 跳转交互未实际验证。
- 已用默认配置执行真实行情回放：`SHSE.600000` 在 2026-01-05 至 2026-03-31 共 56 根日线，前 5 根用于准备历史窗口，产生 51 个样本、306 个有限因子值。
- 使用临时观测包装执行真实 `main()` 和原有事件处理逻辑，确认样本缓存保留 51 个样本、策略引擎实际处理 51 个 FACTOR 事件；逐项核对样本时间，最后一日的 6 个因子与独立算术计算一致。
- 默认策略未激活，本次验证到策略模块接收事件为止；自动生成表达式和进化搜索尚未实现。

**下一步**

- 继续验证多股票及横截面公式，再衔接因子评价和策略信号。

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
