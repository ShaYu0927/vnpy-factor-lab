"""保存个体、适应度评价结果，以及每一代的种群"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from vnpy.alpha.mining.schema import Candidate

# 一个候选公式的评价结果
@dataclass(frozen=True)
class Fitness:
    score: float               # 综合评分，用于比较候选、选择父代和保留精英
    rank_ic: float | None      # Rank IC 指标
    ic_days: int               # 用于评价的有效 IC 天数
    coverage: float            # 数据覆盖率，具体统计口径由评价器决定
    error: str | None = None   # 评价失败时记录错误原因；正常情况下为 None
    ic_std: float | None = None # IC 的标准差，用于反映 IC 在不同日期的波动
    stability: float | None = None  # 稳定性指标，具体计算公式由评价器决定
    complexity: int = 0         # 表达式复杂度，例如节点数

# 一个已完成评价的个体：候选公式 + 成绩 + 来源
@dataclass(frozen=True)
class Individual:
    candidate: Candidate  # 候选公式及其元信息。
    fitness: Fitness      # 对这个候选公式的评价结果。
    origin: str           # 个体来源，例如 random、mutation、crossover
    parents: tuple[str, ...] = ()  # 父代标识，用于追踪这个公式是由哪些个体产生的

    def record(self) -> dict:
        return {
            **self.candidate.as_dict(), "status": "evaluated", "origin": self.origin,
            "parents": list(self.parents), **asdict(self.fitness),
        }


@dataclass(frozen=True)
class Population:
    """一代种群，包含保留的精英和新评价的后代"""

    generation: int
    members: tuple[Individual, ...]

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be nonnegative")
        keys = [item.candidate.fingerprint for item in self.members]
        if len(keys) != len(set(keys)):
            raise ValueError("population contains duplicate expression trees")

    def ranked(self) -> tuple[Individual, ...]:
        return tuple(sorted(self.members, key=lambda item: item.fitness.score, reverse=True))

    def elites(self, count: int) -> tuple[Individual, ...]:
        return self.ranked()[:count]

    def median_score(self) -> float:
        scores = sorted(item.fitness.score for item in self.members)
        return scores[len(scores) // 2] if scores else 0.0
