"""组织因子进化流程，负责候选生成、合法性检查和去重。"""

from __future__ import annotations

from collections import Counter
from random import Random

from vnpy.alpha.expression import Node
from vnpy.alpha.mining.config import SearchSpace
from vnpy.alpha.mining.generator import ExpressionGenerator
from vnpy.alpha.mining.schema import Candidate, tree_fingerprint

from .config import EvolutionConfig
from .crossover import Crossover
from .fitness import FitnessEvaluator
from .mutator import Mutator
from .population import Individual, Population
from .selector import Selector


class EvolutionEngine:
    """逐代构建因子种群，选择过程只使用训练集评分"""

    def __init__(
        self,
        space: SearchSpace,
        config: EvolutionConfig,
        evaluator: FitnessEvaluator,
        *,
        selector: Selector | None = None,
        crossover: Crossover | None = None,
        mutator: Mutator | None = None,
    ) -> None:
        # 搜索空间限定字段、算子、树深度、节点数和回看长度；
        # 配置控制种群规模、代数等；评分器评价候选因子的效果。
        self.space, self.config, self.evaluator = space, config, evaluator

        # 选择、交叉和变异共享随机数生成器，便于固定种子复现。
        self.random = Random(config.seed)

        # 随机表达树生成器使用自己的随机数生成器。
        self.generator = ExpressionGenerator(space, seed=config.seed)

        # 支持注入自定义实现；未传入时使用默认模块。
        self.selector = selector or Selector(
            self.random,
            config.tournament_size,
        )
        self.crossover = crossover or Crossover(self.random)
        self.mutator = mutator or Mutator(space, self.random)

        # 全局记录已尝试的表达树指纹，跨代去重。
        self.seen: set[str] = set()

        # 从空的第 0 代开始，第一代通过随机生成建立初始种群。
        self.population = Population(0, ())

    def _accept(self, tree: Node, generation: int, origin: str, parents: tuple[str, ...] = (),) -> Individual | None:
        """检查、去重并评分；不符合结构约束时返回 None"""
        key = tree_fingerprint(tree)

        # 同一指纹只尝试一次。
        if key in self.seen:
            return None

        # 在验证前登记，因此验证失败的树也不会再次尝试。
        self.seen.add(key)

        try:
            # 限制树深度和节点数量，避免表达式过于复杂。
            if (
                tree.depth > self.space.max_depth
                or sum(1 for _ in tree.walk()) > self.space.max_nodes
            ):
                return None

            # 检查字段、参数类型等，并推导历史数据需求。
            analysis = self.generator.analyzer.analyze(
                tree,
                set(self.space.fields),
            )

            # 拒绝历史数据需求超过搜索空间限制的表达式。
            if analysis.lookback > self.space.max_lookback:
                return None

            # 尝试编译，确认表达树可以转换为受支持的计算表达式。
            self.generator.compiler.compile(
                tree,
                available_fields=set(self.space.fields),
            )

        except (TypeError, ValueError, KeyError):
            # 结构或编译检查失败，丢弃本次候选。
            return None

        # 保存表达树、生成代数、回看长度和依赖字段。
        candidate = Candidate(
            tree,
            generation,
            analysis.lookback,
            tuple(sorted(analysis.fields)),
        )

        # 使用默认的训练集评分，并记录来源和父代指纹。
        # 此处没有最低分门槛，零分候选也可以进入种群。
        return Individual(
            candidate,
            self.evaluator.score(tree),
            origin,
            parents,
        )

    def _propose(self, population: Population,) -> tuple[Node, str, tuple[str, ...]]:
        """提出一个候选表达树，返回树、来源和父代指纹。"""

        # 空种群只能随机生成；已有种群时仍保留 12% 的随机生成概率，
        # 为搜索引入新的表达式结构。
        if not population.members or self.random.random() < 0.12:
            return self.generator.grow_tree(), "random", ()

        # 从当前种群中选择一个父代。
        left = self.selector.select(population)

        # 在非随机生成分支中，以 55% 的概率执行变异。
        if self.random.random() < 0.55:
            return (
                self.mutator.mutate(left.candidate.tree),
                "mutation",
                (left.candidate.fingerprint,),
            )

        # 否则再选择一个父代，尝试交叉；两次选择可能得到同一个个体。
        right = self.selector.select(population)
        child = self.crossover.cross(
            left.candidate.tree,
            right.candidate.tree,
        )

        # 无法交叉时，回退到对第一个父代执行变异。
        if child is None:
            return (
                self.mutator.mutate(left.candidate.tree),
                "mutation",
                (left.candidate.fingerprint,),
            )

        return (
            child,
            "crossover",
            (
                left.candidate.fingerprint,
                right.candidate.fingerprint,
            ),
        )

    def next_generation(self, population: Population,) -> tuple[Population, list[Individual], dict]:
        """保留精英并补充新候选，构建下一代及其统计信息。"""
        generation = population.generation + 1

        # 直接继承上一代的精英，不重新生成或评分。
        elites = population.elites(self.config.elite_size)

        # 新候选与继承的精英一起，补足目标种群规模。
        target_new = self.config.population_size - len(elites)
        created: list[Individual] = []

        attempts = 0

        # 统计成功接收的候选来源，不包含被拒绝的尝试。
        origins: Counter[str] = Counter()

        # 重复提出候选并验证，直到种群补齐或达到尝试次数上限。
        while (len(created) < target_new and attempts < self.config.attempts_per_generation):
            attempts += 1
            tree, origin, parents = self._propose(population)
            item = self._accept(tree, generation, origin, parents)

            if item is not None:
                created.append(item)
                origins[origin] += 1

        # 重复或非法候选过多时，可能无法在限制内补齐种群。
        # 此时明确报错，避免继续运行一个规模不足的种群。
        if len(created) != target_new:
            raise RuntimeError(
                f"generation {generation} exhausted after {attempts} attempts: "
                f"{len(created)}/{target_new} new candidates"
            )

        # 下一代由继承的精英和本代新接收的候选组成。
        next_population = Population(
            generation,
            elites + tuple(created),
        )

        # 汇总本代生成情况及训练评分表现。
        summary = {
            "generation": generation,
            "new_candidates": len(created),
            "attempts": attempts,
            "origins": dict(origins),
            "best_training_score": (
                next_population.ranked()[0].fitness.score
            ),
            "median_training_score": next_population.median_score(),
        }

        return next_population, created, summary

    def run(self, on_generation=None) -> list[Individual]:
        """运行指定代数，返回本次运行中新接收的全部候选。"""
        history: list[Individual] = []

        for _ in range(self.config.generations):
            # 基于当前种群生成下一代。
            population, created, summary = self.next_generation(
                self.population
            )

            # 更新当前种群，供下一轮选择父代。
            self.population = population

            # 只记录新候选；直接继承的精英不重复加入历史。
            history.extend(created)

            # 每代完成后调用可选回调，用于保存候选和统计信息。
            if on_generation:
                on_generation(
                    population.generation,
                    created,
                    summary,
                )

        return history