"""Parent selection from the current population."""

from __future__ import annotations

from random import Random

from .ga_population import Individual, Population


class Selector:
    """Select a parent with a bounded tournament from the current population."""

    def __init__(self, random: Random, tournament_size: int) -> None:
        if tournament_size < 1:
            raise ValueError("tournament_size must be positive")
        self.random = random
        self.tournament_size = tournament_size

    def select(self, population: Population) -> Individual:
        if not population.members:
            raise ValueError("cannot select a parent from an empty population")
        contestants = self.random.sample(population.members,
                                         min(self.tournament_size, len(population.members)))
        return max(contestants, key=lambda item: item.fitness.score)
