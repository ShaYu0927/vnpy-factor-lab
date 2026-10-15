import math
import random


class DiscreteGbmModel:
    """Approximate geometric Brownian motion using discrete up/down steps."""

    def __init__(self, mu: float, sigma: float, seed: int | None = None,) -> None:
        if not math.isfinite(mu):
            raise ValueError("mu must be finite")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")

        # Drift of the log price, following the textbook's convention.
        self.mu = mu
        self.sigma = sigma

        # Initialize once; do not reset the seed for each simulation.
        self.random = random.Random(seed)

    def simulate_path(self, initial_price: float, total_time: float, steps: int,) -> list[float]:
        """Return a price path containing steps + 1 values, including the initial price."""

        if not math.isfinite(initial_price) or initial_price <= 0:
            raise ValueError("initial_price must be finite and positive")
        if not math.isfinite(total_time) or total_time <= 0:
            raise ValueError("total_time must be finite and positive")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")

        dt = total_time / steps

        # Compute the up multiplier, down multiplier, and up probability.
        u = math.exp(self.sigma * math.sqrt(dt))
        d = math.exp(-self.sigma * math.sqrt(dt))
        p = 0.5 * (1 + self.mu / self.sigma * math.sqrt(dt))

        if not 0 <= p <= 1:
            raise ValueError("Up probability is outside [0, 1]; increase steps")

        price = initial_price
        prices = [price]

        # Randomly select an up or down move at each time step.
        for _ in range(steps):
            if self.random.random() < p:
                price *= u
            else:
                price *= d

            prices.append(price)

        return prices

    def estimate(self, initial_price: float, total_time: float, steps: int, simulations: int, threshold: float,) -> dict[str, float]:
        """Estimate the terminal mean and probabilities from simulated paths.

        Return the mean terminal price, the probability of exceeding the
        initial price, and the probability of falling below the threshold.
        """

        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations <= 0
        ):
            raise ValueError("simulations must be a positive integer")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")

        total_final_price = 0.0
        above_initial_count = 0
        below_threshold_count = 0

        for _ in range(simulations):
            path = self.simulate_path(initial_price=initial_price, total_time=total_time, steps=steps,)

            # Only the terminal price is needed for these statistics.
            final_price = path[-1]

            total_final_price += final_price

            if final_price > initial_price:
                above_initial_count += 1

            if final_price < threshold:
                below_threshold_count += 1

        # Estimate the expectation and probabilities using sample averages.
        return {
            "mean_final_price": total_final_price / simulations,
            "prob_above_initial": above_initial_count / simulations,
            "prob_below_threshold": below_threshold_count / simulations,
        }