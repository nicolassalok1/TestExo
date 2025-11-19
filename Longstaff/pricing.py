import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import norm
from dataclasses import dataclass

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class StochasticProcess(ABC):
    """Represente a Stochastic process"""

    @abstractmethod
    def simulate(self):
        ...


@dataclass
class GeometricBrownianMotion(StochasticProcess):
    """
    A classic geometric brownian motion which can be simulated.
    The closed form formula allow a fully vectorized calculation of the paths.
    """

    mu: float
    sigma: float

    def simulate(
        self, s0: float, T: int, n: int, m: int, v0: float = None
    ) -> pd.DataFrame:  # n = number of paths, m = number of discretization points

        dt = T / m
        np.random.seed(0)
        W = np.cumsum(np.sqrt(dt) * np.random.randn(m + 1, n), axis=0)
        W[0] = 0

        T = np.ones(n).reshape(1, -1) * np.linspace(0, T, m + 1).reshape(-1, 1)

        s = s0 * np.exp((self.mu - 0.5 * self.sigma**2) * T + self.sigma * W)

        return s


@dataclass
class HestonProcess(StochasticProcess):
    """
    An Heston process which can be simulated using Milstein schema.
    """

    mu: float
    kappa: float
    theta: float
    eta: float
    rho: float

    def simulate(
        self, s0: float, v0: float, T: int, n: int, m: int
    ) -> pd.DataFrame:  # n = number of paths, m = number of discretization points

        dt = T / m
        z1 = np.random.randn(m, n)
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.randn(m, n)

        s = np.zeros((m + 1, n))
        x = np.zeros((m + 1, n))
        v = np.zeros((m + 1, n))

        s[0] = s0
        v[0] = v0

        for i in range(m):

            v[i + 1] = (
                v[i]
                + self.kappa * (self.theta - v[i]) * dt
                + self.eta * np.sqrt(v[i] * dt) * z1[i]
                + self.eta**2 / 4 * (z1[i] ** 2 - 1) * dt
            )
            v = np.where(v > 0, v, -v)

            x[i + 1] = x[i] + (self.mu - v[i] / 2) * dt + np.sqrt(v[i] * dt) * z2[i]

            s[1:] = s[0] * np.exp(x[1:])

        return s


@dataclass
class Option:
    """
    Representation of an option derivative
    """

    s0: float
    T: int
    K: int
    v0: float = None
    call: bool = True

    def payoff(self, s: np.ndarray) -> np.ndarray:
        payoff = np.maximum(s - self.K, 0) if self.call else np.maximum(self.K - s, 0)
        return payoff


def monte_carlo_simulation(option: Option, process: StochasticProcess, n: int, m: int, alpha: float = 0.05) -> float:
    """
    Given an option and a process followed by the underlying, calculate the classic monte carlo price estimator
    """
    # n = number of paths, m = number of discretization points
    s = process.simulate(s0=option.s0, v0=option.v0, T=option.T, n=n, m=m)
    st = s[-1]
    payoffs = option.payoff(s=st)

    discount = np.exp(-process.mu * option.T)
    price = np.mean(payoffs) * discount

    quantile = norm.ppf(1 - alpha / 2)
    confidence_interval = [
        np.round(price - quantile * np.std(payoffs * discount) / np.sqrt(n), 2),
        np.round(price + quantile * np.std(payoffs * discount) / np.sqrt(n), 2),
    ]

    print(f"The price of {option!r} = {price:.2f}")
    print(f"{(1-alpha)*100}% confidence interval = {confidence_interval}")

    return np.round(price, 2)


def monte_carlo_simulation_LS(option: Option, process: StochasticProcess, n: int, m: int, alpha: float = 0.05) -> float:
    """
    Given an option and a process followed by the underlying, calculate the option value using the Longstaff-Schwartz algorithme
    """
    # n = number of path, m = number of discretization points

    s = process.simulate(s0=option.s0, v0=option.v0, T=option.T, n=n, m=m)

    payoffs = option.payoff(s=s)

    v = np.zeros_like(payoffs)
    v[-1] = payoffs[-1]

    dt = option.T / m
    discount = np.exp(-process.mu * dt)

    for t in range(m - 1, 0, -1):
        polynome = Polynomial.fit(s[t], discount * v[t + 1], 5)
        c = polynome(s[t])
        v[t] = np.where(payoffs[t] > c, payoffs[t], discount * v[t + 1])

    price = discount * np.mean(v[1])

    print(f"The price of {option!r} = {round(price, 4)}")


def black_scholes_merton(r, sigma, option: Option):
    """
    Calculate the price of vanilla options using BSM formula
    """
    d1 = (np.log(option.s0 / option.K) + (r + sigma**2 / 2) * option.T) / (sigma * np.sqrt(option.T))
    d2 = d1 - sigma * np.sqrt(option.T)

    price = option.s0 * norm.cdf(d1) - option.K * np.exp(-r * option.T) * norm.cdf(d2)
    price = price if option.call else price - option.s0 + option.K * np.exp(-r * option.T)

    return np.round(price, 2)


def crr_pricing(r=0.1, sigma=0.2, option: Option = Option(s0=100, T=1, K=100, call=False), n=250):
    """
    Calculate the price of an American option using a Cox–Ross–Rubinstein tree.
    """
    if n <= 0:
        raise ValueError("n must be positive for the CRR tree.")

    dt = option.T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    a = np.exp(r * dt)
    p = (a - d) / (u - d)
    q = 1 - p
    discount = np.exp(-r * dt)

    asset_prices = np.array([option.s0 * (u**j) * (d ** (n - j)) for j in range(n + 1)])
    option_values = option.payoff(asset_prices)

    for step in range(n - 1, -1, -1):
        continuation_values = discount * (p * option_values[1:] + q * option_values[:-1])
        asset_prices = np.array([option.s0 * (u**j) * (d ** (step - j)) for j in range(step + 1)])
        exercise_values = option.payoff(asset_prices)
        option_values = np.maximum(exercise_values, continuation_values)

    return float(option_values[0])
