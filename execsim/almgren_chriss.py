"""
Optimal execution of a large sell order, after Almgren & Chriss (2000), "Optimal
execution of portfolio transactions".

The question: you must sell X shares within T days. Selling fast moves the price
against you (market impact). Selling slowly leaves you exposed to the price
drifting away (risk). What schedule balances the two?

Discrete-time model, N intervals of length tau = T / N, n_k shares sold in interval k,
x_k shares still held after interval k (x_0 = X, x_N = 0):

    price:      S_k  = S_{k-1} + sigma * sqrt(tau) * xi_k - gamma * n_k      (permanent impact)
    you get:    S~_k = S_{k-1} - eps - eta * n_k / tau                       (temporary impact)
    shortfall:  C    = X * S_0 - sum_k n_k * S~_k        (what you gave up versus selling at S_0)

    E[C] = 0.5 * gamma * X^2 + eps * X + (eta_tilde / tau) * sum_k n_k^2,   eta_tilde = eta - 0.5 * gamma * tau
    V[C] = sigma^2 * tau * sum_k x_k^2

The optimal schedule minimizes E[C] + lambda * V[C]:

    x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T),
    kappa = (1/tau) * arccosh(0.5 * kappa_tilde^2 * tau^2 + 1),   kappa_tilde^2 = lambda * sigma^2 / eta_tilde

lambda = 0 gives the straight-line schedule (TWAP). Larger lambda sells earlier.

This is a stylized simulation: impact is LINEAR in trading rate (real impact is closer to a
square root), prices are a random walk, and the parameters are inputs you choose, not
estimates from real trades.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    """Price and impact parameters, in plain dollars and shares.

    s0     starting price ($)
    sigma  price volatility ($ per sqrt(day))
    gamma  permanent impact: each share sold lowers the price by gamma ($ per share)
    eta    temporary impact: selling at a rate of v shares/day costs eta * v $ per share
    eps    fixed cost per share, e.g. half the bid-ask spread ($)
    """
    s0: float
    sigma: float
    gamma: float
    eta: float
    eps: float = 0.0


@dataclass(frozen=True)
class Order:
    shares: float      # X
    days: float        # T
    intervals: int     # N

    @property
    def tau(self) -> float:
        return self.days / self.intervals


def market_from_bps(s0: float, daily_vol_pct: float, adv: float, temp_bps_per_1pct_adv: float,
                    perm_bps_per_1pct_adv: float, half_spread_bps: float) -> Market:
    """Build a Market from quantities a trader thinks in.

    adv                     average daily volume (shares)
    temp_bps_per_1pct_adv   extra cost per share, in basis points of price, when trading at a rate of
                            1% of ADV per day (so 10 bps -> trading at 5% of ADV costs 50 bps)
    perm_bps_per_1pct_adv   permanent price move, in bps, after selling 1% of ADV in total
    """
    one_pct_adv = 0.01 * adv
    return Market(
        s0=s0,
        sigma=s0 * daily_vol_pct / 100.0,
        gamma=s0 * perm_bps_per_1pct_adv / 1e4 / one_pct_adv,
        eta=s0 * temp_bps_per_1pct_adv / 1e4 / one_pct_adv,
        eps=s0 * half_spread_bps / 1e4,
    )


def eta_tilde(m: Market, o: Order) -> float:
    et = m.eta - 0.5 * m.gamma * o.tau
    if et <= 0:
        raise ValueError("eta - 0.5*gamma*tau must be positive: use fewer intervals or a smaller permanent impact")
    return et


def lambda_from_urgency(urgency: float, m: Market, o: Order) -> float:
    """Risk aversion lambda from a dimensionless urgency rho = (kappa_tilde * T)^2.

    rho = 0 is TWAP; rho around 1 front-loads a little; rho around 25 sells most of the order early.
    """
    if urgency < 0:
        raise ValueError("urgency must be >= 0")
    return urgency * eta_tilde(m, o) / (m.sigma ** 2 * o.days ** 2)


# ---------------------------------------------------------------- schedules
def optimal_holdings(m: Market, o: Order, risk_aversion: float) -> np.ndarray:
    """x_0..x_N for the given lambda (length N + 1)."""
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0")
    N, tau, T = o.intervals, o.tau, o.days
    t = np.arange(N + 1) * tau
    if risk_aversion == 0:
        return o.shares * (1 - t / T)
    kt2 = risk_aversion * m.sigma ** 2 / eta_tilde(m, o)
    kappa = np.arccosh(0.5 * kt2 * tau ** 2 + 1) / tau
    # sinh(kappa (T - t)) / sinh(kappa T), written with exponentials so a large kappa*T cannot overflow
    x = (np.exp(-kappa * t) - np.exp(-kappa * (2 * T - t))) / (1 - np.exp(-2 * kappa * T))
    x = o.shares * x
    x[0], x[-1] = o.shares, 0.0
    return x


def twap_holdings(o: Order) -> np.ndarray:
    return o.shares * (1 - np.arange(o.intervals + 1) / o.intervals)


def immediate_holdings(o: Order) -> np.ndarray:
    """Sell everything in the first interval."""
    x = np.zeros(o.intervals + 1)
    x[0] = o.shares
    return x


def trades(x: np.ndarray) -> np.ndarray:
    return x[:-1] - x[1:]


# ---------------------------------------------------------------- expected cost and risk
def expected_cost(m: Market, o: Order, x: np.ndarray) -> float:
    n = trades(x)
    return float(0.5 * m.gamma * o.shares ** 2 + m.eps * o.shares
                 + eta_tilde(m, o) / o.tau * np.sum(n ** 2))


def cost_variance(m: Market, o: Order, x: np.ndarray) -> float:
    return float(m.sigma ** 2 * o.tau * np.sum(x[1:] ** 2))


def frontier(m: Market, o: Order, urgencies) -> list[dict]:
    """Expected cost and risk (std. dev.) of the optimal schedule for each urgency."""
    rows = []
    for rho in urgencies:
        x = optimal_holdings(m, o, lambda_from_urgency(rho, m, o))
        rows.append({"urgency": rho, "expected_cost": expected_cost(m, o, x),
                     "std_dev": np.sqrt(cost_variance(m, o, x))})
    return rows


# ---------------------------------------------------------------- Monte Carlo
def simulate_shortfall(m: Market, o: Order, x: np.ndarray, n_paths: int = 20000, seed: int = 0) -> np.ndarray:
    """Implementation shortfall ($) on n_paths simulated price paths, following the model's price rules."""
    rng = np.random.default_rng(seed)
    n = trades(x)
    N, tau = o.intervals, o.tau
    xi = rng.standard_normal((n_paths, N))
    shocks = m.sigma * np.sqrt(tau) * xi
    # price before interval k: S_{k-1} = S_0 + (random-walk shocks up to k-1) - gamma * (shares already sold)
    sold_before = np.concatenate([[0.0], np.cumsum(n)[:-1]])
    walk_before = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(shocks, axis=1)[:, :-1]], axis=1)
    s_before = m.s0 + walk_before - m.gamma * sold_before
    got = s_before - m.eps - m.eta * n / tau
    return o.shares * m.s0 - (n * got).sum(axis=1)


def to_bps(cost: float, m: Market, o: Order) -> float:
    """A dollar cost as basis points of the order's starting value."""
    return cost / (o.shares * m.s0) * 1e4
