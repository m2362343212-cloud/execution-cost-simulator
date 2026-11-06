import numpy as np
import pytest

from execsim.almgren_chriss import (
    Market, Order, cost_variance, eta_tilde, expected_cost, frontier, immediate_holdings,
    lambda_from_urgency, market_from_bps, optimal_holdings, simulate_shortfall, to_bps, trades, twap_holdings,
)

# 500,000 shares of a $50 stock, 10% of a 5M-share average day, sold within one day in 20 steps.
M = market_from_bps(s0=50, daily_vol_pct=2, adv=5_000_000, temp_bps_per_1pct_adv=10,
                    perm_bps_per_1pct_adv=5, half_spread_bps=2)
O = Order(shares=500_000, days=1, intervals=20)


def test_schedule_starts_full_ends_empty_and_never_buys():
    for rho in [0, 0.5, 5, 50]:
        x = optimal_holdings(M, O, lambda_from_urgency(rho, M, O))
        assert x[0] == O.shares and x[-1] == 0
        assert np.all(trades(x) >= -1e-9)
        assert trades(x).sum() == pytest.approx(O.shares)


def test_zero_risk_aversion_is_twap():
    n = trades(optimal_holdings(M, O, 0.0))
    assert np.allclose(n, O.shares / O.intervals)
    assert np.allclose(optimal_holdings(M, O, 0.0), twap_holdings(O))


def test_closed_form_matches_a_direct_numerical_minimum():
    """Minimize E[C] + lambda*V[C] by solving the linear equations that set its gradient to zero,
    without using the sinh formula, and compare."""
    N, tau, X = O.intervals, O.tau, O.shares
    et = eta_tilde(M, O)
    for rho in [0.3, 4.0, 30.0]:
        lam = lambda_from_urgency(rho, M, O)
        a = et / tau
        b = lam * M.sigma ** 2 * tau
        A = np.zeros((N - 1, N - 1))
        np.fill_diagonal(A, 2 * a + b)
        A[np.arange(N - 2), np.arange(1, N - 1)] = -a
        A[np.arange(1, N - 1), np.arange(N - 2)] = -a
        rhs = np.zeros(N - 1)
        rhs[0] = a * X                                   # x_0 = X sits in the first equation
        interior = np.linalg.solve(A, rhs)
        x = optimal_holdings(M, O, lam)
        assert np.allclose(x[1:-1], interior, rtol=1e-8)


def test_expected_cost_matches_the_expected_price_paid():
    """Cost computed two ways: the closed form, and summing shares x (expected price received)."""
    x = optimal_holdings(M, O, lambda_from_urgency(3.0, M, O))
    n = trades(x)
    expected_price = M.s0 - M.gamma * (O.shares - x[:-1]) - M.eps - M.eta * n / O.tau
    direct = O.shares * M.s0 - np.sum(n * expected_price)
    assert expected_cost(M, O, x) == pytest.approx(direct, rel=1e-9)


def test_twap_cost_has_the_textbook_closed_form():
    expect = 0.5 * M.gamma * O.shares ** 2 + M.eps * O.shares + eta_tilde(M, O) * O.shares ** 2 / O.days
    assert expected_cost(M, O, twap_holdings(O)) == pytest.approx(expect, rel=1e-9)


def test_more_urgency_costs_more_but_is_less_risky():
    rows = frontier(M, O, [0, 0.5, 2, 8, 32])
    costs = [r["expected_cost"] for r in rows]
    risks = [r["std_dev"] for r in rows]
    assert all(b > a for a, b in zip(costs, costs[1:]))
    assert all(b < a for a, b in zip(risks, risks[1:]))


def test_twap_is_the_cheapest_and_selling_at_once_is_the_least_risky():
    twap, now = twap_holdings(O), immediate_holdings(O)
    assert expected_cost(M, O, twap) < expected_cost(M, O, optimal_holdings(M, O, lambda_from_urgency(5, M, O)))
    assert expected_cost(M, O, now) > expected_cost(M, O, twap)
    assert cost_variance(M, O, now) == 0


def test_monte_carlo_agrees_with_the_formulas():
    x = optimal_holdings(M, O, lambda_from_urgency(4.0, M, O))
    c = simulate_shortfall(M, O, x, n_paths=200_000, seed=1)
    assert c.mean() == pytest.approx(expected_cost(M, O, x), rel=0.01)
    assert c.std() == pytest.approx(np.sqrt(cost_variance(M, O, x)), rel=0.02)


def test_unit_conversion_from_bps():
    # trading at 1% of ADV per day should cost exactly temp_bps of price per share
    rate = 0.01 * 5_000_000
    assert M.eta * rate == pytest.approx(50 * 10 / 1e4)
    # selling 1% of ADV in total moves the price by perm_bps
    assert M.gamma * rate == pytest.approx(50 * 5 / 1e4)
    assert M.sigma == pytest.approx(1.0)
    assert to_bps(25_000, M, O) == pytest.approx(10.0)        # $25,000 of a $25,000,000 order = 10 bps


def test_too_many_intervals_for_the_permanent_impact_is_rejected():
    bad = Market(s0=50, sigma=1.0, gamma=1e-3, eta=1e-6)
    with pytest.raises(ValueError):
        optimal_holdings(bad, Order(shares=1000, days=1, intervals=10), 1e-6)


def test_extreme_urgency_does_not_overflow():
    x = optimal_holdings(M, O, lambda_from_urgency(1e6, M, O))
    assert np.all(np.isfinite(x))
    assert x[1] < 1e-3 * O.shares            # over 99.9% of the order is sold in the first interval
