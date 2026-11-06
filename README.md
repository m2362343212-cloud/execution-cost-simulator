# Execution Cost Simulator

How fast should a large sell order be executed?

Selling quickly pushes the price against you (market impact). Selling slowly leaves you exposed to the price moving while you wait (risk). This project simulates that trade-off with the Almgren-Chriss (2000) optimal-execution model and shows, for a given order, what an even-paced sale costs, what selling earlier costs, and how much risk each one removes.

It is a **simulation of a textbook model**. Prices are a random walk and every parameter is an input you choose. Nothing here is calibrated to real trades, and nothing here is a trading system.

```
pip install -r requirements.txt
streamlit run app.py          # interactive page
python -m pytest -q           # 11 tests, under a second
```

## What it shows (default example)

500,000 shares of a $50 stock (10% of an average 5M-share day), sold within one day in 20 steps, 2% daily volatility, 10 bps of temporary impact per 1% of daily volume, 5 bps of permanent impact, 2 bps half-spread:

| Urgency | Expected cost | Risk (std. dev.) |
|---|---|---|
| 0 (even pace, TWAP) | 125.8 bps | 111.1 bps |
| 4 | 144.4 bps | 88.8 bps |
| 16 | 224.7 bps | 63.6 bps |
| 64 | 414.3 bps | 40.6 bps |

So an even pace is cheapest on average but the most exposed to the price moving; each step toward selling earlier buys less risk and costs more. Selling everything in the first interval removes the price risk entirely and costs about 2,000 bps in this setup, which is why nobody does it.

## The model

`N` intervals of length `tau = T/N`; `n_k` shares sold in interval `k`; `x_k` shares still held.

```
price after each trade:   S_k  = S_{k-1} + sigma*sqrt(tau)*xi_k - gamma*n_k      permanent impact
price you actually get:   S~_k = S_{k-1} - eps - eta*n_k/tau                      temporary impact + spread
cost of the sale:         C    = X*S_0 - sum_k n_k*S~_k                           shortfall vs. selling at S_0

E[C] = 0.5*gamma*X^2 + eps*X + (eta_tilde/tau) * sum n_k^2      eta_tilde = eta - 0.5*gamma*tau
V[C] = sigma^2 * tau * sum x_k^2
```

The best schedule minimizes `E[C] + lambda*V[C]` and has the closed form `x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T)`. `lambda = 0` is a straight line (TWAP); larger `lambda` sells earlier. The page uses a dimensionless **urgency** `rho = (kappa_tilde * T)^2` instead of `lambda`, because `lambda` has awkward units: `rho = 0` is TWAP, around 1 is a small tilt toward early selling, and around 25 sells most of the order early.

Inputs are in the units a trader thinks in (price, daily volatility %, average daily volume, impact in bps per 1% of daily volume), converted to the model's `sigma, gamma, eta, eps` in `market_from_bps`.

## How it is checked

The tests do not just re-state the formulas:

- **The closed form is checked against a separate calculation.** The optimal holdings from the sinh formula are compared with the solution of the linear equations that set the gradient of `E + lambda*V` to zero, solved directly with no sinh in it.
- **Expected cost is computed two ways:** the formula above, and by summing shares times the expected price actually received, from the price rules.
- **Monte Carlo agrees:** over 200,000 simulated price paths, the mean cost is within 1% and the standard deviation within 2% of the formulas.
- **Known limits:** urgency 0 gives equal trades; the TWAP cost matches its textbook closed form; more urgency always means higher expected cost and lower risk; selling at once has zero price risk.
- **Edge cases:** the order never turns into buying; huge urgency does not overflow; too many intervals for the permanent impact (`eta - 0.5*gamma*tau <= 0`) is rejected with a message.

## Limitations

- **Impact is linear in trading speed.** Real impact grows more slowly (closer to a square root of the rate), so the model overstates the cost of very large, very fast orders.
- **Constant volatility and volume.** No intraday volume pattern (a VWAP-style schedule would need one), no volatility changes.
- **Other traders do not react,** and there are no limit orders, fees or partial fills.
- **Parameters are not estimated from data.** To use this for real decisions you would fit impact and volatility to a firm's own execution records.

## Where it fits

Portfolio and signal projects decide *what* to hold. This one looks at *how much it costs to trade it*: the cost of a trade depends on how it is executed, and that cost is part of whether a strategy works at all. The [Portfolio Tracker's backtester](https://github.com/m2362343212-cloud/portfolio-tracker) deducts a fixed trading cost for that reason; this project models where such a cost comes from for a single large order.

Reference: R. Almgren and N. Chriss, "Optimal execution of portfolio transactions", *Journal of Risk* 3 (2000).
