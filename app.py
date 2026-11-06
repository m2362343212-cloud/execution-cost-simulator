"""Streamlit page: how fast should a large sell order be executed?   streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from execsim.almgren_chriss import (
    Order, cost_variance, expected_cost, frontier, immediate_holdings, lambda_from_urgency,
    market_from_bps, optimal_holdings, simulate_shortfall, to_bps, trades, twap_holdings,
)

st.set_page_config(page_title="Execution cost simulator", layout="wide")
st.title("How fast should a large sell order be executed?")
st.write(
    "Selling quickly pushes the price against you. Selling slowly leaves you exposed to the price "
    "moving while you wait. This page shows that trade-off with the Almgren-Chriss (2000) model, using "
    "**simulated** prices and parameters you choose."
)

with st.sidebar:
    st.header("The order")
    shares = st.number_input("Shares to sell", 1_000, 50_000_000, 500_000, step=50_000)
    days = st.number_input("Time allowed (trading days)", 0.1, 10.0, 1.0, step=0.1)
    intervals = st.slider("Number of trading intervals", 5, 78, 20)
    st.header("The market")
    s0 = st.number_input("Starting price ($)", 1.0, 1000.0, 50.0)
    vol = st.number_input("Daily volatility (%)", 0.1, 10.0, 2.0, step=0.1)
    adv = st.number_input("Average daily volume (shares)", 10_000, 500_000_000, 5_000_000, step=500_000)
    temp = st.number_input("Temporary impact (bps per 1% of daily volume)", 0.0, 100.0, 10.0, step=1.0,
                           help="Extra cost per share while you are trading fast. Trading at 5% of daily "
                                "volume costs 5x this.")
    perm = st.number_input("Permanent impact (bps per 1% of daily volume sold)", 0.0, 100.0, 5.0, step=1.0,
                           help="How far the price stays moved after you have sold.")
    spread = st.number_input("Half bid-ask spread (bps)", 0.0, 50.0, 2.0, step=0.5)
    st.header("Your urgency")
    urgency = st.slider("Urgency (0 = even pace)", 0.0, 100.0, 4.0, step=0.5,
                        help="0 sells at an even pace (TWAP). Higher values sell earlier to cut price risk.")

market = market_from_bps(s0, vol, adv, temp, perm, spread)
order = Order(float(shares), float(days), int(intervals))
try:
    lam = lambda_from_urgency(urgency, market, order)
    chosen = optimal_holdings(market, order, lam)
except ValueError as e:
    st.error(str(e))
    st.stop()
twap = twap_holdings(order)
now = immediate_holdings(order)

participation = shares / days / adv * 100
st.caption(f"This order is {shares / adv * 100:.1f}% of one average day's volume, "
           f"which is {participation:.1f}% of daily volume if sold at an even pace.")


def bps(c):
    return to_bps(c, market, order)


@st.cache_data(show_spinner=False)
def paths(h_key, _x, n=20000):
    return simulate_shortfall(market, order, _x, n_paths=n, seed=7)


names = ["Your schedule", "Even pace (TWAP)", "Sell at once"]
schedules = [chosen, twap, now]
mc = [bps(paths((i, shares, days, intervals, s0, vol, adv, temp, perm, spread, urgency), x))
      for i, x in enumerate(schedules)]
rows = []
for name, x, c in zip(names, schedules, mc):
    rows.append({"Schedule": name,
                 "Expected cost (bps)": round(bps(expected_cost(market, order, x)), 1),
                 "Risk: std. dev. (bps)": round(bps(np.sqrt(cost_variance(market, order, x))), 1),
                 "Bad day: cost at the 95th percentile (bps)": round(float(np.percentile(c, 95)), 1)})
st.dataframe(rows, hide_index=True, width="stretch")

e_me, e_tw = rows[0]["Expected cost (bps)"], rows[1]["Expected cost (bps)"]
r_me, r_tw = rows[0]["Risk: std. dev. (bps)"], rows[1]["Risk: std. dev. (bps)"]
if urgency == 0:
    st.info("At urgency 0 your schedule is the even-pace schedule.")
else:
    st.info(f"Compared with an even pace, your schedule pays {e_me - e_tw:.1f} bps more in expected cost "
            f"and cuts the risk (standard deviation) by {r_tw - r_me:.1f} bps.")

c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    t = np.arange(order.intervals + 1) * order.tau
    ax.plot(t, chosen / shares * 100, label="Your schedule", lw=2)
    ax.plot(t, twap / shares * 100, label="Even pace", ls="--")
    ax.set_xlabel("Days")
    ax.set_ylabel("Shares still to sell (%)")
    ax.set_title("Selling schedule")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with c2:
    grid = np.concatenate([[0], np.geomspace(0.1, 100, 40)])
    fr = frontier(market, order, grid)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.plot([bps(r["std_dev"]) for r in fr], [bps(r["expected_cost"]) for r in fr], color="gray")
    ax.scatter([r_me], [e_me], s=70, label="Your schedule", zorder=3)
    ax.scatter([r_tw], [e_tw], s=70, marker="s", label="Even pace", zorder=3)
    ax.set_xlabel("Risk: std. dev. of cost (bps)")
    ax.set_ylabel("Expected cost (bps)")
    ax.set_title("Cost vs. risk (best schedules)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with c3:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    hi = float(np.percentile(np.concatenate(mc[:2]), 99.5))
    lo = float(np.percentile(np.concatenate(mc[:2]), 0.5))
    ax.hist(mc[0], bins=60, range=(lo, hi), alpha=0.6, label="Your schedule")
    ax.hist(mc[1], bins=60, range=(lo, hi), alpha=0.6, label="Even pace")
    ax.set_xlabel("Cost of the sale (bps of starting value)")
    ax.set_ylabel("Simulated outcomes")
    ax.set_title("20,000 simulated price paths")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("What this is, and what it is not"):
    st.markdown(
        """
- **A simulation of a textbook model**, not a trading system. Prices are a random walk and every
  number comes from the inputs on the left, not from real trades.
- **Impact is linear** in trading speed here. Real impact grows more slowly than that (closer to a
  square root), so the model overstates the cost of very large, very fast orders.
- **Cost** means *implementation shortfall*: what you gave up compared with selling everything at the
  starting price. Risk is the standard deviation of that cost.
- The even-pace schedule has the lowest expected cost; selling at once has no price risk but a large
  impact cost. The curve in the middle chart is the best you can do for each level of risk.
- Not modelled: volume that changes through the day, other traders reacting, limit orders, fees.
"""
    )
