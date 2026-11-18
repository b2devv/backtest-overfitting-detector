"""
Backtest Overfitting Detector

Idea: generate lots of random trading rules, backtest all of them, keep the best.
The best one looks great even on pure noise. The Deflated Sharpe Ratio (Bailey and
Lopez de Prado, 2014) tells you the probability the result is real after
accounting for how many things you tried.

Usage:
    python detector.py                     # random noise data (proves the point)
    python detector.py --ticker SPY        # real data via yfinance
    python detector.py --ticker SPY --trials 5000 --seed 1
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EULER = 0.5772156649015329
DAYS = 252


# ---------- data ----------
def load_prices(ticker, years, seed):
    if ticker is None:
        rng = np.random.default_rng(seed)
        n = years * DAYS
        rets = rng.normal(0.0003, 0.01, n)  # random walk, no edge by construction
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="noise")
    import yfinance as yf
    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    close = df["Close"].squeeze().dropna()
    if close.empty:
        raise SystemExit(f"No data returned for {ticker}")
    return close


# ---------- random rules ----------
def random_rule(rng):
    kind = rng.choice(["sma_cross", "momentum", "mean_revert", "breakout"])
    if kind == "sma_cross":
        fast = int(rng.integers(3, 60))
        slow = int(rng.integers(fast + 5, 250))
        return kind, {"fast": fast, "slow": slow}
    if kind == "momentum":
        return kind, {"lookback": int(rng.integers(5, 250))}
    if kind == "mean_revert":
        return kind, {"lookback": int(rng.integers(5, 100)), "z": float(rng.uniform(0.5, 2.5))}
    return kind, {"lookback": int(rng.integers(10, 200))}


def positions(prices, kind, p):
    """Return +1 / -1 / 0 position series. Shifted one day later to avoid lookahead."""
    if kind == "sma_cross":
        sig = np.sign(prices.rolling(p["fast"]).mean() - prices.rolling(p["slow"]).mean())
    elif kind == "momentum":
        sig = np.sign(prices.pct_change(p["lookback"]))
    elif kind == "mean_revert":
        m = prices.rolling(p["lookback"]).mean()
        s = prices.rolling(p["lookback"]).std()
        z = (prices - m) / s
        sig = pd.Series(0.0, index=prices.index)
        sig[z > p["z"]] = -1.0
        sig[z < -p["z"]] = 1.0
    else:
        hi = prices.rolling(p["lookback"]).max().shift(1)
        lo = prices.rolling(p["lookback"]).min().shift(1)
        sig = pd.Series(0.0, index=prices.index)
        sig[prices > hi] = 1.0
        sig[prices < lo] = -1.0
        sig = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return sig.shift(1).fillna(0.0)


def backtest(prices, kind, p, cost_bps=1.0):
    r = prices.pct_change().fillna(0.0)
    pos = positions(prices, kind, p)
    trades = pos.diff().abs().fillna(0.0)
    return pos * r - trades * cost_bps / 1e4


# ---------- metrics ----------
def sharpe_period(x):
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 or np.isnan(sd) else x.mean() / sd


def expected_max_sharpe(sr_trials_period, n_trials):
    """Expected best Sharpe from pure luck, given the spread of trial Sharpes."""
    sigma = np.std(sr_trials_period, ddof=1)
    return sigma * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                    + EULER * norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(best_returns, sr0_period):
    """Probability the best strategy's true Sharpe beats the luck benchmark sr0."""
    T = len(best_returns)
    sr = sharpe_period(best_returns)
    g3 = skew(best_returns)
    g4 = kurtosis(best_returns, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0_period) * np.sqrt(T - 1) / denom))


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None, help="e.g. SPY. Omit for random noise.")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="overfit_result.png")
    a = ap.parse_args()

    prices = load_prices(a.ticker, a.years, a.seed)
    rng = np.random.default_rng(a.seed + 1)

    rules, rets = [], []
    for _ in range(a.trials):
        kind, p = random_rule(rng)
        rules.append((kind, p))
        rets.append(backtest(prices, kind, p))

    sr_p = np.array([sharpe_period(r) for r in rets])
    sr_ann = sr_p * np.sqrt(DAYS)
    best = int(np.argmax(sr_p))
    best_ret = rets[best]

    sr0_p = expected_max_sharpe(sr_p, a.trials)
    sr0_ann = sr0_p * np.sqrt(DAYS)
    dsr = deflated_sharpe(best_ret, sr0_p)

    label = a.ticker or "random noise"
    print(f"Data: {label}, {len(prices)} days, {a.trials} random rules")
    print(f"Best rule: {rules[best][0]} {rules[best][1]}")
    print(f"Best Sharpe (annualized):          {sr_ann[best]:.2f}")
    print(f"Sharpe expected from luck alone:   {sr0_ann:.2f}")
    print(f"Deflated Sharpe (prob. it's real): {dsr:.1%}")
    print("Verdict:", "LIKELY REAL" if dsr > 0.95 else "NOT DISTINGUISHABLE FROM LUCK")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(sr_ann, bins=40, color="#2a9d8f")
    ax[0].axvline(sr_ann[best], color="red", label=f"best of {a.trials}: {sr_ann[best]:.2f}")
    ax[0].axvline(sr0_ann, color="orange", ls="--", label=f"expected from luck: {sr0_ann:.2f}")
    ax[0].set_title("Sharpe of every random rule")
    ax[0].set_xlabel("annualized Sharpe")
    ax[0].set_ylabel("number of rules")
    ax[0].legend()
    (1 + best_ret).cumprod().plot(ax=ax[1], color="red", label="best rule")
    (1 + prices.pct_change().fillna(0)).cumprod().plot(ax=ax[1], color="gray", label="buy and hold")
    ax[1].set_title(f"Best rule looks great. Deflated Sharpe says {dsr:.0%} real")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(a.out, dpi=130)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
