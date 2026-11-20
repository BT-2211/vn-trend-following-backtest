"""qlab.py - every function used in the project, defined once.

The functions come from Lab 01-05 of the course "Du an Quant cho Ho so Du hoc".
run_all.py imports this file and runs the whole study in order.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------- settings
SYMBOLS = ["FPT", "VCB", "HPG", "VNM", "MWG", "MBB", "SSI", "VIC", "GAS", "CTG"]
INDEX = "VNINDEX"
START = "2016-01-01"        # backtest start; data before this only warms up MA200
PERIODS = 252               # trading sessions per year
LIMIT = 0.07                # HOSE daily price band (+/-7%)
BUY_COST = 0.0015           # brokerage fee on buy: 0.15%
SELL_COST = 0.0025          # brokerage fee 0.15% + personal income tax 0.10% on sell


# ---------------------------------------------------------------- data (Lab 01-02)
def load_data(path):
    """Read the long-format CSV (one row per date and symbol) and pivot closes to wide format."""
    df = pd.read_csv(path, parse_dates=["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    return df, close


def check_data(df):
    """Data-quality report per symbol. Every non-zero number must be explained in the report."""
    calendar = df.loc[df["symbol"] == INDEX, "date"]
    rows = []
    for sym, g in df.groupby("symbol"):
        g = g.set_index("date").sort_index()
        r = g["close"].pct_change()
        rows.append({
            "symbol": sym,
            "rows": len(g),
            "first_date": g.index[0].date(),
            "last_date": g.index[-1].date(),
            "missing_days": int((~calendar[calendar >= g.index[0]].isin(g.index)).sum()),
            "duplicate_days": int(g.index.duplicated().sum()),
            "bad_ohlc": int(((g["high"] < g[["open", "close"]].max(axis=1))
                             | (g["low"] > g[["open", "close"]].min(axis=1))).sum()),
            "beyond_limit": int((r.abs() > LIMIT + 0.005).sum()),
            "zero_volume": int((g["volume"] == 0).sum()),
        })
    return pd.DataFrame(rows).set_index("symbol")


# ---------------------------------------------------------------- metrics (Lab 01)
def sharpe(r):
    """Annualised Sharpe ratio, risk-free rate assumed 0."""
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(PERIODS) if r.std() > 0 else np.nan


def drawdown(r):
    """Drop from the highest value reached so far."""
    equity = (1 + r.fillna(0)).cumprod()
    return equity / equity.cummax() - 1


def perf_stats(r):
    r = r.dropna()
    monthly = (1 + r).resample("ME").prod() - 1
    return {
        "CAGR": (1 + r).prod() ** (PERIODS / len(r)) - 1,
        "Vol": r.std() * np.sqrt(PERIODS),
        "Sharpe": sharpe(r),
        "MaxDD": drawdown(r).min(),
        "WorstMonth": monthly.min(),
    }


# ---------------------------------------------------------------- indicators and signals (Lab 03)
def sma(s, n):
    return s.rolling(n).mean()


def golden(c):
    """Golden cross: 1 (hold the stock) when MA50 > MA200, else 0 (hold cash)."""
    return (sma(c, 50) > sma(c, 200)).astype(float)


def ma_cross(fast, slow):
    return lambda c: (sma(c, fast) > sma(c, slow)).astype(float)


def price_above(n):
    return lambda c: (c > sma(c, n)).astype(float)


# ---------------------------------------------------------------- Vietnamese rules (Lab 04)
def execute_vn(signal, settlement=2, delay=1):
    """Signal -> actual holding.

    delay=1      : signal uses the close of day t, so the order fills at the ATC of day t+1.
    long-only    : the holding is only ever 0 or 1 (no short selling).
    settlement=2 : shares bought on day i can be sold from day i+2 onward (T+2).
    """
    want = signal.shift(delay).fillna(0).to_numpy()
    held = np.zeros(len(want))
    state, bought = 0, -10**9
    for i, w in enumerate(want):
        if state == 0 and w == 1:
            state, bought = 1, i                          # buy at the ATC of day i
        elif state == 1 and w == 0 and i - bought >= settlement:
            state = 0                                     # sell only once the shares have arrived
        held[i] = state
    return pd.Series(held, index=signal.index)


def run_backtest(signal, r, settlement=2, delay=1, buy_cost=BUY_COST, sell_cost=SELL_COST):
    """One stock. Returns daily holding, gross return, net return (after fees and tax) and trades."""
    held = execute_vn(signal, settlement, delay)
    change = held.diff().fillna(held)                      # +1 = buy, -1 = sell
    cost = (change > 0) * buy_cost + (change < 0) * sell_cost
    gross = held.shift(1).fillna(0) * r.fillna(0)          # earn day t's return if held at close of t-1
    return pd.DataFrame({"held": held, "gross": gross, "net": gross - cost, "trades": change.abs()})


def run_portfolio(strategy, stocks, ret, start=START, **kw):
    """Same strategy on each stock; the portfolio return is the average of the 10 stock returns.

    Note: averaging daily returns means the 10 sleeves are rebalanced back to 1/10 every day.
    """
    per = {s: run_backtest(strategy(stocks[s]), ret[s], **kw).loc[start:] for s in stocks.columns}
    port = pd.concat({s: b["net"] for s, b in per.items()}, axis=1).mean(axis=1)
    return port, per


def buy_and_hold_no_rebalance(stocks, start=START):
    """True buy-and-hold: 1/10 of capital in each stock on the first day, never rebalanced."""
    p = stocks.loc[start:]
    value = (p / p.iloc[0]).mean(axis=1)
    return value.pct_change()


def hypothetical_long_short(signal, r, cost=0.002):
    """What-if test only: short selling is NOT allowed on the Vietnamese cash market."""
    pos = (2 * signal - 1).shift(1).fillna(0)              # 1 = long, -1 = short
    return pos.shift(1).fillna(0) * r.fillna(0) - pos.diff().abs().fillna(pos.abs()) * cost


# ---------------------------------------------------------------- robustness (Lab 05)
def robust_test(port, bench, per, ret, is_share=0.7):
    """Seven pre-set tests. GOLD = at least 6 passed, SILVER = 5, otherwise FAIL."""
    net = port.dropna()
    cut = int(len(net) * is_share)
    s_is, s_oos = sharpe(net.iloc[:cut]), sharpe(net.iloc[cut:])
    monthly = (1 + net).resample("ME").prod() - 1
    wins = sum(sharpe(b["net"]) > sharpe(ret[s].loc[b.index]) for s, b in per.items())
    tests = {
        "T1 OOS Sharpe >= 0.5": s_oos >= 0.5,
        "T2 OOS > 0 in >= 2 of 3 splits": sum(sharpe(net.iloc[int(len(net) * f):]) > 0
                                              for f in (0.5, 0.6, 0.7)) >= 2,
        "T3 IS/OOS <= 2": s_oos > 0 and s_is / s_oos <= 2,
        "T4 worst month >= -15%": monthly.min() >= -0.15,
        "T5 MaxDD >= -30%": drawdown(net).min() >= -0.30,
        "T6 Sharpe >= buy-and-hold": sharpe(net) >= sharpe(bench.reindex(net.index)),
        "T7 beats buy-and-hold in >= 6 of 10 stocks": wins >= 6,
    }
    n = sum(bool(v) for v in tests.values())
    return {"Sharpe": round(float(sharpe(net)), 3), "IS": round(float(s_is), 3), "OOS": round(float(s_oos), 3),
            "OOS_start": net.index[cut].date(), "stocks_won": int(wins),
            **{k: bool(v) for k, v in tests.items()}, "n_pass": int(n),
            "grade": "GOLD" if n >= 6 else "SILVER" if n == 5 else "FAIL"}


def deflated_sharpe(best, all_returns):
    """Bailey & Lopez de Prado (2014): probability that the best Sharpe beats the best expected by luck."""
    r = best.dropna()
    sr, n = r.mean() / r.std(), len(all_returns)
    sd = np.std([x.mean() / x.std() for x in all_returns], ddof=1)
    g = 0.5772156649                                       # Euler-Mascheroni constant
    sr0 = sd * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    return norm.cdf((sr - sr0) * np.sqrt(len(r) - 1)
                    / np.sqrt(1 - r.skew() * sr + (r.kurt() + 2) / 4 * sr ** 2))
