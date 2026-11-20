"""run_all.py - reproduce every table and figure of the study with one command:

    python run_all.py

Reads data/vn_daily.csv, writes CSV tables to outputs/ and charts to figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qlab as q

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# 1. Data and quality check -------------------------------------------------------------
df, close = q.load_data(ROOT / "data" / "vn_daily.csv")
q.check_data(df).to_csv(OUT / "data_check.csv")
stocks, vnindex = close[q.SYMBOLS], close[q.INDEX]
ret = stocks.pct_change()
S = q.START

# 2. Main result: golden cross 50/200 vs benchmarks -------------------------------------
port, per = q.run_portfolio(q.golden, stocks, ret)
hold_ew = ret.loc[S:].mean(axis=1)                         # equal weight, rebalanced daily (course benchmark)
hold_true = q.buy_and_hold_no_rebalance(stocks)            # equal weight on day 1, never rebalanced
hold_idx = vnindex.pct_change().loc[S:]                    # VN-Index (reference only, not directly investable)
nets = {
    "Golden cross 50/200": port,
    "Buy-and-hold 10 stocks (daily rebalanced)": hold_ew,
    "Buy-and-hold 10 stocks (no rebalancing)": hold_true,
    "VN-Index": hold_idx,
}
base = pd.DataFrame({k: q.perf_stats(v) for k, v in nets.items()}).T
base.to_csv(OUT / "baseline_stats.csv")

yearly = pd.DataFrame({k: (1 + v).resample("YE").prod() - 1 for k, v in nets.items()})
yearly.index = yearly.index.year
yearly.to_csv(OUT / "yearly_returns.csv")

# 3. Where does the golden cross lose return? --------------------------------------------
gross = pd.concat({s: b["gross"] for s, b in per.items()}, axis=1).mean(axis=1)
held = pd.concat({s: b["held"] for s, b in per.items()}, axis=1)
trades = pd.concat({s: b["trades"] for s, b in per.items()}, axis=1)
years = len(port.dropna()) / q.PERIODS
net_s = port.dropna()
cut = net_s.index[int(len(net_s) * 0.7)]
details = pd.Series({
    "CAGR before fees and tax": q.perf_stats(gross)["CAGR"],
    "CAGR after fees and tax": q.perf_stats(port)["CAGR"],
    "Share of time in the market": held.mean().mean(),
    "Round trips per stock per year": trades.sum().mean() / 2 / years,
    "Out-of-sample start (70% split)": str(cut.date()),
    "Sharpe in-sample, golden cross": q.sharpe(port.loc[:cut].iloc[:-1]),
    "Sharpe in-sample, buy-and-hold (daily rebalanced)": q.sharpe(hold_ew.loc[:cut].iloc[:-1]),
    "Sharpe out-of-sample, golden cross": q.sharpe(port.loc[cut:]),
    "Sharpe out-of-sample, buy-and-hold (daily rebalanced)": q.sharpe(hold_ew.loc[cut:]),
}, name="value")
details.to_csv(OUT / "golden_cross_details.csv")

per_stock = pd.DataFrame({
    s: {"Sharpe golden cross": q.sharpe(b["net"]),
        "Sharpe buy-and-hold": q.sharpe(ret[s].loc[b.index]),
        "Round trips per year": b["trades"].sum() / 2 / years,
        "Share of time held": b["held"].mean()}
    for s, b in per.items()}).T
per_stock.to_csv(OUT / "per_stock.csv")

# 4. Which Vietnamese rule actually matters? (Sharpe of the portfolio) --------------------
rules = {}
rule_strats = {f"Price > MA{n}": q.price_above(n) for n in (5, 10, 20, 50)}
rule_strats["Golden cross 50/200"] = q.golden
for name, strat in rule_strats.items():
    p_full, per_full = q.run_portfolio(strat, stocks, ret)
    rules[name] = {
        "Look-ahead (wrong)": q.sharpe(q.run_portfolio(strat, stocks, ret, delay=0, settlement=0)[0]),
        "T0, no costs": q.sharpe(q.run_portfolio(strat, stocks, ret, settlement=0, buy_cost=0, sell_cost=0)[0]),
        "T0, costs": q.sharpe(q.run_portfolio(strat, stocks, ret, settlement=0)[0]),
        "T+2, costs": q.sharpe(p_full),
        "Round trips per stock per year": np.mean([b["trades"].sum() / 2 / years for b in per_full.values()]),
    }
pd.DataFrame(rules).T.to_csv(OUT / "vn_rules_effect.csv")

# 5. Cost of long-only: what if short selling were allowed? (hypothetical) -----------------
ls = pd.concat({s: q.hypothetical_long_short(q.golden(stocks[s]), ret[s]).loc[S:] for s in q.SYMBOLS},
               axis=1).mean(axis=1)
pd.DataFrame({"Long-only (actual rules)": q.perf_stats(port),
              "Long-short (hypothetical)": q.perf_stats(ls)}).T.to_csv(OUT / "long_only_vs_short.csv")

# 6. Strategy scan and robustness tests (21 trend-following rules) ------------------------
grid = {f"MA cross {f}/{s}": q.ma_cross(f, s)
        for f in (5, 10, 20, 50) for s in (50, 100, 150, 200) if f < s}
grid.update({f"Price > MA{n}": q.price_above(n) for n in (5, 10, 20, 50, 100, 200)})
board, scan_returns = [], {}
for name, strat in grid.items():
    p, pr = q.run_portfolio(strat, stocks, ret)
    scan_returns[name] = p
    board.append({"strategy": name, **q.robust_test(p, hold_ew, pr, ret)})
board = pd.DataFrame(board).sort_values("Sharpe", ascending=False)
best = board.iloc[0]["strategy"]
board["deflated_sharpe_of_best"] = np.nan
board.loc[board["strategy"] == best, "deflated_sharpe_of_best"] = q.deflated_sharpe(
    scan_returns[best], list(scan_returns.values()))
board.to_csv(OUT / "strategy_leaderboard.csv", index=False)
pd.DataFrame([q.robust_test(port, hold_ew, per, ret)], index=["Golden cross 50/200"]).T.to_csv(
    OUT / "robustness_golden_cross.csv")

# 7. Figures and LaTeX table ----------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for k, v in nets.items():
    (1 + v.fillna(0)).cumprod().plot(ax=ax[0], label=k)
    q.drawdown(v).plot(ax=ax[1])
ax[0].set_ylabel("Value of 1 VND"); ax[0].legend(fontsize=7); ax[1].set_ylabel("Drawdown")
fig.tight_layout(); fig.savefig(FIG / "fig_equity.png", dpi=200); plt.close(fig)

ax = (yearly * 100).plot.bar(figsize=(8, 3.5), ylabel="%")
ax.legend(fontsize=7); ax.figure.tight_layout(); ax.figure.savefig(FIG / "fig_yearly.png", dpi=200)
plt.close(ax.figure)

fasts, slows = (5, 10, 20, 50), (50, 100, 150, 200)
heat = pd.DataFrame([[q.sharpe(scan_returns[f"MA cross {f}/{s}"]) if f < s else np.nan for s in slows]
                     for f in fasts], index=fasts, columns=slows)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(heat.values, cmap="RdYlGn")
ax.set_xticks(range(len(slows)), slows); ax.set_yticks(range(len(fasts)), fasts)
ax.set_xlabel("Slow MA"); ax.set_ylabel("Fast MA"); fig.colorbar(im, label="Sharpe (after costs, T+2)")
for i in range(len(fasts)):
    for j in range(len(slows)):
        if not np.isnan(heat.values[i, j]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_heatmap.png", dpi=200); plt.close(fig)

pct = base.assign(**{c: base[c] * 100 for c in ["CAGR", "Vol", "MaxDD", "WorstMonth"]})
(FIG / "table_main.tex").write_text(pct.to_latex(
    float_format="%.2f", label="tab:main",
    caption="Performance 2016--2026, after fees and taxes, next-day ATC execution, T+2 and long-only. "
            "CAGR, Vol, MaxDD and WorstMonth in percent."), encoding="utf-8")

# 8. Short summary on screen --------------------------------------------------------------
pd.set_option("display.width", 140)
print("Data:", df["date"].min().date(), "to", df["date"].max().date(), "| backtest from", S)
print("\nMain table\n", base.round(3).to_string())
print("\nGolden cross details\n", details.to_string())
print("\nVietnamese rules (Sharpe)\n", pd.DataFrame(rules).T.round(2).to_string())
print("\nStrategies tested:", len(board), "| GOLD:", int((board.grade == "GOLD").sum()),
      "| best:", best, "| deflated Sharpe of best:", round(float(board.iloc[0]["deflated_sharpe_of_best"]), 3))
print("\nDone. Tables in outputs/, figures in figures/.")
