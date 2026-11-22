# Long-Only Trend-Following in Vietnamese Equities under T+2 Settlement and Trading Costs

Backtest of moving-average trend-following rules (main case: the 50/200 golden cross) on 10 large HOSE stocks, 2016–2026, simulating the trading rules of the Vietnamese stock market.

## Research question

Does a moving-average strategy give better risk-adjusted returns than buying and holding the same 10 Vietnamese blue-chip stocks, once Vietnam's real trading rules are applied?

## Vietnamese market rules simulated

| Rule | How it is modelled |
|---|---|
| Next-day execution | The signal uses the close of day *t*, so the order fills at the ATC (closing auction) of day *t+1* |
| T+2 settlement | Shares bought on day *T* can only be sold from day *T+2* |
| Long-only | No short selling: the holding is either the stock (1) or cash (0) |
| Fees and tax | 0.15% brokerage on buys, 0.15% brokerage + 0.10% personal income tax on sells (0.40% per round trip) |

## Data

`data/vn_daily.csv`: daily adjusted prices (thousand VND) of FPT, VCB, HPG, VNM, MWG, MBB, SSI, VIC, GAS, CTG and the VN-Index, from the VCI source via the `vnstock` library, provided as a cleaned file by the course. Data starts in mid-2014; the backtest starts on 1 January 2016 so that the 200-day moving average has enough history.

## How to run

```bash
pip install -r requirements.txt
python run_all.py
```

In Google Colab, one cell:

```
!git clone https://github.com/BT-2211/vn-trend-following-backtest && cd vn-trend-following-backtest && pip install -r requirements.txt && python run_all.py
```

Running twice gives identical results.

## Repository structure

```
qlab.py          all functions (data check, indicators, Vietnamese execution rules, metrics, robustness tests)
run_all.py       runs the full study and writes every table and figure
data/            vn_daily.csv
outputs/         result tables (CSV), created by run_all.py
figures/         charts (PNG) and the LaTeX results table, created by run_all.py
notebooks/       Lab 01–07, the step-by-step notebooks
report/          research report (PDF and LaTeX), added with the final version
```

| Output file | Contents |
|---|---|
| `data_check.csv` | Missing/duplicate days, invalid candles, moves beyond the ±7% band, zero-volume days |
| `baseline_stats.csv` | CAGR, volatility, Sharpe, max drawdown, worst month: golden cross vs benchmarks |
| `golden_cross_details.csv` | Return before vs after costs, time in market, trades per year, in-sample vs out-of-sample Sharpe |
| `vn_rules_effect.csv` | Sharpe with each rule switched on/off (look-ahead, T0, costs, T+2) |
| `long_only_vs_short.csv` | Long-only vs a hypothetical version with short selling |
| `strategy_leaderboard.csv` | 21 trend-following rules, 7 robustness tests, GOLD/SILVER/FAIL, Deflated Sharpe |
| `per_stock.csv`, `yearly_returns.csv`, `robustness_golden_cross.csv` | Supporting tables |

## Main results

<!-- Fill in from outputs/ after running run_all.py. Every number must match the CSV files. -->

- Golden cross 50/200: CAGR 15.7%, Sharpe 1.04, max drawdown −31.6%
- Buy-and-hold 10 stocks (Daily rebalanced): CAGR 21.0%, Sharpe 1.00, max drawdown −40.1%
- Effect of T+2: none on the golden cross (Sharpe 1.04 with and without T+2) | Effect of fees and tax: CAGR 16.1% → 15.7% (about 0.3 percentage points per year); Sharpe 1.06 → 1.04
- Robustness: 5 of 7 tests passed (SILVER); 14 of 21 rules graded GOLD

## Limitations

- Survivorship bias: the 10 stocks were chosen because they are large today.
- All orders fill at the ATC price: no slippage, no 100-share lot size, no limit-down days where selling is impossible.
- Cash earns no interest.
- One period (2016–2026); the 21-rule grid was chosen by the author.
- The 10-stock portfolio averages daily returns, i.e. it is rebalanced to equal weights every day without rebalancing costs. `baseline_stats.csv` also reports a true buy-and-hold (no rebalancing) for comparison.
- The short-selling comparison is hypothetical and simplified: it ignores borrowing costs and T+2 for short positions.
- HOSE suspended trading on 23–24 January 2018 after an order-matching system failure on 22 January. The data keeps these two days with zero volume and unchanged prices; no trades occur in the backtest on these days, so the effect on results is negligible.

## Acknowledgements

Developed during the one-to-one course "Quant Project for Study-Abroad Applications" under the mentorship of Nguyen Van Thanh. The lab notebooks and core functions come from the course materials; `qlab.py` and `run_all.py` combine them into one pipeline.

## References

- Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5), 1731–1764.
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
