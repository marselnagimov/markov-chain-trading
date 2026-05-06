# Markov Chain Multi-Market Trading Strategy

A backtested quantitative trading strategy that uses **Markov chain state transitions** on daily returns to generate long/short signals across global equity markets, with Kelly-criterion sizing, intraday stop losses, and an index-rotation fallback for quiet days.

The backtest benchmarks performance against SPY (total return) and reports CAPM alpha, beta, Sharpe ratio, and CAGR.

---

## Strategy Overview

1. **Universe construction.** Pulls constituents from the S&P 500, Hang Seng, Shanghai, FTSE 100, Nikkei 225, plus a curated set of global ETFs, FX, crypto, and futures.
2. **Volatility prefilter.** Computes 20-year annualized volatility per ticker and keeps only the top 20% most volatile names (plus mandatory benchmarks like SPY/QQQ/IWM/EFA).
3. **State discretization.** Daily returns on the training set (first 70%) are z-scored and bucketed into five states: `Very Down`, `Down`, `Neutral`, `Up`, `Very Up` (thresholds at z = ±0.55 and ±1.15).
4. **Transition matrix.** A 5×5 transition matrix is built from the training data with Laplace smoothing (α = 0.7).
5. **Edge filter.** Only tickers whose Markov signal beats random by ≥55% hit rate on training data (with a 20% probability edge threshold) are kept.
6. **Signal generation.** On the test set, each day's state implies a Long or Short signal when `P(Up)+P(Very Up) − P(Down)+P(Very Down)` exceeds the threshold.
7. **Position sizing.** Kelly-style weights capped at 20%, distributed across up to 3 trades per day, sized within a daily budget of 55% of equity.
8. **Risk management.** Hard stop at −3% intraday (LONG: Low ≤ 0.97 × today's Open; SHORT: High ≥ 1.03 × today's Open). 20 bps round-trip transaction costs.
9. **Index rotation fallback.** On days with no signals, the strategy ranks a basket of country/region ETFs (`SPY, QQQ, IWM, EFA, EEM, EWJ, FXI, EWU, EWG, EWQ, EWZ`) by trend slope / volatility ratio and goes long the strongest / short the weakest using inverse-vol weighting (25% of equity, 5-day hold).

---

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python strategy.py
```

The script will:
- Fetch tickers from each index
- Download 20 years of daily price history via `yfinance`
- Build per-ticker Markov matrices and filter for edge
- Run the daily backtest over the test period
- Print the executed trade log and performance metrics
- Plot the strategy equity curve vs SPY

## Output Metrics

- Sharpe ratio (daily and annualized)
- Annualized return (compounded from daily)
- CAGR (calendar time)
- SPY benchmark return and return spread
- CAPM alpha (linear and compounded annualized) with t-stat and p-value
- Beta and R² vs SPY
- Final bankroll

## Configuration

Key parameters at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `THRESHOLD_DIFF` | 0.20 | Minimum probability edge to take a trade |
| `MIN_HIT_RATE` | 0.55 | Minimum train hit rate to keep a ticker |
| `HOLD_DAYS` | 5 | Holding period for signal trades |
| `ROUND_TRIP_COST` | 0.002 | 20 bps round-trip transaction cost |
| `DAILY_BUDGET_FRAC` | 0.55 | Daily capital deployed on signals |
| `MAX_TRADES_PER_DAY` | 3 | Cap on simultaneous new positions |
| `KELLY_CAP` | 0.20 | Maximum Kelly weight per trade |
| `STOP_LOSS_PCT` | −0.03 | Hard stop loss percentage |
| `FALLBACK_BUDGET_FRAC` | 0.25 | Capital for index-rotation fallback |

## Disclaimer

This is a research/educational backtest. It is not investment advice. Past performance does not guarantee future results. Real-world execution costs, slippage, market impact, and survivorship bias in ticker lists may materially affect live results.

## License

MIT
