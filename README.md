# Markov-Chain Multi-Market Trading Strategy

A research backtester for a discrete-state Markov-chain trading strategy,
applied across the S&P 500, Hang Seng, Shanghai Composite, FTSE 100, Nikkei
225, and a global macro basket (crypto, commodities, leveraged/ETF sector
proxies). The strategy classifies each day's return into one of five
z-scored states (`Very Down`, `Down`, `Neutral`, `Up`, `Very Up`), fits a
train-only transition matrix per ticker, filters for tickers with a
statistically meaningful directional edge, and sizes trades with a
Kelly-capped position weight, fixed holding period, intraday stop losses,
and a quiet-day index-rotation fallback.

A full write-up of the methodology, assumptions, and known limitations is
in [`paper/summary.pdf`](paper/summary.pdf) (built from [`paper/summary.tex`](paper/summary.tex)).

## How it works

1. **Universe construction** (`markov_strategy/data.py`) — scrapes/queries
   current constituents of five major indices plus a fixed global macro
   basket.
2. **Volatility prefilter** (`markov_strategy/universe.py`) — keeps only the
   top 20% most volatile tickers (configurable), since low-vol names rarely
   clear the edge threshold and this keeps runtime manageable.
3. **Markov modeling** (`markov_strategy/markov.py`) — for each ticker,
   splits 20 years of daily returns 70/30 into train/test, z-scores using
   **train-only** mean/std (no look-ahead), buckets into 5 states, and fits
   a Laplace-smoothed transition matrix on the train split.
4. **Edge filter** — a ticker is only traded if, on the train set, its
   one-step-ahead directional signal (`P(up) - P(down)` beyond a threshold)
   beats a minimum hit rate over a minimum sample size.
5. **Signal generation** — on the **test** set, transition-matrix edges are
   converted into sized Long/Short signals with a fixed holding period.
6. **Backtest engine** (`markov_strategy/backtest.py`) — a daily event loop
   that opens signal trades (or falls back to trend/vol-ranked index
   rotation on days with no signal), applies intraday stop losses, marks
   positions to market, and closes positions at the end of their holding
   period, net of round-trip transaction costs.
7. **Performance reporting** (`markov_strategy/metrics.py`) — Sharpe ratio,
   CAGR, and CAPM alpha/beta vs. SPY, plus an equity-curve plot.

## Installation

```bash
git clone <this-repo>
cd markov-trading-strategy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Full run with default parameters
python main.py

# Only backtest the top 10% most volatile tickers (faster)
python main.py --top-frac 0.10

# Exclude the crypto/commodity macro basket
python main.py --no-global

# Save the equity curve instead of opening a plot window
python main.py --save-plot equity_curve.png
```

All strategy parameters (edge thresholds, holding period, transaction
costs, position sizing, stop-loss levels) live in `markov_strategy/config.py`
as a single `StrategyConfig` dataclass — edit or subclass it rather than
hunting through the codebase for magic numbers.

## Running tests

The unit tests use synthetic price data only — no network access or live
market data is required:

```bash
pytest tests/ -v
```

## Project structure

```
markov-trading-strategy/
├── main.py                    # CLI entrypoint
├── markov_strategy/
│   ├── config.py               # StrategyConfig — all tunable parameters
│   ├── data.py                  # ticker universes, price data helpers, caches
│   ├── markov.py                 # state discretization, transition matrix, signals
│   ├── rotation.py                # quiet-day index-rotation fallback
│   ├── universe.py                 # volatility prefiltering
│   ├── backtest.py                  # daily simulation loop, stops, MTM
│   └── metrics.py                    # Sharpe/CAPM/plotting
├── tests/                      # pytest unit tests (synthetic data, no network)
├── paper/
│   ├── summary.tex              # LaTeX write-up of methodology
│   └── summary.pdf              # compiled PDF
├── requirements.txt
└── LICENSE
```

## Known limitations

- **Survivorship bias.** Index constituents are pulled from *current*
  membership lists (Wikipedia / yfinance) and then backtested over 20 years
  of history. Companies that were removed from an index (delisted,
  acquired, bankrupt) are excluded, which biases historical performance
  upward relative to what a real-time strategy would have achieved.
- **No slippage/market-impact model beyond a flat round-trip cost.** Real
  execution costs, especially for less liquid international tickers, will
  differ from the flat 20 bps assumption.
- **Stop-loss realism.** Stops book a fixed `STOP_LOSS_PCT` rather than the
  exact price at which a gap-through would have executed, which
  understates losses on large overnight gaps.
- **Data availability across markets.** Hong Kong, Shanghai, London, and
  Tokyo tickers depend on yfinance's coverage and can have gaps, delisted
  symbols, or FX-related quirks (all prices are in local currency, i.e. no
  FX-adjustment is applied across markets).
- **Runtime.** Downloading and modeling thousands of tickers one at a time
  via `yfinance` is slow and can hit rate limits; the volatility prefilter
  is a mitigation, not a fix.

This is a research/educational backtester, not investment advice, and
historical backtest performance is not indicative of future results.

## License

MIT — see [LICENSE](LICENSE).
