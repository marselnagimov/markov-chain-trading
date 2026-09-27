#!/usr/bin/env python3
"""
main.py
-------
CLI entrypoint for the Markov-chain trading strategy backtest.

Usage:
    python main.py                      # full run with default config
    python main.py --top-frac 0.10      # only backtest top 10% most volatile
    python main.py --no-global          # exclude crypto/commodity ETFs
    python main.py --save-plot out.png  # save equity curve instead of showing it
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from markov_strategy.config import DEFAULT_CONFIG, INDEX_BASKET
from markov_strategy.data import build_ticker_universe, build_close_cache
from markov_strategy.universe import flatten_universe, filter_by_volatility
from markov_strategy.markov import build_markov_matrix, has_edge, analyze_trade_signals
from markov_strategy.backtest import run_backtest, set_ohlc_cache
from markov_strategy.metrics import compute_performance, print_report, plot_equity_curve
from markov_strategy import data as data_mod

warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Markov-chain trading strategy backtest")
    p.add_argument("--top-frac", type=float, default=0.20,
                   help="fraction of most-volatile tickers to keep (default 0.20)")
    p.add_argument("--no-global", action="store_true",
                   help="exclude the crypto/commodity 'Global' macro basket")
    p.add_argument("--save-plot", type=str, default=None,
                   help="path to save the equity-curve plot instead of showing it")
    p.add_argument("--quiet", action="store_true", help="suppress per-ticker progress output")
    p.add_argument("--tickers", type=str, default=None,
                   help="comma-separated ticker list to use INSTEAD of scraping index "
                        "universes (e.g. --tickers SPY,QQQ,AAPL,TSLA) — much faster for "
                        "a first end-to-end smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    verbose = not args.quiet

    if args.tickers:
        volatile_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        print(f"Using manual ticker list ({len(volatile_tickers)}): {volatile_tickers}\n")
    else:
        print("Retrieving tickers...\n")
        ticker_universe = build_ticker_universe(include_global=not args.no_global)
        all_tickers = flatten_universe(ticker_universe)
        if verbose:
            for t in all_tickers:
                print(t)

        volatile_tickers = filter_by_volatility(all_tickers, top_frac=args.top_frac, verbose=verbose)

    # ---------- Signals with edge filter ----------
    all_signals = []
    n_no_data, n_no_edge, n_signals_from = 0, 0, 0
    for ticker in volatile_tickers:
        tm, train_df, test_df = build_markov_matrix(
            ticker, alpha=cfg.alpha_smooth
        )
        if tm is None or train_df is None or test_df is None:
            n_no_data += 1
            if verbose:
                print(f"  {ticker}: no usable price data (download failed or too short history)")
            continue
        if not has_edge(train_df, tm, min_diff=cfg.threshold_diff, min_hit=cfg.min_hit_rate):
            n_no_edge += 1
            if verbose:
                print(f"  {ticker}: no edge on train set (below hit-rate/sample-size threshold)")
            continue
        sigs = analyze_trade_signals(
            ticker, test_df, tm,
            min_diff=cfg.threshold_diff, hold_days=cfg.hold_days, kelly_cap=cfg.kelly_cap,
        )
        if sigs:
            n_signals_from += 1
            if verbose:
                print(f"  {ticker}: edge found, {len(sigs)} test-set signals generated")
        all_signals.extend(sigs)

    print(f"\nSummary: {len(volatile_tickers)} tickers scanned | "
          f"{n_no_data} had no usable data | {n_no_edge} had no edge | "
          f"{n_signals_from} produced signals\n")

    if not all_signals:
        if n_no_data == len(volatile_tickers):
            print("No signals generated because NO ticker returned usable price data.\n"
                  "This almost always means the Yahoo Finance download itself is failing "
                  "(rate limiting, missing curl_cffi, or a network/proxy issue) rather than "
                  "the strategy having no edge. Try:\n"
                  "  pip install --upgrade 'curl_cffi>=0.15' 'yfinance>=0.2.40'\n"
                  "and re-run with a single ticker to isolate it, e.g.:\n"
                  "  python main.py --tickers SPY")
        else:
            print("No signals generated. Exiting.")
        sys.exit(0)

    signal_df = pd.DataFrame(all_signals, columns=["Date", "Ticker", "Position", "EntryPrice", "ExitPrice", "Kelly"])
    signal_df["Date"] = pd.to_datetime(signal_df["Date"]).dt.normalize()
    signal_df = signal_df.sort_values("Date")

    # ---------- Simulation setup ----------
    first_day = signal_df["Date"].min()
    last_day = signal_df["Date"].max() + pd.Timedelta(days=cfg.hold_days)
    bdays = pd.bdate_range(first_day, last_day)

    tickers = sorted(set(signal_df["Ticker"].unique()).union(INDEX_BASKET))
    close_px = build_close_cache(tickers, first_day, last_day, bdays)
    set_ohlc_cache(data_mod.build_ohlc_cache(tickers, first_day, last_day, bdays))

    # ---------- Run backtest ----------
    trade_log_df, daily_df = run_backtest(signal_df, INDEX_BASKET, close_px, cfg)

    if not trade_log_df.empty:
        print("\n--- Executed Trades ---")
        print(trade_log_df[["Ticker", "Position", "Action", "Size", "PnL", "Cash"]].fillna(""))

    # ---------- Reporting ----------
    metrics = compute_performance(daily_df, starting_capital=cfg.starting_capital)
    print_report(metrics)
    plot_equity_curve(metrics, starting_capital=cfg.starting_capital, save_path=args.save_plot)


if __name__ == "__main__":
    main()
