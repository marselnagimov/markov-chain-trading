"""
universe.py
-----------
Prefilters the full ticker universe down to the most volatile names, since
the Markov edge-detection step is expensive to run on thousands of tickers
and low-volatility names rarely clear the edge/hit-rate thresholds anyway.
"""

from __future__ import annotations

from itertools import chain

import pandas as pd

from .data import calc_volatility


def flatten_universe(ticker_universe: dict[str, list[str]]) -> list[str]:
    return list(chain.from_iterable(ticker_universe.values()))


def filter_by_volatility(all_tickers: list[str], top_frac: float = 0.20,
                          always_include: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA"),
                          verbose: bool = True) -> list[str]:
    """
    Compute annualized volatility for every ticker, keep the top `top_frac`
    fraction, and force-include a small always-on liquid core (so the
    fallback index-rotation logic always has data for its basket members).
    """
    volatility_map = {}
    for t in all_tickers:
        vol = calc_volatility(t)
        if vol is not None:
            volatility_map[t] = vol
            if verbose:
                print(f"{t}: {vol:.3f}")

    sorted_vol = sorted(volatility_map.items(), key=lambda x: x[1], reverse=True)
    top_n = max(int(len(sorted_vol) * top_frac), 1)
    volatile_tickers = [t for t, _ in sorted_vol[:top_n]]

    for must in always_include:
        if must not in volatile_tickers and must in all_tickers:
            volatile_tickers.append(must)

    if verbose:
        print(f"\nTop {top_frac:.0%} most volatile tickers ({len(volatile_tickers)} total):")
        print(volatile_tickers)

    return volatile_tickers
