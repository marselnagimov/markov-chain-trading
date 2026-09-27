"""
markov.py
---------
Discretizes daily returns into a 5-state Markov chain (Very Down / Down /
Neutral / Up / Very Up), estimates a train-only transition matrix with
Laplace smoothing, filters tickers for statistical "edge" on the training
set, and converts test-set state transitions into sized trade signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from .data import _as_series, get_close_series

STATE_LABELS = ["Very Down", "Down", "Neutral", "Up", "Very Up"]


def _classify(z: float, down: float, up: float, extreme: float) -> str:
    if z <= -extreme:
        return "Very Down"
    if -extreme < z <= down:
        return "Down"
    if down < z < up:
        return "Neutral"
    if up <= z < extreme:
        return "Up"
    return "Very Up"


def build_markov_matrix(
    ticker: str,
    train_frac: float = 0.7,
    down: float = -0.55,
    up: float = 0.55,
    extreme: float = 1.15,
    alpha: float = 0.7,
    period: str = "20y",
):
    """
    Fit a 5-state Markov transition matrix on the first `train_frac` of a
    ticker's return history (z-scored using TRAIN-only mean/std to avoid
    look-ahead), and return (matrix, train_df, test_df).

    Returns (None, None, None) if data is insufficient.
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None, None, None

        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = _as_series(df[col])
        returns = prices.pct_change().dropna()
        if returns.empty:
            return None, None, None

        split = int(len(returns) * train_frac)
        train_ret, test_ret = returns.iloc[:split], returns.iloc[split:]
        if train_ret.empty or test_ret.empty:
            return None, None, None

        mu, sig = train_ret.mean(), train_ret.std(ddof=0)
        if sig == 0 or pd.isna(sig):
            return None, None, None

        z_train = (train_ret - mu) / sig
        z_test = (test_ret - mu) / sig

        states_train = z_train.apply(lambda z: _classify(z, down, up, extreme))
        states_test = z_test.apply(lambda z: _classify(z, down, up, extreme))

        train_df = pd.DataFrame({"return": train_ret, "state": states_train})
        test_df = pd.DataFrame({"return": test_ret, "state": states_test})

        counts = pd.DataFrame(0.0, index=STATE_LABELS, columns=STATE_LABELS)
        s = states_train.to_numpy()
        for a, b in zip(s[:-1], s[1:]):
            counts.loc[a, b] += 1.0

        counts += alpha  # Laplace smoothing
        matrix = counts.div(counts.sum(axis=1), axis=0)

        return matrix, train_df, test_df

    except Exception:
        return None, None, None


def has_edge(train_df: pd.DataFrame, tm: pd.DataFrame, min_diff: float, min_hit: float) -> bool:
    """
    On the TRAIN set only, check whether next-day directional signals
    (P(up)-P(down) beyond min_diff) beat `min_hit` hit-rate, with a minimum
    sample-size guard to avoid spurious small-sample "edges".
    """
    hits = []
    idx = train_df.index.to_list()
    for i in range(len(idx) - 1):
        s = train_df.loc[idx[i], "state"]
        if s not in tm.index:
            continue
        p = tm.loc[s]
        up = float(p.get("Up", 0) + p.get("Very Up", 0))
        dn = float(p.get("Down", 0) + p.get("Very Down", 0))
        diff = up - dn
        if abs(diff) < min_diff:
            continue
        sign = 1 if diff > 0 else -1
        ret1 = float(train_df.loc[idx[i + 1], "return"])
        hits.append((sign * ret1) > 0)

    if len(hits) < 50:  # minimum sample size
        return False
    return (sum(hits) / len(hits)) >= min_hit


def analyze_trade_signals(
    ticker: str,
    test_df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    min_diff: float,
    hold_days: int,
    kelly_cap: float,
) -> list[tuple]:
    """
    Walk the TEST-set states and emit (date, ticker, position, entry, exit,
    kelly_weight) signals wherever the transition matrix shows an edge
    exceeding `min_diff`.
    """
    signals: list[tuple] = []
    if test_df is None or test_df.empty or transition_matrix is None or transition_matrix.empty:
        return signals

    start = pd.to_datetime(test_df.index.min()).normalize()
    end = (pd.to_datetime(test_df.index.max()) + pd.Timedelta(days=hold_days)).normalize()

    px = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if px is None or px.empty:
        return signals
    close = get_close_series(px)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()

    idx = pd.to_datetime(test_df.index).tz_localize(None).normalize().to_list()

    for i in range(len(idx) - 1):
        d0 = idx[i]
        j = min(i + hold_days, len(idx) - 1)
        d1 = idx[j]

        if d0 not in close.index or d1 not in close.index:
            continue

        s = test_df.loc[test_df.index[i], "state"]
        if s not in transition_matrix.index:
            continue

        probs = transition_matrix.loc[s]
        up = float(probs.get("Up", 0) + probs.get("Very Up", 0))
        dn = float(probs.get("Down", 0) + probs.get("Very Down", 0))

        diff_long, diff_short = up - dn, dn - up
        k_long = diff_long if diff_long >= min_diff else 0.0
        k_short = diff_short if diff_short >= min_diff else 0.0

        if k_long <= 0 and k_short <= 0:
            continue

        position = "Long" if k_long >= k_short else "Short"
        kelly = min(max(k_long, k_short), kelly_cap)

        entry, exit_ = float(close.loc[d0]), float(close.loc[d1])
        signals.append((d0, ticker, position, entry, exit_, kelly))

    return signals
