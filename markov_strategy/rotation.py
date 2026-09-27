"""
rotation.py
-----------
Fallback "index rotation" logic used on days when no Markov-chain signal
fires. Ranks a basket of broad index ETFs by a volatility-adjusted trend
score (annualized log-price slope / realized vol) and goes long the
strongest, short the weakest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from .data import get_close_series


def rank_indices_by_trend(index_list, ref_date, lookback: int = 60, vol_lookback: int = 20):
    """
    Return (best_up_ticker, best_down_ticker, scores, vols) where score is
    the annualized log-price trend slope divided by realized volatility.
    """
    slopes, vols = {}, {}
    start = ref_date - pd.tseries.offsets.BDay(lookback + 5)

    for t in index_list:
        try:
            df = yf.download(t, start=start, end=ref_date + pd.Timedelta(days=2),
                              auto_adjust=True, progress=False)
            s = get_close_series(df).dropna()
            if len(s) < max(10, lookback // 2):
                continue
            y = np.log(s.values[-lookback:])
            x = np.arange(len(y))
            beta = np.polyfit(x, y, 1)[0]
            slopes[t] = float(beta) * 252.0
            r = s.pct_change().dropna()
            vols[t] = float(r.rolling(vol_lookback).std().iloc[-1] * np.sqrt(252))
        except Exception:
            continue

    scores = {t: slopes[t] / vols.get(t, 1e-6) for t in slopes if vols.get(t, 0) > 0}
    if not scores:
        return None, None, {}, {}

    up = max(scores, key=scores.get)
    dn = min(scores, key=scores.get)
    return up, dn, scores, vols
