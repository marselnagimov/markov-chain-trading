import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from markov_strategy.data import get_ohlc, get_close_series


def test_get_ohlc_flat_columns():
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({
        "Open": [1, 2, 3, 4, 5],
        "High": [1.5, 2.5, 3.5, 4.5, 5.5],
        "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "Close": [1.2, 2.2, 3.2, 4.2, 5.2],
        "Volume": [100, 200, 300, 400, 500],
    }, index=idx)

    out = get_ohlc(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close"]
    assert len(out) == 5
    assert out["Close"].iloc[0] == 1.2


def test_get_ohlc_multiindex_columns():
    idx = pd.bdate_range("2024-01-01", periods=3)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    tickers = ["AAA"]
    cols = pd.MultiIndex.from_product([fields, tickers])
    data = np.arange(len(fields) * len(tickers) * len(idx)).reshape(len(idx), -1)
    df = pd.DataFrame(data, index=idx, columns=cols)

    out = get_ohlc(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close"]
    assert len(out) == 3


def test_get_ohlc_empty_input():
    out = get_ohlc(pd.DataFrame())
    assert list(out.columns) == ["Open", "High", "Low", "Close"]
    assert out.empty


def test_get_close_series_prefers_adj_close():
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame({"Close": [1, 2, 3], "Adj Close": [1.1, 2.1, 3.1]}, index=idx)
    s = get_close_series(df)
    assert s.iloc[0] == 1.1


def test_get_close_series_falls_back_to_close():
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame({"Close": [1, 2, 3]}, index=idx)
    s = get_close_series(df)
    assert s.iloc[0] == 1
