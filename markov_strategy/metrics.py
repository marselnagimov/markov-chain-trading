"""
metrics.py
----------
Performance reporting: Sharpe ratio, CAGR, CAPM alpha/beta vs. SPY, and the
equity-curve plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from .data import get_close_series
import yfinance as yf


def compute_performance(daily_df: pd.DataFrame, starting_capital: float, rf_annual: float = 0.05) -> dict:
    """
    Compute Sharpe, CAGR, and CAPM alpha/beta of the strategy vs. SPY.
    Returns a dict of scalar metrics plus the aligned equity/return series
    needed for plotting.
    """
    equity = daily_df["EndBank"].ffill()
    if pd.isna(equity.iloc[0]):
        equity.iloc[0] = starting_capital
    retn = equity.pct_change().dropna()

    spy_df = yf.download(
        "SPY",
        start=equity.index.min() - pd.Timedelta(days=5),
        end=equity.index.max() + pd.Timedelta(days=2),
        auto_adjust=True, progress=False, interval="1d",
    )
    spy_close = get_close_series(spy_df).reindex(equity.index).ffill()
    spy_r = spy_close.pct_change().dropna()

    aligned = pd.DataFrame({"strategy": retn.squeeze(), "spy": spy_r.squeeze()}).dropna()
    retn, spy_r = aligned["strategy"], aligned["spy"]

    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = retn - rf_daily
    spy_excess = spy_r - rf_daily

    ex_std = excess.std(ddof=1)
    sharpe_daily = (excess.mean() / ex_std) if ex_std > 0 else np.nan
    sharpe_annual = sharpe_daily * np.sqrt(252)

    ann_from_daily = (1 + retn).prod() ** (252 / len(retn)) - 1
    years_calendar = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr_calendar = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years_calendar) - 1 if years_calendar > 0 else np.nan
    spy_ann = (1 + spy_r).prod() ** (252 / len(spy_r)) - 1
    return_spread = ann_from_daily - spy_ann

    X = sm.add_constant(spy_excess.values)
    y = excess.values
    model = sm.OLS(y, X).fit()
    alpha_daily = float(model.params[0])
    beta = float(model.params[1])
    alpha_ann_linear = alpha_daily * 252
    alpha_ann_comp = (1 + alpha_daily) ** 252 - 1
    alpha_t = float(model.tvalues[0])
    alpha_p = float(model.pvalues[0])
    r2 = float(model.rsquared)

    return {
        "sharpe_daily": sharpe_daily,
        "sharpe_annual": sharpe_annual,
        "ann_from_daily": ann_from_daily,
        "cagr_calendar": cagr_calendar,
        "spy_ann": spy_ann,
        "return_spread": return_spread,
        "alpha_ann_linear": alpha_ann_linear,
        "alpha_ann_comp": alpha_ann_comp,
        "beta": beta,
        "r2": r2,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "final_bankroll": float(equity.iloc[-1]),
        "equity": equity,
        "spy_r": spy_r,
    }


def print_report(metrics: dict) -> None:
    print(f"\nSharpe (daily): {metrics['sharpe_daily']:.3f}")
    print(f"Sharpe (annualized): {metrics['sharpe_annual']:.3f}")
    print(f"Annualized return (from daily compounding): {metrics['ann_from_daily']:.2%}")
    print(f"CAGR (calendar time): {metrics['cagr_calendar']:.2%}")
    print(f"SPY annualized (aligned): {metrics['spy_ann']:.2%}")
    print(f"Return spread vs SPY (CAGR diff): {metrics['return_spread']:.2%}")
    print(f"Alpha (CAPM, annualized - linear): {metrics['alpha_ann_linear']:.2%}")
    print(f"Alpha (CAPM, annualized - compounded): {metrics['alpha_ann_comp']:.2%}")
    print(f"Beta vs SPY: {metrics['beta']:.2f}   R^2: {metrics['r2']:.2f}")
    print(f"Alpha t-stat: {metrics['alpha_t']:.2f}   p-value: {metrics['alpha_p']:.4f}")
    print(f"Final Bankroll: ${metrics['final_bankroll']:,.2f}")


def plot_equity_curve(metrics: dict, starting_capital: float, save_path: str | None = None) -> None:
    equity = metrics["equity"]
    spy_r = metrics["spy_r"]

    plt.figure()
    plt.plot(equity.index, equity.values, label="Strategy Capital")
    spy_curve = (1 + spy_r).cumprod() * equity.iloc[0]
    plt.plot(spy_curve.index, spy_curve.values, label="SPY (TR, auto-adjusted)")
    plt.legend()
    plt.title("Capital Curve vs SPY")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
