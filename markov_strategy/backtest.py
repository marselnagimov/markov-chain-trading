"""
backtest.py
-----------
The daily event-driven simulation loop: opens Markov-signal trades (or
index-rotation fallback trades on quiet days), applies intraday stop
losses, marks open positions to market, and closes positions whose
holding period has elapsed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import px_close
from .rotation import rank_indices_by_trend

# Global non-adjusted OHLC cache, populated by data.build_ohlc_cache() and
# consulted here for true intraday stop-loss checks (open/high/low, not
# adjusted close).
ohlc_px: dict[str, pd.DataFrame] = {}


def set_ohlc_cache(cache: dict[str, pd.DataFrame]) -> None:
    """Install the OHLC cache built by data.build_ohlc_cache()."""
    ohlc_px.clear()
    ohlc_px.update(cache)


def _o_get(t, d, field):
    df = ohlc_px.get(t)
    if df is None or d not in df.index:
        return None
    v = df.loc[d, field]
    return float(v) if pd.notna(v) else None


def o_open(t, d):
    return _o_get(t, d, "Open")


def o_low(t, d):
    return _o_get(t, d, "Low")


def o_high(t, d):
    return _o_get(t, d, "High")


# --------------------------------------------------------------------------
# Trade actions
# --------------------------------------------------------------------------

def open_signal_trades(d, todays, px, equity_start, cash, open_pos, trade_log, cfg: StrategyConfig):
    """Size Markov-signal trades from a single daily risk pot (daily_budget_frac)."""
    cap_total = max(0.0, cfg.daily_budget_frac * equity_start)
    total_kelly = todays["Kelly"].sum()

    for _, row in todays.sort_values("Kelly", ascending=False).head(cfg.max_trades_per_day).iterrows():
        if cap_total <= 0 or total_kelly <= 0:
            break
        entry_px = px_close(px, row["Ticker"], d)
        if entry_px is None:
            continue
        w = float(row["Kelly"]) / float(total_kelly)
        desired = min(cap_total, w * cap_total)
        if desired <= 0:
            continue
        open_cost = desired * (cfg.round_trip_cost / 2.0)
        cash -= open_cost
        open_pos.append({
            "Ticker": row["Ticker"], "Position": row["Position"], "Size": float(desired),
            "Entry": float(entry_px), "OpenCost": float(open_cost),
            "ExitDay": d + pd.tseries.offsets.BDay(cfg.hold_days),
            "EntryDay": d, "OpenAtEntryDay": o_open(row["Ticker"], d),
            "MTM": 0.0, "Origin": "SIGNAL",
        })
        trade_log.append({
            "Date": d, "Ticker": row["Ticker"], "Position": row["Position"],
            "EntryPrice": float(entry_px), "Size": float(desired),
            "Action": "OPEN_SIGNAL", "Cash": float(cash),
        })
        cap_total -= desired
    return cash, open_pos, trade_log


def open_fallback_trades(d, equity_start, cash, open_pos, trade_log, index_basket, px, cfg: StrategyConfig):
    """On days with no Markov signal, rotate into the strongest/weakest index ETFs."""
    up, dn, _, vols = rank_indices_by_trend(index_basket, d, cfg.index_lookback, cfg.index_vol_lookback)
    if not (up and dn and up in vols and dn in vols):
        return cash, open_pos, trade_log

    used_fallback = sum(float(p["Size"]) for p in open_pos if p.get("Origin") == "FALLBACK")
    fallback_cap = max(0.0, cfg.fallback_budget_frac * equity_start - used_fallback)
    if fallback_cap <= 1e-9:
        return cash, open_pos, trade_log

    inv_up, inv_dn = 1.0 / max(vols[up], 1e-6), 1.0 / max(vols[dn], 1e-6)
    w_up = inv_up / (inv_up + inv_dn)
    w_dn = 1.0 - w_up
    plan = [(up, "Long", w_up * fallback_cap), (dn, "Short", w_dn * fallback_cap)]

    for tkr, posn, desired in plan:
        entry_px = px_close(px, tkr, d)
        if entry_px is None or desired <= 0:
            continue
        open_cost = desired * (cfg.round_trip_cost / 2.0)
        cash -= open_cost
        open_pos.append({
            "Ticker": tkr, "Position": posn, "Size": float(desired),
            "Entry": float(entry_px), "OpenCost": float(open_cost),
            "ExitDay": d + pd.tseries.offsets.BDay(cfg.fallback_hold_days),
            "EntryDay": d, "OpenAtEntryDay": o_open(tkr, d),
            "MTM": 0.0, "Origin": "FALLBACK",
        })
        trade_log.append({
            "Date": d, "Ticker": tkr, "Position": posn,
            "EntryPrice": float(entry_px), "Size": float(desired),
            "Action": "OPEN_FALLBACK", "Cash": float(cash),
        })
    return cash, open_pos, trade_log


def apply_daily_stop_losses(d, cash, open_pos, trade_log, cfg: StrategyConfig):
    """
    LONG stops if today's Low <= stop_long_mult * entry-day Open.
    SHORT stops if today's High >= stop_short_mult * entry-day Open.
    No stops are checked on the entry day itself. A hit books a fixed
    stop_loss_pct loss on notional size, not the exact gap-through price.
    """
    survivors = []
    for pos in open_pos:
        if pd.to_datetime(pos["EntryDay"]).normalize() == d:
            survivors.append(pos)
            continue

        tkr = pos["Ticker"]
        lo, hi = o_low(tkr, d), o_high(tkr, d)
        o0 = pos.get("OpenAtEntryDay")

        if o0 is None or (lo is None and hi is None):
            survivors.append(pos)
            continue

        long_stop = float(o0) * cfg.stop_long_mult
        short_stop = float(o0) * cfg.stop_short_mult

        hit = (
            (pos["Position"] == "Long" and lo is not None and lo <= long_stop)
            or (pos["Position"] == "Short" and hi is not None and hi >= short_stop)
        )

        if hit:
            pnl = -float(pos["Size"]) * float(cfg.stop_loss_pct)
            close_cost = float(pos["Size"]) * float(cfg.round_trip_cost / 2.0)
            cash += pnl - close_cost
            trade_log.append({
                "Date": d, "Ticker": tkr, "Position": pos["Position"],
                "StopRefOpen": float(o0), "PnL": float(pnl - (pos["OpenCost"] + close_cost)),
                "Action": "STOP_OUT", "Cash": float(cash),
            })
        else:
            survivors.append(pos)
    return cash, survivors, trade_log


def mark_to_market(d, px, open_pos):
    """Update each open position's unrealized MTM and return the total."""
    mtm = 0.0
    for pos in open_pos:
        p_now = px_close(px, pos["Ticker"], d)
        if p_now is None:
            pos["MTM"] = 0.0
            continue
        gross = ((p_now - pos["Entry"]) / pos["Entry"]) if pos["Position"] == "Long" \
            else ((pos["Entry"] - p_now) / pos["Entry"])
        pos["MTM"] = pos["Size"] * gross
        mtm += pos["MTM"]
    return mtm


def close_due_positions(d, px, open_pos, cash, trade_log, cfg: StrategyConfig):
    """Close any position whose ExitDay has arrived, realizing PnL net of costs."""
    keep = []
    for pos in open_pos:
        if d >= pd.to_datetime(pos["ExitDay"]).normalize():
            exit_px = px_close(px, pos["Ticker"], d)
            if exit_px is None:
                keep.append(pos)
                continue
            gross = ((exit_px - pos["Entry"]) / pos["Entry"]) if pos["Position"] == "Long" \
                else ((pos["Entry"] - exit_px) / pos["Entry"])
            pnl = pos["Size"] * gross
            close_cost = pos["Size"] * (cfg.round_trip_cost / 2.0)
            cash += pnl - close_cost
            trade_log.append({
                "Date": d, "Ticker": pos["Ticker"], "Position": pos["Position"],
                "ExitPrice": float(exit_px), "PnL": float(pnl - (pos["OpenCost"] + close_cost)),
                "Action": "CLOSE", "Cash": float(cash),
            })
        else:
            keep.append(pos)
    return cash, keep, trade_log


# --------------------------------------------------------------------------
# Main daily loop
# --------------------------------------------------------------------------

def run_backtest(signal_df: pd.DataFrame, index_basket, px, cfg: StrategyConfig):
    """
    Run the full daily event loop over the date range implied by
    `signal_df`, returning (trade_log_df, daily_df).

    `px` must be a close-price cache from data.build_close_cache(), already
    reindexed to the desired business-day range. The module-level OHLC
    cache must be populated via set_ohlc_cache() before calling this.
    """
    cash = cfg.starting_capital
    first_day = signal_df["Date"].min()
    last_day = signal_df["Date"].max() + pd.Timedelta(days=cfg.hold_days)
    bdays = pd.bdate_range(first_day, last_day)

    by_date = {d: g for d, g in signal_df.groupby("Date")}
    open_pos: list[dict] = []
    trade_log: list[dict] = []
    daily_rows: list[dict] = []

    for d in bdays:
        mtm = mark_to_market(d, px, open_pos)
        equity_start = cash + mtm

        todays = by_date.get(d)
        if todays is not None and not todays.empty:
            cash, open_pos, trade_log = open_signal_trades(
                d, todays, px, equity_start, cash, open_pos, trade_log, cfg
            )
        else:
            cash, open_pos, trade_log = open_fallback_trades(
                d, equity_start, cash, open_pos, trade_log, index_basket, px, cfg
            )

        cash, open_pos, trade_log = apply_daily_stop_losses(d, cash, open_pos, trade_log, cfg)
        cash, open_pos, trade_log = close_due_positions(d, px, open_pos, cash, trade_log, cfg)

        equity_end = cash + sum(p["MTM"] for p in open_pos)
        daily_rows.append({
            "Date": d,
            "StartBank": float(equity_start),
            "EndBank": float(equity_end),
            "DayPnL": float(equity_end - equity_start),
            "DailyReturn": 0.0 if equity_start == 0 else float((equity_end - equity_start) / equity_start),
        })

    trade_log_df = pd.DataFrame(trade_log)
    if not trade_log_df.empty:
        trade_log_df["Date"] = pd.to_datetime(trade_log_df["Date"])
        trade_log_df.set_index("Date", inplace=True)

    daily_df = pd.DataFrame(daily_rows).set_index("Date").sort_index()
    return trade_log_df, daily_df
