import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from markov_strategy.config import StrategyConfig
from markov_strategy import backtest as bt


def _cfg():
    return StrategyConfig()


def test_mark_to_market_long_and_short():
    px = {"AAA": pd.Series({pd.Timestamp("2024-01-02"): 110.0})}
    open_pos = [
        {"Ticker": "AAA", "Position": "Long", "Entry": 100.0, "Size": 1000.0, "MTM": 0.0},
        {"Ticker": "AAA", "Position": "Short", "Entry": 100.0, "Size": 1000.0, "MTM": 0.0},
    ]
    total = bt.mark_to_market(pd.Timestamp("2024-01-02"), px, open_pos)
    # long: +10% * 1000 = 100 ; short: -10% * 1000 = -100 ; net 0
    assert abs(total - 0.0) < 1e-9
    assert abs(open_pos[0]["MTM"] - 100.0) < 1e-9
    assert abs(open_pos[1]["MTM"] - (-100.0)) < 1e-9


def test_apply_daily_stop_losses_triggers_long_stop():
    cfg = _cfg()
    d0 = pd.Timestamp("2024-01-02")
    d1 = pd.Timestamp("2024-01-03")

    bt.set_ohlc_cache({
        "AAA": pd.DataFrame(
            {"Open": [100.0, 101.0], "High": [101.0, 101.0], "Low": [99.0, 95.0], "Close": [100.5, 96.0]},
            index=[d0, d1],
        )
    })

    open_pos = [{
        "Ticker": "AAA", "Position": "Long", "Size": 1000.0, "Entry": 100.0,
        "OpenCost": 1.0, "EntryDay": d0, "OpenAtEntryDay": 100.0, "MTM": 0.0, "Origin": "SIGNAL",
    }]
    trade_log = []

    cash, survivors, trade_log = bt.apply_daily_stop_losses(d1, 1000.0, open_pos, trade_log, cfg)

    # Low (95) <= 0.97*100 (97) -> stop should trigger
    assert len(survivors) == 0
    assert any(t["Action"] == "STOP_OUT" for t in trade_log)


def test_apply_daily_stop_losses_no_trigger_on_entry_day():
    cfg = _cfg()
    d0 = pd.Timestamp("2024-01-02")

    bt.set_ohlc_cache({
        "AAA": pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [90.0], "Close": [95.0]}, index=[d0])
    })

    open_pos = [{
        "Ticker": "AAA", "Position": "Long", "Size": 1000.0, "Entry": 100.0,
        "OpenCost": 1.0, "EntryDay": d0, "OpenAtEntryDay": 100.0, "MTM": 0.0, "Origin": "SIGNAL",
    }]
    trade_log = []

    cash, survivors, trade_log = bt.apply_daily_stop_losses(d0, 1000.0, open_pos, trade_log, cfg)

    # even though Low would trip the stop, entry day is exempt
    assert len(survivors) == 1
    assert trade_log == []


def test_close_due_positions_realizes_pnl():
    cfg = _cfg()
    d0 = pd.Timestamp("2024-01-02")
    d1 = pd.Timestamp("2024-01-09")

    px = {"AAA": pd.Series({d1: 110.0})}
    open_pos = [{
        "Ticker": "AAA", "Position": "Long", "Size": 1000.0, "Entry": 100.0,
        "OpenCost": 1.0, "ExitDay": d1, "EntryDay": d0, "MTM": 0.0, "Origin": "SIGNAL",
    }]
    trade_log = []

    cash, keep, trade_log = bt.close_due_positions(d1, px, open_pos, 0.0, trade_log, cfg)

    assert len(keep) == 0
    assert len(trade_log) == 1
    assert trade_log[0]["Action"] == "CLOSE"
    # gross = 10% of 1000 = 100, minus open cost 1 and close cost (0.001*1000=1)
    assert abs(trade_log[0]["PnL"] - (100.0 - 1.0 - 1.0)) < 1e-6
