"""
config.py
---------
All tunable strategy parameters in one place.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    # --- Markov edge detection ---
    threshold_diff: float = 0.2       # required P(up)-P(down) edge
    min_hit_rate: float = 0.55        # required train-set hit rate
    hold_days: int = 5                # holding period, trading days
    alpha_smooth: float = 0.7         # Laplace smoothing constant

    # --- Costs & sizing ---
    round_trip_cost: float = 0.002    # 20 bps round-trip transaction cost
    daily_budget_frac: float = 0.55   # fraction of bank risked per day
    max_trades_per_day: int = 3
    kelly_cap: float = 0.20           # cap on Kelly-derived position weight

    # --- Index-rotation fallback (quiet days with no signals) ---
    index_lookback: int = 60
    index_vol_lookback: int = 20
    fallback_hold_days: int = 5
    fallback_budget_frac: float = 0.25

    # --- Risk management ---
    stop_loss_pct: float = -0.03      # PnL booked when a stop triggers
    stop_long_mult: float = 0.97      # stop if Low <= mult * entry-day Open (LONG)
    stop_short_mult: float = 1.03     # stop if High >= mult * entry-day Open (SHORT)

    # --- Simulation ---
    starting_capital: float = 100_000.0


DEFAULT_CONFIG = StrategyConfig()

INDEX_BASKET = ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "FXI", "EWU", "EWG", "EWQ", "EWZ"]
