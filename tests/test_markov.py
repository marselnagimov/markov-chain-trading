import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from markov_strategy.markov import _classify, has_edge, STATE_LABELS


def test_classify_boundaries():
    down, up, extreme = -0.55, 0.55, 1.15
    assert _classify(-2.0, down, up, extreme) == "Very Down"
    assert _classify(-0.8, down, up, extreme) == "Down"
    assert _classify(0.0, down, up, extreme) == "Neutral"
    assert _classify(0.8, down, up, extreme) == "Up"
    assert _classify(2.0, down, up, extreme) == "Very Up"


def test_has_edge_detects_strong_signal():
    # Construct a synthetic train_df where "Up" always precedes a positive
    # return and the transition matrix says P(up|Up) is high -> should
    # detect edge given a big enough diff.
    n = 200
    states = (["Up", "Neutral"] * (n // 2))
    # has_edge compares state[i] (via the transition matrix) against
    # return[i+1], i.e. the return realized *after* that state. So
    # return[i] should be positive whenever state[i-1] == "Up".
    returns = [0.0]
    for i in range(1, len(states)):
        returns.append(0.01 if states[i - 1] == "Up" else -0.001)

    train_df = pd.DataFrame({"state": states, "return": returns})

    tm = pd.DataFrame(0.0, index=STATE_LABELS, columns=STATE_LABELS)
    tm.loc["Up", "Up"] = 0.8
    tm.loc["Up", "Very Up"] = 0.1
    tm.loc["Up", "Down"] = 0.05
    tm.loc["Up", "Very Down"] = 0.0
    tm.loc["Up", "Neutral"] = 0.05
    tm.loc["Neutral"] = [0.1, 0.1, 0.6, 0.1, 0.1]

    assert has_edge(train_df, tm, min_diff=0.2, min_hit=0.55) is True


def test_has_edge_rejects_insufficient_sample():
    train_df = pd.DataFrame({"state": ["Up"] * 10, "return": [0.01] * 10})
    tm = pd.DataFrame(0.0, index=STATE_LABELS, columns=STATE_LABELS)
    tm.loc["Up", "Up"] = 0.9
    assert has_edge(train_df, tm, min_diff=0.2, min_hit=0.55) is False
