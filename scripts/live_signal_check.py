"""
Out-of-sample check of the live primary signal, using the per-hour decision logs.

Usage (from the repository root):
    python scripts/live_signal_check.py

Method
- Each log row is written at the START of an H1 bar: the OHLC columns hold the freshly opened
  (still forming) bar, while `primary_prob` was computed from the previous, fully closed bar.
- The tradable outcome of a decision is therefore the move from this row's open to the next row's
  open (open[i+1] - open[i]).
- Rows where that move is exactly 0 (market closed / weekend) are dropped.
- Reports ROC-AUC of `primary_prob` against the sign of that move (rank-based, bootstrap 95% CI)
  and the hit-rate of the emitted BUY/SELL signals.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs_sample"


def auc(y, s):
    y = np.asarray(y)
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(auc(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def main():
    for name in ("btc", "gold"):
        d = pd.read_csv(LOGS / f"hourly_decisions_{name}.csv", parse_dates=["timestamp"])
        d = d.sort_values("timestamp").dropna(subset=["open", "primary_prob"]).reset_index(drop=True)
        move = d.open.shift(-1) - d.open
        ok = move.notna() & (move != 0)
        y = (move[ok] > 0).astype(int).values
        s = d.primary_prob[ok].values
        lo, hi = bootstrap_ci(y, s)
        sig = d.primary_signal.isin(["BUY", "SELL"]) & ok
        hit = np.where(d.primary_signal[sig] == "BUY", move[sig] > 0, move[sig] < 0).mean()
        print(f"{name.upper():5s} AUC={auc(y, s):.3f} (95% CI {lo:.3f}-{hi:.3f}), n={ok.sum()} bars | "
              f"BUY/SELL hit-rate={hit:.1%} over {sig.sum()} signals")


if __name__ == "__main__":
    main()
