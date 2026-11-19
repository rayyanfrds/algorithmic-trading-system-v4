"""
Recompute the V4.0 forward-test statistics from the raw MT5 deal history.

Usage (from the repository root):
    python scripts/verify_forward_test.py

Only closing deals (entry == 1) of the V4.0 magic numbers (40101 Gold, 40102 BTC)
are used. Net PnL per deal = profit + commission + swap + fee.
Drawdown is measured on the CLOSED-TRADE balance (floating P/L is not included).
"""
import math
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEALS = ROOT / "logs_sample" / "mt5_deals_history_demo_sept2026.csv"
V4_MAGICS = {40101: "XAUUSDm (Gold)", 40102: "BTCUSDm (BTC)"}
START_BALANCE = 9511.11          # account balance when V4.0 went live (09 Sep 2026)
REASON = {4: "Hard SL (-1.5x ATR)", 5: "Hard TP (+2.0x ATR)"}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def summarize(name, g):
    w, l = g[g.net > 0], g[g.net <= 0]
    gp, gl = w.net.sum(), -l.net.sum()
    lo, hi = wilson(len(w), len(g))
    print(f"{name:18s} n={len(g):3d}  W/L={len(w)}/{len(l)}  WR={len(w)/len(g):.2%} "
          f"(95% CI {lo:.1%}-{hi:.1%})  PF={gp/gl:.2f}  net=${g.net.sum():,.2f}  "
          f"avgW=${w.net.mean():.2f}  avgL=${l.net.mean():.2f}")


def closed_balance_dd(g, start=START_BALANCE):
    eq = pd.concat([pd.Series([start]), start + g.sort_values("time_dt").net.cumsum()], ignore_index=True)
    return (eq / eq.cummax() - 1).min()


def main():
    d = pd.read_csv(DEALS, parse_dates=["time_dt"])
    c = d[(d.entry == 1) & (d.magic.isin(V4_MAGICS))].copy()
    c["net"] = c.profit + c.commission + c.swap + c.fee
    c = c.sort_values("time_dt")

    print(f"V4.0 closing deals: {len(c)}   window: {c.time_dt.min()} -> {c.time_dt.max()}\n")
    summarize("PORTFOLIO", c)
    for m, label in V4_MAGICS.items():
        summarize(label, c[c.magic == m])

    eq = START_BALANCE + c.net.cumsum()
    print(f"\nFinal closed balance : ${eq.iloc[-1]:,.2f}  (return {c.net.sum()/START_BALANCE:+.2%})")
    print(f"Max DD (portfolio)   : {closed_balance_dd(c):.2%}")
    for m, label in V4_MAGICS.items():
        print(f"Max DD (standalone {label}): {closed_balance_dd(c[c.magic == m]):.2%}")
    streak = best = 0
    for x in (c.net <= 0):
        streak = streak + 1 if x else 0
        best = max(best, streak)
    print(f"Max consecutive losses: {best}")

    print("\nExit decomposition:")
    # MT5 truncates the order comment to ~31 chars, e.g. "V4.0 Close (EXIT_EXPECTATIO)"
    cm = c.comment.astype(str)
    c["exit"] = c.reason.map(REASON)
    c.loc[(c.reason == 3) & cm.str.contains("EXPECTATIO"), "exit"] = "Expectation flip"
    c.loc[(c.reason == 3) & cm.str.contains("TIME_STOP"), "exit"] = "Time stop (12 bars)"
    tab = c.groupby("exit").net.agg(["count", "sum", "mean"]).round(2)
    print(tab.to_string())
    gp = c[c.net > 0].net.sum()
    tp = c[c.reason == 5].net.sum()
    print(f"\nTP exits = {tp/gp:.1%} of gross profit (${tp:,.2f} of ${gp:,.2f})")


if __name__ == "__main__":
    main()
