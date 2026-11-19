# -*- coding: utf-8 -*-
"""
live_decision_logger.py
=======================
Granular Hourly Decision Audit Logger for Dual-Asset V4.0 System:
- Records full bar OHLC, Volatility ATR, Spread, Model Probabilities
- Tracks Meta-Model Gate Decisions & Sizing Multipliers
- Logs Rolling Cross-Asset Correlation & Haircut Sizing
- Records In-Flight Position Status & Exit Execution Reasons
- Outputs tabular CSV & structured JSONL for independent audit by Don Smith
"""

import os
import sys
import csv
import json
import pathlib
import datetime
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "mid_price",
    "atr_14",
    "spread_usd",
    "primary_prob",
    "primary_signal",
    "meta_prob",
    "meta_gate_pass",
    "meta_sizing_mult",
    "rolling_correlation_480",
    "corr_haircut_mult",
    "effective_lot",
    "risk_budget_usd",
    "in_flight_position",
    "in_flight_entry_px",
    "in_flight_floating_pts",
    "in_flight_floating_atr",
    "in_flight_bars_held",
    "in_flight_trailing_floor",
    "decision_action",
    "trade_realized_pnl",
    "rationale"
]

class LiveDecisionLogger:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        clean_sym = "gold" if "XAU" in self.symbol or "GOLD" in self.symbol else "btc"
        self.csv_path = LOGS_DIR / f"hourly_decisions_{clean_sym}.csv"
        self.jsonl_path = LOGS_DIR / f"hourly_decisions_{clean_sym}.jsonl"
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    def log(
        self,
        timestamp: datetime.datetime or str,
        open_px: float,
        high_px: float,
        low_px: float,
        close_px: float,
        atr_14: float,
        spread_usd: float,
        primary_prob: float,
        primary_signal: str,
        meta_prob: float,
        meta_gate_pass: bool,
        meta_sizing_mult: float,
        rolling_corr_480: float,
        corr_haircut_mult: float,
        effective_lot: float,
        risk_budget_usd: float,
        in_flight_position: str = "NONE",
        in_flight_entry_px: float = None,
        in_flight_floating_pts: float = None,
        in_flight_floating_atr: float = None,
        in_flight_bars_held: int = 0,
        in_flight_trailing_floor: float = None,
        decision_action: str = "IDLE_NO_SIGNAL",
        trade_realized_pnl: float = None,
        rationale: str = ""
    ):
        ts_str = str(timestamp)

        row_dict = {
            "timestamp": ts_str,
            "symbol": self.symbol,
            "open": round(open_px, 2 if "XAU" in self.symbol else 1),
            "high": round(high_px, 2 if "XAU" in self.symbol else 1),
            "low": round(low_px, 2 if "XAU" in self.symbol else 1),
            "close": round(close_px, 2 if "XAU" in self.symbol else 1),
            "mid_price": round((high_px + low_px) / 2.0, 2 if "XAU" in self.symbol else 1),
            "atr_14": round(atr_14, 2 if "XAU" in self.symbol else 1),
            "spread_usd": round(spread_usd, 2),
            "primary_prob": round(primary_prob, 4),
            "primary_signal": primary_signal,
            "meta_prob": round(meta_prob, 4),
            "meta_gate_pass": meta_gate_pass,
            "meta_sizing_mult": round(meta_sizing_mult, 2),
            "rolling_correlation_480": round(rolling_corr_480, 4),
            "corr_haircut_mult": round(corr_haircut_mult, 2),
            "effective_lot": round(effective_lot, 3),
            "risk_budget_usd": round(risk_budget_usd, 2),
            "in_flight_position": in_flight_position,
            "in_flight_entry_px": round(in_flight_entry_px, 2) if in_flight_entry_px is not None else "",
            "in_flight_floating_pts": round(in_flight_floating_pts, 2) if in_flight_floating_pts is not None else "",
            "in_flight_floating_atr": round(in_flight_floating_atr, 2) if in_flight_floating_atr is not None else "",
            "in_flight_bars_held": in_flight_bars_held,
            "in_flight_trailing_floor": round(in_flight_trailing_floor, 2) if in_flight_trailing_floor is not None else "",
            "decision_action": decision_action,
            "trade_realized_pnl": round(trade_realized_pnl, 2) if trade_realized_pnl is not None else "",
            "rationale": rationale
        }

        # 1. Write to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row_dict)

        # 2. Write to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row_dict) + "\n")

        # 3. Print Clean Audit Log Box to Console
        self._print_console_box(row_dict)

    def _print_console_box(self, r: dict):
        status_color = "🟢" if "ENTRY" in r["decision_action"] or "PROFIT" in r["decision_action"] else ("🔴" if "EXIT_HARD_SL" in r["decision_action"] else "⚪")
        print("\n" + "─"*85)
        print(f"⏱️ [{r['timestamp']}] HOURLY AUDIT: {r['symbol']} | Mid: {r['mid_price']} | ATR(14): {r['atr_14']} | Spread: ${r['spread_usd']}")
        print(f"   • Model Primer   : P_up={r['primary_prob']:.4f} -> Sinyal [{r['primary_signal']}]")
        print(f"   • Meta-Model     : P_meta={r['meta_prob']:.4f} -> Gate Pass: {r['meta_gate_pass']} (Sizing: {r['meta_sizing_mult']}x)")
        print(f"   • Korelasi (480b): rho={r['rolling_correlation_480']:+.4f} -> Haircut Multiplier: {r['corr_haircut_mult']}x")
        print(f"   • Lot & Risk     : Lot Efektif={r['effective_lot']} lot | Risk Budget=${r['risk_budget_usd']}")
        if r['in_flight_position'] != "NONE":
            print(f"   • In-Flight Pos  : [{r['in_flight_position']}] Entry @ {r['in_flight_entry_px']} | Float: {r['in_flight_floating_pts']} pts ({r['in_flight_floating_atr']}x ATR) | Hold: {r['in_flight_bars_held']}h")
        print(f"   {status_color} KEPUTUSAN JAM INI : [{r['decision_action']}]")
        print(f"     └─ Rationale   : {r['rationale']}")
        if r['trade_realized_pnl'] != "":
            pnl_val = float(r['trade_realized_pnl'])
            pnl_icon = "💰" if pnl_val > 0 else "🛑"
            print(f"     {pnl_icon} REALIZED PnL : {pnl_val:+.2f} pts")
        print("─"*85)
