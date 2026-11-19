# -*- coding: utf-8 -*-
"""
live_forward_test_v4_meta.py
============================
Unified Production Live Execution Engine V4.0 (Dual-Asset LightGBM + Meta-System):
- MT5 API Connector (XAUUSD & BTCUSD)
- Production Inference: Primary Classifier + Meta-Model Gate (P >= 0.54)
- Continuous Bet Sizing & 480-Bar Dynamic Proportional Correlation Guard
- Hybrid Decoupled Exit Engine (Expectation Flip + Profit Lock Trailing Floor)
- Clean Modular Delegation to TelegramNotifier and LiveDecisionLogger
"""

import os
import sys
import time
import json
import pathlib
import datetime
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import ta
import MetaTrader5 as mt5
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from live_decision_logger import LiveDecisionLogger
from telegram_notifier import TelegramNotifier

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_SUBDIR = MODELS_DIR / "v4.0_forward_tested" if (MODELS_DIR / "v4.0_forward_tested").exists() else MODELS_DIR
LOGS_DIR     = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "")
MT5_PATH     = r"C:\Program Files\MetaTrader 5\terminal64.exe"

MAGIC_GOLD = 40101
MAGIC_BTC  = 40102

PRIMARY_FEATURES = [
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5",
    "vol_10", "vol_20", "atr_14",
    "rsi_14", "macd", "macd_diff", "roc_10",
    "dist_sma_20", "dist_sma_50",
]

META_FEATURES = [
    "primary_prob", "primary_confidence",
    "vol_ratio", "norm_atr",
    "rsi_14", "macd_diff", "dist_sma_50", "roc_10"
]

class DualAssetV4LiveEngine:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.deployment_start_time = datetime.datetime.now()
        self.logger_gold = LiveDecisionLogger("XAUUSD")
        self.logger_btc  = LiveDecisionLogger("BTCUSD")
        self.notifier    = TelegramNotifier(deployment_start_time=self.deployment_start_time)
        self.load_production_models()
        self.in_flight_state = {
            "XAUUSD": {"trailing_floor": None, "entry_ts": None, "lock_notified": False},
            "BTCUSD": {"trailing_floor": None, "entry_ts": None, "lock_notified": False}
        }
        self.last_daily_recap_date = None

    def is_bot_paused(self) -> bool:
        state_file = PROJECT_ROOT / "data" / "bot_control_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("is_paused", False)
            except Exception:
                pass
        return False

    def load_production_models(self):
        print(f"📦 Memuat model produksi V4.0 dari {MODELS_SUBDIR} ...")
        self.gold_primary = joblib.load(MODELS_SUBDIR / "gold_primary_v4.pkl")
        self.gold_meta    = joblib.load(MODELS_SUBDIR / "gold_meta_v4.pkl")
        self.btc_primary  = joblib.load(MODELS_SUBDIR / "btc_primary_v4.pkl")
        self.btc_meta     = joblib.load(MODELS_SUBDIR / "btc_meta_v4.pkl")
        
        meta_file = MODELS_DIR / "v4_metadata.json" if (MODELS_DIR / "v4_metadata.json").exists() else (MODELS_SUBDIR / "v4_metadata.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print("✅ Seluruh model & metadata V4.0 berhasil dimuat.")

    def connect_mt5(self) -> bool:
        if not mt5.initialize(path=MT5_PATH):
            print(f"[ERROR] Inisialisasi MT5 gagal: {mt5.last_error()}")
            return False
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            acc = mt5.account_info()
            if acc is None or acc.login != MT5_LOGIN:
                if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                    print(f"[ERROR] Login MT5 gagal: {mt5.last_error()}")
                    return False
        acc_info = mt5.account_info()
        if acc_info:
            print(f"🔗 Terhubung ke MT5: Akun #{acc_info.login} ({acc_info.currency}) | Server: {acc_info.server} | Equity: ${acc_info.equity:,.2f}")
            self.notifier.notify_system_online(acc_info.login, acc_info.server, acc_info.balance, acc_info.equity)
        return True

    def find_broker_symbol(self, base_sym: str) -> str:
        candidates = [f"{base_sym}c", f"{base_sym}.c", base_sym, f"{base_sym}m", f"{base_sym}.m", f"{base_sym}raw"]
        for c in candidates:
            info = mt5.symbol_info(c)
            if info is not None:
                mt5.symbol_select(c, True)
                return c
        return base_sym

    def fetch_recent_data(self, symbol: str, n_bars: int = 600) -> pd.DataFrame:
        broker_sym = self.find_broker_symbol(symbol)
        rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_H1, 0, n_bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}, inplace=True)
        df["MidPrice"] = (df["High"] + df["Low"]) / 2.0
        
        sym_info = mt5.symbol_info(broker_sym)
        point = sym_info.point if sym_info else 0.01
        spread_pts = (rates["spread"][-1] * point) if ("spread" in rates.dtype.names) else (0.30 if "XAU" in symbol else 15.0)
        df["Spread"] = spread_pts
        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["log_ret"] = np.log(df["MidPrice"] / df["MidPrice"].shift(1))
        df["target_1b"] = (df["log_ret"].shift(-1) > 0).astype(int)

        for i in range(1, 6):
            df[f"ret_lag_{i}"] = df["log_ret"].shift(i)

        df["vol_10"] = df["log_ret"].rolling(10).std()
        df["vol_20"] = df["log_ret"].rolling(20).std()
        df["vol_ratio"] = df["vol_10"] / (df["vol_20"] + 1e-9)

        df["atr_14"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=14
        ).average_true_range()
        df["norm_atr"] = df["atr_14"] / (df["MidPrice"] + 1e-9)

        df["rsi_14"]    = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd            = ta.trend.MACD(df["Close"])
        df["macd"]      = macd.macd()
        df["macd_diff"] = macd.macd_diff()
        df["roc_10"]    = ta.momentum.ROCIndicator(df["Close"], window=10).roc()

        df["sma_20"]      = df["MidPrice"].rolling(20).mean()
        df["dist_sma_20"] = (df["MidPrice"] - df["sma_20"]) / df["sma_20"]
        df["sma_50"]      = df["MidPrice"].rolling(50).mean()
        df["dist_sma_50"] = (df["MidPrice"] - df["sma_50"]) / df["sma_50"]
        return df.dropna()

    def get_open_position(self, symbol: str, magic: int):
        broker_sym = self.find_broker_symbol(symbol)
        positions = mt5.positions_get(symbol=broker_sym)
        if positions:
            for pos in positions:
                if pos.magic == magic:
                    return pos
        return None

    def calculate_lot_size(self, symbol: str, atr_val: float, meta_sizing_mult: float, corr_haircut_mult: float) -> float:
        acc = mt5.account_info()
        equity = acc.equity if (acc and acc.equity > 0) else 1000.0
        
        # Risk budget 0.75% modal
        risk_budget = equity * 0.0075
        sl_distance_pts = 1.5 * atr_val

        broker_sym = self.find_broker_symbol(symbol)
        sym_info = mt5.symbol_info(broker_sym)
        if not sym_info:
            return 0.01

        contract_size = sym_info.trade_contract_size
        min_lot = sym_info.volume_min
        step_lot = sym_info.volume_step

        # Dynamic profit per point from broker API
        ask = sym_info.ask if sym_info.ask > 0 else 1000.0
        profit_1lot_1pt = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, broker_sym, 1.0, ask, ask + 1.0)

        if profit_1lot_1pt and profit_1lot_1pt > 0:
            loss_1lot = sl_distance_pts * profit_1lot_1pt
        else:
            currency_mult = 100.0 if (acc and acc.currency == "USC" and sym_info.currency_profit == "USD") else 1.0
            loss_1lot = sl_distance_pts * contract_size * currency_mult

        raw_lot = risk_budget / (loss_1lot + 1e-9)
        effective_lot = raw_lot * meta_sizing_mult * corr_haircut_mult
        steps = round(effective_lot / step_lot)
        final_lot = max(min_lot, steps * step_lot)
        return round(final_lot, 2)

    def execute_market_order(self, symbol: str, side: str, lot: float, sl_pts: float, pt_pts: float, magic: int, p_prim: float, p_meta: float, sizing_m: float, rho_val: float) -> bool:
        broker_sym = self.find_broker_symbol(symbol)
        sym_info = mt5.symbol_info(broker_sym)
        if not sym_info:
            return False

        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        curr_px = sym_info.ask if side == "BUY" else sym_info.bid
        sl_price = (curr_px - sl_pts) if side == "BUY" else (curr_px + sl_pts)
        tp_price = (curr_px + pt_pts) if side == "BUY" else (curr_px - pt_pts)

        if self.dry_run:
            print(f"   [DRY-RUN] Simulasi Order {side} {symbol} | Lot: {lot} | Entry: {curr_px:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
            return True

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": lot,
            "type": order_type,
            "price": curr_px,
            "sl": round(sl_price, sym_info.digits),
            "tp": round(tp_price, sym_info.digits),
            "deviation": 20,
            "magic": magic,
            "comment": "V4.0 Meta",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        res = mt5.order_send(request)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[ERROR] Eksekusi order {symbol} gagal: {res.comment} (Retcode: {res.retcode})")
            return False

        print(f"✅ Order Berhasil Terbuka: {side} {symbol} {lot} Lot @ {curr_px} (Ticket #{res.order})")
        self.notifier.notify_order_opened(symbol, side, lot, res.order, curr_px, sl_price, tp_price, p_prim, p_meta, sizing_m, rho_val)
        return True

    def close_position(self, pos, reason: str, symbol: str, float_pts: float, float_atr: float, duration_h: int) -> bool:
        broker_sym = pos.symbol
        sym_info = mt5.symbol_info(broker_sym)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = sym_info.bid if pos.type == mt5.ORDER_TYPE_BUY else sym_info.ask

        if self.dry_run:
            print(f"   [DRY-RUN] Simulasi Close Posisi #{pos.ticket} | Alasan: {reason}")
            return True

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": f"V4.0 Close ({reason[:15]})",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        res = mt5.order_send(request)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[ERROR] Tutup posisi #{pos.ticket} gagal: {res.comment}")
            return False

        pnl_usd = pos.profit
        pnl_icon = "💰" if pnl_usd >= 0 else "🛑"
        print(f"{pnl_icon} Posisi #{pos.ticket} Ditutup: {reason} | PnL: ${pnl_usd:,.2f} ({float_pts:+.2f} pts)")
        self.notifier.notify_position_closed(symbol, pos.ticket, reason, close_price, duration_h, pnl_usd, float_pts, float_atr)
        return True

    def get_active_positions_summary_text(self) -> str:
        pos_gold = self.get_open_position("XAUUSD", MAGIC_GOLD)
        pos_btc  = self.get_open_position("BTCUSD", MAGIC_BTC)
        
        active_list = []
        if pos_gold:
            g_side = "BUY" if pos_gold.type == mt5.ORDER_TYPE_BUY else "SELL"
            active_list.append(f"Gold {g_side} {pos_gold.volume} lot (${pos_gold.profit:+,.2f})")
        if pos_btc:
            b_side = "BUY" if pos_btc.type == mt5.ORDER_TYPE_BUY else "SELL"
            active_list.append(f"BTC {b_side} {pos_btc.volume} lot (${pos_btc.profit:+,.2f})")
        return ", ".join(active_list) if active_list else "Semua Flat (Tidak ada posisi)"

    def evaluate_cycle(self):
        df_gold_raw = self.fetch_recent_data("XAUUSD", n_bars=800)
        df_btc_raw  = self.fetch_recent_data("BTCUSD", n_bars=800)

        if df_gold_raw.empty or df_btc_raw.empty:
            print("[WARN] Data MT5 belum tersedia atau gagal dimuat.")
            return

        df_gold = self.compute_features(df_gold_raw)
        df_btc  = self.compute_features(df_btc_raw)

        common_idx = df_gold.index.intersection(df_btc.index)
        if len(common_idx) >= 60:
            roll_corr = df_gold.loc[common_idx, "log_ret"].rolling(window=480, min_periods=60).corr(df_btc.loc[common_idx, "log_ret"]).iloc[-1]
            if np.isnan(roll_corr): roll_corr = 0.28
        else:
            roll_corr = 0.28

        corr_haircut = np.clip(1.0 - np.maximum(0.0, (abs(roll_corr) - 0.40) / 0.60) * 0.50, 0.50, 1.0)

        acc = mt5.account_info()
        curr_equity = acc.equity if (acc and acc.equity > 0) else 1000.0
        risk_budget = curr_equity * 0.0075

        for sym, df_sub, prim_model, meta_model, logger, magic in [
            ("XAUUSD", df_gold, self.gold_primary, self.gold_meta, self.logger_gold, MAGIC_GOLD),
            ("BTCUSD", df_btc, self.btc_primary, self.btc_meta, self.logger_btc, MAGIC_BTC)
        ]:
            broker_sym = self.find_broker_symbol(sym)
            sym_info = mt5.symbol_info(broker_sym)
            pos = self.get_open_position(sym, magic)

            # Guard penutupan sesi / weekend (cegah spam order saat market closed)
            if sym_info is None or sym_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                if not pos:
                    print(f"⏸️ Pasar {sym} ({broker_sym}) sedang tutup (trade_mode={sym_info.trade_mode if sym_info else 'None'}). Melewati siklus.")
                    continue

            latest_bar = df_sub.iloc[-1]
            ts = df_sub.index[-1]
            px = latest_bar["MidPrice"]
            atr_val = latest_bar["atr_14"]
            sp = latest_bar["Spread"]

            X_prim = df_sub.iloc[[-1]][PRIMARY_FEATURES]
            p_prob = float(prim_model.predict_proba(X_prim)[0][1])

            df_sub_meta = df_sub.iloc[[-1]].copy()
            df_sub_meta["primary_prob"] = p_prob
            df_sub_meta["primary_confidence"] = abs(p_prob - 0.50)
            X_m = df_sub_meta[META_FEATURES]
            m_prob = float(meta_model.predict_proba(X_m)[0][1])

            prim_signal = "BUY" if p_prob >= 0.55 else ("SELL" if p_prob <= 0.45 else "NEUTRAL")
            meta_pass = (m_prob >= 0.54)
            meta_sizing_mult = np.clip(0.60 + ((m_prob - 0.54) / 0.06) * 0.60, 0.60, 1.20) if meta_pass else 0.0

            effective_lot = self.calculate_lot_size(sym, atr_val, meta_sizing_mult, corr_haircut)

            action = "IDLE_NO_SIGNAL"
            rationale = "Tidak ada sinyal arah baru atau pasar sideways."
            realized_pnl = None

            in_flight_pos = "NONE"
            entry_px = None
            float_pts = None
            float_atr = None
            bars_held = 0
            trailing_floor = self.in_flight_state[sym]["trailing_floor"]

            if pos:
                in_flight_pos = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                entry_px = pos.price_open
                pos_dir = 1 if in_flight_pos == "BUY" else -1
                float_pts = pos_dir * (px - entry_px)
                float_atr = float_pts / (atr_val + 1e-9)
                
                pos_open_time = pd.to_datetime(pos.time, unit="s")
                bars_held = max(1, int((ts - pos_open_time).total_seconds() / 3600))

                # Check Trailing Profit Lock
                if float_atr >= 1.5:
                    lock_price = entry_px + pos_dir * (0.20 * atr_val)
                    if trailing_floor is None:
                        trailing_floor = lock_price
                    else:
                        trailing_floor = max(trailing_floor, lock_price) if pos_dir == 1 else min(trailing_floor, lock_price)
                    self.in_flight_state[sym]["trailing_floor"] = trailing_floor

                    if not self.in_flight_state[sym]["lock_notified"]:
                        self.notifier.notify_profit_lock(sym, in_flight_pos, float_atr, trailing_floor)
                        self.in_flight_state[sym]["lock_notified"] = True

                hit_pt = (float_atr >= 2.0)
                hit_sl = (float_atr <= -1.5)
                hit_time = (bars_held >= 12)
                hit_flip = (pos_dir == 1 and p_prob < 0.50) or (pos_dir == -1 and p_prob > 0.50)
                hit_trailing = (trailing_floor is not None) and (
                    (pos_dir == 1 and px <= trailing_floor) or (pos_dir == -1 and px >= trailing_floor)
                )

                if hit_pt:
                    action = "EXIT_HARD_PT"
                    rationale = "Menyentuh Hard Profit Target +2.0x ATR."
                    self.close_position(pos, action, sym, float_pts, float_atr, bars_held)
                    realized_pnl = float_pts
                    self.in_flight_state[sym]["trailing_floor"] = None
                    self.in_flight_state[sym]["lock_notified"] = False
                elif hit_sl:
                    action = "EXIT_HARD_SL"
                    rationale = "Menyentuh Hard Stop Loss -1.5x ATR."
                    self.close_position(pos, action, sym, float_pts, float_atr, bars_held)
                    realized_pnl = float_pts
                    self.in_flight_state[sym]["trailing_floor"] = None
                    self.in_flight_state[sym]["lock_notified"] = False
                elif hit_trailing:
                    action = "EXIT_PROFIT_LOCK"
                    rationale = "Menyentuh lantai Trailing Stop +0.20x ATR (Anti-Bleeding)."
                    self.close_position(pos, action, sym, float_pts, float_atr, bars_held)
                    realized_pnl = float_pts
                    self.in_flight_state[sym]["trailing_floor"] = None
                    self.in_flight_state[sym]["lock_notified"] = False
                elif hit_flip:
                    action = "EXIT_EXPECTATION_FLIP"
                    rationale = f"Probabilitas arah berbalik arah (P={p_prob:.4f}). Exit dini mitigasi risiko."
                    self.close_position(pos, action, sym, float_pts, float_atr, bars_held)
                    realized_pnl = float_pts
                    self.in_flight_state[sym]["trailing_floor"] = None
                    self.in_flight_state[sym]["lock_notified"] = False
                elif hit_time:
                    action = "EXIT_TIME_STOP"
                    rationale = "Mencapai batas waktu holding maksimal 12 jam."
                    self.close_position(pos, action, sym, float_pts, float_atr, bars_held)
                    realized_pnl = float_pts
                    self.in_flight_state[sym]["trailing_floor"] = None
                    self.in_flight_state[sym]["lock_notified"] = False
                else:
                    action = "HOLD_POSITION"
                    rationale = f"Mempertahankan posisi [{in_flight_pos}] | Floating: {float_pts:+.2f} pts ({float_atr:+.2f}x ATR)."
            else:
                if self.is_bot_paused():
                    action = "PAUSED_BY_USER"
                    rationale = "Pembukaan order baru dijeda manual melalui perintah Telegram (/pause)."
                elif prim_signal != "NEUTRAL":
                    if meta_pass:
                        action = f"EXECUTE_ENTRY_{prim_signal}"
                        rationale = f"Sinyal {prim_signal} disetujui Meta-Model (P_meta={m_prob:.4f} >= 0.54, Sizing={meta_sizing_mult:.2f}x)."
                        sl_pts = 1.5 * atr_val
                        pt_pts = 2.0 * atr_val
                        self.execute_market_order(sym, prim_signal, effective_lot, sl_pts, pt_pts, magic, p_prob, m_prob, meta_sizing_mult, roll_corr)
                        self.in_flight_state[sym]["trailing_floor"] = None
                        self.in_flight_state[sym]["lock_notified"] = False
                    else:
                        action = "BLOCKED_BY_META_GATE"
                        rationale = f"Sinyal {prim_signal} ditolak Meta-Gate (P_meta={m_prob:.4f} < 0.54 - Filter Lemparan Koin)."

            # Record to CSV & Console
            row_dict = {
                "timestamp": str(ts),
                "symbol": sym,
                "open": latest_bar["Open"],
                "high": latest_bar["High"],
                "low": latest_bar["Low"],
                "close": latest_bar["Close"],
                "mid_price": round((latest_bar["High"] + latest_bar["Low"]) / 2.0, 2),
                "atr_14": round(atr_val, 2),
                "spread_usd": round(sp, 2),
                "primary_prob": p_prob,
                "primary_signal": prim_signal,
                "meta_prob": m_prob,
                "meta_gate_pass": meta_pass,
                "meta_sizing_mult": meta_sizing_mult,
                "rolling_correlation_480": roll_corr,
                "corr_haircut_mult": corr_haircut,
                "effective_lot": effective_lot,
                "risk_budget_usd": risk_budget,
                "in_flight_position": in_flight_pos,
                "in_flight_entry_px": entry_px,
                "in_flight_floating_pts": float_pts,
                "in_flight_floating_atr": float_atr,
                "in_flight_bars_held": bars_held,
                "in_flight_trailing_floor": trailing_floor,
                "decision_action": action,
                "trade_realized_pnl": realized_pnl,
                "rationale": rationale
            }

            logger.log(
                timestamp=ts,
                open_px=latest_bar["Open"],
                high_px=latest_bar["High"],
                low_px=latest_bar["Low"],
                close_px=latest_bar["Close"],
                atr_14=atr_val,
                spread_usd=sp,
                primary_prob=p_prob,
                primary_signal=prim_signal,
                meta_prob=m_prob,
                meta_gate_pass=meta_pass,
                meta_sizing_mult=meta_sizing_mult,
                rolling_corr_480=roll_corr,
                corr_haircut_mult=corr_haircut,
                effective_lot=effective_lot,
                risk_budget_usd=risk_budget,
                in_flight_position=in_flight_pos,
                in_flight_entry_px=entry_px,
                in_flight_floating_pts=float_pts,
                in_flight_floating_atr=float_atr,
                in_flight_bars_held=bars_held,
                in_flight_trailing_floor=trailing_floor,
                decision_action=action,
                trade_realized_pnl=realized_pnl,
                rationale=rationale
            )

            # Kirim Hourly Telegram Digest via TelegramNotifier
            self.notifier.notify_hourly_digest(row_dict)

    def run_live_loop(self):
        print("\n" + "="*85)
        print("🤖 V4.0 DUAL-ASSET META-TRADING LIVE ENGINE AKTIF")
        print("   • Hourly Telegram Digest: Aktif (Gold saat Market Buka, BTC 24/7)")
        print("   • Daily Recap: Otomatis setiap 00:00 UTC (07:00 WIB)")
        print("   • Hourly Decision Logger: logs/hourly_decisions_gold.csv & btc.csv")
        print("="*85)

        if not self.connect_mt5():
            print("[FATAL] Gagal menghubungkan ke MT5.")
            return

        # 1. Immediate Initial Evaluation on Startup
        print("\n🚀 Menjalankan evaluasi inisial saat startup...")
        self.evaluate_cycle()
        last_evaluated_hour = datetime.datetime.now().hour

        while True:
            try:
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                now_local = datetime.datetime.now()

                # 2. Trigger Evaluasi Candle H1 Baru pada awal jam (minute 00)
                if now_local.minute == 0 and now_local.hour != last_evaluated_hour:
                    print(f"\n🔔 [CANDLE H1 BARU: {now_local.strftime('%Y-%m-%d %H:%M:%S')}] Memulai evaluasi dual-asset...")
                    time.sleep(2)
                    self.evaluate_cycle()
                    last_evaluated_hour = now_local.hour

                # 3. Trigger Daily Recap otomatis setiap 00:00 UTC (07:00 WIB)
                today_date_str = now_utc.strftime("%Y-%m-%d")
                if now_utc.hour == 0 and now_utc.minute == 5 and self.last_daily_recap_date != today_date_str:
                    self.notifier.send_daily_and_overall_recap(get_open_pos_callback=self.get_active_positions_summary_text)
                    self.last_daily_recap_date = today_date_str

                time.sleep(5)
            except KeyboardInterrupt:
                print("\n🛑 Live Engine dihentikan oleh user.")
                break
            except Exception as e:
                print(f"[ERROR] Exception di live loop: {e}")
                time.sleep(10)

def main():
    dry_run = "--dry-run" in sys.argv or "--single-pass" in sys.argv
    engine = DualAssetV4LiveEngine(dry_run=dry_run)
    
    if "--recap" in sys.argv:
        print("📢 Mengirim Rekapan Harian & Keseluruhan Instan ke Telegram...")
        if engine.connect_mt5():
            engine.notifier.send_daily_and_overall_recap(get_open_pos_callback=engine.get_active_positions_summary_text)
            mt5.shutdown()
    elif "--single-pass" in sys.argv:
        print("🧪 Menjalankan 1-Cycle Single Pass Evaluation (Dry-Run / Test Mode)...")
        if engine.connect_mt5():
            engine.evaluate_cycle()
            mt5.shutdown()
    else:
        engine.run_live_loop()

if __name__ == "__main__":
    main()
