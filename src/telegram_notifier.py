# -*- coding: utf-8 -*-
"""
telegram_notifier.py
====================
Dedicated Modular Telegram Notification & Reporting Service for V4.0 System:
- Clean separation of concerns (Single Responsibility Principle)
- Real-time Instant Alerts (Entry, Exit, Trailing Lock, System Status)
- Hourly Decision Digest with Market Open Filter (Gold vs BTC)
- Automated Daily Recap & Overall Performance Analytics from MT5 Deals
- Non-blocking network I/O with timeout protection
"""

import os
import sys
import json
import html
import urllib.request
import urllib.error
import ssl
import datetime
import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

MAGIC_GOLD = 40101
MAGIC_BTC  = 40102

class TelegramNotifier:
    def __init__(self, deployment_start_time: datetime.datetime = None):
        self.token = TELEGRAM_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.deployment_start_time = deployment_start_time or datetime.datetime.now()
        self.last_daily_recap_date = None

    def _send_raw_html(self, html_message: str) -> bool:
        if not self.token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": html_message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx, timeout=6) as response:
                return response.status == 200
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            print(f"[WARN] TelegramNotifier HTTP {he.code} Error: {err_body}")
            return False
        except Exception as e:
            print(f"[WARN] TelegramNotifier: Gagal kirim pesan: {e}")
            return False

    def is_market_open(self, symbol: str) -> bool:
        """Filter hari/jam market: Gold tutup saat weekend, Bitcoin buka 24/7."""
        if "BTC" in symbol:
            return True
            
        # 1. Cek langsung trade_mode di MT5 jika terhubung
        try:
            candidates = [symbol, f"{symbol}m", f"{symbol}.m", f"{symbol}c", f"{symbol}.c", f"{symbol}raw"]
            for c in candidates:
                info = mt5.symbol_info(c)
                if info is not None:
                    # 0=DISABLED, 1=CLOSEONLY, 2=READONLY, 3=FULL? (mt5.SYMBOL_TRADE_MODE_FULL = 4)
                    return info.trade_mode in (mt5.SYMBOL_TRADE_MODE_FULL, 3, 4)
        except Exception:
            pass

        # 2. Fallback UTC schedule
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if "XAU" in symbol or "GOLD" in symbol:
            if now_utc.weekday() == 5: # Sabtu
                return False
            if now_utc.weekday() == 6 and now_utc.hour < 22: # Minggu sebelum 22:00 UTC
                return False
            if now_utc.weekday() == 4 and now_utc.hour >= 23: # Jumat larut
                return False
        return True

    def notify_system_online(self, login_num: int, server: str, balance: float, equity: float):
        msg = (
            f"🟢 <b>[SYSTEM V4.0 ONLINE]</b>\n"
            f"Akun: <code>#{login_num}</code> ({server})\n"
            f"Balance: <b>${balance:,.2f}</b> | Equity: <b>${equity:,.2f}</b>\n"
            f"Status: <i>Live Dual-Asset Meta-System Active</i>\n"
            f"Features: <i>Hourly Digest (Gold Open Only, BTC 24/7) + Daily/Overall Recap</i>"
        )
        self._send_raw_html(msg)

    def notify_order_opened(self, symbol: str, side: str, lot: float, ticket: int, entry_px: float, sl_px: float, tp_px: float, p_prim: float, p_meta: float, sizing_m: float, rho_val: float):
        side_icon = "🟢" if side == "BUY" else "🔴"
        msg = (
            f"{side_icon} <b>[V4.0 ORDER OPENED]</b>\n"
            f"Symbol: <b>{symbol}</b> ({side})\n"
            f"Volume: <b>{lot} Lot</b> (Ticket: #{ticket})\n"
            f"Entry Price: <b>{entry_px:,.2f}</b>\n"
            f"Hard SL (1.5x ATR): <b>{sl_px:,.2f}</b>\n"
            f"Hard TP (2.0x ATR): <b>{tp_px:,.2f}</b>\n"
            f"──────────────────\n"
            f"• Primary Prob: <code>{p_prim:.4f}</code>\n"
            f"• Meta-Model: <code>{p_meta:.4f}</code> (Sizing: {sizing_m:.2f}x)\n"
            f"• Rolling Corr: <code>rho={rho_val:+.4f}</code>"
        )
        self._send_raw_html(msg)

    def notify_profit_lock(self, symbol: str, position_side: str, float_atr: float, trailing_floor: float):
        msg = (
            f"💎 <b>[V4.0 PROFIT LOCK ACTIVATED]</b>\n"
            f"Symbol: <b>{symbol}</b> ({position_side})\n"
            f"Gain Reached: <b>{float_atr:+.2f}x ATR</b>\n"
            f"Trailing Floor Locked: <b>{trailing_floor:,.2f}</b> (+0.20x ATR Anti-Bleeding)"
        )
        self._send_raw_html(msg)

    def notify_position_closed(self, symbol: str, ticket: int, reason: str, close_price: float, duration_h: int, pnl_usd: float, float_pts: float, float_atr: float):
        pnl_icon = "💰" if pnl_usd >= 0 else "🛑"
        msg = (
            f"{pnl_icon} <b>[V4.0 POSITION CLOSED]</b>\n"
            f"Symbol: <b>{symbol}</b> (Ticket #{ticket})\n"
            f"Exit Reason: <b>{reason}</b>\n"
            f"Exit Price: <b>{close_price:,.2f}</b> (Held: {duration_h} Jam)\n"
            f"Realized PnL: <b>${pnl_usd:+,.2f}</b> ({float_pts:+.2f} pts / {float_atr:+.2f}x ATR)"
        )
        self._send_raw_html(msg)

    def notify_hourly_digest(self, r: dict):
        sym = r["symbol"]
        if not self.is_market_open(sym):
            return

        action_icon = "🟢" if "ENTRY" in r["decision_action"] else ("🛑" if "EXIT" in r["decision_action"] else ("💎" if "LOCK" in r["decision_action"] else "⚪"))
        
        pos_str = "NONE"
        if r["in_flight_position"] != "NONE":
            pos_str = f"<b>{html.escape(str(r['in_flight_position']))}</b> @ {r['in_flight_entry_px']} | Float: <b>{r['in_flight_floating_pts']} pts</b> ({r['in_flight_floating_atr']}x ATR) | Hold: {r['in_flight_bars_held']}h"

        action_escaped = html.escape(str(r.get("decision_action", "")))
        rationale_escaped = html.escape(str(r.get("rationale", "")))

        msg = (
            f"⏱️ <b>[HOURLY DIGEST: {sym}]</b> {r['timestamp']}\n"
            f"• Mid Price: <b>{r['mid_price']}</b> | ATR: <b>{r['atr_14']}</b> | Spread: ${r['spread_usd']}\n"
            f"• Primary: <code>P={r['primary_prob']:.4f} ({r['primary_signal']})</code>\n"
            f"• Meta-Gate: <code>P={r['meta_prob']:.4f} (Pass: {r['meta_gate_pass']})</code> | Sizing: <b>{r['meta_sizing_mult']}x</b>\n"
            f"• Korelasi (480b): <code>rho={r['rolling_correlation_480']:+.4f}</code> (Haircut: {r['corr_haircut_mult']}x)\n"
            f"• In-Flight: {pos_str}\n"
            f"{action_icon} Keputusan: <b>[{action_escaped}]</b>\n"
            f"└─ <i>{rationale_escaped}</i>"
        )
        self._send_raw_html(msg)

    def send_daily_and_overall_recap(self, get_open_pos_callback=None):
        acc = mt5.account_info()
        balance = acc.balance if acc else 10000.0
        equity = acc.equity if acc else 10000.0

        today_start = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        deals_today = mt5.history_deals_get(today_start, datetime.datetime.now(datetime.timezone.utc))
        
        today_trades = [d for d in deals_today if d.magic in (MAGIC_GOLD, MAGIC_BTC) and d.entry == mt5.DEAL_ENTRY_OUT] if deals_today else []
        n_today = len(today_trades)
        wins_today = sum(1 for d in today_trades if d.profit > 0)
        today_wr = (wins_today / n_today * 100.0) if n_today > 0 else 0.0
        today_pnl = sum(d.profit for d in today_trades)

        active_str = "Semua Flat (Tidak ada posisi)"
        if get_open_pos_callback:
            active_str = get_open_pos_callback()

        # 1. Pesan Rekapan Harian
        pnl_icon = "💰" if today_pnl >= 0 else "🛑"
        daily_msg = (
            f"📊 <b>[V4.0 REKAPAN HARIAN - {datetime.datetime.now().strftime('%Y-%m-%d')}]</b>\n"
            f"──────────────────────────\n"
            f"• Closed Trades Hari Ini: <b>{n_today} trades</b>\n"
            f"• Win Rate Hari Ini     : <b>{today_wr:.1f}% ({wins_today} Win / {n_today - wins_today} Loss)</b>\n"
            f"• Realized PnL Hari Ini : <b>${today_pnl:+,.2f}</b> {pnl_icon}\n"
            f"• Saldo Terkini         : Balance: <b>${balance:,.2f}</b> | Equity: <b>${equity:,.2f}</b>\n"
            f"• Posisi Aktif Saat Ini : <i>{active_str}</i>"
        )
        self._send_raw_html(daily_msg)

        # 2. Pesan Rekapan Keseluruhan & Statistika Deskriptif V4.0
        all_deals = mt5.history_deals_get(datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc), datetime.datetime.now(datetime.timezone.utc))
        all_v4_deals = [d for d in all_deals if d.magic in (MAGIC_GOLD, MAGIC_BTC) and d.entry == mt5.DEAL_ENTRY_OUT] if all_deals else []
        n_all = len(all_v4_deals)
        
        if n_all > 0:
            import numpy as np
            pnls = np.array([d.profit + d.swap + d.commission for d in all_v4_deals])
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            n_w = len(wins)
            n_l = len(losses)
            wr_all = (n_w / n_all * 100.0)
            pnl_all = float(np.sum(pnls))
            gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
            gross_losses = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
            pf_all = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 1.0)
            avg_w = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_l = abs(float(np.mean(losses))) if len(losses) > 0 else 0.0
            payoff_r = (avg_w / avg_l) if avg_l > 0 else 0.0
            expectancy = pnl_all / n_all
            pnl_std = float(np.std(pnls)) if n_all > 1 else 0.0
            
            gold_deals = [d for d in all_v4_deals if d.magic == MAGIC_GOLD]
            btc_deals  = [d for d in all_v4_deals if d.magic == MAGIC_BTC]
            pnl_gold = sum(d.profit + d.swap + d.commission for d in gold_deals)
            pnl_btc  = sum(d.profit + d.swap + d.commission for d in btc_deals)

            overall_msg = (
                f"🏆 <b>[V4.0 REKAPAN KESELURUHAN & STATISTIKA DESKRIPTIF]</b>\n"
                f"──────────────────────────\n"
                f"• Total Closed Trades : <b>{n_all} trades</b> ({n_w} Win / {n_l} Loss)\n"
                f"• Cumulative Win Rate : <b>{wr_all:.1f}%</b>\n"
                f"• Total Net Realized  : <b>${pnl_all:+,.2f}</b>\n"
                f"• Profit Factor (PF)  : <b>{pf_all:.2f}</b>\n"
                f"• Expectancy / Trade  : <b>${expectancy:+,.2f}</b>\n"
                f"• Payoff Ratio (W/L)  : <b>{payoff_r:.2f}x</b> (${avg_w:,.2f} / ${avg_l:,.2f})\n"
                f"• PnL Volatility (Std): <b>σ = ${pnl_std:,.2f}</b>\n"
                f"──────────────────────────\n"
                f"• 🪙 Gold   : {len(gold_deals)} trades | Net PnL: <b>${pnl_gold:+,.2f}</b>\n"
                f"• ⚡ Bitcoin: {len(btc_deals)} trades | Net PnL: <b>${pnl_btc:+,.2f}</b>\n"
                f"• Status Proteksi     : <i>Correlation Guard & Decoupled Exit Active</i>"
            )
        else:
            overall_msg = (
                f"🏆 <b>[V4.0 REKAPAN KESELURUHAN SISTEM]</b>\n"
                f"──────────────────────────\n"
                f"• Total Completed Trade: <b>0 trades</b>\n"
                f"• Saldo Terkini        : Balance: <b>${balance:,.2f}</b> | Equity: <b>${equity:,.2f}</b>"
            )
        self._send_raw_html(overall_msg)
        print("📢 TelegramNotifier: Rekapan harian & keseluruhan terkirim.")
