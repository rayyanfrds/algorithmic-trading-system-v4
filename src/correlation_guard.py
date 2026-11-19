# -*- coding: utf-8 -*-
"""
correlation_guard.py
====================
Real-Time Dynamic Cross-Asset Correlation Guard for VPS Production:
- Monitors 20-day (480-bar H1) rolling Pearson correlation between XAUUSD and BTCUSD
- Detects regime co-movement tail risk (when both models take directional BUY or SELL)
- Applies smooth linear risk haircut:
    w = clip(1.0 - (|rho| - 0.40) / 0.60 * 0.50, 0.50, 1.0)
- Exposes telemetry for live bot execution and Telegram Commander (/corr)
"""

import logging
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

log = logging.getLogger("CorrelationGuard")

DEFAULT_ROLLING_WINDOW_BARS = 480  # 20 trading days on H1 timeframe
CORRELATION_THRESHOLD = 0.40       # Activation threshold
MAX_HAIRCUT_FRACTION = 0.50        # Maximum 50% lot reduction at |rho| = 1.00

class CrossAssetCorrelationGuard:
    def __init__(self, window_bars: int = DEFAULT_ROLLING_WINDOW_BARS, threshold: float = CORRELATION_THRESHOLD):
        self.window_bars = window_bars
        self.threshold = threshold
        self.last_corr = 0.28
        self.last_bar_time = None
        self.last_haircut_mult = 1.0

    def fetch_recent_returns(self, symbol: str, n_bars: int = 500) -> pd.Series:
        """Mengambil returns H1 terbaru dari MT5 untuk suatu simbol."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars)
        if rates is None or len(rates) < 60:
            log.warning(f"Gagal mengambil bar MT5 untuk correlation check ({symbol})")
            return pd.Series(dtype=float)
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.iloc[:-1]
        mid = (df["high"] + df["low"]) / 2.0
        ret = np.log(mid / mid.shift(1)).dropna()
        return ret

    def compute_current_correlation(self, gold_symbol: str = "XAUUSDm", btc_symbol: str = "BTCUSDm") -> float:
        """
        Menghitung korelasi Pearson rolling 480 bar antara XAUUSD dan BTCUSD.
        """
        try:
            ret_gold = self.fetch_recent_returns(gold_symbol, self.window_bars + 30)
            ret_btc  = self.fetch_recent_returns(btc_symbol, self.window_bars + 30)

            if ret_gold.empty or ret_btc.empty:
                log.warning("Data returns tidak lengkap, menggunakan baseline korelasi historis 0.28")
                return self.last_corr

            df_combined = pd.DataFrame({"gold": ret_gold, "btc": ret_btc}).dropna()
            if len(df_combined) < 60:
                log.warning(f"Jumlah bar bersama ({len(df_combined)}) kurang dari 60 bar.")
                return self.last_corr

            sub = df_combined.iloc[-self.window_bars:]
            corr = float(sub["gold"].corr(sub["btc"]))
            if np.isnan(corr):
                corr = 0.28

            self.last_corr = corr
            self.last_bar_time = sub.index[-1]
            return corr
        except Exception as e:
            log.error(f"Error kalkulasi correlation guard: {e}")
            return self.last_corr

    def get_haircut_multiplier(self, corr: float, is_co_moving: bool = True) -> float:
        """
        Menghitung pengurang lot (haircut) jika korelasi melampaui ambang batas (|rho| > 0.40)
        dan kedua aset berada pada arah yang sama (keduanya BUY atau keduanya SELL).
        """
        abs_corr = abs(corr)
        if not is_co_moving or abs_corr <= self.threshold:
            self.last_haircut_mult = 1.0
            return 1.0

        excess = (abs_corr - self.threshold) / (1.0 - self.threshold)
        haircut = 1.0 - (excess * MAX_HAIRCUT_FRACTION)
        haircut = float(np.clip(haircut, 0.50, 1.0))
        self.last_haircut_mult = haircut
        return haircut

    def apply_correlation_sizing(
        self,
        base_lot: float,
        current_symbol: str,
        proposed_signal: str,
        gold_symbol: str = "XAUUSDm",
        btc_symbol: str = "BTCUSDm",
        gold_magic: int = 100001,
        btc_magic: int = 100002
    ) -> tuple:
        """
        Memeriksa posisi pasangan pair lain di MT5 dan menerapkan haircut lot jika co-moving tinggi.
        Returns: (final_lot, correlation, haircut_multiplier)
        """
        corr = self.compute_current_correlation(gold_symbol, btc_symbol)

        other_symbol = btc_symbol if "XAU" in current_symbol.upper() else gold_symbol
        other_magic  = btc_magic if "XAU" in current_symbol.upper() else gold_magic

        other_pos_dir = 0
        positions = mt5.positions_get(symbol=other_symbol)
        if positions:
            for p in positions:
                if p.magic == other_magic:
                    other_pos_dir = 1 if p.type == mt5.ORDER_TYPE_BUY else -1
                    break

        proposed_dir = 1 if proposed_signal == "BUY" else (-1 if proposed_signal == "SELL" else 0)
        is_co_moving = (proposed_dir != 0 and other_pos_dir != 0 and proposed_dir == other_pos_dir)

        haircut_mult = self.get_haircut_multiplier(corr, is_co_moving=is_co_moving)
        final_lot = round(base_lot * haircut_mult, 2 if "XAU" in current_symbol else 3)
        
        min_lot = 0.01
        if final_lot < min_lot and base_lot >= min_lot:
            final_lot = min_lot

        if haircut_mult < 1.0:
            log.warning(
                f"🛡️ [Correlation Guard ACTIVATED] rho={corr:+.4f} (|rho|>{self.threshold:.2f}) | "
                f"Co-moving: {current_symbol} ({proposed_signal}) & {other_symbol} ({'BUY' if other_pos_dir==1 else 'SELL'}) | "
                f"Haircut: {haircut_mult:.2f}x | Base: {base_lot} -> Final: {final_lot} Lot"
            )

        return final_lot, corr, haircut_mult

    def get_telemetry_status(self, gold_symbol: str = "XAUUSDm", btc_symbol: str = "BTCUSDm") -> dict:
        """Mengembalikan status telemetri lengkap untuk bot dan Telegram."""
        corr = self.compute_current_correlation(gold_symbol, btc_symbol)
        haircut = self.get_haircut_multiplier(corr, is_co_moving=True)
        return {
            "rolling_correlation": corr,
            "window_bars": self.window_bars,
            "threshold": self.threshold,
            "haircut_multiplier": haircut,
            "haircut_pct": (1.0 - haircut) * 100.0,
            "is_elevated": abs(corr) > self.threshold,
            "last_bar_time": str(self.last_bar_time) if self.last_bar_time else "N/A"
        }
