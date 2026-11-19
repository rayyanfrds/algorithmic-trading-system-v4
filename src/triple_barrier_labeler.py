# -*- coding: utf-8 -*-
"""
triple_barrier_labeler.py
=========================
Implementasi Triple Barrier Method (Marcos López de Prado - Advances in Financial Machine Learning, Ch. 3)
untuk Data H1 XAUUSD & BTCUSD:
1. Upper Barrier (Profit Take): +pt_mult * ATR(14)
2. Lower Barrier (Stop Loss)  : -sl_mult * ATR(14)
3. Vertical Barrier (Time)    : max_holding_bars (default: 12 bars)

CATATAN METODOLOGIS & KONTROL DATA LEAKAGE:
Fungsi ini melakukan forward scan sepanjang horizon `max_holding_bars` untuk menentukan barrier mana
yang tersentuh pertama kali. Karena event horizon saling tumpang tindih (overlapping labels),
proses training model wajib menggunakan validasi temporal (seperti TimeSeriesSplit atau Purged K-Fold)
dengan embargo guna mencegah kontaminasi informasi dari masa depan ke masa lalu.
"""

import numpy as np
import pandas as pd
import ta

def compute_atr_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Menghitung Average True Range (ATR) sebagai batas dinamis berbasis volatilitas."""
    atr = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=window
    ).average_true_range()
    return atr

def apply_triple_barrier(
    df: pd.DataFrame,
    primary_side: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding_bars: int = 12,
    atr_window: int = 14
) -> pd.DataFrame:
    """
    Menerapkan Triple Barrier Method pada setiap bar event t:
    - primary_side: Arah posisi yang diusulkan oleh Model Primer (+1 untuk BUY, -1 untuk SELL, 0 untuk PASS)
    - pt_mult     : Pengali ATR untuk target profit
    - sl_mult     : Pengali ATR untuk batas stop loss
    - max_holding_bars: Batas waktu lilin maksimum (Vertical Barrier)
    
    Returns DataFrame dengan kolom:
    - barrier_touch_time: Waktu saat salah satu barrier pertama kali tersentuh
    - barrier_type      : 'PT' (Profit Take), 'SL' (Stop Loss), atau 'TIME' (Vertical Barrier)
    - ret_barrier       : Return aktual dari entry ke exit barrier
    - meta_target       : 1 jika mencapai PT sebelum SL/timeout, 0 jika gagal (kena SL atau rugi di timeout)
    """
    df = df.copy()
    if "MidPrice" not in df.columns:
        df["MidPrice"] = (df["High"] + df["Low"]) / 2.0
        
    atr = compute_atr_volatility(df, window=atr_window)
    df["atr_dyn"] = atr

    n = len(df)
    events = []

    for i in range(n - max_holding_bars):
        side = primary_side.iloc[i]
        if side == 0 or np.isnan(side):
            continue

        t_entry = df.index[i]
        px_entry = df["MidPrice"].iloc[i]
        vol = df["atr_dyn"].iloc[i]

        if np.isnan(vol) or vol <= 0:
            continue

        pt_price = px_entry + side * (pt_mult * vol)
        sl_price = px_entry - side * (sl_mult * vol)

        touch_idx = None
        touch_type = "TIME"
        exit_px = df["MidPrice"].iloc[i + max_holding_bars]
        exit_time = df.index[i + max_holding_bars]

        # Scan jendela ke depan (strictly forward path)
        for step in range(1, max_holding_bars + 1):
            curr_idx = i + step
            curr_high = df["High"].iloc[curr_idx]
            curr_low  = df["Low"].iloc[curr_idx]
            curr_time = df.index[curr_idx]

            if side == 1:  # Long / BUY
                # Cek Stop Loss dulu (Konservatif / Adversarial)
                if curr_low <= sl_price:
                    touch_idx = curr_idx
                    touch_type = "SL"
                    exit_px = sl_price
                    exit_time = curr_time
                    break
                # Cek Profit Take
                elif curr_high >= pt_price:
                    touch_idx = curr_idx
                    touch_type = "PT"
                    exit_px = pt_price
                    exit_time = curr_time
                    break
            elif side == -1:  # Short / SELL
                # Cek Stop Loss dulu
                if curr_high >= sl_price:
                    touch_idx = curr_idx
                    touch_type = "SL"
                    exit_px = sl_price
                    exit_time = curr_time
                    break
                # Cek Profit Take
                elif curr_low <= pt_price:
                    touch_idx = curr_idx
                    touch_type = "PT"
                    exit_px = pt_price
                    exit_time = curr_time
                    break

        raw_ret = side * (exit_px - px_entry) / px_entry
        meta_label = 1 if touch_type == "PT" else (1 if (touch_type == "TIME" and raw_ret > 0.001) else 0)

        events.append({
            "entry_time": t_entry,
            "exit_time": exit_time,
            "side": side,
            "entry_px": px_entry,
            "exit_px": exit_px,
            "touch_type": touch_type,
            "raw_ret": raw_ret,
            "meta_target": meta_label,
            "holding_bars": (touch_idx - i) if touch_idx else max_holding_bars
        })

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df.set_index("entry_time", inplace=True)
    return events_df
