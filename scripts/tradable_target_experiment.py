"""
tradable_target_experiment.py
=============================
Reproduce the comparison between non-tradable mid-range targets and tradable close/open targets.

Usage (from repository root):
    python scripts/tradable_target_experiment.py

Methodology:
- Evaluates 2,500 H1 bars per asset (Gold and Bitcoin) using TimeSeriesSplit (5-fold temporal expansion).
- Compares:
    1. Mid-to-mid: ln(Mid[t+1] / Mid[t]) where Mid = (High + Low) / 2 [non-tradable, leaks intra-bar range]
    2. Close-to-close: ln(Close[t+1] / Close[t]) [tradable entry at bar t close, exit at bar t+1 close]
    3. Next open-to-close: ln(Close[t+1] / Open[t+1]) [tradable entry at next bar open, exit at next close]
- Reports Out-Of-Fold (OOF) ROC-AUC, accuracy, and Brier score.
"""

import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
if not DATA_DIR.exists() and (ROOT.parent / "data").exists():
    DATA_DIR = ROOT.parent / "data"

PRIMARY_FEATURES = [
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5",
    "vol_10", "vol_20", "atr_14",
    "rsi_14", "macd", "macd_diff", "roc_10",
    "dist_sma_20", "dist_sma_50",
]


def load_asset_data(parquet_path):
    df = pd.read_parquet(parquet_path).iloc[-2500:].copy()
    for c in ["open", "high", "low", "close", "tick_volume", "volume"]:
        for act in df.columns:
            if act.lower() == c:
                df.rename(columns={act: c.capitalize() if c != "tick_volume" else "Volume"}, inplace=True)
    df["MidPrice"] = (df["High"] + df["Low"]) / 2.0
    return df


def engineer_features_and_targets(df_raw):
    df = df_raw.copy()
    
    # Target 1: Mid-to-mid (non-tradable)
    df["ret_mid"] = np.log(df["MidPrice"] / df["MidPrice"].shift(1))
    df["target_mid"] = (df["ret_mid"].shift(-1) > 0).astype(int)

    # Target 2: Close-to-close (tradable)
    df["ret_close_to_close"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["target_close_to_close"] = (df["ret_close_to_close"] > 0).astype(int)

    # Target 3: Next open-to-close (tradable)
    df["ret_open_to_close"] = np.log(df["Close"].shift(-1) / df["Open"].shift(-1))
    df["target_open_to_close"] = (df["ret_open_to_close"] > 0).astype(int)

    # Technical features strictly known at bar t
    for i in range(1, 6):
        df[f"ret_lag_{i}"] = df["ret_mid"].shift(i)
    df["vol_10"] = df["ret_mid"].rolling(10).std()
    df["vol_20"] = df["ret_mid"].rolling(20).std()
    df["atr_14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_diff"] = macd.macd_diff()
    df["roc_10"] = ta.momentum.ROCIndicator(df["Close"], 10).roc()
    df["dist_sma_20"] = (df["MidPrice"] - df["MidPrice"].rolling(20).mean()) / df["MidPrice"].rolling(20).mean()
    df["dist_sma_50"] = (df["MidPrice"] - df["MidPrice"].rolling(50).mean()) / df["MidPrice"].rolling(50).mean()

    df.dropna(inplace=True)
    return df


def evaluate_targets_for_symbol(symbol_name, file_name):
    path = DATA_DIR / file_name
    if not path.exists():
        print(f"[SKIP] {symbol_name}: {path} not found. Training data is not in repository.")
        return

    df_raw = load_asset_data(path)
    df = engineer_features_and_targets(df_raw)
    X = df[PRIMARY_FEATURES]

    targets = {
        "Mid-to-mid (original)": df["target_mid"],
        "Close-to-close (tradable)": df["target_close_to_close"],
        "Next open-to-close (tradable)": df["target_open_to_close"],
    }

    print(f"\n{'='*75}")
    print(f"Target Tradability Experiment: {symbol_name} ({len(df):,} valid H1 bars)")
    print(f"{'='*75}")
    print(f"{'Target Type':<32} | {'OOF AUC':<8} | {'OOF Acc':<8} | {'Brier':<8}")
    print(f"{'-'*75}")

    tscv = TimeSeriesSplit(n_splits=5)
    for tgt_name, y in targets.items():
        oof_probs = np.full(len(df), np.nan)
        for tr_idx, val_idx in tscv.split(X):
            model = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.03, max_depth=3,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                random_state=42, verbosity=-1, n_jobs=-1
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof_probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

        valid_m = ~np.isnan(oof_probs)
        y_val = y.iloc[valid_m]
        p_val = oof_probs[valid_m]

        auc = roc_auc_score(y_val, p_val)
        acc = accuracy_score(y_val, (p_val >= 0.50).astype(int))
        brier = brier_score_loss(y_val, p_val)

        print(f"{tgt_name:<32} | {auc:.3f}    | {acc:.1%}    | {brier:.4f}")


def main():
    print("Running out-of-fold tradable target experiment...")
    evaluate_targets_for_symbol("XAUUSD (Gold)", "xauusd_h1_macro.parquet")
    evaluate_targets_for_symbol("BTCUSD (Bitcoin)", "btcusd_h1_raw.parquet")


if __name__ == "__main__":
    main()
