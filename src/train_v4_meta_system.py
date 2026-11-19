# -*- coding: utf-8 -*-
"""
train_v4_meta_system.py
=======================
Production Model Training, Out-Of-Fold (OOF) Validation & Serialization Engine for V4.0:

ANTI-LEAKAGE ARCHITECTURE (Addressing In-Sample Meta Overfitting):
1. Primary Directional Classifier is trained with TimeSeriesSplit (5-fold temporal expansion).
2. Primary probabilities for Meta-Model training are generated STRICTLY OUT-OF-FOLD (OOF).
   The meta-model never sees in-sample primary probabilities, preventing it from learning on
   over-optimistic, memorized training distributions.
3. Offline validation metrics (In-Sample AUC, Out-Of-Fold AUC, Brier Score, Meta Precision Lift)
   are calculated and saved to validate offline generalization prior to deployment.
4. Serializes models (.pkl) and offline validation manifests into models/ directory.
"""

import os
import sys
import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, precision_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Dynamic project paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
if not DATA_DIR.exists() and (PROJECT_ROOT.parent / "data").exists():
    DATA_DIR = PROJECT_ROOT.parent / "data"

MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BTC_PARQUET  = DATA_DIR / "btcusd_h1_raw.parquet"
GOLD_PARQUET = DATA_DIR / "xauusd_h1_macro.parquet"

sys.path.insert(0, str(SRC_DIR))
from triple_barrier_labeler import apply_triple_barrier

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

def load_data():
    if not BTC_PARQUET.exists() or not GOLD_PARQUET.exists():
        raise FileNotFoundError(
            f"Dataset parquet tidak ditemukan di {DATA_DIR}. "
            f"Pastikan btcusd_h1_raw.parquet dan xauusd_h1_macro.parquet tersedia."
        )

    df_btc = pd.read_parquet(BTC_PARQUET)
    df_gold = pd.read_parquet(GOLD_PARQUET)

    for df, sym in [(df_btc, "BTC"), (df_gold, "GOLD")]:
        for c in ["open", "high", "low", "close", "tick_volume", "volume"]:
            for act in df.columns:
                if act.lower() == c:
                    df.rename(columns={act: c.capitalize() if c != "tick_volume" else "Volume"}, inplace=True)
        df["MidPrice"] = (df["High"] + df["Low"]) / 2.0
        df["Spread"] = 15.0 if sym == "BTC" else 0.30

    return df_btc, df_gold

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df.dropna(inplace=True)
    return df

def train_and_validate_symbol(symbol: str, df_feat: pd.DataFrame, train_lookback_bars: int = 2500, n_splits: int = 5):
    print(f"\n" + "="*80)
    print(f"📦 TRAINING & OUT-OF-FOLD (OOF) VALIDATION: {symbol}")
    print("="*80)

    T = len(df_feat)
    train_df = df_feat.iloc[-train_lookback_bars:].copy() if T > train_lookback_bars else df_feat.copy()
    print(f"   • Historical Window: {len(train_df):,} bars ({train_df.index[0]} s/d {train_df.index[-1]})")

    X_primary = train_df[PRIMARY_FEATURES]
    y_primary = train_df["target_1b"]

    # -------------------------------------------------------------------------
    # 1. OUT-OF-FOLD (OOF) PRIMARY MODEL CROSS-VALIDATION
    # -------------------------------------------------------------------------
    # Use TimeSeriesSplit to avoid look-ahead bias and leakages
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_primary_probs = np.full(len(train_df), np.nan)

    for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_primary), start=1):
        fold_model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            random_state=42, verbosity=-1, n_jobs=-1
        )
        fold_model.fit(X_primary.iloc[tr_idx], y_primary.iloc[tr_idx])
        oof_primary_probs[val_idx] = fold_model.predict_proba(X_primary.iloc[val_idx])[:, 1]

    # Evaluate Primary Out-of-Fold Performance
    valid_mask = ~np.isnan(oof_primary_probs)
    y_val_actual = y_primary.iloc[valid_mask]
    val_probs = oof_primary_probs[valid_mask]

    primary_oof_auc = float(roc_auc_score(y_val_actual, val_probs))
    primary_oof_acc = float(accuracy_score(y_val_actual, (val_probs >= 0.50).astype(int)))
    primary_oof_brier = float(brier_score_loss(y_val_actual, val_probs))

    # Fit In-Sample Primary Model to measure overfit gap
    full_primary_model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    full_primary_model.fit(X_primary, y_primary)
    in_sample_probs = full_primary_model.predict_proba(X_primary)[:, 1]
    primary_in_sample_auc = float(roc_auc_score(y_primary, in_sample_probs))

    print(f"   • Primary Model Evaluation (Features: {len(PRIMARY_FEATURES)}):")
    print(f"     ├─ In-Sample AUC (Overfitted) : {primary_in_sample_auc:.4f}")
    print(f"     ├─ Out-Of-Fold (OOF) AUC      : {primary_oof_auc:.4f}  [Overfit Gap: -{(primary_in_sample_auc - primary_oof_auc):.4f}]")
    print(f"     ├─ OOF Directional Accuracy   : {primary_oof_acc*100:.2f}%")
    print(f"     └─ OOF Brier Score (Loss)     : {primary_oof_brier:.4f}")

    # -------------------------------------------------------------------------
    # 2. GENERATE OOF TRIPLE BARRIER EVENTS FOR META-MODEL
    # -------------------------------------------------------------------------
    # The meta-model MUST be trained on OOF primary probabilities, NEVER in-sample!
    valid_indices = np.where(valid_mask)[0]
    df_oof = train_df.iloc[valid_indices].copy()
    df_oof["primary_prob"] = val_probs
    df_oof["primary_confidence"] = np.abs(val_probs - 0.50)

    # Direction side proposed strictly by OOF predictions
    p_side_oof = pd.Series(np.where(val_probs >= 0.50, 1, -1), index=df_oof.index)

    events = apply_triple_barrier(df_oof, p_side_oof, pt_mult=2.0, sl_mult=1.5, max_holding_bars=12)
    df_oof["meta_target"] = events["meta_target"]

    meta_train_data = df_oof.dropna(subset=["meta_target"]).copy()
    X_meta = meta_train_data[META_FEATURES]
    y_meta = meta_train_data["meta_target"].astype(int)

    base_success_rate = float(y_meta.mean() * 100.0)
    print(f"   • Triple Barrier Events Generated: {len(meta_train_data):,} OOF events")
    print(f"     └─ Baseline Success Rate (Raw Signals): {base_success_rate:.2f}%")

    # -------------------------------------------------------------------------
    # 3. META-MODEL VALIDATION (OOF Split on Meta Dataset)
    # -------------------------------------------------------------------------
    tscv_meta = TimeSeriesSplit(n_splits=4)
    oof_meta_probs = np.full(len(meta_train_data), np.nan)

    for tr_m, val_m in tscv_meta.split(X_meta):
        fold_meta_m = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            random_state=42, verbosity=-1, n_jobs=-1
        )
        fold_meta_m.fit(X_meta.iloc[tr_m], y_meta.iloc[tr_m])
        oof_meta_probs[val_m] = fold_meta_m.predict_proba(X_meta.iloc[val_m])[:, 1]

    meta_valid_mask = ~np.isnan(oof_meta_probs)
    y_meta_actual = y_meta.iloc[meta_valid_mask]
    val_meta_probs = oof_meta_probs[meta_valid_mask]

    meta_oof_auc = float(roc_auc_score(y_meta_actual, val_meta_probs))
    meta_pred_54 = (val_meta_probs >= 0.54).astype(int)
    meta_precision_54 = float(precision_score(y_meta_actual, meta_pred_54, zero_division=0) * 100.0)
    meta_rejection_rate = float((1.0 - meta_pred_54.mean()) * 100.0)

    print(f"   • Meta-Model Evaluation (Features: {len(META_FEATURES)}):")
    print(f"     ├─ Meta Out-Of-Fold AUC       : {meta_oof_auc:.4f}")
    print(f"     ├─ Precision @ Threshold ≥0.54: {meta_precision_54:.2f}% vs Baseline {base_success_rate:.2f}%")
    print(f"     ├─ Precision Lift             : {(meta_precision_54 - base_success_rate):+.2f}%")
    print(f"     └─ Signal Rejection Rate (0.54): {meta_rejection_rate:.1f}% filtered out")

    # -------------------------------------------------------------------------
    # 4. TRAIN FINAL PRODUCTION MODELS & SERIALIZE
    # -------------------------------------------------------------------------
    # Final Meta-Model is trained on the full OOF-clean dataset
    final_meta_model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    final_meta_model.fit(X_meta, y_meta)

    prefix = "gold" if "XAU" in symbol or "GOLD" in symbol else "btc"
    primary_path = MODELS_DIR / f"{prefix}_primary_v4.pkl"
    meta_path    = MODELS_DIR / f"{prefix}_meta_v4.pkl"

    joblib.dump(full_primary_model, primary_path)
    joblib.dump(final_meta_model, meta_path)

    print(f"   • Serialized Production Artifacts Saved:")
    print(f"     ├─ Primary : {primary_path}")
    print(f"     └─ Meta    : {meta_path}")

    return {
        "symbol": symbol,
        "prefix": prefix,
        "n_samples": len(train_df),
        "n_oof_samples": len(meta_train_data),
        "primary_features": PRIMARY_FEATURES,
        "meta_features": META_FEATURES,
        "pt_mult": 2.0,
        "sl_mult": 1.5,
        "max_hold_bars": 12,
        "meta_gate_threshold": 0.54,
        "training_end_ts": str(train_df.index[-1]),
        "offline_validation": {
            "primary_in_sample_auc": primary_in_sample_auc,
            "primary_oof_auc": primary_oof_auc,
            "primary_overfit_gap": round(primary_in_sample_auc - primary_oof_auc, 4),
            "primary_oof_accuracy": round(primary_oof_acc * 100.0, 2),
            "primary_oof_brier_score": round(primary_oof_brier, 4),
            "meta_baseline_success_rate": round(base_success_rate, 2),
            "meta_oof_auc": round(meta_oof_auc, 4),
            "meta_precision_at_054": round(meta_precision_54, 2),
            "meta_precision_lift": round(meta_precision_54 - base_success_rate, 2),
            "meta_rejection_rate": round(meta_rejection_rate, 2),
        }
    }

def main():
    print("="*85)
    print("🚀 PIPELINE TRAINING & VALIDASI OUT-OF-FOLD (OOF) PRODUKSI V4.0")
    print("="*85)

    df_btc_raw, df_gold_raw = load_data()
    df_btc_feat = engineer_base_features(df_btc_raw)
    df_gold_feat = engineer_base_features(df_gold_raw)

    report_gold = train_and_validate_symbol("XAUUSD", df_gold_feat)
    report_btc  = train_and_validate_symbol("BTCUSD", df_btc_feat)

    manifest = {
        "system_version": "v4.0_meta_triple_barrier",
        "created_at": pd.Timestamp.now().isoformat(),
        "pt_multiplier": 2.0,
        "sl_multiplier": 1.5,
        "max_holding_bars": 12,
        "meta_gate_threshold": 0.54,
        "anti_leakage_method": "TimeSeriesSplit Out-Of-Fold (OOF) primary probabilities",
        "primary_features": PRIMARY_FEATURES,
        "meta_features": META_FEATURES,
        "correlation_guard": {
            "lookback_bars": 480,
            "normal_threshold": 0.40,
            "max_haircut_threshold": 0.80,
            "max_haircut_pct": 0.50
        },
        "exit_engine": {
            "expectation_flip": True,
            "trailing_profit_lock_trigger_atr": 1.5,
            "trailing_profit_lock_floor_atr": 0.20,
            "time_stop_bars": 12
        },
        "models": {
            "gold": report_gold,
            "btc": report_btc
        }
    }

    manifest_path = MODELS_DIR / "v4_metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    val_report_path = PROJECT_ROOT / "docs" / "OFFLINE_VALIDATION_REPORT.json"
    val_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "gold": report_gold["offline_validation"],
            "btc": report_btc["offline_validation"]
        }, f, indent=2)

    print("\n" + "="*85)
    print("✅ TRAINING & OOF VALIDASI SELESAI DENGAN SUKSES!")
    print(f"   • Metadata Manifest: {manifest_path}")
    print(f"   • Offline Validation Report: {val_report_path}")
    print("="*85)

if __name__ == "__main__":
    main()
