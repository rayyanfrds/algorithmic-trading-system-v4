@echo off
title V4.0 DUAL-ASSET META-TRADING LIVE BOT [XAUUSD + BTCUSD]
color 0A
cd /d "%~dp0"

echo ===============================================================================
echo   ALGORITHMIC TRADING SYSTEM V4 - DUAL-ASSET META-TRADING LIVE SYSTEM
echo   - Model Primer: LightGBM Directional Classifier
echo   - Meta-Model Gate: P(Meta) Threshold 0.54 (Bet Sizing 0.60x - 1.20x)
echo   - Correlation Guard: 480-Bar Dynamic Proportional Haircut
echo   - Exit Engine: Hybrid Decoupled (Expectation Flip + Profit Lock Trailing)
echo   - Audit Logger: logs/hourly_decisions_gold.csv and btc.csv
echo ===============================================================================
echo.

:LOOP
echo [%date% %time%] Memulai Live Forward Test Engine V4.0...
python src/live_forward_test_v4_meta.py

echo.
echo [WARNING] Engine terhenti atau mengalami exception.
echo Melakukan restart otomatis dalam 10 detik... (Tekan Ctrl+C untuk membatalkan)
timeout /t 10 /nobreak
goto LOOP
