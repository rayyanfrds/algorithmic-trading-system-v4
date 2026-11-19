# -*- coding: utf-8 -*-
"""
export_live_data.py
====================
Utility Ekstraksi Data Transaksi MT5 untuk Audit Kuantitatif Out-of-Sample (OOS).
- Mengekstrak seluruh history Deals & Orders dari terminal MT5 Exness ke CSV.
- Menampilkan ringkasan audit cepat (Total Deals, Gross Profit, Win Rate, Total Fee).
- Tidak mematikan atau mengganggu terminal MT5 / Bot Live yang sedang berjalan.
"""

import os
import sys
import datetime
import pathlib
import pandas as pd
import MetaTrader5 as mt5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def namedtuples_to_dataframe(nt_list):
    """Konversi list namedtuples MT5 ke DataFrame dengan mempertahankan nama kolom."""
    if not nt_list or len(nt_list) == 0:
        return pd.DataFrame()
    if hasattr(nt_list[0], "_asdict"):
        return pd.DataFrame([item._asdict() for item in nt_list])
    return pd.DataFrame(list(nt_list))

def main():
    print("=" * 70)
    print("🚀 PRIME QUANT - MT5 LIVE DATA EXPORTER")
    print("=" * 70)

    # 1. Inisialisasi koneksi IPC ke terminal MT5 yang sudah aktif
    if not mt5.initialize():
        print(f"❌ [ERROR] Gagal terhubung ke terminal MT5: {mt5.last_error()}")
        return

    acc_info = mt5.account_info()
    if acc_info is None:
        print("❌ [ERROR] Tidak dapat membaca info akun MT5.")
        mt5.shutdown()
        return

    print(f"🔗 Terhubung ke MT5: Akun #{acc_info.login} | Server: {acc_info.server} | Balance: ${acc_info.balance:,.2f} | Equity: ${acc_info.equity:,.2f}")

    # 2. Tarik histori transaksi deals (sejak awal deploy live)
    from_date = datetime.datetime(2026, 1, 1)
    to_date = datetime.datetime.now() + datetime.timedelta(days=1)

    print(f"⏳ Mengambil data transaksi dari {from_date.strftime('%Y-%m-%d')} s/d {to_date.strftime('%Y-%m-%d')}...")
    deals = mt5.history_deals_get(from_date, to_date)
    orders = mt5.history_orders_get(from_date, to_date)

    # 3. Export Deals History
    if deals and len(deals) > 0:
        df_deals = namedtuples_to_dataframe(deals)
        if "time" in df_deals.columns:
            df_deals["time_dt"] = pd.to_datetime(df_deals["time"], unit="s")
        if "time_msc" in df_deals.columns:
            df_deals["time_msc_dt"] = pd.to_datetime(df_deals["time_msc"], unit="ms")

        deals_file = LOGS_DIR / "mt5_deals_history.csv"
        df_deals.to_csv(deals_file, index=False)
        print(f"✅ Deals Export: {len(df_deals):,} deals tersimpan di: {deals_file.relative_to(PROJECT_ROOT)}")

        # Ringkasan Cepat
        entry_col = "entry" if "entry" in df_deals.columns else None
        profit_col = "profit" if "profit" in df_deals.columns else None

        if entry_col and profit_col:
            # entry: 0=DEAL_ENTRY_IN, 1=DEAL_ENTRY_OUT, 2=DEAL_ENTRY_INOUT, 3=DEAL_ENTRY_OUT_BY
            trade_deals = df_deals[df_deals["entry"] == 1]
            total_profit = float(df_deals["profit"].sum())
            total_commission = float(df_deals["commission"].sum()) if "commission" in df_deals.columns else 0.0
            total_swap = float(df_deals["swap"].sum()) if "swap" in df_deals.columns else 0.0
            net_profit = total_profit + total_commission + total_swap

            print("\n📊 [RINGKASAN AUDIT HISTORI MT5]:")
            print(f"   • Total Closed Trades (Deals Out) : {len(trade_deals):,} transaksi")
            print(f"   • Gross Realized Profit           : ${total_profit:+,.2f}")
            print(f"   • Total Broker Commission         : ${total_commission:,.2f}")
            print(f"   • Total Swap                      : ${total_swap:,.2f}")
            print(f"   • Net Realized PnL (Bersih)       : ${net_profit:+,.2f}")

            if len(trade_deals) > 0:
                wins = trade_deals[trade_deals["profit"] > 0]
                win_rate = (len(wins) / len(trade_deals)) * 100.0
                print(f"   • Realized Win Rate               : {win_rate:.2f}% ({len(wins)} / {len(trade_deals)})")
    else:
        print("⚠️ Tidak ada transaksi Deals ditemukan pada rentang tanggal tersebut.")

    # 4. Export Orders History
    if orders and len(orders) > 0:
        df_orders = namedtuples_to_dataframe(orders)
        if "time_setup" in df_orders.columns:
            df_orders["time_setup_dt"] = pd.to_datetime(df_orders["time_setup"], unit="s")
        if "time_done" in df_orders.columns:
            df_orders["time_done_dt"] = pd.to_datetime(df_orders["time_done"], unit="s")

        orders_file = LOGS_DIR / "mt5_orders_history.csv"
        df_orders.to_csv(orders_file, index=False)
        print(f"✅ Orders Export: {len(df_orders):,} orders tersimpan di: {orders_file.relative_to(PROJECT_ROOT)}")

    # 5. Melepas socket API (MT5 terminal tetap hidup)
    mt5.shutdown()
    print("\n🔒 Socket API berhasil dilepas. Terminal MT5 dan bot trading tetap berjalan normal.")
    print("=" * 70)

if __name__ == "__main__":
    main()
