"""Jalan tiap sore via GitHub Actions: scan watchlist -> cache/ -> commit.

Skeleton dari panduan implementasi (item E). Sesuaikan WATCHLIST dengan
watchlist ril kamu. Juga memanggil update_signal_log_forward_returns() (4c)
supaya forward return signal log terisi otomatis tiap hari tanpa perlu klik
tombol manual di UI — lihat catatan penutup implementasi-review-patch-2.md.
"""
import json
import os

import pandas as pd

from quant_engine import (
    fetch_data, composite_signal, simulate_gbm,
    summarize_paths, run_backtest, classify_liquidity,
    update_signal_log_forward_returns,
)

WATCHLIST = ["BBCA", "TLKM", "UNVR", "ANTM", "SMGR"]  # dst — sesuaikan
os.makedirs("cache", exist_ok=True)

for tkr in WATCHLIST:
    try:
        df = fetch_data("stock_id", tkr, period="2y")
        sig = composite_signal(df)
        mc = summarize_paths(simulate_gbm(df["Close"], n_days=15, n_sims=2000))
        bt = run_backtest(sig, fee_buy_bps=20, fee_sell_bps=30)
        liq = classify_liquidity(df, "stock_id")
        out = {"signal": sig.iloc[-1]["composite_signal"],
               "score": float(sig.iloc[-1]["composite_score"]),
               "mc_prob_profit": mc["prob_profit"],
               "win_rate": bt["metrics"].get("win_rate_pct"),
               "liq_tier": liq["tier"]}
        with open(f"cache/{tkr}.json", "w") as f:
            json.dump(out, f)
        print(f"OK {tkr}")
    except Exception as e:
        print(f"FAIL {tkr}: {e}")

# FIX 4c (lanjutan): isi forward return signal log yang sudah cukup umur,
# supaya live IC di Tab 4 selalu up to date tanpa klik manual tiap hari.
try:
    update_signal_log_forward_returns()
    print("OK signal log forward returns updated")
except Exception as e:
    print(f"FAIL update signal log: {e}")
