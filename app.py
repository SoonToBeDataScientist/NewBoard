"""
app.py
==========================================================================
Quant Dashboard — single deployable file for Streamlit Community Cloud
(or any other host that runs a plain `streamlit run app.py`).

Architecture: this file is ONLY the Streamlit UI. All actual compute
(data fetch, Monte Carlo, signals, ML, backtest, HRP, GA optimizer,
tearsheet, AI narrative) lives in quant_engine.py / advanced_features.py
and is imported below — same modularity as before, just live instead of
precomputed, since there's no more RAM-ceiling reason to split them.

DEPLOY (all via browser, no terminal needed):
  1. Create a GitHub repo, upload these files via the web "Add file ->
     Upload files" button: app.py, quant_engine.py, advanced_features.py,
     requirements.txt.
  2. share.streamlit.io -> New app -> pick that repo -> Deploy.
  3. (Optional) App settings -> Secrets -> add ANTHROPIC_API_KEY if you
     want the AI Narrative button to call the real API instead of the
     template fallback.

GA optimization, the tearsheet, and the AI narrative are all deliberately
behind explicit buttons (not auto-run on every tab open) — Community
Cloud's free tier is ~1GB RAM, and this keeps only one heavy operation
running at a time instead of stacking automatically on every rerun.
==========================================================================
"""

import os
import json
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from quant_engine import *          # noqa: F401,F403 — see module for full list
from quant_engine import (
    _is_rate_limit_error, _yf_call_with_backoff,
    _BLUECHIP_PRESET_IDX, _FALLBACK_IDX, _FALLBACK_US,
    _FALLBACK_CRYPTO_IDR, _FALLBACK_CRYPTO_USDT,
)  # underscore-prefixed names are skipped by `import *`, need explicit import
import advanced_features as adv

# ==========================================================================
# ==== TRADE JOURNAL (I/O ringan — CSV lokal atau Google Sheets kalau ====
# ==== connections.gsheets diisi di Secrets. Sengaja bukan bagian dari ====
# ==== quant_engine.py: ini state yang ditulis manusia, bukan hasil hitung. ====
# ==========================================================================

JOURNAL_FILE = "trade_journal.csv"

# ==========================================================================
# ==== Preset ticker yang SENGAJA diselang-seling antar sektor ====
# ==========================================================================
# Dipakai buat tombol "Isi otomatis" di Aggregate Backtest — ambil N pertama
# dari list ini (bukan dari _BLUECHIP_PRESET_IDX/_FALLBACK_US mentah, yang
# urutannya cenderung numpuk 1 sektor duluan, mis. bank semua di depan)
# supaya prefix berapa pun panjangnya tetap tersebar lintas sektor, bukan
# "10 saham yang gerak bareng doang" (poin yang dibahas soal independensi
# sample untuk bootstrap).
_SECTOR_DIVERSE_IDX = [
    "BBCA", "TLKM", "UNVR", "ANTM", "SMGR", "KLBF", "CPIN", "ASII", "GOTO", "ACES",
    "BBRI", "EXCL", "ICBP", "ADRO", "INTP", "SIDO", "JPFA", "UNTR", "BUKA",
    "BMRI", "ISAT", "INDF", "PTBA", "AKRA",
    "BBNI", "TOWR", "AMRT", "ITMG",
    "BRIS", "TBIG", "MAPI", "PGAS", "MEDC", "EMTK", "MDKA",
]
_SECTOR_DIVERSE_US = [
    "AAPL", "JPM", "JNJ", "XOM", "PG", "CAT", "NFLX", "TSLA",
    "MSFT", "BAC", "UNH", "CVX", "WMT", "BA", "DIS", "GM",
    "GOOGL", "GS", "PFE", "KO", "HON",
    "NVDA", "MA", "ABBV", "MCD",
    "META", "V",
]

JOURNAL_WORKSHEET = "trade_journal"
JOURNAL_COLUMNS = [
    "entry_timestamp", "symbol", "asset_type", "action_type",
    "entry_price", "quantity", "est_value", "composite_score_at_entry",
    "exit_timestamp", "exit_price", "return_pct", "status", "notes",
]


def _journal_backend() -> str:
    try:
        if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
            return "gsheets"
    except Exception:
        pass
    return "csv"


@st.cache_resource(show_spinner=False)
def _gsheets_conn():
    from streamlit_gsheets import GSheetsConnection
    return st.connection("gsheets", type=GSheetsConnection)


def _normalize_journal_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[JOURNAL_COLUMNS]


def load_journal() -> pd.DataFrame:
    if _journal_backend() == "gsheets":
        try:
            df = _gsheets_conn().read(worksheet=JOURNAL_WORKSHEET, ttl=0)
            return _normalize_journal_df(df.dropna(how="all"))
        except Exception as e:
            st.warning(f"⚠️ Gagal baca journal dari Google Sheets ({e}) — fallback ke CSV lokal.")
    if os.path.exists(JOURNAL_FILE):
        try:
            df = pd.read_csv(JOURNAL_FILE, parse_dates=["entry_timestamp", "exit_timestamp"])
            return _normalize_journal_df(df)
        except Exception:
            pass
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def _persist_journal(df: pd.DataFrame):
    if _journal_backend() == "gsheets":
        try:
            _gsheets_conn().update(worksheet=JOURNAL_WORKSHEET, data=df)
            return
        except Exception as e:
            st.error(f"🚨 Gagal tulis ke Google Sheets ({e}) — tetap menulis ke CSV lokal "
                     f"sebagai jaring pengaman (TIDAK persist kalau di-hosting & container "
                     f"sleep/restart — sambungkan Google Sheets di Secrets untuk itu).")
    df.to_csv(JOURNAL_FILE, index=False)


def append_journal_entry(entry: dict):
    df = load_journal()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    _persist_journal(df)


def close_journal_entry(row_index: int, exit_price: float, exit_timestamp):
    df = load_journal()
    entry_price = df.loc[row_index, "entry_price"]
    action = df.loc[row_index, "action_type"]
    return_pct = ((exit_price / entry_price - 1) if action == "BUY"
                  else (entry_price / exit_price - 1)) * 100
    df.loc[row_index, ["exit_price", "exit_timestamp", "return_pct", "status"]] = \
        [exit_price, exit_timestamp, return_pct, "CLOSED"]
    _persist_journal(df)


# ==========================================================================
# ==== BACKTEST LOG — bandingin win rate & akurasi antar run ====
# ==========================================================================
# Beda dari Trade Journal (posisi yang beneran dieksekusi manusia): ini
# nyatet HASIL BACKTEST tiap kali user klik "Simpan run ini" — parameter
# yang dipakai + metrik yang keluar — biar bisa dibandingin lintas ticker/
# parameter dari waktu ke waktu, bukan cuma lihat satu run lalu lupa.

BACKTEST_LOG_FILE = "backtest_log.csv"
BACKTEST_LOG_WORKSHEET = "backtest_log"
BACKTEST_LOG_COLUMNS = [
    "logged_at", "symbol", "asset_type",
    "mr_window", "mr_z_entry", "mom_fast", "mom_slow", "mr_weight",
    "stochbb_weight", "position_size_pct", "stop_loss_pct", "max_holding_days",
    "execution_price", "dynamic_slippage", "use_ml", "ml_model_type",
    "n_trades", "win_rate_pct", "total_return_pct", "sharpe_ratio",
    "max_drawdown_pct", "buy_hold_return_pct", "effective_slippage_bps",
    "is_concentrated", "dominant_trade_pct", "ml_accuracy_pct", "ml_bss",
]


def _normalize_backtest_log_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BACKTEST_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[BACKTEST_LOG_COLUMNS]


def load_backtest_log() -> pd.DataFrame:
    if _journal_backend() == "gsheets":
        try:
            df = _gsheets_conn().read(worksheet=BACKTEST_LOG_WORKSHEET, ttl=0)
            return _normalize_backtest_log_df(df.dropna(how="all"))
        except Exception as e:
            st.warning(f"⚠️ Gagal baca log backtest dari Google Sheets ({e}) — fallback ke CSV lokal.")
    if os.path.exists(BACKTEST_LOG_FILE):
        try:
            df = pd.read_csv(BACKTEST_LOG_FILE, parse_dates=["logged_at"])
            return _normalize_backtest_log_df(df)
        except Exception:
            pass
    return pd.DataFrame(columns=BACKTEST_LOG_COLUMNS)


def append_backtest_log_entry(entry: dict):
    df = load_backtest_log()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    if _journal_backend() == "gsheets":
        try:
            _gsheets_conn().update(worksheet=BACKTEST_LOG_WORKSHEET, data=df)
            df.to_csv(BACKTEST_LOG_FILE, index=False)  # local safety-net copy too
            return
        except Exception as e:
            st.error(f"🚨 Gagal tulis log backtest ke Google Sheets ({e}) — tetap menulis ke CSV lokal.")
    df.to_csv(BACKTEST_LOG_FILE, index=False)


def delete_backtest_log_entries(row_indices: list[int]):
    df = load_backtest_log()
    df = df.drop(index=row_indices).reset_index(drop=True)
    if _journal_backend() == "gsheets":
        try:
            _gsheets_conn().update(worksheet=BACKTEST_LOG_WORKSHEET, data=df)
        except Exception as e:
            st.error(f"🚨 Gagal update Google Sheets ({e}) — tetap dihapus dari CSV lokal.")
    df.to_csv(BACKTEST_LOG_FILE, index=False)


# Catatan: fungsi signal log (load_signal_log, log_signal_snapshot,
# update_signal_log_forward_returns) dipindah ke quant_engine.py (bukan di sini)
# supaya bisa dipanggil headless dari precompute.py tanpa menjalankan seluruh
# UI Streamlit di app.py ini. Sudah otomatis ikut ke-import lewat
# 'from quant_engine import *' di atas.

# ==== SECTION 5: STREAMLIT DASHBOARD ====
# ==========================================================================

st.set_page_config(page_title="Quant Dashboard", layout="wide")

# --------------------------------------------------------------------------
# ---- Bridge Streamlit Cloud Secrets -> os.environ ----
# --------------------------------------------------------------------------
# quant_engine.py / advanced_features.py are deliberately Streamlit-agnostic
# (no `import streamlit` in either — see their module docstrings) and read
# API keys via os.environ.get(...), the normal/portable way. But Streamlit
# Cloud's "Secrets" UI (and local .streamlit/secrets.toml) populates
# st.secrets, NOT os.environ, automatically — a common gotcha (symptom: key
# is saved, app runs fine, but whatever reads os.environ silently gets None
# and falls back to a template/default, with no visible error). This copies
# each key this app actually uses over to os.environ once at startup, ONLY
# if it isn't already set there (so a real OS-level env var still works
# exactly the same, st.secrets is just the Cloud/local-file-based source).
#
# IMPORTANT: unlike the first version of this bridge, failures are now
# surfaced in the sidebar instead of silently swallowed — a malformed
# secrets.toml (bad TOML syntax, wrong location, wrong extension hidden by
# Windows, etc.) used to fail this whole block silently via a bare
# `except: pass`, so nothing ever reached os.environ and there was zero
# indication anything had gone wrong — exactly the "I made the .toml file
# but it's still not being called" symptom this replaces.
_secrets_status = {}
_secrets_load_error = None
try:
    _ = dict(st.secrets)  # forces st.secrets to actually parse secrets.toml now,
                           # so a syntax error surfaces here, not silently later
except Exception as e:
    _secrets_load_error = str(e)

for _key in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "TWELVEDATA_API_KEY",
             "MARKETAUX_API_KEY"):
    if _key in os.environ:
        _secrets_status[_key] = "OS env var"
    elif _secrets_load_error is None:
        try:
            if _key in st.secrets:
                os.environ[_key] = st.secrets[_key]
                _secrets_status[_key] = "secrets.toml/Secrets UI"
            else:
                _secrets_status[_key] = None
        except Exception:
            _secrets_status[_key] = None
    else:
        _secrets_status[_key] = None

with st.sidebar.expander("🔑 Status API Keys", expanded=False):
    if _secrets_load_error:
        st.error(
            f"⚠️ Gagal baca secrets.toml sama sekali: {_secrets_load_error}\n\n"
            f"Cek: (1) file ada persis di `.streamlit/secrets.toml` di folder yang sama "
            f"tempat `streamlit run app.py` dijalankan, (2) ekstensi beneran `.toml` bukan "
            f"`.toml.txt` (Windows suka nyembunyiin ekstensi — aktifkan 'File name extensions' "
            f"di File Explorer buat cek), (3) sintaks TOML-nya benar (`KEY = \"value\"`, pakai "
            f"tanda kutip)."
        )
    else:
        for _key, _source in _secrets_status.items():
            if _source:
                st.success(f"✅ {_key}: ke-load dari {_source}")
            else:
                st.caption(f"⬜ {_key}: tidak diset (opsional — fitur terkait fallback otomatis)")



# --------------------------------------------------------------------------
# ---- MODE COMPUTE: Cloud (hemat resource) vs Lokal (performa penuh) ----
# --------------------------------------------------------------------------
# quant_engine.py & advanced_features.py sama persis di kedua mode — yang
# beda cuma angka default di sini (jumlah simulasi, ukuran scan, populasi
# GA, dst). Auto-terdeteksi lewat path mount khas Streamlit Community Cloud
# ("/mount/src/..."), tapi tetap bisa di-override manual — auto-detect
# bukan hal yang didokumentasikan resmi, jadi jangan 100% diandalkan kalau
# suatu saat host-nya beda lagi.
_looks_like_cloud = os.path.exists("/mount/src") or "STREAMLIT_RUNTIME_ENV" in os.environ

with st.sidebar.expander("💻 Mode Compute", expanded=False):
    compute_mode = st.radio(
        "Jalan di mana?", ["☁️ Cloud (hemat resource)", "💻 Lokal (performa penuh)"],
        index=0 if _looks_like_cloud else 1,
        help="Cloud: batasan default lebih ketat (jumlah simulasi, ukuran scan, populasi GA) "
             "supaya aman di RAM terbatas (Streamlit Community Cloud ~1GB) dan nggak nge-block "
             "user lain kalau ada yang share host yang sama. Lokal: kalau kamu jalanin "
             "`streamlit run app.py` di komputer sendiri, batasan itu boleh lebih longgar "
             "karena RAM/CPU-nya punya kamu sendirian. Auto-terdeteksi, tapi silakan ubah "
             "manual kalau salah tebak."
    )
    st.caption(f"Terdeteksi otomatis: {'Cloud' if _looks_like_cloud else 'Lokal'}")
IS_LOCAL = compute_mode.startswith("💻")

st.sidebar.title("⚙️ Konfigurasi")

asset_type_label = st.sidebar.selectbox(
    "Jenis aset", ["Saham Indonesia (IDX)", "Saham US", "Crypto"]
)
asset_type_map = {"Saham Indonesia (IDX)": "stock_id", "Saham US": "stock_us", "Crypto": "crypto"}
asset_type = asset_type_map[asset_type_label]
trading_days = 365.0 if asset_type == "crypto" else 252.0  # FIX 1e

if asset_type == "stock_id":
    symbol = st.sidebar.text_input("Kode saham (tanpa .JK)", value="BBCA")
    period = st.sidebar.selectbox("Rentang histori", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2,
                                   help="Histori lebih panjang = lebih banyak kesempatan sinyal muncul = "
                                        "lebih banyak trade di backtest, tapi juga lebih mungkin nyampur "
                                        "beberapa rezim pasar berbeda (bull/bear/sideways) jadi satu angka.")
    fetch_kwargs = {"period": period}
elif asset_type == "stock_us":
    symbol = st.sidebar.text_input("Ticker", value="AAPL")
    period = st.sidebar.selectbox("Rentang histori", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2,
                                   help="Histori lebih panjang = lebih banyak kesempatan sinyal muncul = "
                                        "lebih banyak trade di backtest, tapi juga lebih mungkin nyampur "
                                        "beberapa rezim pasar berbeda (bull/bear/sideways) jadi satu angka.")
    fetch_kwargs = {"period": period}
else:
    default_symbol = "BTC/IDR"
    symbol = st.sidebar.text_input("Pair (format EXCHANGE)", value=default_symbol)
    exchange_id = st.sidebar.selectbox(
        "Exchange",
        ["indodax", "binance", "kraken", "coinbase"],
        index=0,
        help="Binance diblokir Kominfo di jaringan ISP Indonesia (redirect ke "
             "Internet Positif). Indodax adalah exchange lokal berizin Bappebti "
             "dan tidak diblokir — jadi ini default. Kraken/Coinbase juga biasanya "
             "aman kalau butuh pair yang tidak ada di Indodax."
    )
    if exchange_id == "indodax":
        st.sidebar.caption(
            "💡 Indodax umumnya quote dalam **IDR**, bukan USDT — pakai format "
            "seperti `BTC/IDR`, `ETH/IDR`. Kalau ganti dari Binance (biasanya "
            "`BTC/USDT`) ke Indodax, ubah juga suffix pair-nya."
        )
    if exchange_id == "binance":
        st.sidebar.warning(
            "⚠️ Binance sering diblokir di jaringan ISP Indonesia (redirect ke "
            "Internet Positif). Kalau fetch data gagal, coba ganti ke Indodax, "
            "Kraken, atau Coinbase — atau pakai VPN/DNS alternatif (mis. "
            "Cloudflare 1.1.1.1) kalau memang butuh Binance spesifik."
        )
    limit = st.sidebar.slider("Jumlah candle harian", 100, 1000, 500)
    fetch_kwargs = {"exchange_id": exchange_id, "timeframe": "1d", "limit": limit}

st.sidebar.markdown("---")
st.sidebar.subheader("Sinyal")

# ---- Trading-style presets ----
# Each preset tunes window/threshold parameters to match a holding-period
# style. These are starting points based on general swing/position-trading
# heuristics (shorter windows = more reactive/noisy, longer = smoother/slower),
# not a guarantee of performance — always sanity-check via the Backtest tab.
TRADING_PRESETS = {
    "Swing (hold 1-5 hari)": {
        "mr_window": 20, "mr_z_entry": 1.5, "mom_fast": 10, "mom_slow": 30,
        "mr_weight": 0.5, "mc_days": 15, "fold_test_days": 60,
        "desc": "Reaktif ke pergerakan jangka pendek. Cocok kalau kamu cek "
                "dashboard tiap hari/beberapa hari sekali.",
    },
    "Posisi/Menengah (hold 1-4 minggu)": {
        "mr_window": 40, "mr_z_entry": 1.75, "mom_fast": 20, "mom_slow": 60,
        "mr_weight": 0.4, "mc_days": 45, "fold_test_days": 120,
        "desc": "Window lebih lebar, lebih sedikit sinyal tapi lebih 'yakin'. "
                "Cocok kalau kamu cek dashboard mingguan.",
    },
    "Long-term/Investasi (hold bulanan+)": {
        "mr_window": 90, "mr_z_entry": 2.0, "mom_fast": 50, "mom_slow": 150,
        "mr_weight": 0.3, "mc_days": 90, "fold_test_days": 252,
        "desc": "Fokus tren jangka panjang, momentum lebih dominan dari "
                "mean-reversion. Sinyal jarang muncul tapi horizon-nya panjang.",
    },
    "Custom (atur manual)": None,
}

PRESET_KEYS = ["mr_window", "mr_z_entry", "mom_fast", "mom_slow", "mr_weight",
               "mc_days", "fold_test_days"]
PRESET_DEFAULTS = TRADING_PRESETS["Swing (hold 1-5 hari)"]  # fallback defaults

for _k in PRESET_KEYS:
    st.session_state.setdefault(_k, PRESET_DEFAULTS[_k])


def _apply_trading_preset():
    choice = st.session_state["trading_style_select"]
    preset = TRADING_PRESETS[choice]
    if preset is not None:
        for k in PRESET_KEYS:
            st.session_state[k] = preset[k]


trading_style = st.sidebar.selectbox(
    "Tipe trading", list(TRADING_PRESETS.keys()),
    key="trading_style_select", on_change=_apply_trading_preset,
    help="Pilih tipe trading buat auto-set parameter window/threshold yang "
         "sesuai. Pilih 'Custom' kalau mau atur manual semua slider di bawah."
)
if TRADING_PRESETS[trading_style] is not None:
    st.sidebar.caption(f"💡 {TRADING_PRESETS[trading_style]['desc']}")

mr_window = st.sidebar.slider("Mean reversion window (hari)", 5, 120, key="mr_window")
mr_z_entry = st.sidebar.slider("Z-score entry threshold", 0.5, 3.0, step=0.1, key="mr_z_entry")
mom_fast = st.sidebar.slider("MA cepat", 3, 60, key="mom_fast")
mom_slow = st.sidebar.slider("MA lambat", 10, 200, key="mom_slow")
mr_weight = st.sidebar.slider("Bobot mean-reversion vs momentum", 0.0, 1.0, step=0.05, key="mr_weight")

st.sidebar.markdown("**Stochastic + Bollinger Band**")
stochbb_weight = st.sidebar.slider(
    "Bobot Stochastic-BB", 0.0, 1.0, 0.0, 0.05,
    help="0 = mati total (perilaku lama, default). Sinyal ini cuma nembak BUY/SELL kalau "
         "Stochastic oversold/overbought DAN harga juga di dekat pinggir Bollinger Band "
         "secara bersamaan — kombinasi klasik, bukan salah satu indikator sendirian."
)
with st.sidebar.expander("Parameter Stochastic-BB (lanjutan)", expanded=False):
    bb_window = st.slider("Bollinger window (hari)", 10, 60, 20, key="bb_window")
    bb_std = st.slider("Bollinger std-dev", 1.0, 3.0, 2.0, 0.1, key="bb_std")
    stoch_window = st.slider("Stochastic %K window (hari)", 5, 30, 14, key="stoch_window")
    stoch_smooth = st.slider("Stochastic %D smoothing", 1, 10, 3, key="stoch_smooth")

st.sidebar.markdown("**Earnings Drift (PEAD)**")
earnings_weight = st.sidebar.slider(
    "Bobot Earnings Drift", 0.0, 1.0, 0.0, 0.05, key="earnings_weight",
    help="0 = mati total (default). Beda dari sinyal lain — ini BUKAN dari harga/volume, "
         "tapi dari kejutan earnings aktual (EPS aktual vs estimasi konsensus). Efek "
         "'drift' pasca-earnings itu anomali pasar yang didokumentasikan puluhan tahun, "
         "beda dari pola teknikal generik yang udah lama di-arbitrase habis. Cakupan data "
         "earnings yfinance jauh lebih bagus buat saham US dibanding IDX — kalau data "
         "nggak tersedia untuk simbol ini, bobot ini otomatis nggak berefek (bukan error)."
)
earnings_drift_window = st.sidebar.slider(
    "Drift window (hari)", 20, 120, 60, 5, key="earnings_drift_window",
    help="Berapa lama efek earnings surprise dianggap masih relevan — makin jauh dari "
         "tanggal earnings, efeknya makin di-decay linear menuju 0."
) if earnings_weight > 0 else 60

use_regime_gate = st.sidebar.checkbox(
    "🌍 Market regime gate (benchmark > MA200)", value=True,
    help="FIX 4a: sinyal BUY di-mute saat benchmark (IHSG/^GSPC/BTC) di bawah "
         "MA-200. Satu filter yang secara historis paling konsisten memperbaiki "
         "strategi mean-reversion/momentum ritel long-only.")

st.sidebar.markdown("---")
st.sidebar.subheader("Sinyal ML")
use_ml = st.sidebar.checkbox("Aktifkan sinyal ML", value=True,
                              help="Sinyal ketiga berbasis machine learning, memprediksi "
                                   "probabilitas harga naik dari fitur teknikal (return, "
                                   "RSI, MACD, Bollinger, ATR, Stochastic, volume, dll). "
                                   "Walk-forward (anti-lookahead), tapi nambah waktu "
                                   "komputasi beberapa detik.")
if use_ml:
    ml_model_label = st.sidebar.selectbox(
        "Model ML", ["LightGBM", "XGBoost", "Ensemble (LightGBM + XGBoost)"],
        help="LightGBM/XGBoost: satu model, cepat. Ensemble: latih DUA model dan rata-ratakan "
             "prediksinya — bisa lebih stabil (nggak kebawa quirk satu model doang), tapi "
             "waktu training kira-kira 2x lipat karena beneran fit dua model tiap retrain."
    )
    ml_model_type = {"LightGBM": "lightgbm", "XGBoost": "xgboost",
                      "Ensemble (LightGBM + XGBoost)": "ensemble"}[ml_model_label]
    ml_weight = st.sidebar.slider("Bobot sinyal ML dalam composite (maksimum)", 0.0, 1.0, 0.3, 0.05,
                                   help="Ini adalah PLAFON/batas atas. Kalau 'Adaptive ML weight' "
                                        "di bawah aktif, bobot aktual yang benar-benar dipakai di "
                                        "composite score bisa lebih kecil dari ini — didiskon "
                                        "otomatis berdasarkan seberapa bagus kalibrasi model "
                                        "(reliability curve) di histori out-of-sample-nya sendiri.")
    adaptive_ml_weight = st.sidebar.checkbox(
        "⚖️ Adaptive ML weight (berdasarkan reliability)", value=True,
        help="Kalau aktif: bobot ML di atas didiskon otomatis pakai Brier Skill Score dari "
             "reliability curve out-of-sample (lihat expander 'Error Analysis' di tab Signal). "
             "Model yang kalibrasinya bagus (BSS mendekati 1) dapat bobot penuh; model yang "
             "kalibrasinya jelek/tidak lebih baik dari nebak base rate (BSS <= 0) otomatis "
             "didiskon sampai ke 0 — bukan cuma dipajang sebagai info, tapi beneran mengubah "
             "composite score & verdict akhir. Nonaktifkan untuk pakai nilai slider apa adanya."
    )
    ml_horizon_days = st.sidebar.slider(
        "Horizon prediksi (hari)", 1, 10, 1,
        help="Model memprediksi 'apakah harga naik dalam N hari ke depan', bukan cuma besok. "
             "Samain kira-kira dengan holding period gaya trading kamu — misal preset Swing "
             "(1-5 hari) cocok pakai horizon 2-3, bukan cuma 1 hari."
    )
    ml_min_train = st.sidebar.slider("Minimal hari training", 60, 300, 100, 10)
    ml_retrain_every = st.sidebar.slider("Retrain setiap N hari", 5, 60, 20, 5)
    st.sidebar.caption(
        "Model di-retrain periodik pakai data masa lalu saja (expanding window) — "
        "prediksi hari T selalu dari model yang cuma tau data sebelum hari T."
    )
else:
    ml_weight, ml_min_train, ml_retrain_every = 0.0, 100, 20
    ml_model_type, ml_horizon_days = "lightgbm", 1
    ml_model_label = "LightGBM"
    adaptive_ml_weight = True

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo")
mc_days = st.sidebar.slider("Horizon simulasi (hari)", 5, 120, key="mc_days")
mc_sims = st.sidebar.slider("Jumlah simulasi", 1000, 200000 if IS_LOCAL else 50000,
                             20000 if IS_LOCAL else 5000, 1000)
st.sidebar.caption(
    "GBM tetap cepat sampai 50.000+ simulasi (vectorized numpy). GARCH punya "
    "fixed-cost fitting ~2-3 detik per run (independen dari jumlah simulasi), "
    "jadi jumlah simulasi besar juga tidak terlalu menambah waktu tunggu. "
    "Catatan statistik: presisi Monte Carlo naik sebanding 1/√n — dari 2.000 "
    "ke 50.000 simulasi (25x lipat) cuma memperkecil error estimasi sekitar "
    "5x, jadi di atas ~10.000 manfaat tambahannya mulai landai (diminishing "
    "returns), tapi tetap murah untuk dijalankan."
)
mc_model = st.sidebar.radio(
    "Model volatilitas",
    ["GBM (konstan)", "GARCH(1,1) (dinamis)", "GJR-GARCH (asimetris/leverage)"],
    help="GJR-GARCH sama seperti GARCH(1,1) tapi volatilitasnya boleh naik "
         "lebih tinggi setelah hari MERAH dibanding hari HIJAU dengan besaran "
         "sama (leverage effect) — biasanya lebih realistis untuk saham dan "
         "terutama crypto, yang cenderung 'panik' lebih keras waktu turun."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")
fee_buy_bps = st.sidebar.number_input(
    "Fee beli (bps)", value=20,
    help="FIX 2a: realita ritel IDX ~0.15-0.25% sisi beli. Backtest yang cuma "
         "profit di fee 10 bps tapi rugi di 20/30 bps = tidak punya edge ril.")
fee_sell_bps = st.sidebar.number_input(
    "Fee jual (bps)", value=30,
    help="Sisi jual lebih mahal: komisi ~0.25-0.35% SUDAH termasuk pajak final "
         "0.1% yang hanya dikenakan saat jual — makanya asimetris.")
fee_bps = fee_buy_bps  # kompatibilitas untuk caller lama yang cuma terima 1 angka
slippage_bps = st.sidebar.number_input("Slippage (bps)", value=5)
dynamic_slippage = st.sidebar.checkbox(
    "Slippage dinamis dari likuiditas ticker", value=False,
    help="Kalau nyala, angka Slippage (bps) di atas DIABAIKAN dan diganti estimasi "
         "per-ticker dari spread OHLC (Corwin-Schultz/Roll/Edge — sama yang dipakai "
         "buat klasifikasi likuiditas di tab Kesimpulan). Saham tipis otomatis dapat "
         "slippage lebih realistis dibanding saham blue-chip, bukan angka flat yang sama."
)
execution_price = st.sidebar.radio(
    "Harga eksekusi", ["Open hari berikutnya (realistis)", "Close hari berikutnya (optimis)"],
    index=0,
    help="Sinyal dihitung dari Close hari T. 'Open hari berikutnya' = eksekusi paling "
         "cepat yang realistis didapat (begitu market buka besoknya). 'Close hari "
         "berikutnya' itu skenario terbaik yang nggak realistis (seolah bisa beli/jual "
         "tepat di harga penutupan besok) — cuma buat perbandingan, bukan angka yang "
         "harus dipercaya."
)
execution_price_kw = "next_open" if execution_price.startswith("Open") else "next_close"

st.sidebar.markdown("**Position Sizing & Risk Management**")
position_size_pct = st.sidebar.slider(
    "% modal per trade", 10, 100, 100, 10,
    help="Default 100% = all-in tiap sinyal BUY (perilaku lama). Turunkan kalau mau "
         "simulasi 'cuma taruh sebagian modal per posisi', gaya manajemen risiko yang "
         "lebih realistis dibanding all-in setiap saat. Ini jadi BASE size kalau "
         "confidence-scaled sizing di bawah diaktifkan — bukan diganti, tapi diskalakan."
) / 100.0
use_confidence_sizing = st.sidebar.checkbox(
    "Skalakan size pakai conviction + ML confidence", value=False,
    help="Opt-in, OFF = perilaku lama (position_size_pct flat tiap trade). Kalau ON: "
         "size per trade diskalakan naik/turun dari base di atas berdasarkan (1) seberapa "
         "jauh composite score dari ambang 0.5 saat itu, (2) Brier Skill Score kalibrasi "
         "ML out-of-sample (sinyal dari model yang belum terbukti skill-nya otomatis "
         "didiskon), dan opsional (3) volatility targeting biar risiko sebanding lintas "
         "ticker. Tidak pernah mengubah arah BUY/SELL, cuma besaran size-nya."
)
use_vol_targeting = (
    st.sidebar.checkbox("+ Volatility targeting", value=False,
                          help="Butuh confidence-scaled sizing aktif. Size makin kecil "
                               "untuk saham dengan volatilitas realized tinggi (mis. "
                               "small-cap IDX yang bisa gerak >5%/hari), makin besar untuk "
                               "yang tenang — supaya risiko per trade sebanding.")
    if use_confidence_sizing else False
)
vol_target_annual_pct = (
    st.sidebar.slider("Target volatilitas tahunan (%)", 10, 80, 25, 5,
                       help="Semacam 'risiko wajar' portofolio kamu. Saham dengan "
                            "volatilitas realized di atas ini di-downsize, di bawah ini "
                            "di-upsize (dalam batas conviction/confidence multiplier di atas).") / 100.0
    if use_vol_targeting else None
)
use_stop_loss = st.sidebar.checkbox("Aktifkan stop-loss", value=False)
stop_loss_pct = (st.sidebar.slider("Stop-loss (%)", 1, 20, 5, 1,
                                    help="Keluar paksa kalau harga Low hari itu breach "
                                         "level ini dari harga entry — independen dari "
                                         "sinyal, sama seperti stop order beneran.") / 100.0
                  if use_stop_loss else None)
use_max_holding = st.sidebar.checkbox("Batasi lama holding", value=False)
max_holding_days = (st.sidebar.slider("Maks hari holding", 1, 60, 15, 1,
                                       help="Keluar paksa setelah N hari kalau sinyal SELL "
                                            "nggak kunjung muncul — hindari posisi 'nyangkut' "
                                            "nunggu sinyal yang mungkin lambat berubah.")
                     if use_max_holding else None)
use_turnover_cap = st.sidebar.checkbox(
    "Batasi size berdasarkan turnover harian", value=(asset_type == "stock_id"),
    help="FIX 2c: nilai posisi di-cap ke sekian % dari rata-rata turnover harian "
         "ticker — mencegah backtest 'menghasilkan' return yang secara fisik "
         "tidak bisa dieksekusi di saham tipis.")
max_turnover_participation = (
    st.sidebar.slider("Maks % turnover harian per posisi", 1, 10, 2,
                       help="2% turnover harian adalah batas konservatif yang umum "
                            "dipakai supaya order kamu tidak menggerakkan harga sendiri.") / 100.0
    if use_turnover_cap else None)
idx_realism = st.sidebar.checkbox(
    "Mode realisme IDX (lot 100, tick size, ARB lock)", value=(asset_type == "stock_id"),
    help="2b: bulatkan size ke kelipatan 1 lot (100 lembar), slippage minimum 1 tick "
         "(saham <Rp200: 1 tick > 0.5%!), dan exit/stop-loss GAGAL fill saat hari "
         "terkunci ARB (tidak ada bid — posisi terbawa ke hari berikutnya). Tier "
         "ARA/ARB & tick mengikuti aturan BEI — verifikasi berkala karena bisa direvisi.")

st.sidebar.markdown("**Walk-Forward Validation**")
wf_method = st.sidebar.radio(
    "Metode pembagian fold",
    ["Jumlah fold tetap (4 fold)", "Panjang fold custom (hari)"],
    help="'Jumlah fold tetap' membagi data jadi 4 bagian sama rata. "
         "'Panjang fold custom' pakai jendela uji dengan panjang tetap (hari) "
         "yang bergeser maju — jumlah fold-nya otomatis menyesuaikan panjang "
         "data. Ini yang biasa dipakai di walk-forward validation quant "
         "beneran, karena panjang periode uji itu yang punya makna praktis "
         "(mis. 'uji per 60 hari trading'), bukan jumlah fold-nya."
)
if wf_method == "Panjang fold custom (hari)":
    fold_test_days = st.sidebar.slider("Panjang tiap fold uji (hari)", 10, 252,
                                        key="fold_test_days")
    fold_min_train = st.sidebar.slider("Minimal hari training sebelum fold pertama",
                                        30, 300, 100, 10)
    fold_n_folds = 4  # tidak dipakai di mode ini (jumlah fold otomatis menyesuaikan panjang data)
else:
    fold_test_days, fold_min_train = None, 100
    fold_n_folds = st.sidebar.slider(
        "Jumlah fold", 2, 12, 4, 1,
        help="Lebih banyak fold = validasi lebih ketat (tiap fold lebih kecil/lebih banyak "
             "titik uji out-of-sample), tapi fold yang kekecilan juga lebih rentan window "
             "rolling (Mean reversion window/MA lambat) belum sempat 'terisi' di awal fold."
    )

run_btn = st.sidebar.button("🚀 Jalankan Analisis", use_container_width=True)

# st.button only returns True for the single rerun it was clicked on — any
# other widget interaction anywhere in the app (e.g. changing the Screener's
# "Rentang histori" dropdown) triggers a fresh rerun where run_btn is False
# again. Persist the "has been run at least once" state so the analysis view
# doesn't collapse back to the placeholder screen on unrelated interactions.
if run_btn:
    st.session_state["analysis_ran"] = True

st.title("📊 Quant Dashboard — Monte Carlo + Signal + Backtest")
st.caption("Untuk swing trading (hold 1-2 hari atau lebih selama sinyal masih green), "
           "bukan day trading intraday. Semua logic ada dalam satu file ini.")

if not st.session_state.get("analysis_ran", False):
    st.info("Atur parameter di sidebar lalu klik **Jalankan Analisis**.")
    st.stop()

with st.spinner(f"Mengambil data untuk {symbol}..."):
    try:
        price_df = fetch_data(asset_type, symbol, **fetch_kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if _is_rate_limit_error(err_str):
            st.error(
                "🚫 **Masih kena rate limit dari Yahoo Finance** setelah otomatis dicoba "
                "ulang beberapa kali dengan jeda (5s, 10s, 20s, 40s) — berarti limitnya "
                "cukup ketat saat ini, bukan cuma kebetulan satu request nyangkut. Ini "
                "biasanya terjadi kalau kamu baru aja pakai fitur yang banyak nembak "
                "yfinance (Screener dengan banyak simbol, Live Gainers dengan "
                "auto-refresh, atau Fundamental)."
            )
            st.info(
                "**Solusinya:** tunggu beberapa menit sampai ~1 jam, biasanya reset otomatis. "
                "Sementara itu, hindari scan Screener dengan Universe besar atau nyalain "
                "Live Gainers auto-refresh sampai limitnya reset."
            )
        else:
            st.error(f"Gagal mengambil data: {e}")
        st.stop()

n_rows = len(price_df)

if n_rows < 15:
    st.error(
        f"Data cuma {n_rows} baris — kemungkinan besar ini saham yang baru IPO "
        f"(belum lama listing di bursa) atau tickernya belum ter-index oleh Yahoo "
        f"Finance. Analisis statistik (Monte Carlo, sinyal, backtest) tidak bisa "
        f"dijalankan secara bermakna dengan data sesedikit ini — semua model di "
        f"sini butuh histori harga untuk mengukur volatilitas dan tren."
    )
    st.stop()

# Adaptive mode: with limited history (typical for recently-IPO'd or thinly-covered
# tickers), automatically shrink rolling windows so the signal/backtest still run,
# but warn clearly that results are low-confidence.
LOW_DATA_THRESHOLD = 60
adaptive_mode = n_rows < LOW_DATA_THRESHOLD

if adaptive_mode:
    st.warning(
        f"⚠️ Data cuma **{n_rows} baris** ({price_df.index[0].date()} s/d "
        f"{price_df.index[-1].date()}) — kemungkinan saham ini baru IPO atau "
        f"tergolong niche/thin-coverage di Yahoo Finance. Parameter window "
        f"(mean-reversion, MA, Monte Carlo) otomatis diperkecil supaya tetap bisa "
        f"jalan, tapi **hasilnya jauh lebih rendah keyakinannya** dibanding saham "
        f"dengan histori panjang. Anggap ini eksplorasi awal, bukan sinyal siap pakai."
    )
    # shrink windows proportionally to available data, with sane floors
    max_window = max(5, n_rows // 3)
    mr_window = min(mr_window, max_window)
    mom_slow = min(mom_slow, max_window)
    mom_fast = min(mom_fast, max(3, mom_slow // 2))
    mc_days = min(mc_days, max(5, n_rows // 4))
else:
    st.success(f"Data berhasil diambil: {n_rows} baris, "
               f"{price_df.index[0].date()} s/d {price_df.index[-1].date()}")
    _suspect = flag_suspicious_prices(price_df)  # FIX 4f
    if len(_suspect) > 0:
        st.warning(
            f"⚠️ Data hygiene: {len(_suspect)} hari dengan |return harian| > 35% "
            f"(di atas batas ARA tier manapun — hampir pasti bad tick / corporate "
            f"action yang salah di yfinance): "
            + ", ".join(f"{d.date()} ({r:+.0%})" for d, r in _suspect.head(5).items())
            + ". Cross-check dengan Twelve Data fallback sebelum percaya angka di hari-hari itu."
        )

# cached_fetch_data() now lives in quant_engine.py (imported via `from
# quant_engine import *` at the top of this file) — this used to be a
# dangling forward-reference that scan_universe_parallel() depended on
# implicitly at call time.


# ==========================================================================
# ==== SECTION 5.4: LIVE GAINERS (raw price movement, not signal-based) ====
# ==========================================================================
# Different semantics from the Screener: this ranks purely by CURRENT %
# price change, no signal/backtest/ML logic involved. "Live" is relative —
# stock quotes via yfinance are delayed ~15-20min same as everywhere else in
# this app; crypto quotes via ccxt's fetch_tickers are close to real-time
# since exchanges push current order-book-derived last price.

@st.cache_data(ttl=20, show_spinner=False)
def fetch_live_quote_stock(ticker: str) -> dict:
    """Single-ticker live-ish quote via yfinance fast_info (much lighter/faster
    than full .info used for fundamentals — just price fields). Cached for a
    short 20s TTL — cheap insurance against Yahoo Finance's rate limiting when
    auto-refresh or repeated scans hit the same tickers in quick succession,
    without meaningfully hurting freshness (auto-refresh interval is 30s anyway).
    Also paced + auto-retried against 429s, see _yf_call_with_backoff."""
    import yfinance as yf

    def _do():
        fi = yf.Ticker(ticker).fast_info
        last = fi.get("lastPrice") or fi.get("last_price")
        prev = fi.get("previousClose") or fi.get("previous_close") or fi.get("regularMarketPreviousClose")
        if last is None or prev is None or prev == 0:
            raise ValueError("Data harga tidak lengkap dari yfinance")
        pct = (last / prev - 1) * 100
        # opportunistic — no extra network call, just extra fields from the same fast_info
        # payload we already fetched. May be None depending on yfinance version/ticker.
        volume = fi.get("lastVolume") or fi.get("last_volume") or fi.get("regularMarketVolume")
        return {"Simbol": ticker, "Harga": last, "Prev Close": prev, "% Perubahan": pct, "Volume": volume}

    return _yf_call_with_backoff(_do)


def fetch_live_gainers_stocks(tickers: list[str], max_workers: int = 8) -> tuple[list, list]:
    """Parallel live-quote fetch across many stock tickers. Concurrency kept
    moderate (not the 15 used for price-history scans) to stay gentler on
    Yahoo Finance's rate limiter — fast_info calls are cheap individually but
    add up fast when hit repeatedly (e.g. auto-refresh) across many tickers."""
    import concurrent.futures
    results, errors = [], []

    def _one(tkr):
        try:
            return ("ok", tkr, fetch_live_quote_stock(tkr))
        except Exception as e:
            return ("error", tkr, str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_one, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            status, tkr, payload = fut.result()
            if status == "ok":
                results.append(payload)
            else:
                errors.append((tkr, payload))
    return results, errors


def fetch_live_gainers_crypto(exchange_id: str, quote: str,
                               symbols: list[str] | None = None) -> tuple[list, list]:
    """
    Live gainers for crypto. Prefers a single bulk fetch_tickers() call
    (much more efficient — one API call returns ALL pairs at once) when the
    exchange supports it; falls back to per-symbol fetch_ticker() via a
    thread pool (slower, more exchange-polite concurrency) otherwise.
    """
    import ccxt
    import concurrent.futures

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    results, errors = [], []

    try:
        ex.load_markets()
        if ex.has.get("fetchTickers"):
            all_tickers = ex.fetch_tickers(symbols) if symbols else ex.fetch_tickers()
            for sym, t in all_tickers.items():
                if not sym.endswith("/" + quote):
                    continue
                pct, last = t.get("percentage"), t.get("last")
                if pct is None or last is None:
                    continue
                # opportunistic — already present in the same bulk response, no extra call
                qvol = t.get("quoteVolume")
                results.append({"Simbol": sym, "Harga": last, "% Perubahan": pct, "Volume": qvol})
        else:
            target_symbols = symbols or [s for s in ex.symbols if s.endswith("/" + quote)]

            def _one(sym):
                try:
                    t = ex.fetch_ticker(sym)
                    pct, last = t.get("percentage"), t.get("last")
                    if pct is None or last is None:
                        return ("error", sym, "Field percentage/last tidak tersedia")
                    return ("ok", sym, {"Simbol": sym, "Harga": last, "% Perubahan": pct,
                                         "Volume": t.get("quoteVolume")})
                except Exception as e:
                    return ("error", sym, str(e))

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_one, s): s for s in target_symbols}
                for fut in concurrent.futures.as_completed(futures):
                    status, sym, payload = fut.result()
                    if status == "ok":
                        results.append(payload)
                    else:
                        errors.append((sym, payload))
    except Exception as e:
        errors.append(("__all__", str(e)))

    return results, errors


# ==========================================================================
# ==== SECTION 5.5: FUNDAMENTAL ANALYSIS (stocks only) ====
# ==========================================================================
# fetch_fundamentals() / score_fundamentals() now live in quant_engine.py
# (this section used to have its OWN duplicate definitions that silently
# shadowed the ones in Section 1.5 — see quant_engine.py module docstring).

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["🎲 Monte Carlo", "📈 Signal", "🔁 Backtest", "✅ Checkpoint & Journal",
     "🔍 Screener", "📑 Fundamental", "🔥 Live Gainers", "🎯 Kesimpulan",
     "📐 Portofolio (Markowitz)"]
)

# ---- TAB 1: MONTE CARLO ----
with tab1:
    st.subheader("Simulasi Monte Carlo")
    with st.spinner("Menjalankan simulasi..."):
        try:
            if mc_model.startswith("GBM"):
                paths = simulate_gbm(price_df["Close"], n_days=mc_days, n_sims=mc_sims)
            else:
                if n_rows < 100:
                    st.info(
                        "GARCH idealnya butuh 100+ hari data untuk fitting yang stabil "
                        f"(data saat ini {n_rows} baris). Tetap dicoba — kalau gagal, "
                        "otomatis fallback ke GBM."
                    )
                try:
                    paths = simulate_garch(price_df["Close"], n_days=mc_days, n_sims=mc_sims,
                                            asymmetric=mc_model.startswith("GJR"))
                except Exception:
                    st.warning(
                        "Fitting GARCH gagal (data terlalu sedikit/tidak stabil untuk "
                        "estimasi volatilitas dinamis) — fallback otomatis ke GBM."
                    )
                    paths = simulate_gbm(price_df["Close"], n_days=mc_days, n_sims=mc_sims)
        except Exception as e:
            st.error(f"Simulasi gagal: {e}")
            st.stop()

    summary = summarize_paths(paths)
    bands = fan_chart_bands(paths)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Harga sekarang", f"{summary['s0']:,.2f}")
    c2.metric(f"Median setelah {mc_days} hari", f"{summary['median_final']:,.2f}",
              f"{(summary['median_final']/summary['s0']-1)*100:+.1f}%")
    c3.metric("Probabilitas profit", f"{summary['prob_profit']*100:.1f}%")
    c4.metric("VaR 95% (potensi rugi)", f"{summary['VaR_95pct']*100:.1f}%")
    c5.metric("ES/CVaR 95%", f"{summary['ES_95pct']*100:.1f}%",
              help="Expected Shortfall: rata-rata kerugian DI LUAR VaR — 'kalau sudah "
                   "masuk 5% skenario terburuk, rata-rata rugi seberapa dalam'.")

    chart_style = st.radio(
        "Tipe visualisasi", ["Fan Chart (persentil)", "Individual Paths (tiap jalur simulasi)"],
        horizontal=True,
        help="Fan Chart: pita persentil, lebih bersih buat baca rentang kemungkinan. "
             "Individual Paths: tiap simulasi digambar satu-satu, warna sesuai hasil akhirnya "
             "(biru=rendah, merah=tinggi) — gaya yang sering dipakai di video/tutorial finance."
    )

    if chart_style == "Fan Chart (persentil)":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p95"], line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p5"], fill="tonexty",
                                  line=dict(width=0), name="90% band",
                                  fillcolor="rgba(99,110,250,0.15)"))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p75"], line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p25"], fill="tonexty",
                                  line=dict(width=0), name="50% band",
                                  fillcolor="rgba(99,110,250,0.3)"))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p50"], line=dict(color="#636efa", width=2),
                                  name="Median path"))
        fig.update_layout(title=f"Fan chart — {mc_model}", xaxis_title="Hari ke depan",
                           yaxis_title="Harga", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        import plotly.express as px

        # Rendering every single simulated path (up to 50,000) would crash the
        # browser — subsample for DISPLAY only. Stats above (prob_profit, VaR,
        # etc.) already come from the FULL simulation set, this subsampling
        # only affects what gets drawn on screen.
        n_display = min(len(paths), 300)
        rng_display = np.random.default_rng(0)
        display_idx = rng_display.choice(len(paths), size=n_display, replace=False)
        display_paths = paths[display_idx]

        finals = display_paths[:, -1]
        f_min, f_max = finals.min(), finals.max()
        norm = (finals - f_min) / (f_max - f_min) if f_max > f_min else np.zeros_like(finals)
        colors = px.colors.sample_colorscale("RdBu_r", norm.tolist())

        x_axis = list(range(display_paths.shape[1]))
        fig2 = go.Figure()
        for i in range(n_display):
            fig2.add_trace(go.Scatter(
                x=x_axis, y=display_paths[i], mode="lines",
                line=dict(color=colors[i], width=0.8), opacity=0.5,
                showlegend=False, hoverinfo="skip"
            ))
        fig2.add_hline(y=summary["s0"], line_dash="dash", line_color="black",
                        annotation_text=f"Initial Stock Price: {summary['s0']:,.2f}")
        # dummy trace purely to render a shared colorbar for the line colors above
        fig2.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(colorscale="RdBu_r", cmin=f_min, cmax=f_max, color=[f_min],
                        showscale=True, colorbar=dict(title="Harga Akhir"), size=0.1),
            showlegend=False, hoverinfo="skip"
        ))
        fig2.update_layout(
            title=f"Monte Carlo Simulation Paths — {symbol} ({mc_model})",
            xaxis_title="Time Steps (Hari)", yaxis_title="Harga Saham", height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
        if len(paths) > n_display:
            st.caption(f"Menampilkan {n_display} dari {len(paths)} simulasi (subsample buat "
                       f"performa render) — statistik di atas (VaR, probabilitas profit, dst) "
                       f"tetap dihitung dari SEMUA {len(paths)} simulasi, bukan cuma yang tampil.")

    st.caption("Catatan: Monte Carlo memberi gambaran **distribusi kemungkinan**, bukan "
               "prediksi pasti. Model GARCH lebih realistis untuk aset volatil (crypto) "
               "karena volatilitas ikut berubah seiring waktu, bukan konstan seperti GBM. "
               "GJR-GARCH menambah satu hal lagi: volatilitas boleh melonjak lebih tinggi "
               "setelah hari merah dibanding hari hijau yang besarnya sama.")

# ---- TAB 2: SIGNAL ----
with tab2:
    st.subheader("Sinyal Trading")

    ml_score_series, ml_final_model, ml_feat_importance, ml_features_df, ml_final_regressor, ml_calibrator, ml_diagnostics = (
        None, None, None, None, None, None, None
    )
    if use_ml:
        if n_rows < ml_min_train + 50:
            st.info(
                f"Data ({n_rows} baris) belum cukup untuk sinyal ML dengan setting "
                f"minimal training {ml_min_train} hari — butuh minimal ~{ml_min_train + 50} "
                f"baris. Sinyal ML dilewati, composite fallback ke mean-reversion+momentum saja."
            )
        else:
            with st.spinner(f"Melatih model {ml_model_label} (walk-forward)..."):
                ml_benchmark = fetch_benchmark_prices(
                    asset_type, symbol, price_df.index,
                    exchange_id=exchange_id if asset_type == "crypto" else "binance"
                )
                ml_score_series, ml_final_model, ml_feat_importance, ml_features_df, ml_final_regressor, ml_calibrator, ml_diagnostics = (
                    walk_forward_ml_signal(price_df, min_train_days=ml_min_train,
                                            retrain_every=ml_retrain_every,
                                            horizon_days=ml_horizon_days,
                                            model_type=ml_model_type,
                                            benchmark=ml_benchmark)
                )
                if ml_benchmark is None:
                    st.caption(
                        "ℹ️ Fitur relative-strength (vs IHSG/S&P500/BTC) tidak tersedia untuk "
                        "run ini (fetch benchmark gagal, atau aset ini sendiri adalah BTC) — "
                        "model tetap jalan pakai fitur teknikal murni seperti biasa."
                    )

    # ---- Adaptive ML weight: discount the composite's ML contribution when
    # the reliability curve says this model's calibration is weak, instead
    # of trusting the sidebar slider blindly regardless of demonstrated skill.
    ml_confidence, ml_confidence_detail = ml_calibration_confidence(ml_diagnostics)
    raw_ml_weight = ml_weight if use_ml else 0.0
    effective_ml_weight = raw_ml_weight * (ml_confidence if adaptive_ml_weight else 1.0)

    # ---- Earnings-drift (PEAD) series — fetched here (not inside
    # composite_signal, which stays a pure/testable dataframe transform
    # with no network calls of its own, same pattern as ml_score above).
    # Only fetched when actually enabled AND for stocks (no earnings for
    # crypto) — zero extra network cost when this feature is off.
    earnings_score_series = None
    if earnings_weight > 0 and asset_type in ("stock_id", "stock_us"):
        try:
            _earnings_ticker = symbol if asset_type == "stock_us" else (
                symbol if symbol.upper().endswith(".JK") else symbol.upper() + ".JK"
            )
            earnings_score_series = earnings_drift_series(
                price_df, _earnings_ticker, drift_window_days=earnings_drift_window)
            if earnings_score_series is None or earnings_score_series.notna().sum() == 0:
                st.caption(
                    "ℹ️ Data earnings surprise nggak tersedia untuk simbol ini (umum buat "
                    "saham IDX di luar cakupan yfinance) — bobot Earnings Drift nggak "
                    "berefek untuk simbol ini, sisanya jalan normal."
                )
        except Exception:
            earnings_score_series = None

    sig_df = composite_signal(
        price_df, mr_window=mr_window, mr_z_entry=mr_z_entry,
        mom_fast=mom_fast, mom_slow=mom_slow, mr_weight=mr_weight,
        stochbb_weight=stochbb_weight, bb_window=bb_window, bb_std=bb_std,
        stoch_window=stoch_window, stoch_smooth=stoch_smooth,
        ml_score=ml_score_series, ml_weight=effective_ml_weight,
        earnings_score=earnings_score_series, earnings_weight=earnings_weight,
    )

    # ---- FIX 4a: market regime gate ----
    if use_regime_gate:
        _bench_gate = fetch_benchmark_prices(
            asset_type, symbol, price_df.index,
            exchange_id=exchange_id if asset_type == "crypto" else "binance")
        if _bench_gate is not None:
            _n_buy_before = int((sig_df["composite_signal"] == "BUY").sum())
            sig_df = apply_regime_gate(sig_df, _bench_gate)
            _n_buy_after = int((sig_df["composite_signal"] == "BUY").sum())
            if not bool(sig_df["regime_ok"].iloc[-1]):
                st.warning(
                    "🌍 **Regime gate aktif**: benchmark saat ini di BAWAH MA-200 "
                    "(rezim bearish) — sinyal BUY di-mute sampai benchmark kembali "
                    "di atas MA-200. Sinyal SELL/HOLD tidak terpengaruh."
                )
            if _n_buy_after < _n_buy_before:
                st.caption(f"🌍 Regime gate menekan {_n_buy_before - _n_buy_after} "
                           f"sinyal BUY historis yang muncul saat rezim bearish.")
        else:
            st.caption("ℹ️ Regime gate dilewati — data benchmark tidak tersedia.")

    # ---- FIX 3a: Information Coefficient ----
    with st.expander("📏 Information Coefficient — apakah score benar-benar prediktif?"):
        ic_df = signal_ic(price_df, sig_df["composite_score"])
        st.dataframe(ic_df.round(4), use_container_width=True)
        st.caption(
            "Rank IC (Spearman) antara composite score hari ini vs return N hari ke depan. "
            "Aturan praktis: **|rank IC| < 0.02 dengan |t| < 2 = tidak ada edge terukur**, "
            "apa pun kata backtest. Ini metrik yang mengukur SINYALNYA sendiri, bukan satu "
            "path trading tertentu."
        )

    # ---- FIX 4c: signal log (forward test otomatis) ----
    try:
        log_signal_snapshot(symbol, asset_type, sig_df)
    except Exception:
        pass  # logging tidak boleh menggagalkan analisis utama

    if use_ml and ml_confidence_detail is not None:
        if adaptive_ml_weight:
            st.info(
                f"⚖️ **Bobot ML disesuaikan otomatis (adaptive weighting)** — slider bobot ML "
                f"kamu di `{raw_ml_weight:.2f}`, tapi Brier Skill Score kalibrasi out-of-sample "
                f"model ini cuma `{ml_confidence_detail['bss']:.2f}` (1.0 = kalibrasi sempurna, "
                f"0.0 = setara nebak base rate historis, negatif = lebih buruk dari itu) → "
                f"bobot ML **efektif** yang benar-benar dipakai di composite score & verdict "
                f"jadi `{effective_ml_weight:.2f}`. Detail reliability curve-nya ada di expander "
                f"🔬 Error Analysis & Model Diagnostics di bawah."
            )
        else:
            st.caption(
                f"ℹ️ Brier Skill Score kalibrasi out-of-sample model ini: "
                f"`{ml_confidence_detail['bss']:.2f}`. Adaptive ML weight nonaktif (sidebar) — "
                f"bobot ML tetap pakai nilai slider apa adanya (`{raw_ml_weight:.2f}`), bukan "
                f"nilai efektif yang sudah didiskon."
            )


    latest = sig_df.iloc[-1]
    sig_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(latest["composite_signal"], "⚪")
    st.markdown(f"### Sinyal terkini: {sig_color} **{latest['composite_signal']}** "
                f"(score: {latest['composite_score']:.2f})")

    # ---- Confidence-scaled position sizing (opt-in, see sidebar) — computed
    # here where composite_score, ml_confidence, and price history are all
    # in scope. Falls back to the flat sidebar % when the toggle is off, so
    # behavior is byte-for-byte identical to before unless the user opts in.
    effective_position_size_pct = position_size_pct
    sizing_detail = None
    if use_confidence_sizing:
        _realized_vol_annual = None
        if use_vol_targeting:
            _ret = price_df["Close"].pct_change().dropna()
            if len(_ret) > 5:
                _realized_vol_annual = float(_ret.tail(60).std() * (trading_days ** 0.5))
        _sizing = confidence_scaled_position_size(
            base_position_pct=position_size_pct,
            composite_score=float(latest["composite_score"]),
            ml_confidence=(ml_confidence if use_ml else None),
            vol_target_annual=vol_target_annual_pct,
            realized_vol_annual=_realized_vol_annual,
        )
        effective_position_size_pct = _sizing["position_size_pct"]
        sizing_detail = _sizing["detail"]

        sc1, sc2 = st.columns(2)
        sc1.metric("% modal per trade — base (sidebar)", f"{position_size_pct*100:.0f}%")
        sc2.metric("% modal per trade — confidence-scaled", f"{effective_position_size_pct*100:.0f}%",
                   f"{(effective_position_size_pct - position_size_pct)*100:+.0f} pts")
        _detail_bits = []
        if "conviction_mult" in sizing_detail:
            _detail_bits.append(f"conviction ×{sizing_detail['conviction_mult']:.2f}")
        if "confidence_mult" in sizing_detail:
            _detail_bits.append(f"ML confidence ×{sizing_detail['confidence_mult']:.2f}")
        if "vol_target_mult" in sizing_detail:
            _detail_bits.append(f"vol targeting ×{sizing_detail['vol_target_mult']:.2f}")
        st.caption(
            "📐 Size ini yang dipakai di tab Backtest (bukan angka slider sidebar mentah) — "
            "rincian: " + " · ".join(_detail_bits) + ". Tetap cuma soal BESARAN, arah "
            "BUY/SELL/HOLD di atas nggak berubah."
        )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sig_df.index, y=sig_df["Close"], name="Close",
                               line=dict(color="#2a2a2a", width=1.5)))
    fig2.add_trace(go.Scatter(x=sig_df.index, y=sig_df["ma_fast"], name=f"MA{mom_fast}",
                               line=dict(color="orange", width=1)))
    fig2.add_trace(go.Scatter(x=sig_df.index, y=sig_df["ma_slow"], name=f"MA{mom_slow}",
                               line=dict(color="purple", width=1)))

    buys = sig_df[sig_df["composite_signal"] == "BUY"]
    sells = sig_df[sig_df["composite_signal"] == "SELL"]
    fig2.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="BUY",
                               marker=dict(color="green", size=9, symbol="triangle-up")))
    fig2.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="SELL",
                               marker=dict(color="red", size=9, symbol="triangle-down")))

    fig2.update_layout(title="Harga + Sinyal Composite", height=500)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Lihat Z-score mean reversion"):
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sig_df.index, y=sig_df["mr_zscore"], name="Z-score"))
        fig3.add_hline(y=mr_z_entry, line_dash="dash", line_color="red")
        fig3.add_hline(y=-mr_z_entry, line_dash="dash", line_color="green")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    if stochbb_weight > 0 and "bb_upper" in sig_df.columns and "stochbb_signal" in sig_df.columns:
        with st.expander("Lihat Stochastic + Bollinger Band", expanded=True):
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=sig_df.index, y=sig_df["bb_upper"], name="BB Upper",
                                         line=dict(color="lightgray", width=1), showlegend=False))
            fig_bb.add_trace(go.Scatter(x=sig_df.index, y=sig_df["bb_lower"], name="BB Lower",
                                         line=dict(color="lightgray", width=1), fill="tonexty",
                                         fillcolor="rgba(200,200,200,0.2)", showlegend=False))
            fig_bb.add_trace(go.Scatter(x=sig_df.index, y=sig_df["bb_mid"], name="BB Mid (MA)",
                                         line=dict(color="gray", width=1, dash="dot")))
            fig_bb.add_trace(go.Scatter(x=sig_df.index, y=sig_df["Close"], name="Close",
                                         line=dict(color="#2a2a2a", width=1.5)))
            stochbb_buys = sig_df[sig_df["stochbb_signal"] == "BUY"]
            stochbb_sells = sig_df[sig_df["stochbb_signal"] == "SELL"]
            fig_bb.add_trace(go.Scatter(x=stochbb_buys.index, y=stochbb_buys["Close"], mode="markers",
                                         name="Stoch-BB BUY", marker=dict(color="green", size=8, symbol="star")))
            fig_bb.add_trace(go.Scatter(x=stochbb_sells.index, y=stochbb_sells["Close"], mode="markers",
                                         name="Stoch-BB SELL", marker=dict(color="red", size=8, symbol="star")))
            band_walk_up = sig_df[sig_df["band_walk_state"] == "up"]
            band_walk_down = sig_df[sig_df["band_walk_state"] == "down"]
            if len(band_walk_up) > 0:
                fig_bb.add_trace(go.Scatter(x=band_walk_up.index, y=band_walk_up["Close"], mode="markers",
                                             name="Band-walk (naik, SELL di-mute)",
                                             marker=dict(color="orange", size=5, symbol="circle")))
            if len(band_walk_down) > 0:
                fig_bb.add_trace(go.Scatter(x=band_walk_down.index, y=band_walk_down["Close"], mode="markers",
                                             name="Band-walk (turun, BUY di-mute)",
                                             marker=dict(color="blue", size=5, symbol="circle")))
            fig_bb.update_layout(title="Harga vs Bollinger Band", height=400)
            st.plotly_chart(fig_bb, use_container_width=True)

            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=sig_df.index, y=sig_df["stoch_k"], name="%K"))
            fig_stoch.add_trace(go.Scatter(x=sig_df.index, y=sig_df["stoch_d"], name="%D"))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
            fig_stoch.update_layout(title="Stochastic Oscillator", height=280)
            st.plotly_chart(fig_stoch, use_container_width=True)
            st.caption(
                "Sinyal Stoch-BB (bintang) cuma muncul kalau %K oversold/overbought (<20 atau >80) "
                "**dan** harga di dekat pinggir Bollinger Band secara bersamaan — kombinasi klasik, "
                "sengaja selektif/jarang nembak. Titik oranye/biru = band-walk terdeteksi (tren kuat, "
                "band melebar, harga nempel di pinggir beberapa hari beruntun) — sinyal reversion yang "
                "berlawanan arah tren di-mute di titik-titik itu, BUKAN dibalik jadi sinyal beli/jual."
            )

    if earnings_weight > 0 and "earnings_drift_score" in sig_df.columns:
        with st.expander("Lihat Earnings Drift (PEAD)", expanded=True):
            _earn_active = sig_df["earnings_drift_score"].notna().sum()
            if _earn_active == 0:
                st.info("Nggak ada earnings surprise historis yang tersedia untuk simbol ini "
                        "dalam rentang data yang di-fetch.")
            else:
                fig_earn = go.Figure()
                fig_earn.add_trace(go.Scatter(
                    x=sig_df.index, y=sig_df["earnings_drift_score"],
                    name="PEAD Score", line=dict(color="#9467bd"), connectgaps=False
                ))
                fig_earn.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_earn.update_layout(
                    title="Earnings Drift Score (aktif cuma dalam drift window pasca-earnings)",
                    height=280, yaxis_range=[-1.05, 1.05]
                )
                st.plotly_chart(fig_earn, use_container_width=True)
                st.caption(
                    f"{_earn_active} dari {len(sig_df)} hari punya sinyal PEAD aktif (sisanya "
                    f"di luar drift window {earnings_drift_window} hari, garis terputus). Beda "
                    f"dari semua sinyal lain di dashboard ini — ini dibangun dari data earnings "
                    f"surprise (EPS aktual vs estimasi), bukan dari harga/volume."
                )

    st.markdown("---")
    st.markdown("#### 📰 Sentimen Berita (Informasional)")
    st.caption(
        "Sentimen berita real-time via Marketaux — **BUKAN** bagian dari composite score "
        "yang divalidasi (nggak ada arsip berita historis gratis buat backtest, jadi ini "
        "nggak bisa diperlakukan sama ketatnya kayak sinyal lain di dashboard ini). Ini "
        "cuma panel informasional yang kamu baca sendiri, bukan angka yang otomatis "
        "menggerakkan sinyal beli/jual. Kuota Marketaux 100 request/hari untuk SELURUH app — "
        "di belakang tombol biar kamu yang kontrol kapan kepakai."
    )
    if st.button("📰 Cek Sentimen Berita", key="check_news_sentiment_btn"):
        _news_query = symbol if asset_type != "stock_id" else (
            symbol.upper().replace(".JK", "")
        )
        with st.spinner("Mengambil berita terbaru..."):
            news_result = fetch_news_sentiment(_news_query)
        st.session_state["news_sentiment_result"] = news_result
        st.session_state["news_sentiment_query"] = _news_query

    if "news_sentiment_result" in st.session_state:
        news_result = st.session_state["news_sentiment_result"]
        if news_result is None:
            st.info(
                "Nggak ada hasil — kemungkinan: `MARKETAUX_API_KEY` belum diisi di Secrets, "
                "kuota harian habis, atau nggak ada berita relevan ditemukan untuk simbol ini "
                "(cakupan berita Indonesia/crypto kecil di Marketaux belum tentu selengkap "
                "saham US besar)."
            )
        else:
            avg = news_result["avg_sentiment"]
            label = "Positif" if avg > 0.15 else "Negatif" if avg < -0.15 else "Netral"
            color = "🟢" if avg > 0.15 else "🔴" if avg < -0.15 else "⚪"
            st.metric(f"{color} Sentimen rata-rata ({news_result['n_articles']} artikel)",
                      f"{label} ({avg:+.2f})")
            for art in news_result["articles"]:
                _s = art["sentiment"]
                _icon = "🟢" if _s > 0.15 else "🔴" if _s < -0.15 else "⚪"
                st.write(f"{_icon} **{art['title']}** — {art['source']} · {art['published_at']} "
                         f"(sentimen: {_s:+.2f})")

    if use_ml and ml_final_model is not None:
        with st.expander(f"Lihat detail sinyal ML ({ml_model_label})", expanded=True):
            last_feat = ml_features_df.iloc[[-1]]
            if not last_feat.isna().any(axis=1).iloc[0]:
                if ml_model_type == "ensemble":
                    p_lgbm = ml_final_model[0].predict_proba(last_feat)[0, 1]
                    p_xgb = ml_final_model[1].predict_proba(last_feat)[0, 1]
                    live_proba = (p_lgbm + p_xgb) / 2
                else:
                    live_proba = ml_final_model.predict_proba(last_feat)[0, 1]

                if ml_calibrator is not None:
                    display_proba = float(ml_calibrator.predict([live_proba])[0])
                else:
                    display_proba = live_proba

                c1, c2 = st.columns(2)
                c1.metric(f"P(harga naik dalam {ml_horizon_days} hari) — model final",
                          f"{display_proba*100:.1f}%")
                c2.metric("ML score", f"{2*display_proba-1:+.2f}", "range -1 (turun) s/d +1 (naik)")

                if ml_calibrator is not None:
                    st.caption(
                        f"Sudah dikalibrasi (isotonic regression, dilatih dari histori prediksi "
                        f"walk-forward yang out-of-sample beneran vs hasil aktualnya) dari "
                        f"probabilitas mentah model **{live_proba*100:.1f}%** → **{display_proba*100:.1f}%** "
                        f"— supaya angka ini lebih mendekati frekuensi empirisnya, bukan cuma "
                        f"skor mentah dari tree ensemble."
                    )
                else:
                    st.caption(
                        "Kalibrasi probabilitas belum tersedia (butuh minimal 50 hari histori "
                        "walk-forward dengan prediksi valid) — angka di atas masih probabilitas "
                        "mentah dari model."
                    )

                if ml_model_type == "ensemble":
                    st.caption(f"Rincian ensemble (sebelum kalibrasi): LightGBM {p_lgbm*100:.1f}% · "
                               f"XGBoost {p_xgb*100:.1f}% → rata-rata {live_proba*100:.1f}%")

                if ml_final_regressor is not None:
                    pred_return = ml_final_regressor.predict(last_feat)[0]
                    st.metric(f"Prediksi magnitude return ({ml_horizon_days} hari)",
                              f"{pred_return*100:+.2f}%",
                              help="Model regresi terpisah (selalu LightGBM) yang memprediksi "
                                   "BESARAN return, bukan cuma arah — pelengkap probabilitas "
                                   "di atas, bukan pengganti. Sama-sama cuma estimasi statistik.")
            else:
                st.info("Fitur hari terakhir mengandung NaN (data belum cukup panjang "
                        "untuk semua indikator), live prediction tidak tersedia.")

            st.caption(
                f"Cakupan walk-forward: {ml_score_series.notna().sum()} dari {n_rows} hari "
                f"punya prediksi ML (hari-hari awal dilewati karena belum cukup data training)."
            )

            if ml_feat_importance is not None:
                fi_title = "Feature Importance (LightGBM)" if ml_model_type == "ensemble" \
                    else f"Feature Importance ({ml_model_label})"
                fig_fi = go.Figure(go.Bar(
                    x=ml_feat_importance.values, y=ml_feat_importance.index,
                    orientation="h", marker_color="#636efa"
                ))
                fig_fi.update_layout(title=fi_title, height=350, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_fi, use_container_width=True)
                if ml_model_type == "ensemble":
                    st.caption("Ensemble melatih dua model (LightGBM + XGBoost) dan "
                               "merata-ratakan prediksinya, tapi chart importance ini "
                               "cuma dari sisi LightGBM-nya — biar satu chart yang jelas "
                               "dibaca, bukan dua yang berpotensi konflik urutannya.")

            if ml_diagnostics is not None:
                with st.expander("🔬 Error Analysis & Model Diagnostics (Reliability Curve)"):
                    st.caption(
                        "Dihitung dari histori walk-forward out-of-sample yang sama dipakai untuk "
                        "melatih kalibrator isotonic di atas — prediksi yang model betul-betul "
                        "belum lihat outcome-nya saat prediksi dibuat, bukan in-sample fit yang "
                        "menyesatkan."
                    )
                    diag_raw = np.asarray(ml_diagnostics["proba_raw"], dtype=float)
                    diag_actual = np.asarray(ml_diagnostics["actual"], dtype=float)
                    diag_calib = np.asarray(ml_diagnostics["proba_calibrated"], dtype=float)

                    def _reliability_bins(pred, actual, n_bins=10):
                        edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
                        if len(edges) < 3:
                            return np.array([pred.mean()]), np.array([actual.mean()]), np.array([len(pred)])
                        bin_idx = np.clip(np.digitize(pred, edges[1:-1], right=True), 0, len(edges) - 2)
                        mp, ma, mn = [], [], []
                        for b in range(len(edges) - 1):
                            mask = bin_idx == b
                            if mask.sum() == 0:
                                continue
                            mp.append(pred[mask].mean())
                            ma.append(actual[mask].mean())
                            mn.append(int(mask.sum()))
                        return np.array(mp), np.array(ma), np.array(mn)

                    n_bins = min(10, max(3, len(diag_raw) // 20))
                    raw_x, raw_y, raw_n = _reliability_bins(diag_raw, diag_actual, n_bins)
                    calib_x, calib_y_, _ = _reliability_bins(diag_calib, diag_actual, n_bins)
                    marker_sizes = 5 + (raw_n / raw_n.max() * 20 if raw_n.max() > 0 else 0)

                    fig_rel = go.Figure()
                    fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                                  line=dict(dash="dash", color="gray"),
                                                  name="Kalibrasi sempurna"))
                    fig_rel.add_trace(go.Scatter(x=raw_x, y=raw_y, mode="lines+markers",
                                                  name="Mentah (sebelum kalibrasi)",
                                                  marker=dict(size=marker_sizes)))
                    fig_rel.add_trace(go.Scatter(x=calib_x, y=calib_y_, mode="lines+markers",
                                                  name="Setelah isotonic calibration"))
                    fig_rel.update_layout(
                        title="Reliability Curve — Probabilitas Prediksi vs Frekuensi Aktual",
                        xaxis_title="Probabilitas rata-rata diprediksi (per bin)",
                        yaxis_title="Frekuensi 'naik' aktual (per bin)",
                        height=400, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)
                    st.caption(
                        "Titik yang jatuh persis di garis putus-putus abu-abu artinya kalibrasi "
                        "sempurna — kalau model bilang '70% naik', secara historis memang ~70% "
                        "dari hari-hari di bin itu benar-benar naik. Ukuran marker (kurva mentah) "
                        "proporsional ke jumlah observasi di bin itu."
                    )

                    brier_raw = float(np.mean((diag_raw - diag_actual) ** 2))
                    brier_calib = float(np.mean((diag_calib - diag_actual) ** 2))
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Brier score — mentah", f"{brier_raw:.4f}",
                               help="Mean squared error antara probabilitas & outcome biner "
                                    "aktual (0/1). Makin rendah makin baik: 0 = sempurna, "
                                    "0.25 = setara nebak 50/50 terus-menerus.")
                    bc2.metric("Brier score — terkalibrasi", f"{brier_calib:.4f}",
                               delta=f"{brier_calib - brier_raw:+.4f}", delta_color="inverse")
                    bc3.metric("N observasi out-of-sample", f"{len(diag_raw)}")

                    resid = diag_actual - diag_calib
                    fig_resid = go.Figure(go.Histogram(x=resid, nbinsx=30, marker_color="#636efa"))
                    fig_resid.update_layout(
                        title="Sebaran Residual (outcome aktual − probabilitas terkalibrasi)",
                        xaxis_title="Residual", yaxis_title="Frekuensi", height=300,
                    )
                    st.plotly_chart(fig_resid, use_container_width=True)
                    st.caption(
                        "Residual dekat 0 = prediksi akurat untuk observasi itu. Distribusi ini "
                        "wajar bimodal (menumpuk dekat 0 atau dekat ±1) karena outcome-nya biner "
                        "(naik/turun), bukan kontinu — bukan tanda model buruk, cuma cara baca "
                        "residual untuk target biner yang beda dari regresi biasa."
                    )
                    st.divider()
                    st.markdown("**⚖️ Dampak ke Weighted Decision (Kesimpulan)**")
                    bss_val = ml_confidence_detail["bss"]
                    st.write(
                        f"Brier Skill Score: **{bss_val:.2f}** vs baseline nebak base rate "
                        f"→ confidence multiplier **{ml_confidence:.2f}** → bobot ML di "
                        f"composite score: `{raw_ml_weight:.2f}` × `{ml_confidence:.2f}` = "
                        f"**`{effective_ml_weight:.2f}`**"
                        + (" *(adaptive weighting nonaktif — angka ini cuma info, bobot "
                           "aktual tetap pakai nilai slider mentah)*" if not adaptive_ml_weight else "")
                        + ". Angka `{:.2f}` inilah yang benar-benar dipakai untuk menghitung "
                          "composite score & verdict di tab 🎯 Kesimpulan, bukan cuma "
                          "ditampilkan sebagai info.".format(effective_ml_weight)
                    )
            elif use_ml and ml_score_series is not None:
                st.caption(
                    "🔬 Error Analysis & Model Diagnostics belum tersedia — butuh minimal 50 "
                    "hari histori walk-forward out-of-sample dengan prediksi valid (syarat sama "
                    "seperti kalibrasi isotonic di atas)."
                )

            st.warning(
                "⚠️ Model ML dilatih dari data historis dengan fitur teknikal (return, "
                "volatilitas, RSI, MACD, Bollinger %B, ATR, Stochastic, rasio MA, volume, "
                "Z-score). Ini bukan model canggih buat production trading — anggap sebagai "
                "eksplorasi tambahan, bukan sumber kebenaran. Selalu cek konsistensinya di "
                "tab Backtest."
            )

    st.markdown("---")
    st.markdown("#### 🧬 Auto-tuning Parameter (Genetic Algorithm)")
    _default_pop, _default_gen = (60, 30) if IS_LOCAL else (24, 15)
    with st.expander("Atur populasi & generasi manual", expanded=False):
        st.caption(
            f"Default otomatis: {_default_pop} populasi × {_default_gen} generasi "
            f"({'mode lokal' if IS_LOCAL else 'mode cloud'}). Naikkan manual kalau mau "
            "pencarian lebih luas (lebih lambat), turunkan kalau mau lebih cepat coba-coba."
        )
        _ga_pop = st.slider("Ukuran populasi", 8, 200, _default_pop, 4, key="ga_pop_manual")
        _ga_gen = st.slider("Jumlah generasi", 3, 100, _default_gen, 1, key="ga_gen_manual")
        _ga_train_frac = st.slider(
            "Porsi data untuk training GA", 0.5, 0.9, 0.7, 0.05, key="ga_train_frac_manual",
            help="Sisanya (holdout) TIDAK PERNAH dilihat GA selama pencarian — dipakai "
                 "cuma sekali di akhir buat ngecek parameter terbaik beneran generalize "
                 "atau cuma noise-fitting periode training."
        )
    st.caption(
        "Cari parameter mean-reversion/momentum otomatis lewat genetic algorithm, "
        "dioptimasi terhadap Sharpe ratio backtest (dengan penalti kalau jumlah trade "
        "terlalu sedikit). Di belakang tombol karena ini jalanin ratusan backtest "
        f"sekaligus — lumayan berat kalau auto-jalan tiap buka tab. "
        f"Populasi {_ga_pop} × {_ga_gen} generasi."
    )
    st.caption(
        "⚠️ **Walk-forward split**: GA cuma boleh 'melihat' & optimasi di data training "
        f"({_ga_train_frac:.0%} awal). Sisa data (holdout) dites SEKALI di akhir dengan "
        "parameter yang menang — itu satu-satunya angka yang boleh dipercaya sebagai "
        "estimasi edge beneran, bukan fitness training-nya."
    )
    if st.button("🚀 Jalankan GA Optimization", key="run_ga_btn"):
        with st.spinner(f"Menjalankan genetic algorithm (~{_ga_gen} generasi)..."):
            ga_progress = st.empty()
            ga_result = adv.optimize_signal_params_ga(
                price_df, composite_signal, run_backtest,
                pop_size=_ga_pop, n_generations=_ga_gen, train_frac=_ga_train_frac,
                progress_callback=lambda g, tot, f: ga_progress.text(
                    f"Generasi {g}/{tot} — fitness training terbaik: {f:.3f}"),
            )
        st.session_state["ga_result"] = ga_result
        ga_progress.empty()

    if "ga_result" in st.session_state:
        ga = st.session_state["ga_result"]
        st.markdown("**Fitness training** (dipakai GA buat nyari — optimis, jangan dipercaya sendirian)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fitness default", f"{ga['baseline_fitness']:.2f}")
        c2.metric("Fitness GA", f"{ga['best_fitness']:.2f}", f"{ga['improvement']:+.2f}")
        c3.metric("Generasi", len(ga["history"]))

        st.markdown(
            f"**Fitness holdout** (data {ga['n_holdout_rows']} baris sejak "
            f"`{ga['split_date']}` — GA nggak pernah lihat ini, angka ini yang jujur)"
        )
        h1, h2, h3 = st.columns(3)
        h1.metric("Fitness default (holdout)", f"{ga['baseline_holdout_fitness']:.2f}")
        h2.metric("Fitness GA (holdout)", f"{ga['holdout_fitness']:.2f}",
                   f"{ga['holdout_improvement']:+.2f}")
        h3.metric("Overfit gap", f"{ga['overfit_gap']:.2f}",
                   help="best_fitness (training) - holdout_fitness. Makin besar & positif, "
                        "makin kuat indikasi GA cuma noise-fitting periode training.")

        if ga["holdout_improvement"] <= 0:
            st.error(
                "🚨 Di data holdout, parameter GA **TIDAK lebih baik** (atau lebih buruk) "
                "dibanding parameter default — indikasi kuat overfitting ke periode training. "
                "Jangan pakai parameter ini apa adanya."
            )
        elif ga["overfit_gap"] > 0.5:
            st.warning(
                "⚠️ Gap antara fitness training dan holdout cukup besar — parameter GA "
                "menang di holdout, tapi sebagian dari keunggulan training-nya kemungkinan "
                "noise. Perlakukan `holdout_improvement` di atas sebagai estimasi edge yang "
                "lebih realistis dibanding `improvement` (training)."
            )
        else:
            st.success(
                "✅ Parameter GA tetap unggul di data holdout yang nggak pernah dilihat "
                "GA, dengan gap training-holdout yang kecil — sinyal yang lebih meyakinkan "
                "dibanding cuma lihat fitness training."
            )

        fig_ga = go.Figure(go.Scatter(y=ga["history"], mode="lines+markers"))
        fig_ga.update_layout(title="Konvergensi GA (training)", xaxis_title="Generasi",
                              yaxis_title="Fitness (Sharpe, penalized)", height=280)
        st.plotly_chart(fig_ga, use_container_width=True)

        params_df = pd.DataFrame([ga["baseline_params"], ga["best_params"]],
                                  index=["Default (sidebar)", "GA-optimized"])
        st.dataframe(params_df, use_container_width=True)

        st.markdown("**📉 Uji statistik tambahan (multiple testing & plateau)**")
        _n_trials = _ga_pop * _ga_gen
        _psr = probabilistic_sharpe_ratio(ga["holdout_fitness"], ga["n_holdout_rows"])
        _max_null = expected_max_sharpe_under_null(_n_trials, ga["n_holdout_rows"])
        pc1, pc2 = st.columns(2)
        pc1.metric("PSR (holdout)", f"{_psr*100:.1f}%" if _psr is not None else "N/A",
                   help="Probabilistic Sharpe Ratio — probabilitas Sharpe holdout benar-benar "
                        "> 0 setelah memperhitungkan ketidakpastian estimasi dari jumlah observasi.")
        pc2.metric("E[max Sharpe] di bawah null", f"{_max_null:.2f}",
                   help=f"Setelah {_n_trials} kombinasi dievaluasi GA, ini Sharpe tertinggi yang "
                        f"DIHARAPKAN muncul murni dari noise. best_fitness GA "
                        f"({ga['best_fitness']:.2f}) harus jauh di atas angka ini, bukan cuma > 0.")
        if st.button("🧱 Jalankan plateau test pada parameter GA", key="plateau_btn"):
            with st.spinner("Menguji tetangga parameter (mr_window ±5, mr_z_entry ±0.2)..."):
                _plateau = parameter_plateau_test(
                    price_df, ga["best_params"], composite_signal, run_backtest,
                    backtest_kwargs={"fee_buy_bps": fee_buy_bps, "fee_sell_bps": fee_sell_bps,
                                      "slippage_bps": slippage_bps, "trading_days": trading_days})
            st.dataframe(_plateau, use_container_width=True, hide_index=True)
            st.caption(
                "Parameter robust duduk di 'dataran': Sharpe di tetangga terdekat (offset ±1 "
                "langkah) tidak boleh runtuh jauh dari nilai base (is_base=True). Kalau runtuh "
                "→ spike noise, bukan edge."
            )
        st.caption(
            "Parameter di atas TIDAK otomatis dipakai di tab lain — kalau mau pakai, "
            "salin nilainya secara manual ke slider di sidebar lalu klik ulang "
            "'Jalankan Analisis'. Ini disengaja supaya kamu yang memutuskan, bukan "
            "auto-apply diam-diam."
        )

    st.markdown("---")
    st.markdown("#### 🏷️ Meta-Labeling — ML sebagai filter trade, bukan sinyal (4b)")
    st.caption(
        "Pendekatan López de Prado: sinyal rule-based TETAP yang memutuskan entry; "
        "model ML sekunder cuma menjawab 'trade yang diusulkan ini layak diambil atau "
        "di-skip?'. Berbeda dari sinyal ML yang di-blend ke composite score — di sini "
        "tugasnya terpisah bersih: primary = arah & timing, secondary = filtering."
    )
    if st.button("🏷️ Latih Meta-Labeling Model", key="meta_label_btn"):
        with st.spinner("Membangun sampel trade & validasi walk-forward..."):
            _ml_samples = build_meta_label_samples(
                price_df,
                signal_params=dict(mr_window=mr_window, mr_z_entry=mr_z_entry,
                                    mom_fast=mom_fast, mom_slow=mom_slow,
                                    mr_weight=mr_weight),
                fee_buy_bps=fee_buy_bps, fee_sell_bps=fee_sell_bps,
                slippage_bps=slippage_bps, stop_loss_pct=stop_loss_pct,
                max_holding_days=max_holding_days or 15)
            st.session_state["meta_label_result"] = train_meta_label_model(_ml_samples)
            st.session_state["meta_label_samples"] = _ml_samples

    if "meta_label_result" in st.session_state:
        mlr = st.session_state["meta_label_result"]
        if not mlr["trained"]:
            st.warning(f"⚠️ {mlr['reason']}")
        else:
            oos = mlr["oos"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win rate SEMUA trade", f"{oos['base_win_rate']:.0f}%")
            c2.metric("Win rate trade LOLOS filter",
                      f"{oos['filtered_win_rate']:.0f}%" if oos["filtered_win_rate"] is not None else "N/A")
            c3.metric("Coverage (lolos filter)", f"{oos['coverage']*100:.0f}%")
            c4.metric("BSS meta-model", f"{oos['bss']:.3f}")
            st.caption(
                f"n={oos['n_oos']} trade out-of-sample · avg return/trade tanpa filter "
                f"{oos['base_avg_return']:+.2f}% vs dengan filter "
                f"{oos['filtered_avg_return']:+.2f}%"
                if oos["filtered_avg_return"] is not None else
                f"n={oos['n_oos']} trade out-of-sample"
            )
            if oos["bss"] <= 0:
                st.warning("⚠️ BSS ≤ 0 — filter meta-labeling TIDAK lebih baik dari nebak "
                           "base rate untuk ticker/parameter ini. Jangan dipakai.")
            elif mlr["n_samples"] < 40:
                st.info(f"ℹ️ Baru {mlr['n_samples']} sampel trade — hasil indikatif, "
                        f"belum konklusif (idealnya 40+).")

            # keputusan untuk sinyal TERKINI
            if latest["composite_signal"] == "BUY":
                _feat_now = compute_ml_features(price_df).iloc[-1].to_dict()
                _feat_now["composite_score"] = float(latest["composite_score"])
                _feat_now["mr_zscore"] = float(latest["mr_zscore"])
                _take_proba = mlr["predict_fn"](_feat_now)
                _take = _take_proba >= oos["take_threshold"]
                if _take:
                    st.success(f"🏷️ Sinyal BUY terkini **LOLOS filter** "
                               f"(P(profit) = {_take_proba*100:.0f}%).")
                else:
                    st.error(f"🏷️ Sinyal BUY terkini **DI-SKIP oleh filter** "
                             f"(P(profit) = {_take_proba*100:.0f}% < threshold "
                             f"{oos['take_threshold']*100:.0f}%). Sistem menyarankan "
                             f"TIDAK mengambil trade ini meski sinyalnya BUY.")

# ---- TAB 3: BACKTEST ----
with tab3:
    st.subheader("Backtest (Walk-Forward Anti-Lookahead)")
    result = run_backtest(sig_df, fee_buy_bps=fee_buy_bps, fee_sell_bps=fee_sell_bps,
                           slippage_bps=slippage_bps,
                           position_size_pct=effective_position_size_pct,
                           stop_loss_pct=stop_loss_pct, max_holding_days=max_holding_days,
                           execution_price=execution_price_kw, dynamic_slippage=dynamic_slippage,
                           trading_days=trading_days,
                           max_turnover_participation=max_turnover_participation,
                           idx_realism=idx_realism)
    metrics = result["metrics"]
    if use_confidence_sizing:
        st.caption(
            f"ℹ️ Backtest ini pakai confidence-scaled position size "
            f"(**{effective_position_size_pct*100:.0f}%**, dari base "
            f"{position_size_pct*100:.0f}% di sidebar) — lihat tab Sinyal untuk rinciannya. "
            f"Matikan toggle di sidebar kalau mau balik ke size flat."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{metrics['total_return_pct']:.1f}%",
              f"vs Buy&Hold {metrics['buy_hold_return_pct']:.1f}%")
    c2.metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] else "N/A")
    c3.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.1f}%")
    c4.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%" if metrics['win_rate_pct'] else "N/A",
              f"{metrics['n_trades']} trades")

    _ext = extended_performance_metrics(
        result["equity_curve"], trading_days=trading_days,
        benchmark_close=fetch_benchmark_prices(
            asset_type, symbol, price_df.index,
            exchange_id=exchange_id if asset_type == "crypto" else "binance"))
    if "error" not in _ext:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Sortino", f"{_ext['sortino']:.2f}" if _ext.get("sortino") is not None else "N/A",
                  help="Seperti Sharpe tapi hanya menghukum volatilitas TURUN.")
        e2.metric("Calmar", f"{_ext['calmar']:.2f}" if _ext.get("calmar") is not None else "N/A",
                  help="CAGR / |max drawdown| — return per unit rasa sakit.")
        e3.metric("Beta vs benchmark", f"{_ext['beta']:.2f}" if _ext.get("beta") is not None else "N/A")
        e4.metric("Alpha tahunan", f"{_ext['alpha_annual_pct']:+.1f}%" if _ext.get("alpha_annual_pct") is not None else "N/A",
                  help="Return strategi di luar yang dijelaskan oleh beta terhadap benchmark.")

    _n_sl = metrics.get("n_stop_loss_exits", 0)
    _n_mh = metrics.get("n_max_holding_exits", 0)
    if _n_sl or _n_mh or metrics.get("position_size_pct", 1.0) < 1.0:
        st.caption(
            f"⚙️ {metrics['position_size_pct']*100:.0f}% modal/trade · "
            f"slippage efektif {metrics['effective_slippage_bps']:.0f}bps"
            + (f" · {_n_sl} exit kena stop-loss" if _n_sl else "")
            + (f" · {_n_mh} exit kena batas holding" if _n_mh else "")
        )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=result["equity_curve"].index, y=result["equity_curve"],
                               name="Strategi", line=dict(color="#636efa", width=2)))
    fig4.add_trace(go.Scatter(x=result["buy_hold_curve"].index, y=result["buy_hold_curve"],
                               name="Buy & Hold", line=dict(color="gray", width=1.5, dash="dot")))
    fig4.update_layout(title="Equity Curve: Strategi vs Buy & Hold", height=450)
    st.plotly_chart(fig4, use_container_width=True)

    if len(result["trades"]) > 0:
        st.markdown("#### Log Trade")
        st.dataframe(result["trades"], use_container_width=True)

        robustness = score_backtest_robustness(result["trades"])
        if robustness["is_concentrated"]:
            st.error(
                f"🚨 **Backtest ini didominasi 1 trade outlier**: satu trade "
                f"menyumbang **{robustness['dominant_trade_pct']:.0f}%** dari total return "
                f"positif. Ini classic small-cap backtest trap — hasil kelihatan bagus di "
                f"kertas cuma karena satu lonjakan harga ekstrem yang kebetulan tertangkap "
                f"sinyal, bukan pola yang robust/repeatable. Jangan terlalu percaya win rate "
                f"atau total return di atas sampai kamu cek trade ini satu-satu."
            )
    else:
        st.info("Belum ada trade yang tereksekusi dalam periode ini dengan parameter saat ini.")

    st.markdown("---")
    st.markdown("#### Validasi Walk-Forward (out-of-sample)")
    if fold_test_days is not None:
        st.caption(
            f"Metode: **panjang fold custom** — tiap fold uji {fold_test_days} hari, "
            f"minimal {fold_min_train} hari training sebelum fold pertama. Jumlah fold "
            f"otomatis menyesuaikan panjang data yang tersedia."
        )
    else:
        st.caption(
            f"Metode: **{fold_n_folds} fold tetap** — data dibagi {fold_n_folds} bagian sama "
            f"rata, tiap bagian 70% training / 30% testing."
        )
    st.caption(
        "ℹ️ Sinyal untuk tiap fold diambil dengan memotong (slice) skor composite yang "
        "sudah dihitung sekali untuk seluruh histori di tab Signal — bukan dihitung ulang "
        "dari nol per fold. Ini penting: kalau dihitung ulang dari nol per fold, window "
        "rolling (Mean reversion window / MA lambat) nggak akan sempat 'terisi' pada fold "
        "yang pendek, sinyal jadi macet di HOLD terus. Slicing tetap valid secara anti-"
        "lookahead karena semua indikator di sini backward-looking (nilai di hari T cuma "
        "pernah lihat data sampai hari T, terlepas dari fold mana dia jatuh)."
    )

    fold_rows = []
    for i, (train, test) in enumerate(
        walk_forward_split(price_df, n_folds=fold_n_folds, test_days=fold_test_days,
                            min_train_days=fold_min_train)
    ):
        # IMPORTANT: slice the already-computed full-history sig_df (from tab
        # Signal) instead of recomputing composite_signal on `test` alone.
        # Recomputing fresh on just the test slice starves rolling windows
        # (mr_window up to 120, mom_slow up to 200) of warmup — if the fold
        # is shorter than those windows, EVERY row stays NaN and the signal
        # is stuck at HOLD for the whole fold (0 trades, 0% return — exactly
        # the symptom of always-empty fold results). Since our indicators
        # are all backward-looking (never use future data), slicing the
        # fully-warmed-up signal by date introduces no lookahead — a row's
        # indicator value only ever depended on that row and earlier dates,
        # whether or not those earlier dates happen to fall in "train".
        test_sig = sig_df.loc[test.index]
        r = run_backtest(test_sig, fee_buy_bps=fee_buy_bps, fee_sell_bps=fee_sell_bps,
                          slippage_bps=slippage_bps,
                          position_size_pct=position_size_pct,
                          stop_loss_pct=stop_loss_pct, max_holding_days=max_holding_days,
                          execution_price=execution_price_kw, dynamic_slippage=dynamic_slippage,
                          trading_days=trading_days, idx_realism=idx_realism)
        fold_rows.append({
            "Fold": i + 1,
            "Periode": f"{test.index[0].date()} - {test.index[-1].date()}",
            "Return (%)": round(r["metrics"].get("total_return_pct", 0), 2),
            "Trades": r["metrics"].get("n_trades", 0),
            "Sharpe": round(r["metrics"]["sharpe_ratio"], 2) if r["metrics"].get("sharpe_ratio") else None,
        })
    if fold_rows:
        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)
        if all(row["Trades"] == 0 for row in fold_rows):
            st.warning(
                "⚠️ Semua fold menunjukkan 0 trade. Kemungkinan penyebab: panjang tiap fold "
                "(atur di sidebar 'Panjang tiap fold uji') masih lebih pendek dari window "
                "sinyal kamu (Mean reversion window / MA lambat) — coba perbesar panjang fold, "
                "atau perkecil window sinyal di sidebar, atau data historisnya memang terlalu "
                "flat/pendek untuk memicu sinyal apapun."
            )
    else:
        st.info("Data belum cukup panjang untuk walk-forward split.")

    st.markdown("---")
    st.markdown("#### 📊 Tearsheet Lengkap")
    st.caption(
        "Laporan satu-halaman ala QuantStats — equity curve, drawdown, dan heatmap "
        "return bulanan. Di belakang tombol karena nge-render beberapa chart Plotly "
        "sekaligus (murah di RAM, tapi nggak perlu auto-generate tiap buka tab)."
    )
    if st.button("📈 Buat Tearsheet", key="build_tearsheet_btn"):
        if len(result["trades"]) == 0:
            st.warning("Belum ada trade di periode ini — tearsheet butuh minimal 1 trade.")
        else:
            with st.spinner("Merangkai tearsheet..."):
                tearsheet_html = adv.build_tearsheet_html(
                    symbol, result["equity_curve"], result["buy_hold_curve"],
                    result["trades"], metrics)
            st.session_state["tearsheet_html"] = tearsheet_html

    if "tearsheet_html" in st.session_state:
        st.components.v1.html(st.session_state["tearsheet_html"], height=1000, scrolling=True)
        st.download_button(
            "⬇️ Download tearsheet (HTML)", data=st.session_state["tearsheet_html"],
            file_name=f"tearsheet_{symbol}_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
        )

    st.warning("⚠️ Hasil backtest historis tidak menjamin performa masa depan. Parameter di "
               "sidebar sengaja bisa diubah — coba beberapa kombinasi dan cek konsistensinya "
               "lewat walk-forward di atas, bukan cuma cari parameter yang 'kebetulan' bagus "
               "di satu periode.")

    st.markdown("---")
    st.markdown("#### 🎲 Bootstrap: Seberapa Bisa Dipercaya Win Rate Ini?")
    st.caption(
        "Acak ulang urutan trade yang SAMA (dengan penggantian) ribuan kali, hitung win rate "
        "& return tiap pengacakan, lihat sebarannya. Beda dari walk-forward di atas (yang tes "
        "apakah strategi generalize ke data belum pernah dilihat) — ini tes seberapa besar "
        "angka headline kamu bergantung ke urutan/kebetulan trade spesifik yang terjadi."
    )
    n_bootstrap = st.slider("Jumlah resampling", 100, 5000, 1000, 100, key="n_bootstrap_slider")
    if st.button("🎲 Jalankan Bootstrap", key="run_bootstrap_btn"):
        if len(result["trades"]) < 3:
            st.warning("Butuh minimal 3 trade untuk bootstrap yang bermakna.")
        else:
            with st.spinner(f"Resampling {n_bootstrap}x..."):
                boot = bootstrap_trade_metrics(result["trades"], n_bootstrap=n_bootstrap,
                                                block_size=3)
            st.session_state["bootstrap_result"] = boot
            st.session_state["bootstrap_source_trades"] = result["trades"]
            st.session_state["bootstrap_source_label"] = f"{symbol} (single ticker)"

    if "bootstrap_result" in st.session_state and "error" not in st.session_state["bootstrap_result"]:
        boot = st.session_state["bootstrap_result"]
        _boot_source = st.session_state.get("bootstrap_source_trades", result["trades"])
        st.caption(f"Sumber trade: {st.session_state.get('bootstrap_source_label', symbol)}")
        wr, tr = boot["win_rate"], boot["total_return_pct"]

        if boot.get("n_outliers", 0) > 0:
            st.info(
                f"🔍 Kedeteksi {boot['n_outliers']} trade outlier (return jauh di luar sebaran "
                f"trade lainnya, metode IQR) — jumlah resampling otomatis dinaikkan dari "
                f"{boot['requested_n_bootstrap']} ke **{boot['n_bootstrap']}x** biar estimasi "
                f"persentil-nya tetap stabil (outlier bikin hasil tiap resampling lebih fluktuatif, "
                f"butuh lebih banyak percobaan buat 'meratakan' fluktuasi itu)."
            )
            _outlier_rows = _boot_source.iloc[boot["outlier_trade_indices"]]
            with st.expander(f"Lihat {boot['n_outliers']} trade outlier yang kedeteksi"):
                st.dataframe(_outlier_rows, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.metric("Win rate — median", f"{wr['p50']:.0f}%", f"rentang 90%: {wr['p5']:.0f}–{wr['p95']:.0f}%")
        c2.metric("Total return — median", f"{tr['p50']:.0f}%", f"rentang 90%: {tr['p5']:.0f}–{tr['p95']:.0f}%")

        _wr_width = wr["p95"] - wr["p5"]
        if _wr_width > 40:
            st.error(
                f"🚨 Rentang win rate {wr['p5']:.0f}–{wr['p95']:.0f}% itu LEBAR ({_wr_width:.0f} "
                f"poin) — angka headline-nya kemungkinan besar cuma kebetulan urutan trade, "
                f"bukan pola yang beneran robust. Wajar terjadi kalau jumlah trade masih sedikit "
                f"({boot['n_trades']})."
            )
        elif _wr_width > 20:
            st.warning(f"⚠️ Rentang win rate {wr['p5']:.0f}–{wr['p95']:.0f}% — lumayan lebar, "
                       f"hati-hati terlalu percaya diri sama angka headline-nya.")
        else:
            st.success(f"✅ Rentang win rate {wr['p5']:.0f}–{wr['p95']:.0f}% — relatif stabil "
                       f"across resampling.")

        fig_boot = go.Figure()
        fig_boot.add_trace(go.Histogram(x=boot["win_rate_samples"], nbinsx=40, name="Win rate (%)"))
        fig_boot.add_vline(x=metrics.get("win_rate_pct", 0), line_dash="dash", line_color="red",
                            annotation_text="Win rate historis (1x)")
        fig_boot.update_layout(title=f"Sebaran Win Rate — {boot['n_bootstrap']}x resampling",
                                height=350, xaxis_title="Win rate (%)")
        st.plotly_chart(fig_boot, use_container_width=True)
        st.caption(
            f"Garis merah putus-putus = win rate dari SATU urutan historis yang beneran "
            f"terjadi ({metrics.get('win_rate_pct', 0):.0f}%). Kalau garis itu ada di ujung "
            f"sebaran (bukan di tengah), kemungkinan urutan trade historis kamu itu kebetulan "
            f"lebih bagus/jelek dari rata-rata skenario yang mungkin terjadi."
        )

    st.markdown("---")
    st.markdown("#### 📊 Backtest Multi-Ticker (Sample Lebih Besar)")
    st.caption(
        "Daripada cuma ngandelin trade dari SATU ticker (rawan kekecilan sample-nya), "
        "jalanin parameter signal & backtest yang SAMA ke banyak ticker sekaligus, gabung "
        "semua trade jadi satu sample besar. Ini nguji apakah STRATEGI-nya punya edge "
        "across watchlist — bukan cuma 'apakah ticker ini kebetulan bagus'. Sinyal ML "
        "dilewati di sini (retrain per-ticker across watchlist terlalu lambat) — cuma "
        "signal rule-based (mean-reversion/momentum/Stochastic-BB)."
    )
    st.caption(
        "⚠️ Caveat survivorship bias (3e): universe/preset di sini adalah konstituen "
        "HARI INI — saham yang sudah delisting/gagal tidak ada di daftar, jadi hasil "
        "agregat optimistis secara sistematis. Tidak bisa di-fix gratis secara sempurna; "
        "perlakukan win rate gabungan sebagai batas ATAS, bukan estimasi netral."
    )
    if asset_type == "stock_id":
        _agg_default = ", ".join(_BLUECHIP_PRESET_IDX[:15])
    elif asset_type == "stock_us":
        _agg_default = ", ".join(_FALLBACK_US[:15])
    else:
        _agg_default = symbol

    agg_max_tickers = st.slider("Batasi maksimal ticker", 3, 60 if IS_LOCAL else 30,
                                 15 if IS_LOCAL else 10, 1, key="agg_max_tickers")

    c_fill1, c_fill2 = st.columns([3, 1])
    with c_fill2:
        st.markdown("")  # spacer biar tombol sejajar vertikal sama text area
        st.markdown("")
        if st.button("🔄 Isi otomatis", key="agg_autofill_btn",
                     help="Isi kotak di kiri dengan N ticker (sesuai slider di atas) dari "
                          "preset yang SENGAJA diselang-seling antar sektor — bukan cuma "
                          "ambil N pertama dari daftar yang bisa aja numpuk 1 sektor doang."):
            if asset_type == "stock_id":
                _preset = _SECTOR_DIVERSE_IDX
            elif asset_type == "stock_us":
                _preset = _SECTOR_DIVERSE_US
            else:
                _preset = [symbol]
            _n = min(agg_max_tickers, len(_preset))
            st.session_state["agg_tickers_input"] = ", ".join(_preset[:_n])
            if agg_max_tickers > len(_preset):
                st.warning(f"Preset cuma punya {len(_preset)} ticker unik — sisanya "
                           f"({agg_max_tickers - len(_preset)}) nggak keisi, tambahin manual "
                           f"kalau perlu.")
    with c_fill1:
        agg_tickers_input = st.text_area(
            "Daftar ticker (pisah koma)", value=_agg_default, key="agg_tickers_input",
            help="Default diisi preset blue-chip biar nggak kosong. Klik 'Isi otomatis' di "
                 "kanan buat generate ulang sejumlah slider di atas (diselang-seling sektor), "
                 "atau edit manual sendiri."
        )
    agg_use_min_turnover = st.checkbox("Filter turnover harian minimum", value=(asset_type == "stock_id"),
                                        key="agg_use_min_turnover",
                                        help="Skip ticker illiquid dari sample gabungan — sama seperti "
                                             "filter di tab Screener, biar trade dari saham yang gampang "
                                             "'digerakkan' 1-2 lot nggak ikut mengotori sample.")
    agg_min_turnover = None
    if agg_use_min_turnover:
        if asset_type == "stock_id":
            agg_min_turnover = st.slider("Turnover minimum (Miliar Rp/hari)", 0.5, 20.0, 5.0, 0.5,
                                          key="agg_min_turnover_slider") * 1_000_000_000
        else:
            agg_min_turnover = st.slider("Turnover minimum (Juta $/hari)", 0.1, 20.0, 1.0, 0.1,
                                          key="agg_min_turnover_slider") * 1_000_000
    if st.button("📊 Jalankan Aggregate Backtest", key="run_aggregate_btn"):
        agg_ticker_list = [t.strip().upper() for t in agg_tickers_input.split(",") if t.strip()]
        agg_ticker_list = agg_ticker_list[:agg_max_tickers]
        if len(agg_ticker_list) < 2:
            st.warning("Minimal 2 ticker untuk aggregate backtest.")
        else:
            _signal_params = dict(
                mr_window=mr_window, mr_z_entry=mr_z_entry, mom_fast=mom_fast,
                mom_slow=mom_slow, mr_weight=mr_weight, stochbb_weight=stochbb_weight,
                bb_window=bb_window, bb_std=bb_std, stoch_window=stoch_window,
                stoch_smooth=stoch_smooth,
            )
            _backtest_params = dict(
                fee_buy_bps=fee_buy_bps, fee_sell_bps=fee_sell_bps,
                slippage_bps=slippage_bps,
                position_size_pct=position_size_pct, stop_loss_pct=stop_loss_pct,
                max_holding_days=max_holding_days, execution_price=execution_price_kw,
                dynamic_slippage=dynamic_slippage, trading_days=trading_days,
                max_turnover_participation=max_turnover_participation,
                idx_realism=idx_realism,
            )
            agg_progress = st.empty()
            with st.spinner(f"Backtest {len(agg_ticker_list)} ticker..."):
                agg_result = run_aggregate_backtest(
                    agg_ticker_list, asset_type, {"period": period}, _signal_params,
                    _backtest_params, min_turnover=agg_min_turnover,
                    progress_callback=lambda done, total, tkr: agg_progress.text(
                        f"{done}/{total} ({tkr})"),
                )
            agg_progress.empty()
            st.session_state["aggregate_backtest_result"] = agg_result

    if "aggregate_backtest_result" in st.session_state:
        agg = st.session_state["aggregate_backtest_result"]
        pooled_trades = agg["trades"]
        st.caption(
            f"{agg['n_tickers_ok']} ticker berhasil, {agg['n_tickers_failed']} gagal · "
            f"total {len(pooled_trades)} trade terkumpul (vs {metrics['n_trades']} dari "
            f"ticker {symbol} sendirian)."
        )
        if len(pooled_trades) > 0:
            _agg_win_rate = (pooled_trades["return_pct"] > 0).mean() * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Win rate gabungan", f"{_agg_win_rate:.0f}%")
            c2.metric("Total trade", len(pooled_trades))
            c3.metric("Rata-rata return/trade", f"{pooled_trades['return_pct'].mean():.1f}%")

            st.dataframe(agg["per_ticker"].sort_values("win_rate_pct", ascending=False, na_position="last"),
                         use_container_width=True, hide_index=True)

            if st.button("🎲 Bootstrap sample gabungan ini", key="bootstrap_aggregate_btn"):
                with st.spinner("Resampling..."):
                    agg_boot = bootstrap_trade_metrics(pooled_trades, n_bootstrap=1000,
                                                        block_size=3)
                st.session_state["bootstrap_result"] = agg_boot
                st.session_state["bootstrap_source_trades"] = pooled_trades
                st.session_state["bootstrap_source_label"] = (
                    f"{agg['n_tickers_ok']} ticker gabungan ({len(pooled_trades)} trade)")
                st.info("Hasil bootstrap di section 🎲 di atas sudah diganti pakai sample "
                        "gabungan ini — scroll ke atas untuk lihat.")
        if agg["errors"]:
            with st.expander(f"⚠️ {len(agg['errors'])} ticker gagal"):
                st.dataframe(pd.DataFrame(agg["errors"], columns=["symbol", "error"]),
                             use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 💼 Backtest Level Portofolio (modal bersama)")
    st.caption(
        "Aggregate backtest di atas men-pool trade tapi mengasumsikan tiap ticker all-in "
        "dengan modal sendiri. Di sini: **satu rekening, modal dibagi, maksimal N posisi "
        "konkuren, sinyal BUY di-skip kalau slot penuh** — equity curve yang jauh lebih "
        "jujur untuk strategi multi-ticker. Entry diprioritaskan dari composite score "
        "tertinggi saat slot terbatas."
    )
    pc1, pc2, pc3 = st.columns(3)
    port_capital = pc1.number_input("Modal awal portofolio", min_value=1_000_000.0,
                                     value=100_000_000.0, step=10_000_000.0, key="port_capital")
    port_max_pos = pc2.slider("Maks posisi konkuren", 2, 10, 5, key="port_max_pos")
    port_pos_pct = pc3.slider("% equity per posisi", 5, 50, 20, 5, key="port_pos_pct") / 100.0

    if st.button("💼 Jalankan Portfolio Backtest", key="run_portfolio_btn"):
        _ptickers = [t.strip().upper() for t in agg_tickers_input.split(",") if t.strip()][:agg_max_tickers]
        if len(_ptickers) < 2:
            st.warning("Minimal 2 ticker (pakai daftar yang sama dengan aggregate backtest di atas).")
        else:
            _sig_params = dict(mr_window=mr_window, mr_z_entry=mr_z_entry, mom_fast=mom_fast,
                                mom_slow=mom_slow, mr_weight=mr_weight,
                                stochbb_weight=stochbb_weight, bb_window=bb_window, bb_std=bb_std,
                                stoch_window=stoch_window, stoch_smooth=stoch_smooth)
            _sigs, _fails = {}, []
            _prog = st.empty()
            for _i, _t in enumerate(_ptickers):
                _prog.text(f"Menyiapkan sinyal {_t}... ({_i+1}/{len(_ptickers)})")
                try:
                    _df_t = cached_fetch_data(asset_type, _t, **({"period": period} if asset_type != "crypto"
                                                                   else {"exchange_id": exchange_id,
                                                                          "timeframe": "1d", "limit": 500}))
                    if len(_df_t) >= mr_window + 10:
                        _sigs[_t] = composite_signal(_df_t, **_sig_params)
                except Exception as _e:
                    _fails.append((_t, str(_e)))
            _prog.empty()
            if len(_sigs) < 2:
                st.error(f"Kurang dari 2 ticker berhasil disiapkan. Gagal: {_fails}")
            else:
                with st.spinner("Menjalankan portfolio backtest..."):
                    _port_result = run_portfolio_backtest(
                        _sigs, initial_capital=port_capital, max_positions=port_max_pos,
                        position_pct=port_pos_pct, fee_buy_bps=fee_buy_bps,
                        fee_sell_bps=fee_sell_bps, slippage_bps=slippage_bps,
                        stop_loss_pct=stop_loss_pct, max_holding_days=max_holding_days,
                        trading_days=trading_days,
                        lot_size=100 if (asset_type == "stock_id" and idx_realism) else None,
                        max_turnover_participation=max_turnover_participation)
                st.session_state["portfolio_backtest_result"] = _port_result
                if _fails:
                    st.caption(f"⚠️ {len(_fails)} ticker gagal: " +
                               ", ".join(t for t, _ in _fails))

    if "portfolio_backtest_result" in st.session_state:
        _pr = st.session_state["portfolio_backtest_result"]
        _pm = _pr["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total return portofolio", f"{_pm['total_return_pct']:.1f}%")
        m2.metric("Sharpe", f"{_pm['sharpe_ratio']:.2f}" if _pm.get("sharpe_ratio") else "N/A")
        m3.metric("Max drawdown", f"{_pm['max_drawdown_pct']:.1f}%")
        m4.metric("Sinyal di-skip (slot penuh)", _pm["n_skipped_full_slots"],
                  help="Berapa kali ada sinyal BUY tapi semua slot posisi sedang terisi — "
                       "opportunity cost yang TIDAK terlihat di aggregate backtest biasa.")
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=_pr["equity_curve"].index, y=_pr["equity_curve"],
                                       name="Portofolio strategi", line=dict(color="#636efa", width=2)))
        fig_port.add_trace(go.Scatter(x=_pr["buy_hold_curve"].index, y=_pr["buy_hold_curve"],
                                       name="Equal-weight Buy & Hold", line=dict(color="gray", dash="dot")))
        fig_port.update_layout(title="Equity Curve Portofolio vs Equal-Weight B&H", height=420)
        st.plotly_chart(fig_port, use_container_width=True)
        if len(_pr["trades"]) > 0:
            with st.expander(f"Log trade portofolio ({len(_pr['trades'])} trade)"):
                st.dataframe(_pr["trades"], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📋 Log & Perbandingan Backtest")
    st.caption(
        "Simpan hasil run ini (parameter + metrik) ke log, biar bisa dibandingin lintas "
        "ticker/parameter dari waktu ke waktu — bukan cuma lihat satu run lalu lupa."
    )
    if st.button("💾 Simpan run ini ke log", key="save_backtest_log_btn"):
        _robustness_for_log = score_backtest_robustness(result["trades"]) if len(result["trades"]) > 0 \
            else {"is_concentrated": None, "dominant_trade_pct": None}
        _ml_bss = ml_confidence_detail["bss"] if (use_ml and ml_confidence_detail is not None) else None
        _ml_acc = None
        if use_ml and ml_confidence_detail is not None:
            try:
                _actual = np.asarray(ml_diagnostics["actual"], dtype=float)
                _calib = np.asarray(ml_diagnostics["proba_calibrated"], dtype=float)
                _ml_acc = float((((_calib > 0.5).astype(float)) == _actual).mean() * 100)
            except Exception:
                _ml_acc = None
        append_backtest_log_entry({
            "logged_at": datetime.now().isoformat(), "symbol": symbol, "asset_type": asset_type,
            "mr_window": mr_window, "mr_z_entry": mr_z_entry, "mom_fast": mom_fast,
            "mom_slow": mom_slow, "mr_weight": mr_weight, "stochbb_weight": stochbb_weight,
            "position_size_pct": position_size_pct, "stop_loss_pct": stop_loss_pct,
            "max_holding_days": max_holding_days, "execution_price": execution_price_kw,
            "dynamic_slippage": dynamic_slippage, "use_ml": use_ml,
            "ml_model_type": ml_model_type if use_ml else None,
            "n_trades": metrics["n_trades"], "win_rate_pct": metrics.get("win_rate_pct"),
            "total_return_pct": metrics["total_return_pct"], "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "buy_hold_return_pct": metrics["buy_hold_return_pct"],
            "effective_slippage_bps": metrics.get("effective_slippage_bps"),
            "is_concentrated": _robustness_for_log["is_concentrated"],
            "dominant_trade_pct": _robustness_for_log["dominant_trade_pct"],
            "ml_accuracy_pct": _ml_acc, "ml_bss": _ml_bss,
        })
        st.success("Tersimpan ke log.")

    backtest_log_df = load_backtest_log()
    if len(backtest_log_df) > 0:
        with st.expander(f"Lihat log ({len(backtest_log_df)} run tersimpan)", expanded=False):
            _min_trades_trust = st.slider(
                "Anggap 'kurang bisa dipercaya' kalau jumlah trade di bawah", 1, 30, 10,
                key="log_min_trust",
                help="Win rate dari sample kecil gampang menyesatkan — satu-dua trade "
                     "kebetulan untung bisa bikin win rate kelihatan tinggi padahal cuma "
                     "kebetulan statistik, bukan pola yang beneran berulang."
            )
            log_display = backtest_log_df.copy()
            log_display["cukup_trade?"] = log_display["n_trades"].fillna(0) >= _min_trades_trust
            sort_col = st.selectbox(
                "Urutkan berdasarkan", ["win_rate_pct", "sharpe_ratio", "total_return_pct",
                                        "ml_accuracy_pct", "logged_at"],
                index=0, key="log_sort_col"
            )
            log_display = log_display.sort_values(sort_col, ascending=False, na_position="last")
            st.dataframe(log_display, use_container_width=True, hide_index=True)

            _low_trust = log_display[~log_display["cukup_trade?"]]
            if len(_low_trust) > 0:
                st.caption(
                    f"⚠️ {len(_low_trust)} run punya trade di bawah {_min_trades_trust} — "
                    f"win rate-nya ditandai 'kurang bisa dipercaya' di kolom terakhir, bukan "
                    f"berarti salah, cuma sample-nya kekecilan buat disimpulkan sebagai pola."
                )

            to_delete = st.multiselect(
                "Hapus baris (pilih index)", options=list(range(len(backtest_log_df))),
                key="log_delete_select"
            )
            if to_delete and st.button("🗑️ Hapus baris terpilih", key="delete_log_rows_btn"):
                delete_backtest_log_entries(to_delete)
                st.success("Baris terhapus.")
                st.rerun()

# ---- TAB 4: HUMAN CHECKPOINT & JOURNAL ----
with tab4:
    st.subheader("✅ Checkpoint Manusia — Order Ticket & Journal")
    _journal_backend_active = _journal_backend()
    _journal_storage_desc = ("Google Sheets (cloud, persist lewat hibernasi Streamlit Cloud)"
                              if _journal_backend_active == "gsheets"
                              else "CSV lokal (`trade_journal.csv` — HILANG saat container "
                                   "Streamlit Cloud sleep/reboot; aman kalau dijalankan lokal)")
    st.caption(
        "Sistem cuma mengusulkan aksi + ukuran posisi. **Tidak ada eksekusi otomatis "
        "ke exchange/broker** — kamu yang harus konfirmasi manual di sini, lalu place "
        f"order-nya sendiri di app trading kamu. Konfirmasi di sini cuma mencatat "
        f"keputusan ke jurnal, bukan mengirim order beneran. Backend penyimpanan aktif: "
        f"**{_journal_storage_desc}**."
    )

    latest_row = sig_df.iloc[-1]
    latest_action = latest_row["composite_signal"]
    latest_score = latest_row["composite_score"]
    latest_price = float(latest_row["Close"])
    now_ts = pd.Timestamp.now()

    journal_df = load_journal()
    open_mask = (journal_df["symbol"] == symbol) & (journal_df["status"] == "OPEN")
    open_positions = journal_df[open_mask]

    st.markdown("#### 📋 Ukuran Posisi")
    c1, c2 = st.columns(2)
    capital = c1.number_input("Modal tersedia untuk simbol ini", min_value=0.0,
                               value=10_000_000.0, step=100_000.0,
                               help="Dalam satuan mata uang yang sama dengan harga aset "
                                    "(IDR untuk saham IDX/Indodax, USD untuk saham US, dst).")
    risk_pct = c2.slider("Risiko per posisi (% dari modal)", 1, 100, 10,
                          help="Fixed-fractional sizing sederhana: proporsi modal yang "
                               "dialokasikan ke satu posisi ini. Bukan Kelly criterion — "
                               "makin kecil, makin konservatif.")
    suggested_value = capital * (risk_pct / 100)
    suggested_qty = suggested_value / latest_price if latest_price > 0 else 0

    st.markdown("#### 🎫 Order Ticket")
    if len(open_positions) > 0:
        pos = open_positions.iloc[0]
        pos_idx = open_positions.index[0]
        unrealized_pct = (latest_price / pos["entry_price"] - 1) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Posisi terbuka sejak", pd.Timestamp(pos["entry_timestamp"]).strftime("%Y-%m-%d"))
        c2.metric("Harga entry", f"{pos['entry_price']:,.2f}")
        c3.metric("Unrealized P&L", f"{unrealized_pct:+.2f}%")

        if latest_action in ("SELL", "EXIT"):
            st.success(f"🔴 Sinyal **{latest_action}** terdeteksi — sistem sarankan tutup posisi.")
            st.write(f"**Harga exit usulan:** {latest_price:,.2f} (harga close terakhir)")
            if st.button("✅ Konfirmasi EXIT & Catat ke Journal", type="primary"):
                close_journal_entry(pos_idx, latest_price, now_ts)
                st.success("Dicatat sebagai CLOSED. Jangan lupa eksekusi order jual di "
                           "app trading kamu kalau belum.")
                st.rerun()
        else:
            st.info(f"Posisi masih terbuka. Sinyal saat ini: **{latest_action}** "
                    f"(belum ada sinyal EXIT).")
    else:
        if latest_action == "BUY":
            st.success("🟢 Sinyal **BUY** terdeteksi — belum ada posisi terbuka untuk simbol ini.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Harga entry usulan", f"{latest_price:,.2f}")
            c2.metric("Qty usulan", f"{suggested_qty:,.4f}")
            c3.metric("Nilai posisi", f"{suggested_value:,.0f}")
            notes = st.text_input("Catatan (opsional)", key="entry_notes")
            if st.button("✅ Konfirmasi ENTRY & Catat ke Journal", type="primary"):
                append_journal_entry({
                    "entry_timestamp": now_ts, "symbol": symbol, "asset_type": asset_type,
                    "action_type": "BUY", "entry_price": latest_price,
                    "quantity": suggested_qty, "est_value": suggested_value,
                    "composite_score_at_entry": latest_score,
                    "exit_timestamp": None, "exit_price": None, "return_pct": None,
                    "status": "OPEN", "notes": notes,
                })
                st.success("Dicatat sebagai OPEN. Jangan lupa eksekusi order beli di "
                           "app trading kamu kalau belum.")
                st.rerun()
        else:
            st.info(f"Tidak ada sinyal ENTRY saat ini (sinyal: **{latest_action}**). "
                    "Tidak ada aksi yang disarankan untuk simbol ini sekarang.")

    st.markdown("---")
    st.markdown("#### 📖 Journal (semua simbol)")
    if len(journal_df) > 0:
        display_df = journal_df.sort_values("entry_timestamp", ascending=False).copy()
        st.dataframe(display_df, use_container_width=True)

        closed = journal_df[journal_df["status"] == "CLOSED"]
        if len(closed) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Trade tercatat (closed)", len(closed))
            c2.metric("Win rate", f"{(closed['return_pct'] > 0).mean()*100:.1f}%")
            c3.metric("Avg return/trade", f"{closed['return_pct'].mean():+.2f}%")

        csv_bytes = journal_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download journal (.csv)", csv_bytes, "trade_journal.csv", "text/csv")
    else:
        st.info("Journal masih kosong. Konfirmasi sinyal BUY/SELL di atas untuk mulai mencatat.")

    st.markdown("---")
    st.markdown("#### 📡 Signal Log — Forward Test Otomatis")
    st.caption(
        "Tiap kali analisis dijalankan, score + verdict sistem hari itu dicatat otomatis "
        "(1 baris per simbol per hari). Forward return 5 hari diisi saat kamu klik tombol "
        "update di bawah (setelah datanya cukup umur). **Ini satu-satunya validasi yang "
        "tidak bisa dibantah**: kalau live IC ≈ 0 padahal backtest bagus → backtest overfit."
    )
    if st.button("🔄 Update forward returns", key="update_fwd_returns_btn"):
        with st.spinner("Mengisi forward return untuk baris yang sudah cukup umur..."):
            update_signal_log_forward_returns()
        st.success("Selesai.")
    _siglog = load_signal_log()
    if len(_siglog) > 0:
        st.dataframe(_siglog.sort_values("log_date", ascending=False),
                     use_container_width=True, hide_index=True)
        _filled = _siglog.dropna(subset=["fwd_return_5d_pct"])
        if len(_filled) >= 10:
            _live_ic = _filled["composite_score"].corr(_filled["fwd_return_5d_pct"],
                                                        method="spearman")
            st.metric("Live IC (rank, horizon 5 hari)", f"{_live_ic:.3f}",
                      f"n={len(_filled)} observasi")
            if abs(_live_ic) < 0.02:
                st.error("🚨 Live IC ≈ 0 — sinyal TIDAK menunjukkan edge di data live, "
                         "apa pun kata backtest. Jangan pakai uang ril.")
            else:
                st.success(f"✅ Live IC {_live_ic:+.3f} — ada korelasi terukur antara score "
                           f"dan return ke depan. Tetap lanjutkan paper trading 3-6 bulan.")
        else:
            st.info(f"Baru {len(_filled)} baris punya forward return — butuh ≥10 untuk "
                    f"live IC yang bermakna. Klik update seminggu sekali.")
    else:
        st.info("Signal log masih kosong — otomatis terisi tiap kali kamu menjalankan analisis.")

# ---- TAB 5: SCREENER ----
with tab5:
    st.subheader("🔍 Screener — Cari Sinyal Terkuat")
    st.warning(
        "⚠️ **Ini bukan jaminan profit.** Screener ini cuma nge-rank aset berdasarkan "
        "skor composite yang sama dengan tab Signal (mean-reversion + momentum) — logic "
        "yang sama yang sudah kamu backtest, dengan segala keterbatasannya. Skor tinggi "
        "artinya 'secara historis pola ini pernah mendahului kenaikan', bukan 'dijamin "
        "naik'. Selalu cek likuiditas & fundamental sebelum eksekusi apapun."
    )

    DEFAULT_WATCHLISTS = {
        "stock_id": "BBCA,BBRI,BMRI,BBNI,TLKM,ASII,UNVR,ICBP,ANTM,ADRO,"
                     "PGAS,INDF,KLBF,SMGR,GOTO,MDKA,PTBA,AKRA,CPIN,UNTR",
        "stock_us": "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,NFLX,AMD,JPM",
        "crypto": ("BTC/IDR,ETH/IDR,SOL/IDR,BNB/IDR,XRP/IDR,ADA/IDR,DOGE/IDR"
                   if asset_type == "crypto" and exchange_id == "indodax" else
                   "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,DOGE/USDT"),
    }

    st.markdown("#### 📋 Sumber Watchlist")
    source_mode = st.radio(
        "Pilih sumber simbol", ["Watchlist custom (manual)", "Universe otomatis (semua/banyak simbol)"],
        horizontal=True,
        help="Universe otomatis mengambil daftar lengkap (semua saham IDX, S&P 500, atau "
             "semua pair aktif di exchange) — bukan cuma watchlist kecil yang kamu tulis manual."
    )

    if source_mode == "Watchlist custom (manual)":
        watchlist_str = st.text_area(
            "Watchlist (pisahkan koma)", value=DEFAULT_WATCHLISTS[asset_type], height=70,
            help="Edit bebas — tambah/hapus simbol sesuai yang mau kamu scan. Untuk saham "
                 "IDX cukup kode 4 huruf tanpa .JK (otomatis ditambahkan)."
        )
        watchlist = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]
    else:
        with st.spinner("Mengambil daftar universe..."):
            if asset_type == "stock_id":
                universe = fetch_idx_universe()
                universe_label = f"Seluruh saham IDX ({len(universe)} kode terdeteksi)"
            elif asset_type == "stock_us":
                universe = fetch_sp500_universe()
                universe_label = f"S&P 500 ({len(universe)} kode terdeteksi)"
            else:
                universe = fetch_crypto_universe(exchange_id)
                universe_label = f"Semua pair aktif di {exchange_id} ({len(universe)} pair terdeteksi)"

        is_fallback = (
            (asset_type == "stock_id" and universe == _FALLBACK_IDX) or
            (asset_type == "stock_us" and universe == _FALLBACK_US) or
            (asset_type == "crypto" and universe in (_FALLBACK_CRYPTO_IDR, _FALLBACK_CRYPTO_USDT))
        )
        if is_fallback:
            st.warning(
                f"⚠️ Sumber universe utama gagal diakses, fallback ke daftar kecil bawaan "
                f"({len(universe)} simbol). Coba lagi nanti, atau pakai mode Watchlist custom."
            )
        else:
            st.caption(f"Universe: **{universe_label}**")

        if asset_type == "stock_id" and not is_fallback:
            index_filter = st.radio(
                "Pre-filter sebelum batasi jumlah di bawah",
                ["Semua universe IDX", "Blue-chip preset (~30 kode)"],
                horizontal=True,
                help="Blue-chip preset adalah daftar statis nama large-cap yang lama "
                     "konsisten aktif diperdagangkan — BUKAN pull otomatis dari komposisi "
                     "resmi LQ45/IDX30 (yang BEI review tiap Feb & Agu dan kita tidak punya "
                     "sumber gratis real-time untuk itu). Berguna buat scan cepat & sopan ke "
                     "rate limit, bukan pengganti cek komposisi index resmi kalau itu yang "
                     "kamu butuhkan."
            )
            if index_filter.startswith("Blue-chip"):
                filtered = [t for t in _BLUECHIP_PRESET_IDX if t in universe]
                universe = filtered or _BLUECHIP_PRESET_IDX
                st.caption(f"Universe dipersempit ke **Blue-chip preset ({len(universe)} kode)**.")

        n_universe = len(universe)
        if n_universe == 0:
            st.error("Universe kosong — tidak ada simbol yang bisa di-scan. Coba lagi nanti "
                      "atau pakai mode Watchlist custom.")
            max_scan = 0
        elif n_universe <= 10:
            # slider needs min_value < max_value — with this few symbols (e.g. the
            # 10-symbol US fallback, or a 7-symbol crypto fallback) there's no
            # meaningful range to pick from anyway, so just scan all of them.
            max_scan = n_universe
            st.caption(f"Universe cuma berisi {n_universe} simbol — semuanya akan di-scan "
                       f"(slider batas dilewati karena jumlahnya sudah sekecil ini).")
        else:
            max_scan = st.slider(
                "Batasi maksimal simbol untuk di-scan", 10, n_universe,
                min(300 if IS_LOCAL else 50, n_universe),
                help="Semakin banyak simbol, semakin lama scan-nya (meski sudah paralel). "
                     "Default lebih tinggi di mode Lokal karena nggak ada batasan RAM 1GB "
                     "yang perlu dijaga. Naikkan bertahap kalau mau scan lebih luas."
            )
        watchlist = universe[:max_scan]
        est_min = len(watchlist) * (2.0 if asset_type == "crypto" else 0.5) / 60
        st.caption(f"Akan scan **{len(watchlist)} simbol** — estimasi ~{est_min:.1f} menit "
                   f"(fetch paralel, jadi jauh lebih cepat dari sekuensial).")

    c1, c2 = st.columns(2)
    if asset_type in ("stock_id", "stock_us"):
        scan_period = c1.selectbox("Rentang histori per simbol", ["6mo", "1y", "2y"], index=1)
        scan_kwargs_extra = {"period": scan_period}
    else:
        scan_limit = c1.slider("Candle per simbol", 100, 500, 200)
        scan_kwargs_extra = {"exchange_id": exchange_id, "timeframe": "1d", "limit": scan_limit}
    top_n = c2.slider("Tampilkan top-N kandidat", 3, 15, 5)
    min_turnover_screener = None  # default; bisa di-override di bawah kalau compute_overall_toggle nyala

    st.markdown("#### 🎯 Ranking Overall (Signal + Monte Carlo + Backtest)")
    compute_overall_toggle = st.checkbox(
        "Hitung Overall % per simbol (rank berdasarkan probabilitas kenaikan terbaik)",
        value=True,
        help="Menggabungkan Signal, Monte Carlo (GBM cepat), dan backtest single-period per "
             "simbol jadi satu skor — metodologi sama seperti tab Kesimpulan. Cuma nambah "
             "~0.05 detik komputasi per simbol (fetch data tetap yang paling makan waktu), "
             "jadi tetap ringan meski di-scan ke banyak simbol sekaligus."
    )
    if compute_overall_toggle:
        oc1, oc2 = st.columns(2)
        screener_mc_days = oc1.slider("Horizon Monte Carlo (hari)", 5, 60, 15,
                                       help="Sengaja terpisah dari slider Monte Carlo utama di "
                                            "sidebar — dibatasi lebih kecil karena dihitung untuk "
                                            "SETIAP simbol di watchlist, bukan cuma satu.")
        screener_mc_sims = oc2.slider("Jumlah simulasi MC per simbol", 200, 3000, 1000, 100,
                                       help="Lebih rendah dari slider utama demi kecepatan scan "
                                            "banyak simbol. 1000 sudah cukup stabil untuk ranking.")
        # reuse the same weights set in tab Kesimpulan if already configured, else equal default
        w_signal_s = st.session_state.get("w_signal", 0.34)
        w_mc_s = st.session_state.get("w_mc", 0.33)
        w_bt_s = st.session_state.get("w_bt", 0.33)
        w_fund_s = st.session_state.get("w_fund", 0.0)
        weight_caption = (
            f"Bobot dipakai (sama seperti tab Kesimpulan): Signal {w_signal_s:.2f} · "
            f"Monte Carlo {w_mc_s:.2f} · Backtest {w_bt_s:.2f}"
        )
        if asset_type != "crypto":
            weight_caption += f" · Fundamental {w_fund_s:.2f}"
        weight_caption += ". Ubah di tab 🎯 Kesimpulan kalau mau beda."
        st.caption(weight_caption)

        use_min_turnover = st.checkbox(
            "Filter turnover harian minimum", value=(asset_type == "stock_id"),
            help="Skip simbol yang turnover (Volume × Close) rata-rata hariannya di bawah "
                 "batas — SEBELUM sinyal/ML dihitung, bukan cuma disaring dari hasil akhir. "
                 "Ini yang bikin sinyal ML nggak nongol sama sekali di saham yang harganya "
                 "gampang 'digerakkan' 1-2 lot doang."
        )
        min_turnover_screener = None
        if use_min_turnover:
            if asset_type == "stock_id":
                min_turnover_b = st.slider("Turnover minimum (Miliar Rp/hari)", 0.5, 20.0, 5.0, 0.5,
                                            key="min_turnover_screener_slider")
                min_turnover_screener = min_turnover_b * 1_000_000_000
            else:
                min_turnover_m = st.slider("Turnover minimum (Juta $/hari)", 0.1, 20.0, 1.0, 0.1,
                                            key="min_turnover_screener_slider")
                min_turnover_screener = min_turnover_m * 1_000_000

        use_news_sentiment_screener = st.checkbox(
            "Ikutkan sentimen berita (maks 10 ticker/scan, kuota Marketaux ketat)",
            value=False, key="use_news_sentiment_screener",
            help="Kuota Marketaux cuma 100 request/hari untuk SELURUH app — dibatasi keras "
                 "maks 10 ticker teratas (setelah filter lain) per scan biar nggak abis "
                 "sekali jalan. Ditampilkan sebagai kolom informasional terpisah, BUKAN "
                 "bagian dari Overall %/Verdict (nggak ada arsip berita historis buat "
                 "backtest, jadi ini nggak divalidasi seketat sinyal lain)."
        )

        compute_ml_toggle = st.checkbox(
            "Ikutkan sinyal ML per simbol — LAMBAT, bisa beberapa menit",
            value=True,
            help="Default ON — kamu bilang loading lama nggak masalah asal nggak crash. "
                 "Ini aman karena scan-nya jalan per-batch (lihat 'Batching & Kontrol Scan' "
                 "di bawah), jadi walau lambat, prosesnya nggak pernah 'ngeblok' dalam satu "
                 "panggilan raksasa yang bisa di-kill platform. Matikan manual kalau di sesi "
                 "tertentu kamu justru butuh hasil cepat."
        )
        if compute_ml_toggle:
            screener_ml_model_label = st.selectbox(
                "Model ML untuk scan", ["LightGBM", "XGBoost", "Ensemble (LightGBM + XGBoost)"],
                index=["LightGBM", "XGBoost", "Ensemble (LightGBM + XGBoost)"].index(ml_model_label),
                help="Default-nya ngikutin pilihan Model ML di sidebar, tapi bisa dioverride "
                     "khusus buat scan ini. Ensemble melatih DUA model per simbol (bukan cuma "
                     "sekali untuk keseluruhan scan) — jadi ini benar-benar ~2x lebih lambat "
                     "dari LightGBM/XGBoost sendirian, bukan cuma sedikit lebih lambat."
            )
            screener_ml_model_type = {"LightGBM": "lightgbm", "XGBoost": "xgboost",
                                       "Ensemble (LightGBM + XGBoost)": "ensemble"}[screener_ml_model_label]
            per_ticker_sec = 5.0 if screener_ml_model_type == "ensemble" else 2.5
            est_ml_min = len(watchlist) * per_ticker_sec / 60
            st.warning(
                f"⏳ Dengan ML aktif ({screener_ml_model_label}), estimasi waktu scan "
                f"**{len(watchlist)} simbol ≈ {est_ml_min:.1f} menit** (bukan detik lagi) — "
                f"training model per simbol tidak paralel seefektif fetch data"
                + (", dan Ensemble melatih dua model sekaligus jadi makin lama"
                   if screener_ml_model_type == "ensemble" else "")
                + f". Simbol dengan data terlalu pendek (< {ml_min_train + 50} baris) otomatis "
                  f"fallback ke Signal klasik tanpa ML."
            )
        else:
            screener_ml_model_type = "lightgbm"

        if asset_type == "crypto":
            compute_fund_toggle = False
            st.caption("ℹ️ Fundamental tidak berlaku untuk crypto (tidak ada laporan "
                       "keuangan/earnings) — komponen ini otomatis dilewati.")
        else:
            compute_fund_toggle = st.checkbox(
                "Ikutkan skor Fundamental per simbol — lebih lambat dari fetch harga biasa",
                value=True,
                help="Default ON, sama alasannya dengan toggle ML di atas — fetch yfinance "
                     ".info per simbol + skoring aturan umum (PE, PBV, ROE, DER, dst). Simbol "
                     "yang gagal/data kosong otomatis fallback ke netral 50%, tidak "
                     "menggagalkan scan."
            )
            if compute_fund_toggle:
                est_fund_min = len(watchlist) * 1.0 / 60  # rough estimate ~1s/ticker, 8 workers
                st.warning(
                    f"⏳ Dengan Fundamental aktif, tambahan estimasi waktu **≈ {est_fund_min:.1f} "
                    f"menit** untuk {len(watchlist)} simbol (concurrency dibatasi lebih rendah "
                    f"untuk sopan ke rate limit Yahoo Finance)."
                )
    else:
        compute_ml_toggle = False
        compute_fund_toggle = False
        use_news_sentiment_screener = False  # FIX 1a: dibaca di blok render hasil

    compute_accumulation_toggle = st.checkbox(
        "🔎 Deteksi akumulasi dini (volume/volatility footprint) — eksperimental",
        value=False,
        help="Skor 0-1 per simbol dari 4 komponen OHLCV-only: relative volume, divergensi "
             "OBV vs harga, penyempitan Bollinger Band ('coiling'), dan rasio volume "
             "up-day/down-day. Dirancang buat nyaring simbol yang MUNGKIN lagi diakumulasi "
             "sebelum bergerak besar (sebelum ARA) — bukan sinyal beli, dan BUKAN literally "
             "'melihat bandar' (dashboard ini nggak punya data order book/broker summary)."
    )
    if compute_accumulation_toggle:
        st.warning(
            "⚠️ Skor ini murni pola statistik volume/volatilitas historis — akumulasi asli "
            "dan FOMO ritel murni bisa menghasilkan footprint yang mirip di data ini, dan "
            "skor ini belum divalidasi presisi/recall-nya khusus di small-cap IDX. Pakai "
            "sebagai penyaring buat investigasi lanjut (lihat chart, berita, broker summary "
            "kalau ada), bukan sinyal beli berdiri sendiri. Saham tipis yang jadi target fitur "
            "ini justru saham dengan risiko manipulasi & risiko ARB (nggak ada bid buat keluar) "
            "paling tinggi — skor tinggi itu alasan buat investigasi lebih lanjut, bukan alasan "
            "buat naikkan size."
        )

    st.markdown("#### 📦 Batching & Kontrol Scan")
    st.caption(
        "Scan dijalankan per-batch (bukan 900+ simbol sekaligus dalam satu proses Python) — "
        "ini mencegah Streamlit Community Cloud kena execution timeout / auto-kill di tengah "
        "scan besar, karena UI sempat merender ulang di antara tiap batch. Konsekuensinya: "
        "tombol **Stop Scan** cuma bisa 'nyambar' di antara batch, bukan di tengah satu batch "
        "yang sedang jalan — batch lebih kecil = lebih responsif untuk dihentikan, tapi sedikit "
        "overhead lebih banyak karena start/stop thread pool lebih sering."
    )
    # Ketika ML/Fundamental nyala, tiap simbol jauh lebih berat (retrain model / fetch .info)
    # dibanding cuma fetch harga — batch default diperkecil supaya tiap "giliran" tetap
    # selesai dalam waktu wajar dan UI tetap sempat rerun & cek Stop di antaranya, bukan
    # menunggu 50 simbol berat kelar sekaligus dalam satu batch.
    _heavy_scan = compute_ml_toggle or compute_fund_toggle
    _default_batch = min((30 if IS_LOCAL else 10) if _heavy_scan else (100 if IS_LOCAL else 50),
                         max(10, len(watchlist)))
    batch_size = st.slider("Ukuran batch per giliran", 10, 200, _default_batch, 10,
                            help="Jumlah simbol yang di-scan per rerun Streamlit sebelum UI "
                                 "sempat merender ulang & cek tombol Stop. Default diperkecil "
                                 "otomatis kalau ML/Fundamental nyala, karena tiap simbol jauh "
                                 "lebih berat — biar tiap batch tetap kelar dalam waktu wajar.")

    scan_running = st.session_state.get("scan_queue") is not None
    bc1, bc2 = st.columns([3, 1])
    scan_btn = bc1.button(f"🔍 Scan {len(watchlist)} Simbol", use_container_width=True,
                           disabled=(len(watchlist) == 0 or scan_running))
    stop_btn = bc2.button("⏹️ Stop Scan", use_container_width=True, disabled=not scan_running)

    if stop_btn:
        st.session_state["scan_stop_requested"] = True

    if scan_btn:
        scan_extra_kwargs = {}
        if compute_overall_toggle:
            scan_extra_kwargs = dict(
                compute_overall=True, mc_days=screener_mc_days, mc_sims=screener_mc_sims,
                fee_bps=fee_bps, slippage_bps=slippage_bps,
                fee_buy_bps=fee_buy_bps, fee_sell_bps=fee_sell_bps,
                trading_days=trading_days,
                w_signal=w_signal_s, w_mc=w_mc_s, w_bt=w_bt_s,
                compute_ml=compute_ml_toggle, ml_weight=ml_weight,
                ml_min_train=ml_min_train, ml_retrain_every=ml_retrain_every,
                ml_model_type=screener_ml_model_type, adaptive_ml_weight=adaptive_ml_weight,
                compute_fundamental=compute_fund_toggle, w_fund=w_fund_s,
                min_turnover=min_turnover_screener,
                compute_accumulation=compute_accumulation_toggle,
            )
        else:
            scan_extra_kwargs = dict(compute_overall=False, min_turnover=min_turnover_screener,
                                      compute_accumulation=compute_accumulation_toggle)

        # Seed the batch queue in session_state — frozen at click-time so a
        # long multi-batch scan doesn't drift if the person nudges a slider
        # while it's still running across several reruns.
        st.session_state["scan_queue"] = list(watchlist)
        st.session_state["scan_total"] = len(watchlist)
        st.session_state["scan_batch_results"] = []
        st.session_state["scan_batch_errors"] = []
        st.session_state["scan_batch_diagnostics"] = []
        st.session_state["scan_stop_requested"] = False
        st.session_state["scan_kwargs_extra"] = scan_kwargs_extra
        st.session_state["scan_extra_kwargs"] = scan_extra_kwargs
        st.session_state["scan_batch_size"] = batch_size
        st.rerun()

    queue = st.session_state.get("scan_queue")
    if queue is not None:
        total_n = st.session_state["scan_total"]
        done_n = total_n - len(queue)

        if st.session_state.get("scan_stop_requested") or not queue:
            was_stopped = bool(st.session_state.get("scan_stop_requested")) and bool(queue)
            st.session_state["screener_results"] = st.session_state["scan_batch_results"]
            st.session_state["screener_errors"] = st.session_state["scan_batch_errors"]
            st.session_state["screener_diagnostics"] = st.session_state["scan_batch_diagnostics"]
            st.session_state["scan_queue"] = None
            if was_stopped:
                st.info(f"⏹️ Scan dihentikan manual — {done_n}/{total_n} simbol sempat "
                        f"ter-scan sebelum Stop ditekan. Hasil sebagian tetap ditampilkan "
                        f"di bawah (bukan dibuang).")
        else:
            this_batch_size = st.session_state.get("scan_batch_size", batch_size)
            batch = queue[:this_batch_size]
            progress = st.progress(done_n / total_n if total_n else 0.0,
                                    text=f"Batch scan... ({done_n}/{total_n} simbol)")

            def _update_progress(completed, total, tkr):
                progress.progress(min(1.0, (done_n + completed) / total_n) if total_n else 0.0,
                                   text=f"Scanning {tkr}... ({done_n + completed}/{total_n})")

            batch_results, batch_errors, batch_diag = scan_universe_parallel(
                batch, asset_type, st.session_state["scan_kwargs_extra"],
                mr_window=mr_window, mr_z_entry=mr_z_entry,
                mom_fast=mom_fast, mom_slow=mom_slow, mr_weight=mr_weight,
                progress_callback=_update_progress, **st.session_state["scan_extra_kwargs"],
            )
            st.session_state["scan_batch_results"].extend(batch_results)
            st.session_state["scan_batch_errors"].extend(batch_errors)
            st.session_state["scan_batch_diagnostics"].append(batch_diag)
            st.session_state["scan_queue"] = queue[this_batch_size:]
            st.rerun()

    results = st.session_state.get("screener_results")
    errors = st.session_state.get("screener_errors")
    diag_list = st.session_state.get("screener_diagnostics")

    if diag_list:
        _total_secs = sum(d["total_seconds"] for d in diag_list)
        _prefetch_secs = sum(d["prefetch_seconds"] for d in diag_list)
        _scan_secs = sum(d["scan_seconds"] for d in diag_list)
        _retries = sum(d["rate_limit_retries"] for d in diag_list)
        _backoff_secs = sum(d["seconds_spent_in_backoff"] for d in diag_list)
        _hits = sum(d["batch_prefetch_hits"] for d in diag_list)
        _misses = sum((d["batch_prefetch_misses"] or 0) for d in diag_list)
        with st.expander(f"⏱️ Diagnostik waktu scan (total {_total_secs:.0f}s untuk "
                          f"{sum(d['n_tickers'] for d in diag_list)} simbol)", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Batch pre-fetch harga", f"{_prefetch_secs:.0f}s")
            c2.metric("Tahap scan (signal/ML/fund)", f"{_scan_secs:.0f}s")
            c3.metric("Retry kena rate-limit", f"{_retries}x ({_backoff_secs:.0f}s)")
            if _hits + _misses > 0:
                st.caption(f"Batch fetch harga: {_hits} simbol berhasil sekaligus, "
                           f"{_misses} simbol fallback ke fetch satu-satu.")
            if _retries > 0:
                st.warning(
                    f"⚠️ Kena rate-limit Yahoo Finance {_retries}x selama scan ini "
                    f"(total nunggu backoff {_backoff_secs:.0f}s). Ini kemungkinan "
                    f"penyebab utama scan terasa lambat — coba turunkan jumlah simbol "
                    f"atau matikan toggle ML/Fundamental untuk sementara."
                )
            elif _scan_secs > _prefetch_secs * 2:
                st.info(
                    "ℹ️ Nggak ada retry rate-limit, tapi tahap scan (ML/Fundamental/"
                    "backtest per simbol) jauh lebih lama dari fetch harga — ini "
                    "compute-bound (CPU), bukan network. Matikan toggle ML/Fundamental "
                    "kalau mau lebih cepat, atau ini memang wajar kalau keduanya nyala."
                )

    if results is not None:
        if results:
            res_df_all = pd.DataFrame(results)
            has_overall = "Overall %" in res_df_all.columns
            sort_col = "Overall %" if has_overall else "Score"
            res_df_full = res_df_all.sort_values(sort_col, ascending=False).reset_index(drop=True)

            st.markdown("#### 🎚️ Filter Harga")
            data_min, data_max = float(res_df_full["Harga"].min()), float(res_df_full["Harga"].max())
            fc1, fc2 = st.columns(2)
            min_price = fc1.number_input(
                "Harga minimum", min_value=0.0, value=0.0, step=100.0,
                help="0 = tanpa batas bawah. Contoh: isi 900 untuk cuma tampilkan harga ≥ 900."
            )
            max_price = fc2.number_input(
                "Harga maksimum", min_value=0.0, value=0.0, step=100.0,
                help="0 = tanpa batas atas. Contoh: isi 1000 untuk cuma tampilkan harga ≤ 1000."
            )
            st.caption(f"Rentang harga di hasil scan saat ini: {data_min:,.2f} – {data_max:,.2f}")

            res_df = res_df_full.copy()
            if min_price > 0:
                res_df = res_df[res_df["Harga"] >= min_price]
            if max_price > 0:
                res_df = res_df[res_df["Harga"] <= max_price]
            res_df = res_df.reset_index(drop=True)

            if (min_price > 0 or max_price > 0) and len(res_df) < len(res_df_full):
                st.caption(f"Filter aktif: {len(res_df)} dari {len(res_df_full)} simbol lolos filter harga.")

            if len(res_df) == 0:
                st.warning("Tidak ada simbol yang lolos filter harga ini — coba longgarkan rentangnya.")
            else:
                def _emoji(sig):
                    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig, "⚪")
                res_df["Sinyal"] = res_df["Sinyal"].apply(lambda s: f"{_emoji(s)} {s}")

                verdict_col = "Verdict" if has_overall else "Sinyal"
                if has_overall:
                    res_df["Verdict"] = res_df["Verdict"].apply(lambda s: f"{_emoji(s)} {s}")
                    header = f"#### Top {top_n} Kandidat (Overall % kenaikan tertinggi)"
                    cols_order = ["Simbol", "Harga", "Overall %", "Verdict", "Liquid", "Score", "Sinyal",
                                  "ML", "Fund", "Fund %", "MC %", "Backtest %", "N Trades BT",
                                  "Z-score", "Turnover", "Accum Score", "Accum Flag"]
                    cols_order = [c for c in cols_order if c in res_df.columns]
                    res_df = res_df[cols_order]
                else:
                    header = f"#### Top {top_n} Kandidat (composite score tertinggi)"

                sort_by_accum = False
                if "Accum Score" in res_df.columns:
                    sort_by_accum = st.checkbox(
                        "🔎 Urutkan berdasarkan Accum Score, bukan " +
                        ("Overall %" if has_overall else "Score"),
                        value=False,
                        help="Naikkan simbol dengan skor akumulasi tertinggi ke atas — "
                             "buat nyari kandidat yang MUNGKIN lagi diakumulasi sekarang, "
                             "belum tentu udah punya Overall %/Score tinggi (karena harga "
                             "belum bergerak signifikan)."
                    )
                    if sort_by_accum:
                        res_df = res_df.sort_values("Accum Score", ascending=False, na_position="last")

                if use_news_sentiment_screener:
                    _sentiment_symbols = res_df.head(min(top_n, 10))["Simbol"].tolist()
                    with st.spinner(f"Cek sentimen berita {len(_sentiment_symbols)} ticker teratas..."):
                        _sentiment_map = {}
                        for _sym in _sentiment_symbols:
                            _sym_query = (_sym.upper().replace(".JK", "")
                                          if asset_type == "stock_id" else _sym)
                            _news = fetch_news_sentiment(_sym_query)
                            _sentiment_map[_sym] = (
                                f"{_news['avg_sentiment']:+.2f} ({_news['n_articles']}art)"
                                if _news else "n/a"
                            )
                    res_df["Sentimen"] = res_df["Simbol"].map(_sentiment_map).fillna("—")
                    st.caption(
                        f"📰 Sentimen berita dicek untuk {len(_sentiment_symbols)} ticker "
                        f"teratas saja (batas kuota Marketaux 100/hari) — sisanya ditandai "
                        f"'—'. Informasional, TIDAK ikut menentukan Overall %/Verdict."
                    )

                st.markdown(header + (" — diurutkan Accum Score" if sort_by_accum else ""))
                st.dataframe(res_df.head(top_n), use_container_width=True, hide_index=True)

                n_buy = (res_df[verdict_col].str.contains("BUY")).sum()
                n_sell = (res_df[verdict_col].str.contains("SELL")).sum()
                n_hold = (res_df[verdict_col].str.contains("HOLD")).sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 BUY", n_buy)
                c2.metric("🔴 SELL", n_sell)
                c3.metric("⚪ HOLD", n_hold)

                if not has_overall:
                    st.caption(
                        "💡 Aktifkan 'Hitung Overall %' di atas (sebelum scan) untuk ranking "
                        "berdasarkan probabilitas kenaikan gabungan Signal+MonteCarlo+Backtest, "
                        "bukan cuma composite score sinyal saja."
                    )

                with st.expander(f"Lihat semua {len(res_df)} hasil scan (sudah difilter)"):
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada hasil — semua simbol gagal di-fetch atau datanya kurang.")

        if errors:
            with st.expander(f"⚠️ {len(errors)} simbol gagal/dilewati"):
                for tkr, msg in errors:
                    st.caption(f"**{tkr}**: {msg}")

        st.caption(
            "Catatan: scan ini pakai parameter sinyal yang sama dengan sidebar (Mean "
            "reversion window, Z-score threshold, MA, dst) — ubah di sidebar lalu scan "
            "ulang kalau mau coba parameter lain. ML signal tidak diikutkan di screener "
            "untuk menjaga kecepatan scan banyak simbol sekaligus."
        )
    else:
        st.info("Atur watchlist di atas lalu klik **Scan Watchlist** untuk mulai.")

# ---- TAB 6: OVERALL CONCLUSION ----
# ---- TAB 6: FUNDAMENTAL ----
with tab6:
    st.subheader("📑 Analisis Fundamental")

    if asset_type == "crypto":
        st.info(
            "Fundamental tradisional (PE, ROE, dst) tidak berlaku untuk crypto — "
            "tidak ada laporan keuangan/earnings. Tab ini otomatis dilewati untuk "
            "aset crypto; kalau butuh analisis on-chain/tokenomics, itu di luar "
            "cakupan dashboard ini untuk sekarang."
        )
    else:
        st.warning(
            "⚠️ Skoring di sini pakai **aturan umum generik** (PE rendah = baik, "
            "ROE tinggi = baik, dst) — **TIDAK disesuaikan per sektor**. Bank "
            "secara wajar punya DER tinggi (itu model bisnisnya), saham growth "
            "punya PE tinggi karena investor bayar mahal untuk pertumbuhan masa "
            "depan. Jangan telan skor ini mentah-mentah tanpa konteks sektor."
        )

        full_symbol = symbol if asset_type == "stock_us" else (
            symbol if symbol.upper().endswith(".JK") else symbol.upper() + ".JK"
        )
        with st.spinner(f"Mengambil data fundamental {full_symbol}..."):
            try:
                fund = fetch_fundamentals(full_symbol)
            except Exception as e:
                fund = None
                st.error(f"Gagal mengambil data fundamental: {e}")

        if fund is not None:
            fund_score, fund_subs = score_fundamentals(fund)

            st.markdown("#### Metrik Mentah")
            fc1, fc2, fc3, fc4 = st.columns(4)
            def _fmt(v, suffix=""):
                return f"{v:,.2f}{suffix}" if v is not None else "N/A"
            fc1.metric("PE Ratio", _fmt(fund["PE"]))
            fc2.metric("PBV", _fmt(fund["PBV"]))
            fc3.metric("ROE", _fmt(fund["ROE"]*100 if fund["ROE"] is not None else None, "%"))
            fc4.metric("DER", _fmt(fund["DER"]))
            fc5, fc6, fc7 = st.columns(3)
            fc5.metric("Profit Margin", _fmt(fund["Profit Margin"]*100 if fund["Profit Margin"] is not None else None, "%"))
            fc6.metric("Revenue Growth", _fmt(fund["Revenue Growth"]*100 if fund["Revenue Growth"] is not None else None, "%"))
            fc7.metric("Dividend Yield", _fmt(fund["Dividend Yield"]*100 if fund["Dividend Yield"] is not None else None, "%"))

            n_missing = sum(1 for v in fund.values() if v is None)
            if n_missing > 0:
                st.caption(f"ℹ️ {n_missing} dari {len(fund)} metrik tidak tersedia di Yahoo "
                           f"Finance untuk simbol ini (umum terjadi untuk saham IDX yang "
                           f"lebih kecil) — metrik yang hilang dilewati, bukan dianggap buruk.")

            st.markdown("#### Skor Fundamental")
            if fund_score is not None:
                st.metric("Skor Fundamental Keseluruhan", f"{fund_score*100:.1f}%",
                          help="Rata-rata sub-skor yang tersedia, masing-masing 0-100%.")
                with st.expander("Lihat rincian sub-skor"):
                    for label, sc in fund_subs.items():
                        st.write(f"- **{label}**: {sc*100:.0f}%")
            else:
                st.warning("Tidak ada metrik fundamental yang tersedia sama sekali untuk "
                           "simbol ini — skor fundamental tidak bisa dihitung (akan "
                           "dianggap netral 50% di tab Kesimpulan).")
        else:
            fund_score = None

# ---- TAB 7: OVERALL CONCLUSION ----
# ---- TAB 7: LIVE GAINERS ----
with tab7:
    st.subheader("🔥 Live Gainers — Sedang Naik Sekarang")
    st.caption(
        "Diurutkan murni dari **% perubahan harga saat ini** — BUKAN berdasarkan sinyal, "
        "backtest, atau skor apapun di tab lain. Ini pergerakan harga mentah."
    )
    st.warning(
        "⚠️ **Harga yang lagi naik bukan berarti bagus untuk entry.** Bisa jadi kamu udah "
        "telat (FOMO) — banyak saham/coin yang lagi naik kencang malah lebih berisiko untuk "
        "dibeli di titik itu, bukan lebih aman. Gunakan tab ini buat AWARENESS, gabungkan "
        "dengan tab Signal/Kesimpulan sebelum memutuskan apapun. Data saham delay ~15-20 "
        "menit (standar Yahoo Finance, sama seperti tab lain); data crypto mendekati "
        "real-time tergantung exchange."
    )

    st.caption(
        f"Watchlist dipakai dari tab 🔍 Screener saat ini: **{len(watchlist)} simbol** "
        f"({source_mode})."
    )

    lc1, lc2 = st.columns(2)
    _ = lc1  # top_n now lives inside the fragment for responsive filtering
    auto_refresh = lc2.checkbox("Auto-refresh tiap 30 detik", value=False, key="live_auto_refresh")

    @st.fragment(run_every="30s" if auto_refresh else None)
    def _live_gainers_fragment():
        now_str = pd.Timestamp.now().strftime("%H:%M:%S")
        top_row = st.columns([3, 1])
        top_row[0].caption(f"Update terakhir: {now_str}" + (" · auto-refresh aktif" if auto_refresh else ""))
        top_row[1].button("🔄 Refresh Sekarang", key="live_refresh_btn", use_container_width=True)

        with st.spinner("Mengambil harga terkini..."):
            if asset_type == "crypto":
                quote = "IDR" if exchange_id == "indodax" else "USDT"
                syms = watchlist if source_mode == "Watchlist custom (manual)" else None
                live_results, live_errors = fetch_live_gainers_crypto(exchange_id, quote, symbols=syms)
            else:
                query_list = (
                    [t if t.upper().endswith(".JK") else t.upper() + ".JK" for t in watchlist]
                    if asset_type == "stock_id" else watchlist
                )
                live_results, live_errors = fetch_live_gainers_stocks(query_list)

        if live_results:
            gdf_full = pd.DataFrame(live_results).sort_values("% Perubahan", ascending=False).reset_index(drop=True)
            gdf_full["% Perubahan"] = gdf_full["% Perubahan"].round(2)
            has_volume = "Volume" in gdf_full.columns and gdf_full["Volume"].notna().any()

            st.markdown("#### 🎚️ Filter")
            fc1, fc2, fc3 = st.columns(3)
            min_price = fc1.number_input("Harga minimum", min_value=0.0, value=0.0, step=100.0,
                                          key="live_min_price", help="0 = tanpa batas bawah")
            max_price = fc2.number_input("Harga maksimum", min_value=0.0, value=0.0, step=100.0,
                                          key="live_max_price", help="0 = tanpa batas atas")
            min_change = fc3.number_input(
                "Min % perubahan", value=0.0, step=0.5, key="live_min_change",
                help="Cuma tampilkan yang naik minimal segini % (isi negatif kalau mau "
                     "ikutan lihat yang lagi turun juga)."
            )
            top_n_gainers = st.slider("Tampilkan top-N", 5, 50, 10, key="live_top_n")

            gdf = gdf_full.copy()
            if min_price > 0:
                gdf = gdf[gdf["Harga"] >= min_price]
            if max_price > 0:
                gdf = gdf[gdf["Harga"] <= max_price]
            if min_change != 0.0:
                gdf = gdf[gdf["% Perubahan"] >= min_change]
            gdf = gdf.reset_index(drop=True)

            if (min_price > 0 or max_price > 0 or min_change != 0.0) and len(gdf) < len(gdf_full):
                st.caption(f"Filter aktif: {len(gdf)} dari {len(gdf_full)} simbol lolos.")

            if len(gdf) == 0:
                st.warning("Tidak ada simbol yang lolos filter ini — coba longgarkan rentangnya.")
            else:
                st.markdown(f"#### Top {top_n_gainers} Gainers")
                top_gdf = gdf.head(top_n_gainers).copy()
                if has_volume:
                    top_gdf["Volume"] = top_gdf["Volume"].apply(
                        lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A")
                top_gdf["% Perubahan"] = top_gdf["% Perubahan"].apply(
                    lambda v: f"🟢 +{v:.2f}%" if v > 0 else (f"🔴 {v:.2f}%" if v < 0 else f"⚪ {v:.2f}%")
                )
                st.dataframe(top_gdf, use_container_width=True, hide_index=True)

                n_up = (gdf["% Perubahan"] > 0).sum()
                n_down = (gdf["% Perubahan"] < 0).sum()
                n_flat = (gdf["% Perubahan"] == 0).sum()
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("🟢 Naik", n_up)
                mc2.metric("🔴 Turun", n_down)
                mc3.metric("⚪ Flat", n_flat)

                if not has_volume:
                    st.caption("ℹ️ Data volume tidak tersedia dari sumber ini untuk sebagian/semua "
                               "simbol — kolom Volume dilewati.")

                with st.expander(f"Lihat semua {len(gdf)} hasil (sudah difilter)"):
                    st.dataframe(gdf, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada data — coba watchlist lain, cek koneksi, atau exchange/simbolnya "
                    "mungkin tidak mendukung quote live.")

        if live_errors:
            with st.expander(f"⚠️ {len(live_errors)} simbol gagal/dilewati"):
                for tkr, msg in live_errors:
                    st.caption(f"**{tkr}**: {msg}")

    _live_gainers_fragment()

# ---- TAB 8: OVERALL CONCLUSION ----
with tab8:
    st.subheader("🎯 Kesimpulan Keseluruhan")
    st.warning(
        "⚠️ **Angka di bawah ini adalah rata-rata tertimbang sederhana**, bukan probabilitas "
        "tervalidasi secara statistik. Komponen-komponennya masing-masing sudah dihitung di "
        "tab lain dengan keterbatasannya sendiri — menggabungkannya jadi satu angka TIDAK "
        "menghilangkan keterbatasan itu, cuma meringkasnya. Backtest historis tidak menjamin "
        "performa masa depan. Ini bukan nasihat keuangan."
    )

    # ---- Liquidity check (proxy for real orderbook depth, which we can't get for free) ----
    liq_stats = classify_liquidity(price_df, asset_type)
    liq_threshold = liq_stats["threshold"]
    is_illiquid = liq_stats["tier"] in ("thin", "hard_illiquid")
    is_hard_illiquid = liq_stats["tier"] == "hard_illiquid"

    st.markdown("#### 💧 Cek Likuiditas")
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Rata-rata turnover harian (20 hari)", f"{liq_stats['avg_turnover']:,.0f}")
    lc2.metric("Volume hari ini vs rata-rata",
               f"{liq_stats['volume_vs_avg_ratio']:.2f}x" if liq_stats["volume_vs_avg_ratio"] else "N/A")
    spread_display = (f"{liq_stats['consensus_spread_pct']:.2f}%"
                       if liq_stats["consensus_spread_pct"] is not None else "N/A")
    lc3.metric("Estimasi spread bid-ask (OHLC)", spread_display,
               help="Median dari 3 estimator akademis — EDGE (Ardia/Guidotti/Kroencke, JFE 2024), "
                    "Corwin-Schultz (2012), Roll (1984) — yang mengestimasi bid-ask spread cuma "
                    "dari OHLC, bukan orderbook beneran (yang kita tidak punya akses gratisnya). "
                    "Makin lebar, makin mahal biaya implisit masuk/keluar posisi.")

    if is_hard_illiquid:
        st.error(
            f"⛔ **Likuiditas SANGAT tipis (hard block)** — rata-rata turnover harian "
            f"({liq_stats['avg_turnover']:,.0f}) jauh di bawah threshold `{liq_threshold:,.0f}` "
            f"(< {HARD_ILLIQUID_TURNOVER_RATIO*100:.0f}%-nya), DAN/ATAU "
            f"**{liq_stats['frozen_days_ratio']*100:.0f}% dari 20 hari terakhir sama sekali tidak "
            f"ada pergerakan harga intraday** (High == Low — bukan cuma sepi, tapi hampir tidak "
            f"ada transaksi riil). Ini persis pola *value trap*: sinyal teknikal/ML bisa kelihatan "
            f"bagus di atas kertas, padahal secara riil kamu mungkin tidak bisa masuk/keluar posisi "
            f"tanpa menggerakkan harga sendiri. **Verdict BUY otomatis di-cap ke HOLD** — lihat "
            f"🔒 Liquidity Guard di bawah."
        )
    elif is_illiquid:
        st.warning(
            f"🚨 **Likuiditas tipis** — rata-rata turnover harian ({liq_stats['avg_turnover']:,.0f}) "
            f"di bawah threshold {liq_threshold:,.0f}. Belum cukup parah untuk otomatis nge-block "
            f"verdict, tapi tetap rawan **value trap**: sinyal teknikal bisa kelihatan bagus, tapi "
            f"begitu kamu coba masuk/keluar posisi, harga gampang bergerak melawan kamu sendiri "
            f"karena order-mu sendiri yang gerakin pasar tipis ini. **Kami tidak punya akses data "
            f"orderbook/bid-ask depth beneran (butuh data premium seperti Stockbit Pro/RTI)** — "
            f"turnover dan estimasi spread di atas cuma proxy statistik, bukan pengganti cek "
            f"langsung ketebalan bid/ask di aplikasi trading kamu."
        )
    elif liq_threshold is not None:
        st.success(f"✅ Turnover harian di atas threshold ({liq_threshold:,.0f}) — likuiditas "
                   f"terlihat cukup berdasarkan proxy volume dan spread ini.")
    else:
        st.caption("Cek likuiditas dilewati untuk crypto (bervariasi terlalu besar antar exchange "
                   "untuk satu threshold yang masuk akal).")

    latest_score = float(sig_df.iloc[-1]["composite_score"])
    vote_signal = (latest_score + 1) / 2  # -1..+1 -> 0..1

    vote_mc = float(summary["prob_profit"])  # already 0..1

    bt_metrics = result["metrics"]
    win_rate = bt_metrics.get("win_rate_pct")
    n_trades_bt = bt_metrics.get("n_trades", 0)
    if win_rate is not None and n_trades_bt >= 3:
        vote_bt = win_rate / 100
        bt_note = f"Win rate historis: {win_rate:.1f}% dari {n_trades_bt} trade tercatat"
    else:
        vote_bt = 0.5
        bt_note = (f"Cuma {n_trades_bt} trade tercatat di backtest (terlalu sedikit untuk "
                   f"dipercaya) — komponen ini dianggap netral (50%)")

    has_fundamental = asset_type != "crypto"
    if has_fundamental:
        if fund_score is not None:
            vote_fund = fund_score
            fund_note = f"Skor fundamental dari tab 📑 Fundamental: {fund_score*100:.1f}%"
        else:
            vote_fund = 0.5
            fund_note = "Data fundamental tidak tersedia — komponen ini dianggap netral (50%)"

    st.markdown("#### Komponen Penyusun")
    cols = st.columns(4 if has_fundamental else 3)
    cols[0].metric("📈 Signal", f"{vote_signal*100:.0f}%",
                    help="Mean-reversion + Momentum + ML (kalau aktif), dinormalisasi dari composite score.")
    cols[1].metric("🎲 Monte Carlo", f"{vote_mc*100:.0f}%",
                    help=f"Probabilitas harga di atas level sekarang setelah {mc_days} hari simulasi.")
    cols[2].metric("🔁 Backtest", f"{vote_bt*100:.0f}%", help=bt_note)
    if has_fundamental:
        cols[3].metric("📑 Fundamental", f"{vote_fund*100:.0f}%", help=fund_note)

    st.markdown("#### Bobot Tiap Komponen")
    st.caption("Default sama rata — geser kalau kamu lebih percaya salah satu komponen dibanding yang lain.")

    regime, ann_vol = classify_volatility_regime(price_df, trading_days=trading_days)
    with st.expander(f"🔄 Sarankan bobot berdasarkan regime volatilitas (saat ini: {regime})"):
        st.caption(
            f"Volatilitas historis (20 hari, tahunan): **{ann_vol*100:.1f}%** → regime: **{regime}**. "
            f"Ini heuristik sederhana (threshold tunggal 40%), bukan model regime-switching "
            f"yang divalidasi — anggap sebagai starting point, bukan aturan pasti. Klik tombol "
            f"untuk isi ulang slider bobot di bawah, tetap bisa kamu geser manual setelahnya."
        )

        def _apply_regime_weights():
            if regime == "Tinggi / Trending":
                # feedback: trending/high-vol -> boost Signal & Backtest, reduce Fundamental
                st.session_state["w_signal"] = 0.35
                st.session_state["w_bt"] = 0.35
                st.session_state["w_mc"] = 0.20
                st.session_state["w_fund"] = 0.10
            else:
                # feedback: sideways/low-vol -> boost Monte Carlo, reduce Fundamental & Signal
                st.session_state["w_signal"] = 0.20
                st.session_state["w_mc"] = 0.40
                st.session_state["w_bt"] = 0.25
                st.session_state["w_fund"] = 0.15

        st.button(f"Terapkan bobot untuk regime '{regime}'", on_click=_apply_regime_weights,
                  disabled=not has_fundamental,
                  help=None if has_fundamental else "Preset ini didesain untuk 4 komponen "
                                                       "(termasuk Fundamental) — tidak tersedia untuk crypto.")

    if has_fundamental:
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_signal = wc1.slider("Bobot Signal", 0.0, 1.0, 0.25, 0.05, key="w_signal")
        w_mc = wc2.slider("Bobot Monte Carlo", 0.0, 1.0, 0.25, 0.05, key="w_mc")
        w_bt = wc3.slider("Bobot Backtest", 0.0, 1.0, 0.25, 0.05, key="w_bt")
        w_fund = wc4.slider("Bobot Fundamental", 0.0, 1.0, 0.25, 0.05, key="w_fund")
    else:
        wc1, wc2, wc3 = st.columns(3)
        w_signal = wc1.slider("Bobot Signal", 0.0, 1.0, 0.34, 0.05, key="w_signal")
        w_mc = wc2.slider("Bobot Monte Carlo", 0.0, 1.0, 0.33, 0.05, key="w_mc")
        w_bt = wc3.slider("Bobot Backtest", 0.0, 1.0, 0.33, 0.05, key="w_bt")
        w_fund = 0.0

    total_w = w_signal + w_mc + w_bt + w_fund

    if total_w == 0:
        st.error("Total bobot tidak boleh 0 — geser minimal satu slider di atas 0.")
    else:
        weighted_sum = vote_signal * w_signal + vote_mc * w_mc + vote_bt * w_bt
        votes = [vote_signal, vote_mc, vote_bt]
        if has_fundamental:
            weighted_sum += vote_fund * w_fund
            votes.append(vote_fund)
        overall = weighted_sum / total_w
        spread = max(votes) - min(votes)

        if overall >= 0.55:
            verdict, color = "BUY", "🟢"
        elif overall <= 0.45:
            verdict, color = "SELL", "🔴"
        else:
            verdict, color = "HOLD", "⚪"

        # ---- Guard stacking: Z-score overbought guard + Liquidity hard-block guard ----
        # Per feedback: don't let Fundamental/Backtest dominate into a BUY when
        # the stock is already statistically overbought relative to its own
        # recent history, OR when it's essentially not trading (TIRT case) —
        # cap the verdict regardless of how high Overall % is. Both guards are
        # independent and stack in the label if both fire simultaneously.
        raw_verdict = verdict
        latest_z = sig_df.iloc[-1]["mr_zscore"]
        z_guard_triggered = raw_verdict == "BUY" and pd.notna(latest_z) and latest_z >= mr_z_entry
        liq_guard_triggered = raw_verdict == "BUY" and is_hard_illiquid

        guard_names = (["Z-Guard"] if z_guard_triggered else []) + \
                      (["Liquidity-Guard"] if liq_guard_triggered else [])
        if guard_names:
            verdict, color = f"HOLD ({' + '.join(guard_names)})", "🟡"

        st.markdown("---")
        st.markdown(f"## {color} Kesimpulan: **{verdict}**")
        st.markdown(f"### Kecenderungan bullish: **{overall*100:.1f}%**")

        if z_guard_triggered:
            st.warning(
                f"🟡 **Z-Score Overbought Guard aktif**: Overall score menunjukkan {overall*100:.1f}% "
                f"(secara mentah masuk kategori BUY), TAPI Z-score saat ini `{latest_z:.2f}` sudah "
                f"di atas threshold overbought (`{mr_z_entry:.2f}`) — harga secara statistik sudah "
                f"jauh di atas rata-ratanya sendiri. Verdict otomatis diturunkan ke HOLD supaya kamu "
                f"nggak kejebak beli di puncak. Kalau kamu tetap yakin, ini keputusan sadar kamu "
                f"sendiri, bukan sistem yang bilang aman."
            )

        if liq_guard_triggered:
            st.warning(
                f"🔒 **Liquidity Guard aktif** (fix untuk kasus TIRT): Overall score menunjukkan "
                f"{overall*100:.1f}% (secara mentah masuk kategori BUY), TAPI turnover harian "
                f"({liq_stats['avg_turnover']:,.0f}) jauh di bawah threshold likuiditas DAN/ATAU "
                f"{liq_stats['frozen_days_ratio']*100:.0f}% dari 20 hari terakhir harganya sama "
                f"sekali tidak bergerak intraday. Verdict otomatis diturunkan ke HOLD supaya "
                f"skor komposit yang tinggi di atas kertas nggak disalahartikan sebagai sinyal "
                f"yang bisa dieksekusi di pasar yang nyaris tidak ada transaksinya ini."
            )

        n_comp = len(votes)
        if spread > 0.4:
            st.warning(
                f"⚠️ Komponen-komponen ini **saling berlawanan arah** (rentang {spread*100:.0f} "
                f"poin persentase antar {n_comp} komponen) — sinyal saat ini tidak solid/konsisten. "
                f"Angka overall di atas kurang bisa diandalkan ketika komponen-komponennya "
                f"saling bertentangan seperti ini; pertimbangkan untuk HOLD dulu sampai lebih jelas."
            )
        elif spread > 0.2:
            st.info(f"ℹ️ Ada sedikit perbedaan arah antar komponen (rentang {spread*100:.0f} poin persentase).")
        else:
            st.success(f"✅ Komponen-komponen ini relatif **sepakat arah** (rentang cuma {spread*100:.0f} poin persentase).")

        with st.expander("Lihat detail perhitungan"):
            detail_md = f"""
- **Signal** composite score saat ini: `{latest_score:.3f}` → dinormalisasi jadi `{vote_signal*100:.1f}%`
- **Monte Carlo** P(profit) setelah {mc_days} hari: `{vote_mc*100:.1f}%`
- **Backtest**: {bt_note} → `{vote_bt*100:.1f}%`"""
            if has_fundamental:
                detail_md += f"\n- **Fundamental**: {fund_note} → `{vote_fund*100:.1f}%`"
            calc_str = f"({w_signal:.2f}×{vote_signal*100:.0f}% + {w_mc:.2f}×{vote_mc*100:.0f}% + {w_bt:.2f}×{vote_bt*100:.0f}%"
            if has_fundamental:
                calc_str += f" + {w_fund:.2f}×{vote_fund*100:.0f}%"
            calc_str += f") / {total_w:.2f} = **{overall*100:.1f}%**"
            detail_md += f"\n- **Overall** = {calc_str}"
            st.markdown(detail_md)

        st.error(
            "🚨 Sekali lagi: ini adalah **agregasi heuristik**, bukan model probabilitas yang "
            "divalidasi. Komponen-komponen ini juga tidak independen satu sama lain (Signal "
            "dan Backtest sama-sama berasal dari logic composite yang sama), jadi 'kesepakatan' "
            "antar komponen tidak seharusnya diartikan sebagai bukti independen yang saling "
            "menguatkan. Gunakan ini sebagai ringkasan cepat, bukan pengganti analisis dan "
            "risk management kamu sendiri."
        )
        st.caption(
            "ℹ️ Catatan (2e): `vote_mc` (prob_profit GBM) pada dasarnya fungsi deterministik "
            "dari tren & vol terakhir — ia double-count dengan momentum di `vote_signal`, dan "
            "`vote_bt` berasal dari logic composite yang sama. Tiga 'suara' ini berkorelasi "
            "tinggi; rata-ratanya terlihat seperti konsensus padahal bukan bukti independen."
        )

        st.markdown("---")
        st.markdown("#### 🧪 Meta-Model Tervalidasi (Alternatif Eksperimental)")
        st.caption(
            "Ganti bobot 0.25/0.25/0.25 yang ditebak di atas dengan logistic regression yang "
            "BENERAN dilatih dari histori: 'kombinasi Signal+MC+Backtest hari itu, apakah "
            "harga beneran naik dalam N hari ke depan?' — divalidasi walk-forward, dikalibrasi "
            "pakai Brier Skill Score (standar sama yang dipakai sinyal ML di tab Signal). "
            "**Fundamental sengaja dikeluarkan** — nggak ada data fundamental historis "
            "point-in-time yang tersedia (yfinance cuma kasih snapshot sekarang), jadi nggak "
            "bisa divalidasi dengan cara ini."
        )
        meta_horizon = st.slider("Horizon prediksi (hari)", 5, 40, 15, 1, key="meta_horizon")
        if st.button("🧪 Latih & Validasi Meta-Model", key="train_meta_model_btn"):
            _meta_signal_params = dict(
                mr_window=mr_window, mr_z_entry=mr_z_entry, mom_fast=mom_fast,
                mom_slow=mom_slow, mr_weight=mr_weight,
            )
            with st.spinner("Melatih & memvalidasi (walk-forward)..."):
                meta_result = train_meta_model(price_df, _meta_signal_params,
                                                horizon_days=meta_horizon)
            st.session_state["meta_model_result"] = meta_result

        if "meta_model_result" in st.session_state:
            meta = st.session_state["meta_model_result"]
            if not meta["trained"]:
                st.warning(f"⚠️ Nggak bisa dilatih: {meta['reason']}")
            else:
                cal = meta["calibration"]
                bss = cal["bss"]
                bss_ci = cal.get("bss_ci", {})
                bss_sig = cal.get("bss_significant", False)
                meta_pred = meta["predict_fn"](vote_signal, vote_mc, vote_bt)

                c1, c2, c3 = st.columns(3)
                c1.metric("Prediksi meta-model", f"{meta_pred*100:.0f}%",
                          help="P(harga lebih tinggi) menurut logistic regression, "
                               "sudah dikalibrasi — bandingkan sama Overall heuristik di atas.")
                if bss_ci.get("p5") is not None:
                    c2.metric("Brier Skill Score", f"{bss:.3f}",
                              f"CI90%: [{bss_ci['p5']:.3f}, {bss_ci['p95']:.3f}]",
                              help="0 = sama persis kayak nebak pakai rata-rata historis. CI "
                                   "dari block bootstrap (blok berurutan, bukan baris random "
                                   "— karena target N-hari-ke-depan itu overlap antar hari "
                                   "berturut-turut, resampling baris biasa TERLALU percaya "
                                   "diri). Kalau CI meliputi 0, jangan anggap ini edge beneran.")
                else:
                    c2.metric("Brier Skill Score", f"{bss:.3f}",
                              help="CI nggak bisa dihitung (sample OOS terlalu sedikit).")
                c3.metric("Sample", f"{meta['n_samples']}", f"{meta['n_oos_predictions']} out-of-sample")

                if bss_ci.get("p5") is not None:
                    if bss_sig:
                        st.success(
                            f"✅ CI 90% ({bss_ci['p5']:.3f} sampai {bss_ci['p95']:.3f}) **seluruhnya "
                            f"di atas nol** — kombinasi 3 komponen ini kemungkinan beneran punya "
                            f"skill prediktif, bukan cuma kebetulan statistik. Masih model sederhana "
                            f"(3 fitur, sample terbatas) — bukan jaminan profit, tapi ini standar "
                            f"pembuktian yang jauh lebih ketat dari sekadar 'BSS positif'."
                        )
                    else:
                        st.info(
                            f"ℹ️ CI 90% ({bss_ci['p5']:.3f} sampai {bss_ci['p95']:.3f}) **meliputi "
                            f"nol** — walau titik estimasi BSS positif ({bss:.3f}), secara statistik "
                            f"ini TIDAK bisa dibedakan dari kebetulan/noise. Ini hasil yang jujur, "
                            f"bukan bug — dengan overlap antar hari yang tinggi (target {meta['horizon_days']} "
                            f"hari ke depan dihitung tiap hari), sample independen efektifnya jauh "
                            f"lebih sedikit dari jumlah baris yang kelihatan."
                        )
                else:
                    st.warning("⚠️ CI nggak bisa dihitung — treat titik estimasi BSS di atas "
                               "dengan skeptis, jangan dianggap terbukti.")

                _meta_run_count = st.session_state.get("meta_model_run_count", 0) + 1
                st.session_state["meta_model_run_count"] = _meta_run_count
                if _meta_run_count > 1:
                    st.warning(
                        f"⚠️ **Peringatan multiple-testing**: kamu udah coba {_meta_run_count}x "
                        f"kombinasi (ticker/parameter/horizon) di sesi ini. Makin banyak yang "
                        f"dicoba, makin besar kemungkinan salah satu 'kelihatan signifikan' murni "
                        f"karena kebetulan — bahkan di data acak murni, ~5-10% percobaan bakal "
                        f"lolos ambang CI ini cuma karena keberuntungan. Jangan cuma pegang hasil "
                        f"terbaik dari banyak percobaan tanpa validasi tambahan (data/periode lain)."
                    )

                with st.expander("Lihat koefisien model"):
                    coef = meta["coefficients"]
                    st.write(f"Intercept: `{coef['intercept']:.3f}`")
                    for k in ["vote_signal", "vote_mc", "vote_bt"]:
                        st.write(f"- {k}: `{coef[k]:+.3f}`")
                    st.caption(
                        "Koefisien positif = makin tinggi komponen itu, makin besar prediksi "
                        "'harga naik'. Koefisien mendekati 0 = model belajar komponen itu "
                        "nggak terlalu informatif untuk simbol/parameter ini — beda dari "
                        "heuristik di atas yang MEMAKSA semua komponen dapat bobot sama besar "
                        "nggak peduli seberapa informatif komponen itu sebenarnya."
                    )

        # ---- Scalp vs Swing dual score ----
        # Per feedback: a single "BUY" hides the fact that fast-reacting signals
        # (momentum/ML/volume) and slow-reacting signals (fundamental/backtest/
        # Monte Carlo) can point different directions depending on your holding
        # horizon. Split into two separate scores instead of forcing one verdict.
        st.markdown("---")
        st.markdown("#### ⚡ Scalp/Day-Trade vs 📊 Swing/Position — Skor Terpisah")
        st.caption(
            "Bukan cuma satu 'BUY' — dipecah jadi dua skor sesuai horizon waktu, karena "
            "sinyal cepat (momentum/ML/volume) dan sinyal lambat (fundamental/backtest/"
            "Monte Carlo) bisa menunjuk arah berbeda. **Catatan jujur**: idealnya skor scalp "
            "pakai data orderbook/bid-ask real-time dan bandarmology (net buy/sell broker) — "
            "kita tidak punya akses data itu (butuh provider berbayar), jadi 'Scalp Score' di "
            "sini cuma proxy dari momentum + Z-score + lonjakan volume, BUKAN order-flow "
            "beneran. Kurang cocok dipakai untuk keputusan scalping sungguhan."
        )

        vol_ratio = liq_stats["volume_vs_avg_ratio"]
        vote_volume_spike = min(1.0, vol_ratio / 3) if vol_ratio is not None else 0.5
        vote_z_directional = max(0.0, min(1.0, (-latest_z / 3 + 1) / 2)) if pd.notna(latest_z) else 0.5

        scalp_score = (vote_signal + vote_volume_spike + vote_z_directional) / 3
        # Swing: fundamental + backtest + monte carlo only (drop technical signal),
        # per the feedback's suggested composition — renormalized among available parts.
        swing_parts = [vote_bt, vote_mc] + ([vote_fund] if has_fundamental else [])
        swing_score = sum(swing_parts) / len(swing_parts)

        sc1, sc2 = st.columns(2)
        with sc1:
            scalp_verdict = "BUY" if scalp_score >= 0.55 else "SELL" if scalp_score <= 0.45 else "NEUTRAL"
            st.metric("⚡ Scalp/Day-Trade Score", f"{scalp_score*100:.1f}%", scalp_verdict)
            st.caption(f"Komponen: Signal {vote_signal*100:.0f}% · Volume spike "
                       f"{vote_volume_spike*100:.0f}% · Z-score arah {vote_z_directional*100:.0f}%")
        with sc2:
            swing_verdict = "BUY" if swing_score >= 0.55 else "SELL" if swing_score <= 0.45 else "NEUTRAL"
            st.metric("📊 Swing/Position Score", f"{swing_score*100:.1f}%", swing_verdict)
            fund_part = f" · Fundamental {vote_fund*100:.0f}%" if has_fundamental else ""
            st.caption(f"Komponen: Backtest {vote_bt*100:.0f}% · Monte Carlo "
                       f"{vote_mc*100:.0f}%{fund_part}")

        if scalp_verdict != swing_verdict and "NEUTRAL" not in (scalp_verdict, swing_verdict):
            st.info(
                f"ℹ️ Scalp ({scalp_verdict}) dan Swing ({swing_verdict}) **berbeda arah**. Ini "
                f"normal dan justru informatif — misalnya kondisi lagi 'BUY' untuk akumulasi "
                f"jangka menengah (fundamental/backtest bagus) tapi 'SELL/NEUTRAL' untuk momentum "
                f"jangka pendek (lagi dalam fase koreksi teknikal). Pilih skor yang sesuai gaya "
                f"trading kamu, jangan campur keduanya."
            )

    st.markdown("---")
    st.markdown("#### 🤖 Ringkasan Naratif (AI)")
    st.caption(
        "Menggabungkan semua tab di atas jadi 1-2 paragraf bahasa manusia. Coba GEMINI_API_KEY "
        "dulu (gratis, di Secrets), lalu GROQ_API_KEY (gratis, kuota terpisah — fallback kalau "
        "Gemini kehabisan kuota harian), lalu ANTHROPIC_API_KEY kalau ada (berbayar kecil per klik), "
        "baru fallback ke ringkasan berbasis template (gratis, cuma kurang mengalir) kalau "
        "dua-duanya nggak ada/gagal — di belakang tombol biar kamu yang kontrol kapan API "
        "kepanggil."
    )
    if st.button("✍️ Buat Ringkasan", key="ai_narrative_btn"):
        robustness_n = score_backtest_robustness(result["trades"])
        signal_row_n = {"composite_signal": sig_df.iloc[-1]["composite_signal"],
                         "composite_score": latest_score}
        fund_for_narrative = fund if (has_fundamental and fund_score is not None) else None
        with st.spinner("Merangkai narasi..."):
            narrative = adv.generate_ai_narrative(
                symbol, signal_row_n, summary, bt_metrics, robustness_n,
                fundamentals=fund_for_narrative,
            )
        st.session_state["ai_narrative"] = narrative

    if "ai_narrative" in st.session_state:
        st.markdown(st.session_state["ai_narrative"])


# ==========================================================================
# ==== SECTION 5.6: PORTFOLIO OPTIMIZATION (MARKOWITZ) ====
# ==========================================================================
# New tab per feedback: dashboard sejauh ini fokus ke satu aset per waktu.
# Ini melengkapi dengan sisi alokasi ANTAR aset — ambil beberapa kandidat
# terbaik (mis. dari hasil Screener) dan cari bobot yang efisien secara
# risk/return, murni pakai aljabar matriks NumPy (inverse kovarians +
# eigen-decomposition) — jauh lebih ringan dibanding menjalankan ratusan
# simulasi tambahan, jadi tetap aman untuk batas memori Streamlit Cloud.

# ---- TAB 9: PORTFOLIO OPTIMIZATION (MARKOWITZ) ----
with tab9:
    st.subheader("📐 Portofolio Optimization — Markowitz / Efficient Frontier")
    st.warning(
        "⚠️ Ini historical mean-variance optimization (Markowitz, 1952) klasik — TIDAK "
        "memperhitungkan biaya transaksi, pajak, ukuran lot, atau likuiditas per aset "
        "(cek dulu tab 🎯 Kesimpulan masing-masing simbol untuk itu). Return & kovarians "
        "historis dipakai sebagai proxy untuk masa depan, padahal keduanya TIDAK stabil "
        "dari waktu ke waktu — bobot 'optimal' di sini gampang berubah signifikan beberapa "
        "bulan ke depan. Anggap sebagai alat eksplorasi trade-off risk/return antar aset "
        "yang sudah kamu screen, bukan alokasi final yang harus diikuti mentah-mentah."
    )

    st.markdown("#### 🎯 Pilih Aset")
    screener_res = st.session_state.get("screener_results")
    if screener_res:
        sdf_p = pd.DataFrame(screener_res)
        sort_col_p = "Overall %" if "Overall %" in sdf_p.columns else "Score"
        default_port_tickers = sdf_p.sort_values(sort_col_p, ascending=False)["Simbol"].head(5).tolist()
        st.caption(f"Default terisi dari top-5 hasil tab 🔍 Screener terakhir (diurutkan "
                   f"berdasarkan {sort_col_p}). Edit bebas di bawah.")
    else:
        default_port_tickers = DEFAULT_WATCHLISTS[asset_type].split(",")[:5]
        st.caption("Belum ada hasil dari tab 🔍 Screener — default diisi dari watchlist "
                   "bawaan. Edit bebas, atau jalankan Screener dulu buat kandidat yang "
                   "lebih relevan (mis. yang lolos Liquidity Guard).")

    port_tickers_str = st.text_area(
        "Simbol untuk dioptimasi (pisahkan koma — 3-8 simbol disarankan; lebih dari itu, "
        "matriks kovarians makin sulit diestimasi stabil dari histori terbatas)",
        value=",".join(default_port_tickers), height=60,
    )
    port_tickers = [s.strip().upper() for s in port_tickers_str.split(",") if s.strip()]

    pmc1, pmc2 = st.columns(2)
    port_period = pmc1.selectbox("Rentang histori", ["6mo", "1y", "2y"], index=1,
                                  disabled=(asset_type == "crypto"),
                                  help="Tidak berlaku untuk crypto — pakai jumlah candle di "
                                       "sidebar sebagai gantinya.")
    rf_rate_annual = pmc2.number_input(
        "Risk-free rate tahunan (%, untuk Sharpe ratio)", 0.0, 20.0, 6.0, 0.5,
        help="Perkiraan kasar — untuk IDX bisa pakai acuan BI7DRR/yield SBN pendek, "
             "untuk US pakai yield T-Bill. Cuma memengaruhi titik Max-Sharpe, bukan "
             "bentuk efficient frontier-nya."
    ) / 100

    if len(port_tickers) < 2:
        st.info("Masukkan minimal 2 simbol untuk optimasi portofolio.")
    run_opt = st.button("📐 Hitung Efficient Frontier", disabled=(len(port_tickers) < 2))

    if run_opt:
        with st.spinner(f"Mengambil histori {len(port_tickers)} simbol & menghitung..."):
            price_series = {}
            failed = []
            for t in port_tickers:
                try:
                    if asset_type == "crypto":
                        df_t = cached_fetch_data(asset_type, t, exchange_id=exchange_id,
                                                  timeframe="1d", limit=500)
                    else:
                        df_t = cached_fetch_data(asset_type, t, period=port_period)
                    price_series[t] = df_t["Close"]
                except Exception as e:
                    failed.append((t, str(e)))

            if failed:
                st.warning("⚠️ Gagal fetch: " + ", ".join(f"{t} ({e})" for t, e in failed))

            valid_tickers = list(price_series.keys())
            if len(valid_tickers) < 2:
                st.error("Kurang dari 2 simbol berhasil di-fetch — tidak bisa optimasi.")
            else:
                price_panel = pd.DataFrame(price_series).dropna(how="any")
                if len(price_panel) < 30:
                    st.error(
                        f"Cuma {len(price_panel)} hari overlap antar simbol setelah tanggalnya "
                        f"disamakan — terlalu pendek untuk estimasi kovarians yang masuk akal. "
                        f"Coba simbol lain atau rentang histori lebih panjang."
                    )
                else:
                    log_ret = np.log(price_panel / price_panel.shift(1)).dropna()
                    trading_days = 365 if asset_type == "crypto" else 252
                    mu = log_ret.mean().values * trading_days       # annualized expected return
                    cov = log_ret.cov().values * trading_days       # annualized covariance Σ
                    n_assets = len(valid_tickers)

                    st.markdown("#### 🧮 Matriks Kovarians Σ (Tahunan)")
                    st.dataframe(
                        pd.DataFrame(cov, index=valid_tickers, columns=valid_tickers).round(4),
                        use_container_width=True
                    )

                    # ---- Correlation matrix (heatmap) — more directly readable than
                    # covariance for "how much do these two actually move together",
                    # since it's normalized to [-1, 1] and not distorted by each
                    # asset's own volatility scale the way covariance is. ----
                    st.markdown("#### 🔗 Matriks Korelasi")
                    corr_df_full = log_ret.corr()
                    fig_corr = go.Figure(go.Heatmap(
                        z=corr_df_full.values, x=valid_tickers, y=valid_tickers,
                        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                        text=corr_df_full.round(2).values, texttemplate="%{text}",
                        colorbar=dict(title="Korelasi"),
                    ))
                    fig_corr.update_layout(title="Korelasi Return Harian (Full-Sample)", height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption(
                        "Merah pekat (mendekati +1) = dua aset ini cenderung bergerak **searah** "
                        "— mengoleksi banyak pasangan begini di satu portofolio berarti "
                        "diversifikasi kamu lebih tipis dari yang kelihatan dari jumlah simbolnya. "
                        "Biru pekat (mendekati -1) = cenderung berlawanan arah (jarang terjadi "
                        "murni antar saham/crypto, lebih umum antar kelas aset). Putih/pucat "
                        "(mendekati 0) = pergerakannya relatif independen — pasangan inilah yang "
                        "sebenarnya memberi manfaat diversifikasi."
                    )

                    with st.expander("📉 Korelasi Rolling (per waktu) — cek kestabilan korelasinya"):
                        st.caption(
                            "Angka korelasi di atas itu rata-rata SATU periode penuh — bisa "
                            "menyembunyikan pergeseran rezim. Dua aset bisa rata-rata berkorelasi "
                            "0.3 sepanjang setahun, tapi sempat melonjak ke 0.8-0.9 pas market lagi "
                            "crash bareng-bareng — justru di momen itu diversifikasi paling "
                            "dibutuhkan, dan angka korelasi rata-rata bikin kamu kira kamu masih "
                            "punya proteksi padahal saat itu enggak."
                        )
                        if n_assets > 8:
                            st.info("Lebih dari 8 simbol — pilih satu pasangan di bawah untuk "
                                    "dilihat rolling correlation-nya (semua pasangan sekaligus "
                                    "terlalu ramai untuk satu chart).")
                        rc1, rc2, rc3 = st.columns(3)
                        pair_a = rc1.selectbox("Aset A", valid_tickers, index=0, key="rc_pair_a")
                        _default_b_idx = 1 if len(valid_tickers) > 1 else 0
                        pair_b = rc2.selectbox("Aset B", valid_tickers, index=_default_b_idx, key="rc_pair_b")
                        roll_window = rc3.slider("Window (hari)", 20, 120, 60, 10, key="rc_window")

                        if pair_a == pair_b:
                            st.warning("Pilih dua simbol yang berbeda.")
                        else:
                            roll_corr_series = log_ret[pair_a].rolling(roll_window).corr(log_ret[pair_b])
                            fig_rc = go.Figure(go.Scatter(
                                x=roll_corr_series.index, y=roll_corr_series.values,
                                mode="lines", line=dict(color="#636efa"),
                            ))
                            fig_rc.add_hline(y=0, line_dash="dot", line_color="gray")
                            _full_corr_ab = float(corr_df_full.loc[pair_a, pair_b])
                            fig_rc.add_hline(y=_full_corr_ab, line_dash="dash", line_color="orange",
                                              annotation_text=f"Full-sample: {_full_corr_ab:.2f}")
                            fig_rc.update_layout(
                                title=f"Rolling {roll_window}-hari Korelasi: {pair_a} vs {pair_b}",
                                yaxis=dict(range=[-1, 1], title="Korelasi"), height=350
                            )
                            st.plotly_chart(fig_rc, use_container_width=True)
                            _recent_corr = roll_corr_series.dropna()
                            if len(_recent_corr) > 0:
                                _now_corr = float(_recent_corr.iloc[-1])
                                _corr_range = float(_recent_corr.max() - _recent_corr.min())
                                st.caption(
                                    f"Korelasi rolling terkini: **{_now_corr:.2f}** (vs full-sample "
                                    f"**{_full_corr_ab:.2f}**). Rentang naik-turunnya sepanjang "
                                    f"histori ini: **{_corr_range:.2f}** — makin lebar rentangnya, "
                                    f"makin nggak stabil korelasi kedua aset ini dari waktu ke "
                                    f"waktu, artinya makin hati-hati mengandalkan angka full-sample "
                                    f"tunggal buat asumsi diversifikasi ke depan."
                                )

                    # ---- Eigenvalue decomposition of the correlation matrix: risk factor breakdown ----
                    corr = corr_df_full.values
                    eigvals, eigvecs = np.linalg.eigh(corr)
                    order = np.argsort(eigvals)[::-1]
                    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                    var_explained = eigvals / eigvals.sum()

                    st.markdown("#### 🔬 Dekomposisi Risiko (Eigenvalue Matriks Korelasi)")
                    st.caption(
                        "PC1 biasanya merepresentasikan 'faktor pasar bersama' antar aset yang "
                        "kamu pilih — kalau porsi variance-explained-nya besar, aset-aset ini "
                        "cenderung bergerak bareng (diversifikasi antar mereka kurang efektif); "
                        "kalau tersebar ke banyak komponen, aset-asetnya relatif lebih independen."
                    )
                    fig_eig = go.Figure(go.Bar(
                        x=[f"PC{i+1}" for i in range(n_assets)], y=var_explained * 100,
                        marker_color="#636efa"
                    ))
                    fig_eig.update_layout(title="Variance Explained per Principal Component",
                                           yaxis_title="% variance explained", height=300)
                    st.plotly_chart(fig_eig, use_container_width=True)
                    st.caption(f"PC1 sendiri menjelaskan **{var_explained[0]*100:.1f}%** dari "
                               f"total variance korelasi antar aset yang kamu pilih.")

                    with st.expander("Lihat loading PC1 (bobot tiap aset di faktor risiko dominan)"):
                        pc1_load = pd.Series(eigvecs[:, 0], index=valid_tickers)
                        pc1_load = pc1_load.reindex(pc1_load.abs().sort_values(ascending=False).index)
                        st.dataframe(pc1_load.to_frame("Loading PC1"), use_container_width=True)

                    # ---- Closed-form portfolios via matrix algebra ----
                    try:
                        inv_cov = np.linalg.inv(cov)
                    except np.linalg.LinAlgError:
                        inv_cov = np.linalg.pinv(cov)  # singular fallback (near-duplicate assets)

                    ones = np.ones(n_assets)
                    w_gmv = inv_cov @ ones / (ones @ inv_cov @ ones)

                    excess_mu = mu - rf_rate_annual
                    denom = ones @ inv_cov @ excess_mu
                    w_tangency = (inv_cov @ excess_mu / denom) if abs(denom) > 1e-10 else w_gmv.copy()
                    w_equal = ones / n_assets

                    # ---- Hierarchical Risk Parity (Lopez de Prado, 2016) ----
                    # Unlike GMV/Tangency above, HRP never inverts the covariance matrix —
                    # it clusters assets by correlation then allocates risk top-down through
                    # the cluster tree. Always non-negative (no short selling implied),
                    # historically more stable out-of-sample than the closed-form portfolios
                    # above when assets are highly correlated or the sample is short.
                    w_hrp_series = adv.hrp_weights(log_ret)
                    w_hrp = w_hrp_series.reindex(valid_tickers).values

                    def _port_stats(w):
                        ret = float(w @ mu)
                        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
                        sharpe = (ret - rf_rate_annual) / vol if vol > 0 else float("nan")
                        return ret, vol, sharpe

                    # ---- Random long-only portfolios, for the frontier scatter ----
                    n_sim = 4000
                    rng = np.random.default_rng(42)
                    rand_w = rng.dirichlet(np.ones(n_assets), size=n_sim)
                    rand_ret = rand_w @ mu
                    rand_vol = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov, rand_w))
                    rand_sharpe = (rand_ret - rf_rate_annual) / np.where(rand_vol > 0, rand_vol, np.nan)

                    gmv_ret, gmv_vol, gmv_sharpe = _port_stats(w_gmv)
                    tan_ret, tan_vol, tan_sharpe = _port_stats(w_tangency)
                    eq_ret, eq_vol, eq_sharpe = _port_stats(w_equal)
                    hrp_ret, hrp_vol, hrp_sharpe = _port_stats(w_hrp)

                    st.markdown("#### 📈 Efficient Frontier")
                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=rand_vol, y=rand_ret, mode="markers",
                        marker=dict(size=4, color=rand_sharpe, colorscale="Viridis",
                                    showscale=True, colorbar=dict(title="Sharpe")),
                        name="Portofolio acak (long-only)", opacity=0.5,
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[gmv_vol], y=[gmv_ret], mode="markers",
                        marker=dict(size=16, color="blue", symbol="star"),
                        name=f"Min-Variance (Sharpe {gmv_sharpe:.2f})"
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[tan_vol], y=[tan_ret], mode="markers",
                        marker=dict(size=16, color="red", symbol="star"),
                        name=f"Max-Sharpe / Tangency (Sharpe {tan_sharpe:.2f})"
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[eq_vol], y=[eq_ret], mode="markers",
                        marker=dict(size=14, color="gray", symbol="diamond"),
                        name=f"Equal-Weight (Sharpe {eq_sharpe:.2f})"
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[hrp_vol], y=[hrp_ret], mode="markers",
                        marker=dict(size=16, color="green", symbol="star"),
                        name=f"HRP (Sharpe {hrp_sharpe:.2f})"
                    ))
                    fig_ef.update_layout(
                        xaxis_title="Volatilitas Tahunan (risk)",
                        yaxis_title="Return Tahunan (expected)",
                        xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=500,
                    )
                    st.plotly_chart(fig_ef, use_container_width=True)

                    st.markdown("#### ⚖️ Bobot Portofolio")
                    weights_df = pd.DataFrame({
                        "Simbol": valid_tickers,
                        "Min-Variance (%)": np.round(w_gmv * 100, 1),
                        "Max-Sharpe (%)": np.round(w_tangency * 100, 1),
                        "Equal-Weight (%)": np.round(w_equal * 100, 1),
                        "HRP (%)": np.round(w_hrp * 100, 1),
                    })
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)

                    if (w_gmv < 0).any() or (w_tangency < 0).any():
                        st.info(
                            "ℹ️ Ada bobot negatif di Min-Variance/Max-Sharpe — itu artinya short "
                            "selling secara matematis 'optimal' menurut model closed-form ini "
                            "(yang TIDAK dibatasi long-only). **HRP di kolom sebelahnya selalu "
                            "non-negatif** (long-only by construction, karena dibangun dari "
                            "recursive bisection, bukan inverse matriks) — kalau kamu cuma "
                            "bisa/mau long, HRP kemungkinan lebih relevan dipakai langsung "
                            "dibanding harus 'membuang' bobot negatif secara manual."
                        )

                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Return tahunan (Max-Sharpe)", f"{tan_ret*100:.1f}%")
                    sc2.metric("Volatilitas tahunan (Max-Sharpe)", f"{tan_vol*100:.1f}%")
                    sc3.metric("Sharpe Ratio (Max-Sharpe)", f"{tan_sharpe:.2f}")

                    st.error(
                        "🚨 Sekali lagi: bobot di atas dihitung murni dari matriks kovarians & "
                        "return historis — tidak tahu apa-apa soal fundamental, likuiditas "
                        "(cek dulu tab 🎯 Kesimpulan tiap simbol), atau perubahan rezim pasar "
                        "ke depan. Ini alat eksplorasi trade-off, bukan rekomendasi alokasi final."
                    )
