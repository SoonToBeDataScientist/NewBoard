"""
strategy_arena.py
==========================================================================
Arena multi-strategi: uji BEBERAPA strategi berdampingan di data yang
sama, lacak performanya secara live (out-of-sample sejak tracking
dimulai), lalu peringkatkan dengan sistem poin.

Tiga hal yang sengaja dipisah karena menjawab pertanyaan berbeda:

  1. evaluate_arena()          -> backtest semua strategi di histori yang
                                   sama. Cepat, tapi in-sample — rawan
                                   "strategi X menang karena kebetulan
                                   cocok sama periode ini".
  2. update_arena_log()        -> live/forward tracking. Tiap run, posisi
     + live_leaderboard()         tiap strategi dicatat per hari ke CSV;
                                   performa dihitung HANYA dari baris yang
                                   tercatat sejak tracking dimulai. Ini
                                   satu-satunya angka yang jujur soal
                                   "strategi mana yang beneran jalan".
                                   Dipanggil OTOMATIS dari app.py tiap
                                   analisis jalan (idempotent) DAN dari
                                   precompute.py tiap sore via Actions.
  3. Sistem poin (2 lapis):
       - Poin komposit 0-100   -> kualitas absolut (excess return, Sharpe,
                                   win rate, profit factor, drawdown,
                                   konsistensi) dengan bobot tetap +
                                   diskon aktivitas (strategi yang hampir
                                   tak pernah trade tidak bisa dibedakan
                                   dari kebetulan).
       - Poin liga             -> relatif: tiap periode (mingguan untuk
                                   live, sekali untuk backtest) strategi
                                   di-rank, juara 1 dapat 10 poin, dst.
                                   Strategi yang "selalu lumayan" bisa
                                   mengalahkan yang "sekali jackpot".

Menambah strategi baru cuma butuh 2 langkah:
    def strategi_baru(df, param1=10):
        ...return pd.Series posisi 0/1 ter-index seperti df...
    register_strategy("Strategi Baru", strategi_baru, {"param1": 10})

Semua strategi long-only posisi 0/1, backward-looking (nilai hari T cuma
boleh lihat data s/d hari T), dieksekusi T+1 — standar anti-lookahead
yang sama dengan quant_engine.py.
==========================================================================
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ==========================================================================
# ==== 1. STRATEGI-STRATEGI ====
# ==========================================================================

def _stateful_position(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Ubah kondisi entry/exit (boolean) jadi posisi 0/1 yang BERTAHAN
    sampai kondisi exit muncul — untuk strategi yang sinyalnya berbasis
    kejadian (RSI oversold, breakout), bukan kondisi kontinu."""
    e = entry.fillna(False).values
    x = exit_.fillna(False).values
    pos = np.zeros(len(e))
    in_pos = False
    for i in range(len(e)):
        if not in_pos and e[i]:
            in_pos = True
        elif in_pos and x[i]:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=entry.index)


def strat_buy_hold(df: pd.DataFrame) -> pd.Series:
    """Benchmark — selalu posisi penuh. Bukan 'strategi', tapi wajib ada
    di leaderboard sebagai pembanding jujur."""
    return pd.Series(1.0, index=df.index)


def strat_ma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    ma_f = df["Close"].rolling(fast).mean()
    ma_s = df["Close"].rolling(slow).mean()
    return (ma_f > ma_s).astype(float).where(ma_s.notna(), 0.0)


def strat_rsi_reversion(df: pd.DataFrame, period: int = 14,
                        oversold: float = 30, exit_level: float = 55) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return _stateful_position(rsi < oversold, rsi > exit_level)


def strat_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
               signal: int = 9) -> pd.Series:
    ema_f = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_s = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd > sig).astype(float)


def strat_bollinger_breakout(df: pd.DataFrame, window: int = 20,
                             n_std: float = 2.0) -> pd.Series:
    mid = df["Close"].rolling(window).mean()
    sd = df["Close"].rolling(window).std(ddof=1)
    upper = mid + n_std * sd
    return _stateful_position(df["Close"] > upper, df["Close"] < mid)


def strat_momentum_roc(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    roc = df["Close"].pct_change(lookback)
    return (roc > 0).astype(float).where(roc.notna(), 0.0)


def strat_stoch_bb(df: pd.DataFrame) -> pd.Series:
    """Pakai stoch_bb_signal dari quant_engine (import lazy — kalau engine
    nggak ke-load, strategi lain tetap jalan)."""
    from quant_engine import stoch_bb_signal
    out = stoch_bb_signal(df)
    return _stateful_position(out["stochbb_signal"] == "BUY",
                              out["stochbb_signal"] == "SELL")


def strat_composite_engine(df: pd.DataFrame, mr_window: int = 20,
                           mr_z_entry: float = 1.5, mom_fast: int = 10,
                           mom_slow: int = 30, mr_weight: float = 0.5) -> pd.Series:
    """Composite signal yang sudah ada di quant_engine, diperlakukan sebagai
    SATU peserta arena — supaya bisa dibandingkan head-to-head dengan
    strategi sederhana. Kalau composite kalah dari MA crossover polos di
    live tracking, itu temuan penting."""
    from quant_engine import composite_signal
    sig = composite_signal(df, mr_window=mr_window, mr_z_entry=mr_z_entry,
                           mom_fast=mom_fast, mom_slow=mom_slow, mr_weight=mr_weight)
    return _stateful_position(sig["composite_signal"] == "BUY",
                              sig["composite_signal"].isin(["SELL", "EXIT"]))


STRATEGIES: dict[str, dict] = {
    "Buy & Hold (benchmark)": {"fn": strat_buy_hold, "params": {},
                                "benchmark": True,
                                "desc": "Baseline — strategi lain harus mengalahkan ini SETELAH biaya."},
    "MA Crossover (10/30)":   {"fn": strat_ma_crossover, "params": {"fast": 10, "slow": 30},
                                "desc": "Trend-following klasik golden/death cross."},
    "RSI Mean Reversion":     {"fn": strat_rsi_reversion, "params": {},
                                "desc": "Beli saat RSI<30, tahan sampai RSI>55."},
    "MACD Trend":             {"fn": strat_macd, "params": {},
                                "desc": "Posisi selama MACD di atas signal line."},
    "Bollinger Breakout":     {"fn": strat_bollinger_breakout, "params": {},
                                "desc": "Beli saat close tembus upper band, keluar di mid-band."},
    "Momentum ROC (20)":      {"fn": strat_momentum_roc, "params": {"lookback": 20},
                                "desc": "Posisi selama rate-of-change 20 hari positif."},
    "Stoch-BB (engine)":      {"fn": strat_stoch_bb, "params": {},
                                "desc": "Stochastic+Bollinger dari quant_engine, lengkap dengan band-walk filter."},
    "Composite (engine)":     {"fn": strat_composite_engine, "params": {},
                                "desc": "Sinyal composite MR+momentum yang sudah ada — peserta, bukan wasit."},
}


def register_strategy(name: str, fn, params: dict | None = None,
                      desc: str = "", benchmark: bool = False):
    """Daftarkan strategi baru ke arena tanpa edit file ini.
    fn: callable(df, **params) -> pd.Series posisi 0/1 ter-index seperti df."""
    STRATEGIES[name] = {"fn": fn, "params": params or {},
                        "benchmark": benchmark, "desc": desc}


# ==========================================================================
# ==== 2. ENGINE BACKTEST (vectorized, anti-lookahead) ====
# ==========================================================================
# Posisi hari T diputuskan dari data s/d close hari T -> return-nya baru
# dirasakan di periode close T -> close T+1 (pos.shift(1)). Biaya (fee
# asimetris + slippage) dibebankan saat perubahan posisi dieksekusi, juga
# di T+1. Konvensi yang sama dengan execution_price="next_open" di
# run_backtest quant_engine, versi close-to-close.

def run_strategy(df: pd.DataFrame, position: pd.Series,
                 fee_buy_bps: float = 20.0, fee_sell_bps: float = 30.0,
                 slippage_bps: float = 5.0, trading_days: float = 252.0,
                 initial_capital: float = 100.0) -> dict:
    pos = position.reindex(df.index).fillna(0.0).clip(0.0, 1.0)
    ret = df["Close"].pct_change().fillna(0.0)

    strat_gross = pos.shift(1).fillna(0.0) * ret

    delta = pos.diff()
    delta.iloc[0] = pos.iloc[0]
    exec_delta = delta.shift(1).fillna(0.0)  # diputuskan T, dieksekusi T+1
    cost = (exec_delta.clip(lower=0) * (fee_buy_bps + slippage_bps) / 1e4
            + (-exec_delta.clip(upper=0)) * (fee_sell_bps + slippage_bps) / 1e4)

    strat_net = strat_gross - cost
    equity = (1 + strat_net).cumprod() * initial_capital
    equity.iloc[0] = initial_capital
    equity.name = "equity"

    # ---- ekstrak trade dari segmen posisi (return net sudah termasuk biaya) ----
    trades = []
    in_pos, entry_i = False, None
    pos_vals = pos.values
    for i in range(1, len(pos_vals)):
        if not in_pos and pos_vals[i] > 0:
            in_pos, entry_i = True, i
        elif in_pos and pos_vals[i] == 0:
            seg = strat_net.iloc[entry_i:i + 1]
            trades.append({"entry_date": df.index[entry_i], "exit_date": df.index[i],
                           "return_pct": float(((1 + seg).prod() - 1) * 100),
                           "exit_reason": "signal"})
            in_pos = False
    if in_pos and entry_i is not None:
        seg = strat_net.iloc[entry_i:]
        trades.append({"entry_date": df.index[entry_i], "exit_date": df.index[-1],
                       "return_pct": float(((1 + seg).prod() - 1) * 100),
                       "exit_reason": "open"})

    return {"equity": equity, "trades": pd.DataFrame(trades),
            "position": pos, "daily_net": strat_net}


def compute_arena_metrics(result: dict, trading_days: float = 252.0) -> dict:
    equity, trades_df = result["equity"], result["trades"]
    daily = equity.pct_change().dropna()
    if len(daily) < 2:
        return {"error": "equity terlalu pendek"}

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    daily_std = daily.std(ddof=1)
    sharpe = (float(daily.mean() / daily_std * np.sqrt(trading_days))
              if daily_std > 0 else None)
    max_dd = float(((equity - equity.cummax()) / equity.cummax()).min() * 100)

    n = len(trades_df)
    if n > 0:
        r = trades_df["return_pct"]
        win_rate = float((r > 0).mean() * 100)
        gross_profit = float(r[r > 0].sum())
        gross_loss = float(abs(r[r < 0].sum()))
        profit_factor = (gross_profit / gross_loss if gross_loss > 0
                         else (3.0 if gross_profit > 0 else 0.0))  # tanpa loss -> cap di 3
        avg_trade = float(r.mean())
    else:
        win_rate, profit_factor, avg_trade = None, None, None

    monthly = (1 + daily).resample("ME").prod() - 1
    consistency = float((monthly > 0).mean() * 100) if len(monthly) > 0 else None
    exposure = float(result["position"].mean() * 100)

    return {"total_return_pct": float(total_ret), "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd, "n_trades": int(n),
            "win_rate_pct": win_rate, "profit_factor": profit_factor,
            "avg_trade_return_pct": avg_trade, "consistency_pct": consistency,
            "exposure_pct": exposure}


# ==========================================================================
# ==== 3. SISTEM POIN ====
# ==========================================================================
# Lapis 1 — POIN KOMPOSIT (0-100): kualitas absolut. Tiap komponen
# dinormalisasi ke [0,1] dengan cap yang masuk akal untuk swing trading,
# lalu dikali bobot. Bobot di bawah sengaja tidak menaruh return sebagai
# satu-satunya yang penting — strategi dengan return tinggi tapi drawdown
# 45% dan 2 trade tidak boleh menang.

POINTS_WEIGHTS = {
    "excess_return": 25,   # total return di ATAS buy & hold
    "sharpe":        20,   # return per unit risiko
    "win_rate":      15,
    "profit_factor": 15,   # gross profit / gross loss
    "max_drawdown":  15,   # penalti rasa sakit
    "consistency":   10,   # % bulan positif
}


def _norm(x: float | None, lo: float, hi: float,
          neutral: float | None = None) -> float:
    """Normalisasi linear ke [0,1] dengan clip. None -> neutral (default 0)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0 if neutral is None else neutral
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def compute_points(metrics: dict, bench_return_pct: float,
                   min_trades: int = 3, is_benchmark: bool = False) -> dict:
    """Returns {"total": 0-100, "breakdown": {komponen: poin_kontribusi}}."""
    if "error" in metrics:
        return {"total": None, "breakdown": {}}
    excess = metrics["total_return_pct"] - bench_return_pct
    comp = {
        # -20% di bawah B&H -> 0 ... +20% di atas B&H -> 1
        "excess_return": _norm(excess, -20, 20),
        # Sharpe -1 -> 0 ... +3 -> 1
        "sharpe":        _norm(metrics.get("sharpe_ratio"), -1, 3, neutral=0.25),
        # win rate 20% -> 0 ... 70% -> 1
        "win_rate":      _norm(metrics.get("win_rate_pct"), 20, 70),
        # PF 0 -> 0 ... 3 -> 1
        "profit_factor": _norm(metrics.get("profit_factor"), 0, 3),
        # maxDD 50% -> 0 ... 0% -> 1
        "max_drawdown":  1.0 - _norm(abs(metrics.get("max_drawdown_pct", 0)), 0, 50),
        "consistency":   _norm(metrics.get("consistency_pct"), 0, 100),
    }
    breakdown = {k: round(comp[k] * POINTS_WEIGHTS[k], 1) for k in comp}
    total = sum(breakdown.values())

    # Aktivitas: strategi yang hampir tidak pernah trade tidak bisa
    # dibedakan dari kebetulan statistik — skornya didiskon, bukan
    # dihapus. Benchmark (B&H) dikecualikan: 1 posisi permanen memang
    # kodratnya begitu.
    if not is_benchmark:
        n = metrics.get("n_trades", 0)
        total *= 0.3 + 0.7 * min(1.0, n / min_trades)

    return {"total": round(float(total), 1), "breakdown": breakdown}


# Lapis 2 — POIN LIGA (relatif, gaya F1): tiap periode, strategi di-rank
# berdasarkan return periode itu. Menghargai konsistensi antar periode,
# bukan cuma total akhir.
LEAGUE_POINTS = [10, 8, 6, 5, 4, 3, 2, 1]


def award_league_points(returns_by_strategy: pd.Series) -> dict:
    ranked = returns_by_strategy.dropna().sort_values(ascending=False)
    return {name: (LEAGUE_POINTS[i] if i < len(LEAGUE_POINTS) else 1)
            for i, name in enumerate(ranked.index)}


# ==========================================================================
# ==== 4. EVALUASI ARENA (backtest historis, semua strategi sekaligus) ====
# ==========================================================================

def evaluate_arena(df: pd.DataFrame, fee_buy_bps: float = 20.0,
                   fee_sell_bps: float = 30.0, slippage_bps: float = 5.0,
                   trading_days: float = 252.0,
                   strategies: dict | None = None) -> dict:
    """Jalankan SEMUA strategi di data yang sama, hitung metrik + poin +
    poin liga. IN-SAMPLE — pakai untuk screening cepat & pengembangan
    strategi, bukan bukti edge. Bukti edge = live tracking di bawah."""
    strategies = strategies or STRATEGIES
    results = {}
    bench_ret = None

    for name, spec in strategies.items():
        try:
            pos = spec["fn"](df, **spec["params"])
            res = run_strategy(df, pos, fee_buy_bps, fee_sell_bps,
                               slippage_bps, trading_days)
            res["metrics"] = compute_arena_metrics(res, trading_days)
            results[name] = res
            if spec.get("benchmark") and "error" not in res["metrics"]:
                bench_ret = res["metrics"]["total_return_pct"]
        except Exception as e:
            results[name] = {"error": str(e)}

    # Fallback jujur: kalau benchmark gagal (harusnya tidak pernah — cuma
    # rolling mean), excess return dihitung vs B&H mentah tanpa biaya.
    if bench_ret is None:
        bench_ret = float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)

    rows = []
    for name, res in results.items():
        if "error" in res or "error" in res.get("metrics", {}):
            rows.append({"Strategi": name, "Poin": None,
                         "Error": res.get("error") or res["metrics"].get("error")})
            continue
        m = res["metrics"]
        pts = compute_points(m, bench_ret,
                             is_benchmark=strategies[name].get("benchmark", False))
        rows.append({
            "Strategi": name, "Poin": pts["total"],
            **{f"  · {k}": v for k, v in pts["breakdown"].items()},
            "Total Return %": round(m["total_return_pct"], 1),
            "vs B&H (pts)": round(m["total_return_pct"] - bench_ret, 1),
            "Sharpe": round(m["sharpe_ratio"], 2) if m["sharpe_ratio"] is not None else None,
            "Max DD %": round(m["max_drawdown_pct"], 1),
            "Win Rate %": round(m["win_rate_pct"], 1) if m["win_rate_pct"] is not None else None,
            "Profit Factor": round(m["profit_factor"], 2) if m["profit_factor"] is not None else None,
            "Konsistensi %": round(m["consistency_pct"], 0) if m["consistency_pct"] is not None else None,
            "Trades": m["n_trades"], "Exposure %": round(m["exposure_pct"], 0),
        })

    lb = pd.DataFrame(rows).sort_values("Poin", ascending=False,
                                        na_position="last").reset_index(drop=True)
    lb.index = lb.index + 1
    lb.insert(0, "Rank", lb.index)

    league = award_league_points(
        pd.Series({r["Strategi"]: r.get("Total Return %") for r in rows}))
    lb["Poin Liga"] = lb["Strategi"].map(league)

    return {"leaderboard": lb,
            "equities": {n: r["equity"] for n, r in results.items() if "equity" in r},
            "trades": {n: r["trades"] for n, r in results.items() if "trades" in r},
            "bench_return_pct": bench_ret}


# ==========================================================================
# ==== 5. LIVE TRACKING — forward test yang sebenarnya ====
# ==========================================================================
# Pola yang sama dengan signals_log.csv (4c): posisi tiap strategi dicatat
# per hari; performa live dihitung HANYA dari baris yang tercatat sejak
# tracking dimulai. Karena semua strategi backward-looking, posisi hari T
# yang dihitung hari ini dari histori penuh IDENTIK dengan yang akan
# dihitung live hari T — jadi backfill saat pertama kali tracking itu sah
# secara deterministik, tapi tetap dibedakan dari "benar-benar live" via
# tanggal mulai tracking yang ditampilkan di UI.
#
# AUTOMATION (ini yang bikin "auto testing live" jalan beneran):
#   - app.py memanggil update_arena_log() OTOMATIS tiap kali analisis
#     dijalankan (idempotent, dedupe per tanggal — aman di-call berulang).
#   - precompute.py memanggilnya per watchlist tiap sore via GitHub
#     Actions — log terisi harian tanpa interaksi sama sekali.

ARENA_LOG_FILE = "strategy_arena_log.csv"
ARENA_LOG_COLUMNS = ["date", "symbol", "asset_type", "strategy", "position", "close"]


def load_arena_log() -> pd.DataFrame:
    if os.path.exists(ARENA_LOG_FILE):
        try:
            df = pd.read_csv(ARENA_LOG_FILE, parse_dates=["date"])
            return df[ARENA_LOG_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=ARENA_LOG_COLUMNS)


def update_arena_log(symbol: str, asset_type: str, df: pd.DataFrame,
                     strategies: dict | None = None) -> dict:
    """Tambahkan baris harian baru per strategi sejak tanggal terakhir yang
    tercatat untuk (symbol, strategy). Idempotent — aman dipanggil tiap
    rerun; baris duplikat tidak ditulis dua kali."""
    strategies = strategies or STRATEGIES
    log = load_arena_log()
    new_rows, per_strategy = [], {}

    for name, spec in strategies.items():
        try:
            pos = spec["fn"](df, **spec["params"])
        except Exception as e:
            per_strategy[name] = f"error: {e}"
            continue
        existing = log[(log["symbol"] == symbol) & (log["strategy"] == name)]
        last_date = existing["date"].max() if len(existing) > 0 else None

        sub = pd.DataFrame({"date": df.index, "position": pos.values,
                            "close": df["Close"].values})
        if last_date is not None:
            sub = sub[sub["date"] > last_date]
        if len(sub) == 0:
            per_strategy[name] = "up-to-date"
            continue
        sub["symbol"], sub["asset_type"], sub["strategy"] = symbol, asset_type, name
        new_rows.append(sub[ARENA_LOG_COLUMNS])
        per_strategy[name] = f"+{len(sub)} hari"

    if new_rows:
        log = pd.concat([log] + new_rows, ignore_index=True)
        log.to_csv(ARENA_LOG_FILE, index=False)
    return per_strategy


def live_leaderboard(symbol: str, fee_buy_bps: float = 20.0,
                     fee_sell_bps: float = 30.0, slippage_bps: float = 5.0,
                     trading_days: float = 252.0,
                     min_days: int = 10) -> dict | None:
    """Leaderboard dari log live SAJA (out-of-sample sejak tracking dimulai)
    + poin liga mingguan yang terakumulasi. None kalau log belum cukup umur."""
    log = load_arena_log()
    sub = log[log["symbol"] == symbol]
    if len(sub) == 0:
        return None

    equities, weekly_ret = {}, {}
    for name, grp in sub.groupby("strategy"):
        grp = grp.sort_values("date").drop_duplicates("date", keep="last")
        if len(grp) < min_days:
            continue
        df_g = grp.set_index("date")[["close"]].rename(columns={"close": "Close"})
        res = run_strategy(df_g, grp.set_index("date")["position"],
                           fee_buy_bps, fee_sell_bps, slippage_bps, trading_days)
        m = compute_arena_metrics(res, trading_days)
        if "error" in m:
            continue
        res["metrics"] = m
        equities[name] = res
        weekly_ret[name] = res["equity"].resample("W").last().pct_change()

    if not equities:
        return None

    bench_name = next((n for n in equities if "benchmark" in n.lower()
                       or "Buy & Hold" in n), None)
    bench_ret = (equities[bench_name]["metrics"]["total_return_pct"]
                 if bench_name else 0.0)

    # poin liga: akumulasi per minggu
    wdf = pd.DataFrame(weekly_ret).dropna(how="all")
    league_totals = {n: 0 for n in equities}
    if len(wdf) > 0:
        for _, row in wdf.iterrows():
            for name, pts in award_league_points(row).items():
                if name in league_totals:
                    league_totals[name] += pts

    rows = []
    for name, res in equities.items():
        m = res["metrics"]
        pts = compute_points(m, bench_ret, is_benchmark=(name == bench_name))
        rows.append({"Strategi": name, "Poin": pts["total"],
                     "Poin Liga (akumulasi)": league_totals[name],
                     "Total Return %": round(m["total_return_pct"], 1),
                     "vs B&H (pts)": round(m["total_return_pct"] - bench_ret, 1),
                     "Sharpe": round(m["sharpe_ratio"], 2) if m["sharpe_ratio"] is not None else None,
                     "Max DD %": round(m["max_drawdown_pct"], 1),
                     "Win Rate %": round(m["win_rate_pct"], 1) if m["win_rate_pct"] is not None else None,
                     "Trades": m["n_trades"]})

    lb = pd.DataFrame(rows).sort_values(
        ["Poin Liga (akumulasi)", "Poin"], ascending=False).reset_index(drop=True)
    lb.index = lb.index + 1
    lb.insert(0, "Rank", lb.index)

    n_days = int(sub.groupby("strategy")["date"].nunique().max())
    return {"leaderboard": lb,
            "equities": {n: r["equity"] for n, r in equities.items()},
            "weekly_returns": wdf, "n_live_days": n_days,
            "tracking_since": str(sub["date"].min().date())}