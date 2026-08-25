"""
quant_engine.py
==========================================================================
Pure-compute quant engine: data fetching, fundamentals, Monte Carlo,
signals, ML walk-forward, backtest, robustness scoring, portfolio
construction (incl. HRP), and parameter auto-tuning (GA optimizer).

DELIBERATELY has no Streamlit UI code and no hard dependency on
Streamlit being installed — this is the "heavy" half of the
precompute/viewer split:

    quant_engine.py  <-- you are here (RAM/CPU-heavy, run locally)
    precompute.py    <-- runs the engine for a watchlist, writes cache/
    viewer_app.py     <-- lightweight Streamlit app, only reads cache/

`@_cache_data` from the original single-file app is replaced by
`@_cache_data` below: if Streamlit is present it behaves exactly like
before (so this module still works fine if ever imported *inside* a
running Streamlit app), but if it's not installed, it degrades to a
plain in-process lru_cache — so `precompute.py` can run this file on
a bare Python environment without needing `streamlit` at all.

Two bugs fixed vs. the original single-file version during this split:
  1. `fetch_fundamentals` was defined TWICE in the original file (once
     here, once again in the old Streamlit "Fundamental Analysis" tab
     section) — the second definition silently shadowed this one at
     module load, so the richer curated dict (longName/sector/summary)
     below was dead code. There is now only one definition, and it's
     this curated one.
  2. `fetch_sp500_universe` had two stacked `@_cache_data` decorators
     (ttl=86400 wrapping ttl=3600) — leftover from a TTL change where
     the old decorator wasn't removed. Now just one, ttl=86400.
==========================================================================
"""

import os
import json
import csv
import io
import random
import threading
import time
import warnings
from collections import deque
import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache_data = st.cache_data
    _cache_resource = st.cache_resource
except ImportError:
    import functools

    def _cache_data(*dargs, **dkwargs):
        def decorator(fn):
            return functools.lru_cache(maxsize=256)(fn)
        return decorator

    def _cache_resource(*dargs, **dkwargs):
        def decorator(fn):
            return functools.lru_cache(maxsize=8)(fn)
        return decorator


# ==========================================================================
# ==== SECTION 0: SHARED YFINANCE PACING (proactive) + BACKOFF (reactive) ====
# ==========================================================================
# Yahoo Finance doesn't publish an official rate limit, but hammering it with
# many concurrent requests (Screener/Live Gainers scanning dozens-hundreds of
# tickers via a thread pool) reliably 429s within seconds. Two complementary,
# DELIBERATELY conservative strategies — "a bit slower but reliable" beats
# "fast until it 429s and half the scan comes back empty":
#
#   1. _YF_RATE_LIMITER paces every outgoing yfinance-hitting call to at most
#      a few per second, shared across ALL worker threads app-wide, so we
#      stay under Yahoo's limit proactively instead of finding the ceiling
#      the hard way.
#   2. _yf_call_with_backoff wraps the call so that if a 429 slips through
#      anyway, it waits with exponentially increasing delay (5s, 10s, 20s,
#      40s by default) and retries a few times instead of giving up on that
#      ticker immediately. This is the "kasih jeda, lanjut lagi, walau nambah
#      waktu loading" behavior.

class _RateLimiter:
    """Thread-safe sliding-window limiter: at most `max_calls` calls allowed
    within any rolling `period` seconds, shared across every caller."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._lock = threading.Lock()
        self._calls = deque()
    def wait(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self._calls[0] + self.period - now + 0.02
            time.sleep(max(sleep_for, 0.02))


# Conservative default: max 3 yfinance requests/second app-wide, regardless
# of how many threads a scan spins up. Turn down (e.g. max_calls=1-2) if 429s
# still show up often in practice; turn up cautiously only if scans feel
# unnecessarily slow AND 429s are already rare.
_YF_RATE_LIMITER = _RateLimiter(max_calls=3, period=1.0)


class _DiagnosticsCounters:
    """Thread-safe counters so a Screener scan can report WHERE its time
    actually went (limiter wait vs 429-backoff vs real work), instead of
    everyone guessing from 'it feels slow'. Reset at the start of each
    scan_universe_parallel() call, read back into its diagnostics dict."""

    def __init__(self):
        self._lock = threading.Lock()
        self.retry_count = 0
        self.backoff_seconds = 0.0

    def reset(self):
        with self._lock:
            self.retry_count = 0
            self.backoff_seconds = 0.0

    def record_backoff(self, seconds: float):
        with self._lock:
            self.retry_count += 1
            self.backoff_seconds += seconds

    def snapshot(self) -> dict:
        with self._lock:
            return {"retry_count": self.retry_count, "backoff_seconds": round(self.backoff_seconds, 1)}


_YF_DIAGNOSTICS = _DiagnosticsCounters()


def _is_rate_limit_error(err_str: str) -> bool:
    s = err_str.lower()
    return "too many requests" in s or "rate limit" in s or "429" in s


def _yf_call_with_backoff(fn, *args, max_retries: int = 4, base_delay: float = 5.0, **kwargs):
    """Run a yfinance-hitting callable behind the shared pacing limiter, and
    if a 429 slips through anyway, retry with exponential backoff instead of
    failing that ticker outright. Non-rate-limit errors (bad ticker, empty
    data, etc.) are raised immediately — no point backing off from those."""
    last_err = None
    for attempt in range(max_retries + 1):
        _YF_RATE_LIMITER.wait()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(str(e)) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)
                _YF_DIAGNOSTICS.record_backoff(delay)
                time.sleep(delay)
                continue
            raise
    raise last_err


# ==========================================================================
# ==== SECTION 1: DATA FETCHER ====
# ==========================================================================
# Unified OHLCV fetcher for:
#   - Indonesian stocks (IDX)  -> yfinance, ticker format "BBCA.JK"
#   - US stocks                -> yfinance, ticker format "AAPL"
#   - Crypto                   -> ccxt, symbol format "BTC/USDT"

def fetch_stock(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch daily OHLCV for a stock via yfinance — paced + auto-retried
    against rate limiting (see _yf_call_with_backoff in Section 0).
    Falls back to Twelve Data (if TWELVEDATA_API_KEY is set) when yfinance
    fails outright — this is where yfinance's known IDX coverage gaps
    show up. Single-ticker use only (Monte Carlo/Signal/Backtest/
    Fundamental tabs) — NOT used by the Screener's bulk scan, which stays
    yfinance-only via fetch_stock_batch() to not burn through Twelve
    Data's tight free quota (800 calls/day) on a 50-symbol scan."""
    import yfinance as yf

    try:
        df = _yf_call_with_backoff(
            lambda: yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        )
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'. Check the ticker symbol/suffix.")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df.dropna()
    except Exception as yf_error:
        if os.environ.get("TWELVEDATA_API_KEY"):
            try:
                return fetch_stock_twelvedata(ticker, period=period)
            except Exception:
                pass  # Twelve Data fallback also failed — raise the ORIGINAL yfinance error below,
                      # it's usually more informative than a secondary fallback's error.
        raise yf_error


def fetch_stock_twelvedata(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fallback OHLCV source for when yfinance fails/returns nothing —
    Twelve Data's free tier (800 calls/day, 8/min — see
    https://twelvedata.com/pricing) explicitly covers the Indonesia Stock
    Exchange, which is where yfinance's gaps are worst. Price data ONLY —
    Twelve Data's free tier does not appear to include fundamentals
    (those need a paid plan), so this does not help with missing
    PE/ROE/etc. fields, only missing/incomplete price history.

    NOTE: built against Twelve Data's documented REST pattern
    (https://twelvedata.com/docs#time-series) but not verified against a
    live account — if `values`/field names below don't match what your
    key actually returns, check the docs and adjust.
    """
    import requests

    api_key = os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise ValueError("TWELVEDATA_API_KEY tidak diset.")

    clean_symbol = ticker.upper().replace(".JK", "")
    params = {
        "symbol": clean_symbol,
        "interval": "1day",
        "outputsize": 5000,  # Twelve Data caps this server-side regardless; harmless to over-ask
        "apikey": api_key,
        "format": "JSON",
    }
    if ticker.upper().endswith(".JK"):
        params["exchange"] = "IDX"

    resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, dict) or "values" not in data:
        raise ValueError(f"Twelve Data error untuk '{ticker}': {data.get('message', data) if isinstance(data, dict) else data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise ValueError(f"Twelve Data returned no usable rows for '{ticker}'.")
    return df


def fetch_stock_batch(tickers: list[str], period: str = "1y",
                       chunk_size: int = 50) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for MANY stock tickers in ONE yfinance call, instead of
    one HTTP round-trip + rate-limiter wait PER ticker like fetch_stock()
    does. This is the fix for a Screener scan that's slow with no visible
    CPU usage — that symptom means the bottleneck is network/pacing wait
    (time.sleep in the shared rate limiter, or 429-backoff), NOT compute,
    and fetch_stock's one-request-per-ticker pattern is exactly what makes
    an N-ticker scan take at least N/3 seconds of pure waiting before any
    real work even starts. Batching turns that into ~1 paced call total.

    Still goes through the SAME shared rate limiter + backoff as everything
    else (still one real network operation hitting Yahoo), just once for
    the whole batch instead of once per ticker.

    Returns {ticker: df} for tickers that returned usable data; tickers
    with no/empty data are simply omitted (caller/scan treats a missing key
    as "failed", same as a fetch_stock() exception would).
    """
    import yfinance as yf

    if not tickers:
        return {}

    # FIX 1d: threads=True membuat yfinance menembak N request HTTP konkuren
    # secara internal — semuanya mem-bypass _YF_RATE_LIMITER. Untuk scan 900
    # ticker IDX itu cara tercepat kena 429 massal. Solusinya: chunk per 50
    # ticker, threads=False, tiap chunk tetap lewat limiter + backoff.
    out = {}
    for _ci in range(0, len(tickers), chunk_size):
        chunk = tickers[_ci:_ci + chunk_size]
        raw = _yf_call_with_backoff(
            lambda: yf.download(tickers=chunk, period=period, interval="1d",
                                 auto_adjust=True, group_by="ticker",
                                 threads=False, progress=False)
        )
        is_multi = isinstance(raw.columns, pd.MultiIndex)
        for tkr in chunk:
            try:
                sub = raw[tkr] if is_multi else raw  # single-ticker call returns flat columns
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                if sub.empty or sub["Close"].dropna().empty:
                    continue
                sub = sub.dropna()
                sub.index = pd.to_datetime(sub.index).tz_localize(None)
                sub.index.name = "Date"
                out[tkr] = sub
            except (KeyError, IndexError):
                continue  # this ticker wasn't in the batch response at all (delisted/typo/etc.)
    return out


def fetch_crypto(symbol: str = "BTC/USDT", exchange_id: str = "binance",
                  timeframe: str = "1d", limit: int = 730) -> pd.DataFrame:
    """Fetch OHLCV for a crypto pair via ccxt."""
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        err_str = str(e).lower()
        network_hints = ("connection", "timeout", "resolve", "refused",
                          "unreachable", "network", "ssl", "certificate")
        if any(h in err_str for h in network_hints):
            raise ConnectionError(
                f"Gagal konek ke {exchange_id} ({e}). Kalau exchange_id='binance', "
                f"ini kemungkinan besar karena Binance diblokir Kominfo di jaringan "
                f"ISP Indonesia (redirect ke Internet Positif) — coba ganti exchange "
                f"ke 'indodax', 'kraken', atau 'coinbase' di sidebar."
            ) from e
        raise

    if not ohlcv:
        raise ValueError(f"No data returned for symbol '{symbol}' on '{exchange_id}'. "
                          f"Cek format pair — Indodax umumnya pakai quote IDR (mis. "
                          f"'BTC/IDR'), bukan USDT.")

    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.set_index("Date").drop(columns=["Timestamp"])
    return df.dropna()


def fetch_data(asset_type: str, symbol: str, **kwargs) -> pd.DataFrame:
    """
    Single entry point.
    asset_type: "stock_id", "stock_us", "crypto"
    """
    if asset_type == "stock_id":
        if not symbol.upper().endswith(".JK"):
            symbol = symbol.upper() + ".JK"
        return fetch_stock(symbol, **kwargs)
    elif asset_type == "stock_us":
        return fetch_stock(symbol.upper(), **kwargs)
    elif asset_type == "crypto":
        return fetch_crypto(symbol, **kwargs)
    else:
        raise ValueError("asset_type must be one of: 'stock_id', 'stock_us', 'crypto'")


# ==========================================================================
# ==== SECTION 1.5: FUNDAMENTAL DATA (stocks only — no such thing for crypto) ====
# ==========================================================================
# Pulls valuation/profitability/health ratios from yfinance's Ticker.info.
# Fundamentals change slowly (quarterly reports), so this uses a much longer
# cache TTL than price data. Coverage and field completeness vary a lot,
# especially for smaller IDX stocks outside LQ45 — missing fields are shown
# as "n/a" rather than crashing the page.
#
# NOTE ON THE MERGE: the original single-file app had TWO competing
# fetch_fundamentals() implementations that silently shadowed each other
# (see module docstring at the top) — a "curated" nested-dict version here,
# and a flat-dict version living in the old Streamlit UI section, which is
# the one `score_fundamentals()` actually reads (`f.get("PE")` etc. — flat
# top-level access). This merged version returns ONE flat dict compatible
# with score_fundamentals, PLUS the descriptive fields (longName/sector/
# industry/summary) and a `_formatted` dict of display-ready strings that
# the old curated version had but that were previously unreachable dead
# code. No caller needs to change.

FUNDAMENTAL_FIELDS = {
    # display label -> (yfinance info key, format kind)
    "PE": ("trailingPE", "ratio"),
    "PBV": ("priceToBook", "ratio"),
    "ROE": ("returnOnEquity", "pct"),
    "DER": ("debtToEquity", "ratio100"),
    "Profit Margin": ("profitMargins", "pct"),
    "Revenue Growth": ("revenueGrowth", "pct"),
    "Dividend Yield": ("dividendYield", "pct"),
    "Market Cap": ("marketCap", "money"),
}


def _fmt_fundamental(value, kind: str) -> str:
    if value is None:
        return "n/a"
    try:
        if kind == "ratio":
            return f"{value:.2f}x"
        elif kind == "ratio100":
            return f"{value/100:.2f}x"  # yfinance reports debtToEquity as e.g. 45.3 meaning 0.453x
        elif kind == "pct":
            return f"{value*100:.1f}%" if abs(value) < 5 else f"{value:.1f}%"
        elif kind == "money":
            if value >= 1e12:
                return f"{value/1e12:.2f}T"
            elif value >= 1e9:
                return f"{value/1e9:.2f}B"
            elif value >= 1e6:
                return f"{value/1e6:.2f}M"
            return f"{value:,.0f}"
        return str(value)
    except Exception:
        return "n/a"


@_cache_data(ttl=21600, show_spinner=False)
def fetch_news_sentiment(query: str, max_articles: int = 10) -> dict | None:
    """
    Fetches recent news + entity-level sentiment via Marketaux
    (marketaux.com) — a legitimate, ToS-compliant news API. Deliberately
    NOT a scraper of TradingView/news sites directly — same ToS-fragility
    reasoning already applied elsewhere in this project (see the
    TradingView-scraping discussion this codebase's history is built on):
    an official free-tier API beats scraping when one exists.

    INFORMATIONAL ONLY — deliberately NOT wired into composite_signal or
    any backtested/validated score. Unlike earnings_drift_series (which
    has a free historical archive via yfinance and so COULD be properly
    walk-forward validated), there is no free historical news archive
    available — only "sentiment right now". Feeding an unvalidated,
    unbacktestable number into the same composite score that's been held
    to walk-forward/bootstrap/BSS standards throughout this codebase
    would undermine exactly the rigor the rest of it is built around.
    This stays a separate display panel the person reads themselves —
    see fetch_news_sentiment's callers for how it's kept out of any
    scored/blended signal.

    Free tier: 100 requests/day TOTAL for the whole app (Marketaux) —
    cached long (6h) here, and callers must NEVER auto-fire this across
    a whole Screener scan (would burn the daily quota in one run) — only
    single-ticker/opt-in/hard-capped use, enforced by the caller.

    `query` is a free-text search term (ticker, company name, or crypto
    symbol like "BTC") rather than a strict entity-symbol filter —
    deliberately more format-tolerant, since Marketaux's exact symbol
    format for IDX stocks/crypto pairs hasn't been verified against a
    live account. If results come back empty for a symbol, try the
    company name instead.

    Returns None if unavailable (no MARKETAUX_API_KEY, no results, or
    request failure), else:
      {"avg_sentiment": float in [-1,1], "n_articles": int,
       "articles": [{"title","sentiment","source","url","published_at"}, ...]}
    """
    import requests

    api_key = os.environ.get("MARKETAUX_API_KEY")
    if not api_key:
        return None

    params = {
        "search": query, "filter_entities": "true", "language": "en,id",
        "limit": max_articles, "api_token": api_key,
    }
    try:
        resp = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    articles = data.get("data", [])
    if not articles:
        return None

    sentiments, parsed = [], []
    for art in articles:
        entities = art.get("entities") or []
        scores = [e.get("sentiment_score") for e in entities if e.get("sentiment_score") is not None]
        if not scores:
            continue
        art_sent = sum(scores) / len(scores)
        sentiments.append(art_sent)
        parsed.append({
            "title": art.get("title"), "sentiment": float(art_sent),
            "source": art.get("source"), "url": art.get("url"),
            "published_at": art.get("published_at"),
        })

    if not sentiments:
        return None

    return {
        "avg_sentiment": float(sum(sentiments) / len(sentiments)),
        "n_articles": len(sentiments),
        "articles": parsed,
    }


@_cache_data(ttl=21600, show_spinner=False)
def fetch_earnings_history(ticker: str) -> pd.DataFrame | None:
    """
    All available past earnings surprises (actual EPS vs. consensus
    estimate) via yfinance, typically the last 4-8 quarters — needed to
    build a HISTORICAL, backtestable PEAD feature (see
    earnings_drift_series below), not just a live snapshot.

    HONEST LIMITATION: yfinance's earnings-calendar coverage skews
    heavily toward US stocks — for many IDX tickers this returns None
    (not an error, just "not available"), same graceful-degradation
    pattern as fetch_fundamentals()'s IDX coverage gaps elsewhere in
    this file. Cached longer (6h) than price data since this changes
    only once per quarter.
    """
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        edf = _yf_call_with_backoff(lambda: t.earnings_dates)
        if edf is None or edf.empty:
            return None
        edf = edf.copy()
        edf.index = pd.to_datetime(edf.index)
        try:
            edf.index = edf.index.tz_localize(None)
        except TypeError:
            pass  # already tz-naive
        edf = edf.sort_index()
        col_actual = next((c for c in edf.columns if "Reported" in c and "EPS" in c), None)
        col_est = next((c for c in edf.columns if "Estimate" in c and "EPS" in c), None)
        if col_actual is None or col_est is None:
            return None
        edf = edf.dropna(subset=[col_actual, col_est])
        edf = edf[edf[col_est] != 0]
        if edf.empty:
            return None
        edf["surprise_pct"] = (edf[col_actual] - edf[col_est]) / edf[col_est].abs() * 100
        return edf[["surprise_pct"]]
    except Exception:
        return None


def earnings_drift_series(df: pd.DataFrame, ticker: str, drift_window_days: int = 60,
                           ) -> pd.Series | None:
    """
    Post-Earnings-Announcement-Drift (PEAD) score, as a full historical
    series aligned to df's index — for each day, reflects the MOST
    RECENT past earnings surprise (if any) within `drift_window_days`,
    magnitude-scaled and linearly decayed to 0 by the window edge (the
    drift effect is strongest right after the surprise, documented to
    fade over subsequent weeks — this is a genuine, decades-replicated
    market anomaly, unlike generic technical-indicator patterns which
    are heavily arbitraged by now — see composite_signal's
    earnings_weight docs for why this is opt-in and additive, not a
    replacement for the existing signals).

    Unlike every OTHER signal in this file, this is NOT built from OHLCV
    at all — it needs actual earnings surprise data, which is where the
    genuinely new information comes from, not another transform of the
    same price series everyone else already has too.

    Returns None if earnings history isn't available for this ticker
    (see fetch_earnings_history) — composite_signal treats this exactly
    like ml_score=None: zero effect, no error, no silent guessing.
    """
    earnings_hist = fetch_earnings_history(ticker)
    if earnings_hist is None or earnings_hist.empty:
        return None

    score = pd.Series(np.nan, index=df.index)
    for i, today in enumerate(df.index):
        past = earnings_hist[earnings_hist.index <= today]
        if past.empty:
            continue
        days_since = (today - past.index[-1]).days
        if days_since < 0 or days_since > drift_window_days:
            continue
        surprise_pct = float(past["surprise_pct"].iloc[-1])
        magnitude = np.clip(surprise_pct / 20.0, -1.0, 1.0)  # ±20% surprise -> full ±1
        decay = 1.0 - (days_since / drift_window_days)
        score.iloc[i] = magnitude * decay
    return score


@_cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental info for one stock ticker as a FLAT dict (label ->
    raw value), which is what score_fundamentals() below reads directly —
    plus descriptive fields (longName/sector/industry/summary/currency) and
    a `_formatted` sub-dict of display-ready strings for the viewer.
    Paced + auto-retried against rate limiting, see _yf_call_with_backoff."""
    import yfinance as yf
    info = _yf_call_with_backoff(lambda: yf.Ticker(ticker).info)
    if not info or len(info) < 5:
        raise ValueError("Data fundamental tidak tersedia untuk ticker ini di Yahoo Finance.")

    flat: dict = {
        "longName": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": info.get("longBusinessSummary"),
        "currency": info.get("currency"),
    }
    formatted = {}
    for label, (key, kind) in FUNDAMENTAL_FIELDS.items():
        raw = info.get(key)
        flat[label] = raw
        formatted[label] = _fmt_fundamental(raw, kind)
    flat["_formatted"] = formatted
    return flat


def score_fundamentals(f: dict) -> tuple:
    """
    Rule-based fundamental scoring. Each available metric contributes a
    0..1 sub-score; metrics that are missing are simply skipped (not
    penalized — many small-cap IDX tickers have sparse Yahoo Finance data).
    Sector-agnostic rules of thumb, NOT a real fundamental-analysis model —
    a bank's naturally high DER or a growth stock's high PE isn't
    automatically "bad".
    Returns (overall_score in 0..1 or None if nothing available, sub-scores dict).
    """
    subs = {}

    pe = f.get("PE")
    if pe is not None and pe > 0:
        subs["PE"] = 1.0 if pe < 15 else 0.6 if pe < 25 else 0.3 if pe < 40 else 0.1

    pbv = f.get("PBV")
    if pbv is not None and pbv > 0:
        subs["PBV"] = 1.0 if pbv < 1 else 0.6 if pbv < 3 else 0.3

    roe = f.get("ROE")
    if roe is not None:
        subs["ROE"] = 1.0 if roe > 0.20 else 0.6 if roe > 0.10 else 0.3 if roe > 0 else 0.0

    der = f.get("DER")
    if der is not None:
        der_ratio = der / 100 if der > 10 else der  # yfinance sometimes reports as %, sometimes ratio
        subs["DER"] = 1.0 if der_ratio < 0.5 else 0.6 if der_ratio < 1.0 else 0.3 if der_ratio < 2.0 else 0.1

    pm = f.get("Profit Margin")
    if pm is not None:
        subs["Profit Margin"] = 1.0 if pm > 0.15 else 0.6 if pm > 0.05 else 0.3 if pm > 0 else 0.0

    rg = f.get("Revenue Growth")
    if rg is not None:
        subs["Revenue Growth"] = 1.0 if rg > 0.15 else 0.6 if rg > 0.05 else 0.4 if rg > 0 else 0.1

    if not subs:
        return None, {}
    return sum(subs.values()) / len(subs), subs


# ==========================================================================
# ==== SECTION 2: MONTE CARLO ====
# ==========================================================================
# GBM (constant volatility) and GARCH(1,1) (dynamic, mean-reverting
# volatility) price-path simulation.

def _daily_log_returns(prices: pd.Series) -> np.ndarray:
    return np.diff(np.log(prices.values))


def simulate_gbm(prices: pd.Series, n_days: int = 30, n_sims: int = 2000,
                  seed: int | None = 42, mu_shrink: float = 1.0) -> np.ndarray:
    """Classic GBM Monte Carlo. Returns array shape (n_sims, n_days+1).

    mu_shrink: estimasi drift dari mean historis secara statistik sangat
    tidak stabil (hasil klasik Merton — butuh puluhan tahun data untuk
    estimasi drift yang bermakna). mu_shrink=1.0 = perilaku lama;
    mu_shrink=0.0 = pure risk view (mu=0), disarankan untuk VaR/risk."""
    rng = np.random.default_rng(seed)
    log_ret = _daily_log_returns(prices)

    mu = log_ret.mean() * mu_shrink
    sigma = log_ret.std(ddof=1)
    s0 = prices.values[-1]

    z = rng.standard_normal((n_sims, n_days))
    daily_log_ret = (mu - 0.5 * sigma ** 2) + sigma * z

    log_paths = np.cumsum(daily_log_ret, axis=1)
    paths = s0 * np.exp(log_paths)
    return np.hstack([np.full((n_sims, 1), s0), paths])


def simulate_garch(prices: pd.Series, n_days: int = 30, n_sims: int = 2000,
                    seed: int | None = 42, asymmetric: bool = False) -> np.ndarray:
    """
    GARCH(1,1)-based Monte Carlo with time-varying, clustered volatility.

    asymmetric=True fits GJR-GARCH(1,1,1) instead (adds a leverage term,
    o=1): volatility reacts MORE to a negative-return shock than to a
    positive one of the same size. This "leverage effect" is well
    documented in equities and especially crypto (sharp drops tend to be
    followed by higher vol than equally-sized rallies) — plain symmetric
    GARCH(1,1) has no way to represent that asymmetry at all, so its fan
    chart understates downside risk and overstates upside risk right after
    a selloff.
    """
    from arch import arch_model

    log_ret = _daily_log_returns(prices) * 100  # arch works better in % units
    s0 = prices.values[-1]

    o = 1 if asymmetric else 0
    # FIX 2d: Student-t — fat tails. Fan chart normal-distribusi understate
    # ekor kiri persis di aset yang paling berisiko (crypto & small-cap IDX).
    am = arch_model(log_ret, vol="Garch", p=1, o=o, q=1, dist="studentst", mean="constant")
    res = am.fit(disp="off")

    sim = res.forecast(horizon=n_days, method="simulation",
                        simulations=n_sims, reindex=False)
    sim_returns_pct = sim.simulations.values[0]  # (n_sims, n_days)
    sim_log_ret = sim_returns_pct / 100.0

    log_paths = np.cumsum(sim_log_ret, axis=1)
    paths = s0 * np.exp(log_paths)
    return np.hstack([np.full((n_sims, 1), s0), paths])


def summarize_paths(paths: np.ndarray, confidence: float = 0.95) -> dict:
    """Summarize simulated paths: expected final price, bands, VaR."""
    final_prices = paths[:, -1]
    s0 = paths[0, 0]

    alpha = 1 - confidence
    lower_pct, upper_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    returns = final_prices / s0 - 1
    var = -np.percentile(returns, alpha * 100)
    # FIX 2e: Expected Shortfall / CVaR — rata-rata kerugian DI LUAR VaR.
    # VaR cuma bilang "ambang rugi terburuk 5%"; ES bilang "kalau sudah
    # masuk 5% terburuk itu, rata-rata rugi seberapa dalam".
    tail_losses = returns[returns <= -var]
    es = float(-tail_losses.mean()) if len(tail_losses) > 0 else float(var)

    return {
        "s0": float(s0),
        "mean_final": float(final_prices.mean()),
        "median_final": float(np.median(final_prices)),
        "p_lower": float(np.percentile(final_prices, lower_pct)),
        "p_upper": float(np.percentile(final_prices, upper_pct)),
        "prob_profit": float((final_prices > s0).mean()),
        f"VaR_{int(confidence*100)}pct": float(var),
        f"ES_{int(confidence*100)}pct": es,
        "confidence": confidence,
    }


def fan_chart_bands(paths: np.ndarray, bands=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """Percentile bands per day, for fan chart visualization."""
    pct = np.percentile(paths, [b * 100 for b in bands], axis=0)
    df = pd.DataFrame(pct.T, columns=[f"p{int(b*100)}" for b in bands])
    df.index.name = "day"
    return df


# ==========================================================================
# ==== SECTION 3: SIGNALS ====
# ==========================================================================
# Mean-reversion (Z-score) + momentum (MA crossover) + composite score,
# suited for swing-style holds rather than intraday.

def mean_reversion_signal(df: pd.DataFrame, window: int = 20, z_entry: float = 1.5,
                           z_exit: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    roll_mean = out["Close"].rolling(window).mean()
    roll_std = out["Close"].rolling(window).std(ddof=1)
    z = (out["Close"] - roll_mean) / roll_std

    signal = pd.Series("HOLD", index=out.index)
    signal[z <= -z_entry] = "BUY"
    signal[z >= z_entry] = "SELL"
    signal[z.abs() <= z_exit] = "EXIT"

    out["mr_zscore"] = z
    out["mr_signal"] = signal
    return out


def momentum_signal(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    out = df.copy()
    out["ma_fast"] = out["Close"].rolling(fast).mean()
    out["ma_slow"] = out["Close"].rolling(slow).mean()

    diff = out["ma_fast"] - out["ma_slow"]
    prev_diff = diff.shift(1)

    signal = pd.Series("HOLD", index=out.index)
    signal[(prev_diff <= 0) & (diff > 0)] = "BUY"
    signal[(prev_diff >= 0) & (diff < 0)] = "SELL"

    out["mom_diff"] = diff
    out["mom_signal"] = signal
    return out


def stoch_bb_signal(df: pd.DataFrame, bb_window: int = 20, bb_std: float = 2.0,
                     stoch_window: int = 14, stoch_smooth: int = 3,
                     band_walk_window: int = 5, band_walk_min_touches: int = 3,
                     band_walk_width_lookback: int = 10) -> pd.DataFrame:
    """
    Stochastic Oscillator confirmed by Bollinger Bands — a classic
    mean-reversion confirmation combo, not either indicator alone. The
    idea: an oversold Stochastic reading is a much stronger signal when
    price is ALSO trading near the lower Bollinger Band (the move is
    statistically stretched, not just short-term noise) than when
    Stochastic is oversold but price is nowhere near the band.

    BAND-WALK REGIME FILTER: Bollinger Bands widen with rolling volatility
    rather than staying fixed, so during a genuinely strong trend price
    can ride/"walk" along the upper (or lower) band for many days without
    reverting — a well-known failure mode of reading "price at the band"
    as an overbought/oversold reversal signal (see John Bollinger's own
    writing on "the walk"). This detects that regime — band WIDENING +
    price repeatedly touching the same band over `band_walk_window` days
    (not just one touch) — and MUTES the opposing reversion signal rather
    than flipping it into a continuation/BUY call: an uptrend walk mutes
    the SELL leg (stops calling a strong uptrend "overbought"), a
    downtrend walk mutes the BUY leg. It does not itself claim "keep
    buying because it's walking the band" — that's a different, more
    aggressive claim this function deliberately does not make.

    Adds columns: bb_upper/bb_mid/bb_lower/bb_pctb, stoch_k/stoch_d,
    band_walk_state ("up"/"down"/None), stochbb_score (continuous,
    [-1,+1], +1 = strongly oversold/bullish setup — clipped toward 0 on
    the side being muted during a detected band-walk), stochbb_signal
    (categorical BUY/SELL/HOLD — fires only when BOTH legs agree, the
    Stochastic is in the classic 20/80 zone, AND no opposing band-walk
    is muting that side).
    """
    out = df.copy()
    close = out["Close"]

    bb_mid = close.rolling(bb_window).mean()
    bb_sd = close.rolling(bb_window).std(ddof=1)
    bb_upper = bb_mid + bb_std * bb_sd
    bb_lower = bb_mid - bb_std * bb_sd
    bb_pctb = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    low_k = out["Low"].rolling(stoch_window).min()
    high_k = out["High"].rolling(stoch_window).max()
    stoch_k = (close - low_k) / (high_k - low_k).replace(0, np.nan) * 100
    stoch_d = stoch_k.rolling(stoch_smooth).mean()

    # ---- Band-walk regime detection ----
    # Width normalized by the mid-band so it's comparable across price
    # levels/tickers. Baseline uses .shift(1) so "is today's width above
    # the recent baseline" never peeks at today's own value while forming
    # that baseline — backward-looking only, same anti-lookahead standard
    # as the rest of this file.
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    bb_width_baseline = bb_width.rolling(band_walk_width_lookback).mean().shift(1)
    band_expanding = bb_width > bb_width_baseline

    near_upper = bb_pctb >= 0.85
    near_lower = bb_pctb <= 0.15
    touches_upper_recent = near_upper.rolling(band_walk_window).sum()
    touches_lower_recent = near_lower.rolling(band_walk_window).sum()

    band_walk_up = band_expanding & (touches_upper_recent >= band_walk_min_touches)
    band_walk_down = band_expanding & (touches_lower_recent >= band_walk_min_touches)

    band_walk_state = pd.Series(None, index=out.index, dtype="object")
    band_walk_state[band_walk_up] = "up"
    band_walk_state[band_walk_down] = "down"

    # Both legs normalized to [-1, +1]: +1 = strongly oversold (bullish
    # mean-reversion setup), -1 = strongly overbought (bearish). Averaging
    # them means a case where the two legs DISAGREE (e.g. price pierces
    # the lower band but Stochastic isn't actually oversold yet) gets
    # muted toward 0 instead of firing on either leg alone — that
    # disagreement is itself informative ("wait for confirmation").
    bb_component = (0.5 - bb_pctb) * 2
    stoch_component = (50 - stoch_k) / 50
    stochbb_score = ((bb_component.clip(-1.5, 1.5) + stoch_component.clip(-1.5, 1.5)) / 2).clip(-1, 1)
    # Mute the side being contradicted by an active band-walk — an uptrend
    # walk means "don't call this overbought", so the score can't go
    # negative (SELL-leaning) while band_walk_up is active, and vice versa.
    stochbb_score = stochbb_score.where(~band_walk_up, stochbb_score.clip(lower=0))
    stochbb_score = stochbb_score.where(~band_walk_down, stochbb_score.clip(upper=0))

    signal = pd.Series("HOLD", index=out.index)
    bullish_confirmed = (bb_pctb <= 0.1) & (stoch_k <= 20) & (stoch_k >= stoch_d) & (~band_walk_down)
    bearish_confirmed = (bb_pctb >= 0.9) & (stoch_k >= 80) & (stoch_k <= stoch_d) & (~band_walk_up)
    signal[bullish_confirmed] = "BUY"
    signal[bearish_confirmed] = "SELL"

    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_pctb"] = bb_pctb
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    out["band_walk_state"] = band_walk_state
    out["stochbb_score"] = stochbb_score
    out["stochbb_signal"] = signal
    return out


def composite_signal(df: pd.DataFrame, mr_window: int = 20, mr_z_entry: float = 1.5,
                      mr_z_exit: float = 0.5, mom_fast: int = 10, mom_slow: int = 30,
                      mr_weight: float = 0.5,
                      stochbb_weight: float = 0.0, bb_window: int = 20, bb_std: float = 2.0,
                      stoch_window: int = 14, stoch_smooth: int = 3,
                      ml_score: pd.Series | None = None,
                      ml_weight: float = 0.0,
                      earnings_score: pd.Series | None = None,
                      earnings_weight: float = 0.0) -> pd.DataFrame:
    """Combine mean-reversion + momentum (+ optional Stochastic-Bollinger,
    + optional ML, + optional earnings-drift/PEAD) into composite score
    in [-1, +1].

    stochbb_weight=0.0 (default) means ZERO behavior change from before
    this was added — the Stochastic-Bollinger leg is opt-in, blended in
    on top of the classic MR/momentum composite the same way ml_score is
    (see stoch_bb_signal() for what it computes).

    earnings_weight=0.0 (default) is the same pattern again — pass in a
    series from earnings_drift_series() (computed OUTSIDE this function,
    same as ml_score, so this stays a pure/testable dataframe transform
    with no network calls of its own) and it blends in on top of
    everything else already combined. Unlike the other components, this
    one is built from GENUINELY different information (actual earnings
    surprises, not another transform of price/volume) — see
    earnings_drift_series() docs for why that matters."""
    out = mean_reversion_signal(df, mr_window, mr_z_entry, mr_z_exit)
    out = momentum_signal(out, mom_fast, mom_slow)

    mr_score = out["mr_zscore"].clip(-3, 3) / -3.0
    mom_score = np.sign(out["mom_diff"]).fillna(0)

    classic_composite = mr_weight * mr_score + (1 - mr_weight) * mom_score
    classic_composite = classic_composite.clip(-1, 1)

    if stochbb_weight > 0:
        stochbb_out = stoch_bb_signal(df, bb_window, bb_std, stoch_window, stoch_smooth)
        for col in ("bb_upper", "bb_mid", "bb_lower", "bb_pctb", "stoch_k", "stoch_d",
                    "band_walk_state", "stochbb_score", "stochbb_signal"):
            out[col] = stochbb_out[col].reindex(out.index)
        stochbb_aligned = out["stochbb_score"]
        has_stochbb = stochbb_aligned.notna()
        classic_composite = classic_composite.copy()
        classic_composite[has_stochbb] = ((1 - stochbb_weight) * classic_composite[has_stochbb]
                                           + stochbb_weight * stochbb_aligned[has_stochbb])
        classic_composite = classic_composite.clip(-1, 1)

    if ml_score is not None and ml_weight > 0:
        ml_aligned = ml_score.reindex(out.index)
        # where ML has no prediction (NaN, e.g. not enough training history yet),
        # fall back fully to the classic composite for that row
        has_ml = ml_aligned.notna()
        composite = classic_composite.copy()
        composite[has_ml] = ((1 - ml_weight) * classic_composite[has_ml]
                              + ml_weight * ml_aligned[has_ml])
        composite = composite.clip(-1, 1)
    else:
        composite = classic_composite

    if earnings_score is not None and earnings_weight > 0:
        earnings_aligned = earnings_score.reindex(out.index)
        has_earnings = earnings_aligned.notna()
        out["earnings_drift_score"] = earnings_aligned
        composite = composite.copy()
        composite[has_earnings] = ((1 - earnings_weight) * composite[has_earnings]
                                    + earnings_weight * earnings_aligned[has_earnings])
        composite = composite.clip(-1, 1)

    sig = pd.Series("HOLD", index=out.index)
    sig[composite >= 0.5] = "BUY"
    sig[composite <= -0.5] = "SELL"

    out["classic_composite_score"] = classic_composite
    out["composite_score"] = composite
    out["composite_signal"] = sig
    return out


# ==========================================================================
# ==== SECTION 3.5: ML SIGNAL (LightGBM, walk-forward, anti-lookahead) ====
# ==========================================================================
# Optional third signal source. Uses LightGBM classifier to predict
# P(next-day close > today's close) from engineered technical features.
#
# Anti-lookahead design: the model used to predict day i is trained ONLY on
# (features[j], target[j]) for j < i, retrained periodically (expanding
# window) rather than every single day, to keep runtime reasonable for an
# interactive dashboard.

@_cache_data(ttl=900, show_spinner=False)
def _fetch_benchmark_raw(asset_type: str, crypto_symbol: str = "",
                          exchange_id: str = "binance", n_bars: int = 500):
    """Best-effort raw OHLCV fetch for a relative-strength benchmark. Cached
    per (asset_type, crypto_symbol's quote currency, exchange) so a bulk
    screener/scan doesn't re-hit the same index dozens of times — every IDX
    ticker shares one ^JKSE fetch, every US ticker shares one ^GSPC fetch.
    Returns None on any failure; never raises."""
    try:
        if asset_type == "stock_id":
            return fetch_stock("^JKSE", period="2y")
        elif asset_type == "stock_us":
            return fetch_stock("^GSPC", period="2y")
        elif asset_type == "crypto":
            if crypto_symbol.upper().startswith("BTC/"):
                return None  # asset itself IS the benchmark — nothing to compare
            quote = crypto_symbol.upper().split("/")[-1] if "/" in crypto_symbol else "USDT"
            return fetch_crypto(f"BTC/{quote}", exchange_id=exchange_id,
                                 timeframe="1d", limit=n_bars + 30)
        return None
    except Exception:
        return None


def fetch_benchmark_prices(asset_type: str, symbol: str, index: pd.DatetimeIndex,
                            exchange_id: str = "binance") -> pd.Series | None:
    """
    Best-effort relative-strength benchmark series, reindexed/ffilled to the
    main asset's dates. Never raises — this is a nice-to-have context feature
    for the ML model (single-asset technicals alone can't tell "stock is down
    3% because the whole market is down 4%" from "stock is down 3% while the
    market is flat"), not a hard requirement, so a flaky benchmark silently
    degrades rather than breaking the main analysis.
        stock_id -> ^JKSE (IHSG)   stock_us -> ^GSPC (S&P 500)
        crypto   -> BTC/<same quote as symbol> (skipped for BTC pairs themselves)
    """
    raw = _fetch_benchmark_raw(asset_type, crypto_symbol=symbol if asset_type == "crypto" else "",
                                exchange_id=exchange_id, n_bars=len(index))
    if raw is None or raw.empty:
        return None
    bench = raw["Close"].reindex(index).ffill()
    return bench if bench.notna().sum() > 30 else None


def compute_ml_features(df: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Engineer technical features from OHLCV (all backward-looking, no lookahead).
    Pass `benchmark` (a price series aligned to df's index) to add relative-
    strength / rolling-beta features against a market benchmark — optional,
    everything else works identically without it."""
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # --- returns at multiple horizons ---
    feat["ret_1d"] = close.pct_change(1)
    feat["ret_3d"] = close.pct_change(3)
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)
    feat["ret_20d"] = close.pct_change(20)

    # --- volatility ---
    feat["vol_10d"] = close.pct_change().rolling(10).std()
    feat["vol_20d"] = close.pct_change().rolling(20).std()

    # --- moving average ratios ---
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    feat["ma_ratio_10"] = close / ma10 - 1
    feat["ma_ratio_20"] = close / ma20 - 1
    feat["ma_ratio_50"] = close / ma50 - 1

    # --- RSI(14) ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD (12,26,9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd_line - macd_signal) / close  # normalized by price for scale-invariance

    # --- Bollinger %B (20, 2std): 0 = at lower band, 1 = at upper band ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # --- ATR(14) normalized by price (volatility-of-range, distinct from close-to-close vol) ---
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feat["atr_pct"] = atr14 / close

    # --- Stochastic %K(14) ---
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    feat["stoch_k"] = (close - low14) / (high14 - low14).replace(0, np.nan) * 100

    # --- volume ---
    vol_ma20 = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # --- mean-reversion z-score (same concept as the classic signal, useful as an ML feature too) ---
    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    feat["zscore_20"] = (close - roll_mean) / roll_std.replace(0, np.nan)

    # --- price acceleration (2nd derivative of returns — is momentum speeding up or slowing down) ---
    feat["accel_5d"] = feat["ret_1d"].rolling(5).mean().diff(5)

    # --- relative strength / rolling beta vs a market benchmark (optional) ---
    # Everything above only looks at the asset in isolation. A -3% day means
    # something very different when the benchmark is -4% (asset is actually
    # relatively STRONG) versus when the benchmark is flat (asset is weak on
    # its own) — that distinction is invisible to single-asset technicals,
    # so it's a genuinely new source of predictive information, not just a
    # restatement of the features above.
    if benchmark is not None and benchmark.notna().sum() > 30:
        bench_ret_5d = benchmark.pct_change(5)
        bench_ret_20d = benchmark.pct_change(20)
        feat["rel_strength_5d"] = feat["ret_5d"] - bench_ret_5d
        feat["rel_strength_20d"] = feat["ret_20d"] - bench_ret_20d
        asset_d = close.pct_change()
        bench_d = benchmark.pct_change()
        roll_cov = asset_d.rolling(60).cov(bench_d)
        roll_var = bench_d.rolling(60).var()
        feat["beta_60d"] = roll_cov / roll_var.replace(0, np.nan)

    return feat


def walk_forward_ml_signal(df: pd.DataFrame, min_train_days: int = 100,
                            retrain_every: int = 20, n_estimators: int = 150,
                            max_depth: int = 4, horizon_days: int = 1,
                            model_type: str = "lightgbm",
                            benchmark: pd.Series | None = None):
    """
    Anti-lookahead walk-forward ML signal with a configurable prediction
    HORIZON instead of always next-day. For a swing-trading dashboard (hold
    1-5+ days), predicting "does price rise over the NEXT `horizon_days`
    trading days" is a much better match than a pure next-day classifier —
    horizon_days should generally track the trading-style preset's holding
    period, not stay fixed at 1.

    Every retrain purges/embargoes the last `horizon_days` labeled rows
    before the retrain point (see comment in the loop below) — without that,
    walk-forward "accuracy" is quietly inflated by labels that peek at
    information from on/after the day being predicted.

    model_type: "lightgbm" (default), "xgboost", or "ensemble" (trains BOTH
    and averages their predicted probabilities — costs ~2x the compute of a
    single model, since it's literally fitting two models at every retrain,
    but can smooth out idiosyncrasies of either individual model).

    benchmark: optional price series (same index as df) of a benchmark asset
    — IHSG/S&P500 for stocks, BTC for altcoins — used to add relative-
    strength / rolling-beta features in compute_ml_features. Pass None to
    skip (features degrade gracefully either way).

    Also fits a companion regressor predicting the forward return MAGNITUDE
    (not just up/down), so callers can show "predicted +2.3% over 5 days"
    alongside the probability, not just a bare direction. Regressor always
    uses LightGBM regardless of model_type (XGBoost regressor adds complexity
    for a secondary/optional output — not worth doubling here too).

    Returns:
        score: pd.Series in [-1, +1] (2*P(up)-1), NaN where no model available
            yet. Unlike before, this now also covers the live tail (the most
            recent `horizon_days` rows, filled in using final_model) instead
            of leaving today's row empty.
        final_model: fitted classifier trained on ALL labeled history (for live
            "today" prediction) — a single model, or a (lgbm, xgb) tuple when
            model_type="ensemble"
        feature_importance: pd.Series, sorted descending (LightGBM's importances
            are used as the representative one even in ensemble mode, for a
            single consistent chart rather than two conflicting ones)
        features: the raw feature DataFrame (for live prediction row lookup)
        final_regressor: LGBMRegressor predicting forward return magnitude (or None)
        calibrator: fitted IsotonicRegression mapping raw P(up) -> calibrated
            P(up), trained on the honest out-of-sample walk-forward history
            (or None if there isn't enough history yet — needs >=50 rows)
        diagnostics: dict with "proba_raw"/"actual"/"proba_calibrated" arrays
            from that same out-of-sample history — feeds the reliability
            curve / residual plots in the ML tab (or None, same condition
            as calibrator).
    """
    from lightgbm import LGBMClassifier, LGBMRegressor

    def _make_classifier(kind: str):
        if kind == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=0.05, eval_metric="logloss",
                                  verbosity=0, random_state=42)
        return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               learning_rate=0.05, verbosity=-1, random_state=42)

    def _fit_model(X_train, y_train):
        if model_type == "ensemble":
            m_lgbm = _make_classifier("lightgbm")
            m_xgb = _make_classifier("xgboost")
            m_lgbm.fit(X_train, y_train)
            m_xgb.fit(X_train, y_train)
            return (m_lgbm, m_xgb)
        m = _make_classifier(model_type)
        m.fit(X_train, y_train)
        return m

    def _predict_proba_up(model, x_row):
        if model_type == "ensemble":
            p1 = model[0].predict_proba(x_row)[0, 1]
            p2 = model[1].predict_proba(x_row)[0, 1]
            return (p1 + p2) / 2
        return model.predict_proba(x_row)[0, 1]

    features = compute_ml_features(df, benchmark=benchmark)
    fwd_return = df["Close"].shift(-horizon_days) / df["Close"] - 1
    target = (fwd_return > 0).astype(int)
    n = len(df)
    proba = pd.Series(np.nan, index=df.index)

    model = None
    last_train_idx = -10**9

    for i in range(min_train_days, n - horizon_days):
        if model is None or (i - last_train_idx) >= retrain_every:
            # ---- purge / embargo (Lopez de Prado, "Advances in Financial
            # Machine Learning" — see mlfinlab / AFML in awesome-quant) ----
            # target[j] = 1{Close[j+horizon_days] > Close[j]}, so as of day i
            # we have only "really" observed target[j] once
            # j + horizon_days <= i - 1. The previous version trained on
            # features.iloc[:i] directly, which for j close to i (within
            # horizon_days) uses a label that peeks at Close[>= i] — i.e. at
            # or after the very day being predicted. That's a subtle
            # lookahead leak: it doesn't crash anything, it just quietly
            # inflates the walk-forward "accuracy" versus what a live model
            # could actually have known. Embargoing the last `horizon_days`
            # labeled rows before every retrain removes that leak.
            train_end = i - horizon_days
            X_train = features.iloc[:train_end].dropna()
            y_train = target.iloc[:train_end].reindex(X_train.index)
            if len(X_train) < 30 or y_train.nunique() < 2:
                continue
            model = _fit_model(X_train, y_train)
            last_train_idx = i

        x_i = features.iloc[[i]]
        if not x_i.isna().any(axis=1).iloc[0]:
            proba.iloc[i] = _predict_proba_up(model, x_i)

    X_all = features.iloc[:-horizon_days].dropna()
    y_all = target.iloc[:-horizon_days].reindex(X_all.index)
    y_ret_all = fwd_return.iloc[:-horizon_days].reindex(X_all.index)
    final_model, feature_importance, final_regressor = None, None, None
    if len(X_all) >= 30 and y_all.nunique() >= 2:
        final_model = _fit_model(X_all, y_all)
        lgbm_for_importance = final_model[0] if model_type == "ensemble" else final_model
        feature_importance = pd.Series(lgbm_for_importance.feature_importances_,
                                        index=X_all.columns).sort_values(ascending=False)

        # companion regressor for expected return magnitude — trained on the
        # same rows, separate model since classification and regression
        # losses optimize different things (and always LightGBM regardless
        # of model_type, to keep this secondary output simple)
        y_ret_clean = y_ret_all.dropna()
        X_ret = X_all.loc[y_ret_clean.index]
        if len(X_ret) >= 30:
            final_regressor = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                             learning_rate=0.05, verbosity=-1, random_state=42)
            final_regressor.fit(X_ret, y_ret_clean)

    # ---- fill the "live tail" using final_model ----
    # The walk-forward loop above deliberately stops at n - horizon_days,
    # because rows after that don't have a resolved label yet (there's no
    # future price to check). That silently left the most recent
    # `horizon_days` rows — i.e. TODAY, exactly the row the whole dashboard's
    # final verdict reads from — without an ML score, so composite_signal's
    # "no ML prediction yet -> fall back to classic-only" path was quietly
    # eating the live ML view on the one row that matters most. final_model
    # only ever trained on rows up to n - horizon_days - 1, so applying it to
    # these newer rows is a legitimate forward prediction, not a leak.
    if final_model is not None:
        tail_start = max(min_train_days, n - horizon_days)
        for i in range(tail_start, n):
            x_i = features.iloc[[i]]
            if not x_i.isna().any(axis=1).iloc[0]:
                proba.iloc[i] = _predict_proba_up(final_model, x_i)

    score = 2 * proba - 1  # map P(up) in [0,1] to score in [-1,+1]

    # ---- post-hoc probability calibration ----
    # Tree ensembles like LightGBM/XGBoost are usually good at RANKING
    # (who's more likely to go up) but their raw predict_proba output is
    # often poorly calibrated as an actual probability — e.g. among all days
    # scored "70% up" the true up-rate might really be 55%. The walk-forward
    # `proba` values above (excluding the live tail just filled in, which by
    # construction has no resolved label yet) are honest, leak-free
    # out-of-sample predictions with known outcomes, which is exactly the
    # right dataset to fit a monotonic calibration correction on. Applied
    # later to the live prediction so the probability shown to the user is
    # closer to its true empirical frequency instead of the model's raw score.
    calibrator = None
    calib_mask = proba.notna()
    if final_model is not None:
        calib_mask.iloc[tail_start:] = False  # exclude the live tail just filled in above
    diagnostics = None
    if calib_mask.sum() >= 50:
        calib_y = target.reindex(proba.index)[calib_mask]
        if calib_y.nunique() == 2:
            from sklearn.isotonic import IsotonicRegression
            # FIX 1c: calibrator TIDAK boleh di-fit dan dievaluasi di data yang
            # sama (BSS jadi optimistis -> adaptive_ml_weight kebablasan).
            # Split OOS jadi dua bagian WAKTU: fit di paruh pertama, evaluasi
            # diagnostics/BSS HANYA di paruh kedua.
            calib_X = proba[calib_mask].values
            calib_yv = calib_y.values
            mid = len(calib_X) // 2
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(calib_X[:mid], calib_yv[:mid])
            # ---- diagnostics payload for the Error Analysis / Model Diagnostics
            # expander in the ML tab: the same honest, leak-free out-of-sample
            # (raw proba, actual outcome) pairs used to fit the calibrator above,
            # plus the calibrated version of those same predictions — enough to
            # draw a reliability curve and inspect residuals without re-running
            # the walk-forward loop a second time.
            diag_proba_raw = calib_X[mid:]  # evaluasi HANYA di paruh kedua (unseen oleh calibrator)
            diagnostics = {
                "proba_raw": diag_proba_raw,
                "actual": calib_yv[mid:],
                "proba_calibrated": calibrator.predict(diag_proba_raw),
            }

    return score, final_model, feature_importance, features, final_regressor, calibrator, diagnostics


def ml_calibration_confidence(diagnostics: dict | None) -> tuple[float, dict | None]:
    """
    Turns the walk-forward out-of-sample calibration diagnostics (the same
    ones behind the reliability curve / Error Analysis expander in the
    Signal tab) into a single [0, 1] confidence multiplier, via the Brier
    Skill Score (BSS) against a "no-skill" baseline that always predicts
    the empirical base rate of the window:

        BSS = 1 - brier(calibrated predictions) / brier(base-rate-only)

    BSS = 1 -> perfect calibration; BSS = 0 -> no better than always
    guessing the historical base rate; BSS < 0 -> actively worse than that
    baseline. Clipped to [0, 1] here — a model with negative skill gets
    ZERO say in the composite rather than being allowed to actively hurt
    it (there's no principled reason a worse-than-baseline model should
    still get partial credit).

    Returns (confidence, detail_dict). detail_dict is None only when
    `diagnostics` itself is None (not enough out-of-sample history yet,
    same >=50-row threshold as the calibrator/reliability curve) — in that
    case confidence is 1.0, i.e. "no evidence either way, don't
    second-guess the user's slider value."
    """
    if diagnostics is None:
        return 1.0, None

    actual = np.asarray(diagnostics["actual"], dtype=float)
    calib = np.asarray(diagnostics["proba_calibrated"], dtype=float)
    base_rate = float(actual.mean())
    brier_model = float(np.mean((calib - actual) ** 2))
    brier_baseline = base_rate * (1 - base_rate)

    if brier_baseline <= 1e-9:
        # degenerate window: outcome was ~always the same class, BSS isn't
        # meaningful (any constant predictor "wins" trivially) — treat as
        # neutral rather than either rewarding or punishing the model for it
        bss = 0.0
    else:
        bss = 1 - brier_model / brier_baseline

    confidence = float(np.clip(bss, 0.0, 1.0))
    return confidence, {
        "bss": bss, "brier_model": brier_model,
        "brier_baseline": brier_baseline, "confidence": confidence,
    }


def confidence_scaled_position_size(base_position_pct: float,
                                     composite_score: float | None = None,
                                     ml_confidence: float | None = None,
                                     min_scale: float = 0.25,
                                     max_scale: float = 1.5,
                                     vol_target_annual: float | None = None,
                                     realized_vol_annual: float | None = None) -> dict:
    """
    Scales a base position-size fraction up/down based on how much genuine
    conviction actually backs a given signal, instead of every BUY getting
    the same flat `position_size_pct` regardless of how marginal or
    strong it is. Three independent multipliers, each optional:

    1. `composite_score` conviction: composite_signal's BUY/SELL threshold
       fires at |score| > 0.5. A score of 0.52 just barely crossed that
       line; a score of 0.95 is close to maximum conviction. Linearly
       rescales |score| in [0.5, 1.0] to a size multiplier in
       [min_scale, max_scale] — a razor-thin signal gets sized small, a
       strong one gets sized larger, both still capped at max_scale.

    2. `ml_confidence`: the [0, 1] Brier-Skill-Score-based confidence from
       `ml_calibration_confidence()`. This is deliberately a SEPARATE
       input from composite_score, not folded into it — a ticker/window
       where the ML leg has shown ZERO genuine out-of-sample skill
       (confidence near 0) gets pulled toward min_scale regardless of how
       extreme the raw composite score looks, because part of that score
       may be coming from an ML component that isn't actually predictive
       right now. confidence=1.0 (perfect skill, or no evidence yet —
       see ml_calibration_confidence's own docstring) leaves this
       multiplier untouched.

    3. Inverse-volatility targeting (`vol_target_annual` /
       `realized_vol_annual`, both optional, both needed together to
       activate): the SAME position_size_pct risks very different amounts
       of money on a calm blue-chip vs. a small-cap that swings 5-10%/day.
       Multiplies by vol_target_annual / realized_vol_annual (clipped to
       [0.25, 2.0]) so a stated risk budget (e.g. "I want ~20% annualized
       portfolio vol from this position") is comparable across tickers,
       instead of an illiquid small-cap silently eating a much larger
       risk budget than a blue-chip at the identical slider setting.

    This function is a SIZING multiplier only — it never overrides the
    BUY/SELL/HOLD decision itself (composite_signal's job alone). Result
    is always clipped to [0.01, 1.0]: never negative or above 100% of
    allocated capital, and never driven all the way to zero (a signal
    that already decided to fire shouldn't silently become a phantom
    zero-size trade).

    Returns a dict: {position_size_pct, scale, detail} where `detail`
    breaks down each multiplier applied, for display/debugging.
    """
    scale = 1.0
    detail = {}

    if composite_score is not None:
        conviction = float(np.clip((abs(composite_score) - 0.5) / 0.5, 0.0, 1.0))
        conviction_mult = min_scale + conviction * (max_scale - min_scale)
        scale *= conviction_mult
        detail["conviction_mult"] = conviction_mult

    if ml_confidence is not None:
        conf = float(np.clip(ml_confidence, 0.0, 1.0))
        confidence_mult = min_scale + conf * (1.0 - min_scale)
        scale *= confidence_mult
        detail["confidence_mult"] = confidence_mult

    if (vol_target_annual is not None and realized_vol_annual is not None
            and realized_vol_annual > 1e-6):
        vol_mult = float(np.clip(vol_target_annual / realized_vol_annual, 0.25, 2.0))
        scale *= vol_mult
        detail["vol_target_mult"] = vol_mult

    position_size_pct = float(np.clip(base_position_pct * scale, 0.01, 1.0))
    detail["combined_scale"] = float(scale)
    return {"position_size_pct": position_size_pct, "scale": float(scale), "detail": detail}


# ==========================================================================
# ==== SECTION 4: BACKTEST ====
# ==========================================================================
# Long-only backtest engine with anti-lookahead execution (signal from
# close[T] is actioned at close[T+1]), transaction costs, and walk-forward
# validation splitting.

def run_backtest(df_with_signals: pd.DataFrame, signal_col: str = "composite_signal",
                  fee_bps: float = 10, slippage_bps: float = 5,
                  fee_buy_bps: float | None = None, fee_sell_bps: float | None = None,
                  trading_days: float = 252.0,
                  max_turnover_participation: float | None = None,
                  idx_realism: bool = False, lot_size: int = 100,
                  initial_capital: float = 100.0,
                  position_size_pct: float = 1.0,
                  stop_loss_pct: float | None = None,
                  max_holding_days: int | None = None,
                  execution_price: str = "next_open",
                  dynamic_slippage: bool = False,
                  spread_window: int = 20) -> dict:
    """
    Simulates trading `signal_col` (BUY/SELL/EXIT/HOLD) against OHLC data.

    Realism knobs (all optional, tuned to sane defaults — see each param):
      - position_size_pct: fraction (0-1] of available cash deployed per
        entry. Default 1.0 = all-in (matches the original behavior, kept
        as default so existing callers — GA optimizer, Screener — don't
        silently change meaning). Set lower (e.g. 0.3) to model "only risk
        part of the account per trade" instead of always going all-in.
      - stop_loss_pct: if set (e.g. 0.05 = 5%), force-exits the FIRST day
        the day's Low breaches entry_price*(1-stop_loss_pct) — independent
        of what the signal says, same as a real resting stop order. Exit
        fills AT the stop level itself (standard backtest convention —
        assumes the stop order fills near the trigger, not that day's
        close/open, which would be optimistic or pessimistic bias
        depending on direction).
      - max_holding_days: force-exits after N days in a position even if
        the signal never says SELL/EXIT — avoids "backtest waits forever
        for an exit signal while the position quietly bleeds."
      - execution_price: "next_open" (default) — a signal computed off day
        T's Close is only realistically actionable at day T+1's Open, the
        earliest fill you could actually get. "next_close" reproduces the
        OLD default (fills at day T+1's Close) — an optimistic best-case
        fill kept only for side-by-side comparison, not recommended as
        the number you trust.
      - dynamic_slippage: if True, slippage_bps is estimated PER TICKER
        from OHLC-based spread estimators (see estimate_spread_stats)
        instead of one flat number for every ticker — a thin/illiquid
        stock gets realistically wider slippage than a blue-chip. Falls
        back to the flat `slippage_bps` if no estimate is available (e.g.
        too little history).
    """
    df = df_with_signals.copy()

    price_col = "Open" if execution_price == "next_open" else "Close"
    action_price = df[price_col].shift(-1)
    signal_actioned = df[signal_col]

    # FIX 2a: fee asimetris — sisi jual IDX lebih mahal (komisi lebih tinggi
    # + pajak final 0.1% hanya di sisi jual). Default fallback ke fee_bps
    # lama supaya caller lama (GA, dsb) tidak berubah perilaku.
    fee_buy_bps = fee_bps if fee_buy_bps is None else fee_buy_bps
    fee_sell_bps = fee_bps if fee_sell_bps is None else fee_sell_bps

    _slippage_samples: list[float] = []

    def _slippage_bps_at(i: int) -> float:
        """FIX 1f: estimasi spread per-tanggal, HANYA dari data s/d hari i.
        Versi lama memakai df.tail(20) dari AKHIR sampel untuk mensimulasikan
        trade di awal histori — lookahead halus."""
        if not dynamic_slippage:
            return slippage_bps
        est = estimate_spread_stats(df.iloc[:i + 1], window=spread_window).get("consensus_spread_pct")
        slip = est * 100 if est is not None else slippage_bps
        if idx_realism:
            # 2b: slippage minimum efektif = 1 tick. Untuk saham < Rp200,
            # 1 tick = Rp1 = > 0.5% — jauh di atas slippage flat default.
            px = df["Close"].iloc[i]
            if px > 0:
                slip = max(slip, idx_tick_size(px) / px * 1e4)
        return slip

    cost_rate_bh = (fee_buy_bps + slippage_bps) / 10000.0  # untuk kurva buy & hold
    position_size_pct = min(max(position_size_pct, 0.01), 1.0)  # guard against 0 or >100%

    cash = initial_capital
    idle_cash = 0.0        # portion deliberately held back per position_size_pct
    position = 0.0
    equity = []
    trades = []
    in_position = False
    entry_price = None
    entry_date = None
    entry_idx = None
    stop_price = None
    dates = df.index
    lows = df["Low"].values
    closes = df["Close"].values

    def _close_trade(exit_price, exit_date, reason, exit_i: int):
        nonlocal cash, position, in_position, entry_price, entry_date, entry_idx, stop_price, idle_cash
        slip = _slippage_bps_at(exit_i)
        _slippage_samples.append(slip)
        proceeds = position * exit_price * (1 - (fee_sell_bps + slip) / 10000.0)
        trades.append({
            "entry_date": entry_date, "exit_date": exit_date,
            "entry_price": entry_price, "exit_price": exit_price,
            "return_pct": (exit_price / entry_price - 1) * 100,
            "exit_reason": reason,
        })
        cash = proceeds + idle_cash
        idle_cash = 0.0
        position = 0.0
        in_position = False
        entry_price = None
        entry_date = None
        entry_idx = None
        stop_price = None

    for i in range(len(df) - 1):
        sig = signal_actioned.iloc[i]
        price = action_price.iloc[i]

        # ---- Stop-loss / max-holding checks happen FIRST, using TODAY's
        # own Low / holding length — these are risk controls independent
        # of signal timing, unlike entries/signal-exits which wait for the
        # next bar's execution price. ----
        if in_position and stop_loss_pct is not None and not pd.isna(lows[i]) and stop_price is not None:
            if lows[i] <= stop_price:
                if idx_realism and _is_locked_arb(df, i):
                    # 2b: terkunci ARB — tidak ada bid, stop tidak bisa fill.
                    # Posisi terbawa ke hari berikutnya (gap risk ril).
                    equity.append(cash + idle_cash + position * closes[i])
                    continue
                _close_trade(stop_price, dates[i], "stop_loss", i)
                equity.append(cash)
                continue

        if (in_position and max_holding_days is not None and entry_idx is not None
                and not pd.isna(price) and (i - entry_idx) >= max_holding_days):
            _close_trade(price, dates[i + 1], "max_holding_days", i + 1)
            equity.append(cash + position * closes[i])
            continue

        if pd.isna(price):
            equity.append(cash + idle_cash + position * closes[i])
            continue

        if not in_position and sig == "BUY":
            slip = _slippage_bps_at(i)
            _slippage_samples.append(slip)
            deploy_cash = cash * position_size_pct
            if max_turnover_participation is not None:
                # FIX 2c: cap nilai posisi ke fraksi turnover harian rata-rata
                # (trailing 20 hari s/d hari sinyal — anti-lookahead). Mencegah
                # backtest "menghasilkan" return yang secara fisik tidak bisa
                # dieksekusi di saham tipis.
                _to_avg = float((df["Close"] * df["Volume"]).iloc[:i + 1].tail(20).mean())
                deploy_cash = min(deploy_cash, max_turnover_participation * _to_avg)
            idle_cash = cash - deploy_cash
            units = (deploy_cash * (1 - (fee_buy_bps + slip) / 10000.0)) / price
            if idx_realism:
                # 2b: 1 lot = 100 lembar. Dengan modal 10jt & saham Rp9.000,
                # rounding lot mengubah sizing secara material.
                units = float(np.floor(units / lot_size) * lot_size)
                if units <= 0:
                    equity.append(cash + idle_cash + position * closes[i])
                    continue  # modal tidak cukup untuk 1 lot — trade batal
                spent = units * price / (1 - (fee_buy_bps + slip) / 10000.0)
                idle_cash = cash - spent
            position = units
            cash = 0.0
            in_position = True
            entry_price = price
            entry_date = dates[i + 1]
            entry_idx = i + 1
            stop_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct is not None else None

        elif in_position and sig in ("SELL", "EXIT"):
            if idx_realism and _is_locked_arb(df, i + 1):
                # 2b: hari eksekusi terkunci ARB — exit gagal, coba lagi besok.
                equity.append(cash + idle_cash + position * closes[i])
                continue
            _close_trade(price, dates[i + 1], "signal", i + 1)

        equity.append(cash + idle_cash + position * closes[i])

    if in_position:
        last_price = closes[-1]
        _close_trade(last_price, dates[-1], "end_of_data", len(df) - 1)
        equity.append(cash)

    equity_curve = pd.Series(equity, index=dates[:len(equity)], name="equity")
    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(equity_curve, trades_df, initial_capital,
                                trading_days=trading_days)
    metrics["effective_slippage_bps"] = (float(np.mean(_slippage_samples))
                                          if _slippage_samples else float(slippage_bps))
    metrics["fee_buy_bps"] = float(fee_buy_bps)
    metrics["fee_sell_bps"] = float(fee_sell_bps)
    metrics["position_size_pct"] = float(position_size_pct)
    if len(trades_df) > 0 and "exit_reason" in trades_df.columns:
        metrics["n_stop_loss_exits"] = int((trades_df["exit_reason"] == "stop_loss").sum())
        metrics["n_max_holding_exits"] = int((trades_df["exit_reason"] == "max_holding_days").sum())

    bh_units = (initial_capital * (1 - cost_rate_bh)) / df["Close"].iloc[0]
    buy_hold_curve = bh_units * df["Close"].iloc[:len(equity)]
    metrics["buy_hold_final_value"] = float(buy_hold_curve.iloc[-1])
    metrics["buy_hold_return_pct"] = float((buy_hold_curve.iloc[-1] / initial_capital - 1) * 100)

    return {
        "equity_curve": equity_curve,
        "buy_hold_curve": buy_hold_curve,
        "trades": trades_df,
        "metrics": metrics,
    }


def _compute_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame,
                      initial_capital: float, trading_days: float = 252.0) -> dict:
    if len(equity_curve) < 2:
        return {"error": "not enough data to compute metrics"}

    daily_ret = equity_curve.pct_change().dropna()
    n_days = len(equity_curve)
    years = n_days / trading_days

    final_value = equity_curve.iloc[-1]
    total_return_pct = (final_value / initial_capital - 1) * 100
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(trading_days)
              if daily_ret.std(ddof=1) > 0 else np.nan)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    if len(trades_df) > 0:
        win_rate = (trades_df["return_pct"] > 0).mean() * 100
        avg_trade_return = trades_df["return_pct"].mean()
        n_trades = len(trades_df)
    else:
        win_rate, avg_trade_return, n_trades = np.nan, np.nan, 0

    return {
        "final_value": float(final_value),
        "total_return_pct": float(total_return_pct),
        "cagr_pct": float(cagr) if not np.isnan(cagr) else None,
        "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) else None,
        "max_drawdown_pct": float(max_dd_pct),
        "n_trades": int(n_trades),
        "win_rate_pct": float(win_rate) if not np.isnan(win_rate) else None,
        "avg_trade_return_pct": float(avg_trade_return) if not np.isnan(avg_trade_return) else None,
    }


def compute_liquidity_stats(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Proxy for real orderbook/liquidity depth, which isn't available for free
    via yfinance (unlike crypto exchanges, IDX/US stock bid-ask depth isn't
    exposed). Uses average daily turnover value (Volume × Close) over the
    recent window as a rough liquidity indicator — thin turnover is a decent
    proxy for "hard to enter/exit without moving the price yourself", even
    without seeing the actual order book.

    CAVEAT (see estimate_spread_stats/classify_liquidity below): turnover
    alone can look non-trivially large off a single block trade even on a
    ticker that otherwise barely moves — this is why turnover is combined
    with OHLC-based spread/frozen-day evidence before anything gets
    hard-blocked, not used alone.
    """
    recent = df.tail(window)
    turnover = recent["Close"] * recent["Volume"]
    return {
        "avg_turnover": float(turnover.mean()),
        "median_turnover": float(turnover.median()),
        "avg_volume": float(recent["Volume"].mean()),
        "latest_volume": float(df["Volume"].iloc[-1]),
        "volume_vs_avg_ratio": float(df["Volume"].iloc[-1] / recent["Volume"].mean())
        if recent["Volume"].mean() > 0 else None,
    }


# ---- OHLC-only bid-ask spread estimators ---------------------------------
# We don't have real orderbook/tick data (would need a paid provider), so
# these estimate the effective bid-ask spread purely from Open/High/Low/
# Close — three published, peer-reviewed academic estimators, from oldest/
# crudest to newest/most robust. Having three lets us take a consensus
# (median) instead of trusting a single estimator's assumptions on names
# with unusual trading patterns.

def _edge_spread_estimate(open_, high, low, close) -> float:
    """
    EDGE estimator — Ardia, Guidotti & Kroencke, "Efficient Estimation of
    Bid-Ask Spreads from Open, High, Low, and Close Prices", Journal of
    Financial Economics (2024), https://doi.org/10.1016/j.jfineco.2024.103916.
    Adapted (single-window form) from the reference implementation at
    github.com/eguidotti/bidask (MIT License) so this stays a single-file
    app with no extra pip dependency. Returns the estimated relative spread
    (0.01 = 1%), or NaN when the window has too little genuine price
    variation to identify a spread at all — which, for a near-dead ticker,
    is itself the important signal (see frozen_days_ratio below).
    """
    o = np.log(np.asarray(open_, dtype=float))
    h = np.log(np.asarray(high, dtype=float))
    l = np.log(np.asarray(low, dtype=float))
    c = np.log(np.asarray(close, dtype=float))
    if len(o) < 3:
        return float("nan")

    m = (h + l) / 2.0
    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]
    r1, r2, r3, r4, r5 = m - o, o - m1, m - c1, c1 - m1, o - c1

    tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan, (h != l) | (l != c1))
    po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, o != h)
    po2 = tau * np.where(np.isnan(o) | np.isnan(l), np.nan, o != l)
    pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
    pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pt = np.nanmean(tau)
        po = np.nanmean(po1) + np.nanmean(po2)
        pc = np.nanmean(pc1) + np.nanmean(pc2)
        if np.nansum(tau) < 2 or po == 0 or pc == 0:
            return float("nan")
        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau
        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4
        e1, e2 = np.nanmean(x1), np.nanmean(x2)
        v1 = np.nanmean(x1 ** 2) - e1 ** 2
        v2 = np.nanmean(x2 ** 2) - e2 ** 2

    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0
    return float(np.sqrt(abs(s2)))


def _corwin_schultz_spread_estimate(high, low) -> float:
    """
    Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from
    Daily High and Low Prices", Journal of Finance 67(2). Uses only
    High/Low over rolling 2-day windows, averaged across the estimation
    window. Negative daily estimates (the estimator isn't bounded below by
    construction) are clipped to 0 before averaging, per the paper.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    n = len(h)
    if n < 3:
        return float("nan")
    k = 3 - 2 * np.sqrt(2)
    ests = []
    for t in range(n - 1):
        beta = np.log(h[t] / l[t]) ** 2 + np.log(h[t + 1] / l[t + 1]) ** 2
        h2, l2 = max(h[t], h[t + 1]), min(l[t], l[t + 1])
        if h2 <= 0 or l2 <= 0:
            continue
        gamma = np.log(h2 / l2) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(max(gamma, 0) / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        ests.append(max(0.0, s))
    return float(np.mean(ests)) if ests else float("nan")


def _roll_spread_estimate(close) -> float:
    """
    Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread
    in an Efficient Market", Journal of Finance 39(4). Based on the
    (negative) serial covariance of consecutive log-price changes — only
    well-defined when that covariance is negative, as bid-ask bounce
    implies. Returns NaN otherwise, which happens often on thin names that
    barely move day to day — the least robust of the three estimators here,
    kept mainly as a classic reference point / sanity check on the other two.
    """
    c = np.asarray(close, dtype=float)
    if len(c) < 3:
        return float("nan")
    r = np.diff(np.log(c))
    cov = float(np.cov(r[:-1], r[1:])[0, 1])
    if cov >= 0:
        return float("nan")
    return float(2 * np.sqrt(-cov))


def estimate_spread_stats(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Combines the three OHLC-only spread estimators above into one consensus
    (median of whichever return a valid number) plus a directly observable
    "frozen_days_ratio": the fraction of the last `window` sessions where
    High == Low (i.e. the price didn't even tick intraday). This is a more
    direct "is this actually trading" signal than turnover value alone,
    since a single block trade can make turnover look non-trivial on a
    stock that otherwise never moves — exactly the TIRT-style case (flat
    price, near-empty volume histogram, but turnover computed as > 0).
    """
    recent = df.tail(window)
    o = recent["Open"].values
    h = recent["High"].values
    l = recent["Low"].values
    c = recent["Close"].values

    edge_spread = _edge_spread_estimate(o, h, l, c)
    cs_spread = _corwin_schultz_spread_estimate(h, l)
    roll_spread = _roll_spread_estimate(c)
    frozen_days_ratio = float(np.mean(np.isclose(h, l))) if len(h) else float("nan")

    valid = [x for x in (edge_spread, cs_spread, roll_spread) if pd.notna(x)]
    consensus = float(np.median(valid)) if valid else float("nan")

    return {
        "edge_spread_pct": edge_spread * 100 if pd.notna(edge_spread) else None,
        "corwin_schultz_spread_pct": cs_spread * 100 if pd.notna(cs_spread) else None,
        "roll_spread_pct": roll_spread * 100 if pd.notna(roll_spread) else None,
        "consensus_spread_pct": consensus * 100 if pd.notna(consensus) else None,
        "frozen_days_ratio": frozen_days_ratio,
    }


LIQUIDITY_THRESHOLDS = {
    "stock_id": 1_000_000_000,   # Rp 1 Miliar/hari, sesuai saran umum di feedback
    "stock_us": 1_000_000,       # $1M/hari, padanan kasar untuk small-cap US
    "crypto": None,              # dilewati — likuiditas crypto sangat bervariasi per exchange
}

# Per feedback (kasus TIRT: ditandai BUY walau harga flat & volume nyaris
# kosong hampir tiap hari, sementara label "Tipis" cuma informational dan
# tidak nge-block verdict-nya). Threshold di atas tetap dipakai untuk label
# "Tipis" (soft, informational — perilaku lama tidak berubah), tapi sekarang
# ada tier kedua yang benar-benar nge-cap verdict BUY jadi HOLD:
HARD_ILLIQUID_TURNOVER_RATIO = 0.25     # turnover < 25% dari threshold "Tipis"
HARD_ILLIQUID_FROZEN_DAYS_RATIO = 0.30  # >= 30% dari `window` hari terakhir sama sekali tidak ada pergerakan intraday (High == Low)


def classify_liquidity(df: pd.DataFrame, asset_type: str, window: int = 20) -> dict:
    """
    Single entry point used by both the Screener table and the Kesimpulan
    (Tab 8) verdict — combines turnover-based and OHLC-spread-based checks
    into a 3-tier classification:
      - "liquid":         tidak ada concern berdasarkan data yang ada
      - "thin":            label informational saja (perilaku lama, tidak berubah)
      - "hard_illiquid":   verdict BUY di-cap ke HOLD (Liquidity Guard) — baru
    """
    liq_stats = compute_liquidity_stats(df, window=window)
    spread_stats = estimate_spread_stats(df, window=window)
    threshold = LIQUIDITY_THRESHOLDS.get(asset_type)

    thin = threshold is not None and liq_stats["avg_turnover"] < threshold
    hard = threshold is not None and (
        liq_stats["avg_turnover"] < threshold * HARD_ILLIQUID_TURNOVER_RATIO
        or spread_stats["frozen_days_ratio"] >= HARD_ILLIQUID_FROZEN_DAYS_RATIO
    )
    tier = "hard_illiquid" if hard else ("thin" if thin else "liquid")

    return {**liq_stats, **spread_stats, "tier": tier, "threshold": threshold}


def detect_accumulation_signals(df: pd.DataFrame, vol_window: int = 20,
                                 obv_window: int = 15, squeeze_window: int = 60,
                                 rvol_lookback: int = 3) -> dict:
    """
    Volume/volatility-based "early accumulation" proxy score in [0, 1],
    built from four established technical-analysis concepts (Wyckoff /
    Volume Spread Analysis lineage) using only OHLCV.

    HONEST FRAMING FIRST: this dashboard has no order-book, broker-summary,
    or foreign/local-flow data — there is no such thing as literally
    "detecting bandar" from OHLCV alone. What this function CAN do is flag
    statistically unusual volume/volatility footprints that often (not
    reliably, not provably) precede a directional move — the closest
    honest technical-analysis proxy for what's colloquially called
    "gerak-gerik bandar" in Indonesian retail trading discourse. Genuine
    informed accumulation and pure retail FOMO/rumor can produce
    near-identical footprints in this data; treat this as a screening
    filter to investigate further, never as a standalone entry signal.

    Components (each normalized to roughly [0, 1] before averaging):
      1. rvol           — relative volume: last `rvol_lookback` days'
                           average volume vs the trailing `vol_window`-day
                           average. Well above 1.0 = unusual interest.
      2. obv_divergence  — On-Balance Volume trend vs price trend over the
                           last `obv_window` days, normalized by OBV's own
                           volatility. Positive = volume flowing in (OBV
                           rising) while price is still flat/mild — the
                           classic accumulation-phase signature.
      3. squeeze         — volatility compression: current Bollinger Band
                           width's percentile rank within its own trailing
                           `squeeze_window`-day history. Low percentile
                           (tight bands = "coiling") historically often
                           precedes a bigger-than-recent move — direction
                           NOT implied, only that a bigger move than the
                           recent range suggests is more likely soon.
      4. up_down_volume  — ratio of volume on up-close days vs down-close
                           days over `vol_window` (VSA "effort vs result"):
                           volume concentrating on up-days without price
                           having broken out yet is a mild accumulation tell.

    Returns dict: {accumulation_score, components (the 4 sub-scores),
    flag (bool, score > 0.65), detail (plain-language note on which
    components fired)}. accumulation_score/components are None when
    volume data is missing or history is too short.

    REAL RISK THIS DOESN'T CAPTURE: this is aimed at exactly the segment
    (thin IDX small-caps) that carries the highest manipulation AND
    exit-liquidity risk — a name can be walked up on trivial volume by a
    handful of players, and the moment that stops, the auto-rejection-
    bawah (ARB) daily limit itself can leave a position MECHANICALLY
    unsellable ("no bid") for one or more sessions, unlike a liquid
    blue-chip where an exit is always available at some price. A high
    score here is a reason to look closer, not a reason to size up.
    """
    if "Volume" not in df.columns or len(df) < max(vol_window, squeeze_window, obv_window) + 5:
        return {"accumulation_score": None, "components": None, "flag": False,
                "detail": "Data volume tidak tersedia atau riwayat terlalu pendek."}

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    if volume.tail(vol_window).sum() <= 0:
        return {"accumulation_score": None, "components": None, "flag": False,
                "detail": "Volume nol/kosong sepanjang window — kemungkinan data macet total."}

    # 1. Relative volume
    vol_ma = volume.rolling(vol_window).mean()
    recent_vol = volume.tail(rvol_lookback).mean()
    rvol_raw = float(recent_vol / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0
    rvol_score = float(np.clip((rvol_raw - 1.0) / 2.0, 0.0, 1.0))  # ~1x -> 0, ~3x -> 1

    # 2. OBV divergence vs price (accumulation-phase signature)
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    obv_recent = obv.tail(obv_window)
    price_recent = close.tail(obv_window)
    obv_hist_std = float(obv.tail(vol_window).diff().std())
    if len(obv_recent) >= 5 and obv_hist_std > 0:
        obv_slope = float(np.polyfit(range(len(obv_recent)), obv_recent.values, 1)[0])
        obv_slope_norm = obv_slope / obv_hist_std
    else:
        obv_slope_norm = 0.0
    price_change_pct = (float(price_recent.iloc[-1] / price_recent.iloc[0] - 1)
                         if len(price_recent) > 1 and price_recent.iloc[0] > 0 else 0.0)
    # Discount the divergence heavily if price has already moved a lot —
    # the point is catching it BEFORE the move, not confirming after.
    divergence_raw = obv_slope_norm if price_change_pct < 0.05 else obv_slope_norm * 0.3
    obv_score = float(np.clip(divergence_raw / 3.0, 0.0, 1.0))

    # 3. Volatility squeeze (Bollinger Band width percentile)
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_width = (4 * bb_std / bb_ma).dropna()
    if len(bb_width) >= squeeze_window:
        hist = bb_width.tail(squeeze_window)
        current_width = bb_width.iloc[-1]
        percentile = float((hist < current_width).mean())
        squeeze_score = float(np.clip(1.0 - percentile, 0.0, 1.0))
    else:
        squeeze_score = 0.0

    # 4. Up/down volume ratio (VSA "effort vs result")
    ret = close.pct_change()
    window_ret = ret.tail(vol_window)
    window_vol = volume.tail(vol_window)
    up_vol = float(window_vol[window_ret > 0].sum())
    down_vol = float(window_vol[window_ret < 0].sum())
    updown_ratio = (up_vol / down_vol) if down_vol > 0 else (2.0 if up_vol > 0 else 1.0)
    updown_score = float(np.clip((updown_ratio - 1.0) / 2.0, 0.0, 1.0))

    components = {
        "rvol": round(rvol_score, 3),
        "obv_divergence": round(obv_score, 3),
        "squeeze": round(squeeze_score, 3),
        "up_down_volume": round(updown_score, 3),
    }
    accumulation_score = float(np.mean(list(components.values())))
    fired = [k for k, v in components.items() if v > 0.6]
    detail = ("Komponen dominan: " + ", ".join(fired)) if fired else "Tidak ada komponen yang menonjol."

    return {
        "accumulation_score": round(accumulation_score, 3),
        "components": components,
        "flag": accumulation_score > 0.65,
        "detail": detail,
    }


def score_backtest_robustness(trades_df: pd.DataFrame) -> dict:
    """
    Flags a common small-cap backtest trap: total return dominated by one or
    two outlier trades (a lucky spike), which makes the backtest look great
    on paper but isn't a robust, repeatable pattern.
    """
    if trades_df is None or len(trades_df) == 0:
        return {"n_trades": 0, "dominant_trade_pct": None, "is_concentrated": False}

    returns = trades_df["return_pct"].values
    total_positive = returns[returns > 0].sum()
    if total_positive <= 0:
        return {"n_trades": len(trades_df), "dominant_trade_pct": None, "is_concentrated": False}

    max_trade = returns.max()
    dominant_pct = (max_trade / total_positive * 100) if total_positive > 0 else None
    is_concentrated = (dominant_pct is not None and dominant_pct > 50 and len(trades_df) >= 2)
    return {
        "n_trades": len(trades_df), "dominant_trade_pct": dominant_pct,
        "is_concentrated": is_concentrated,
    }


def bootstrap_trade_metrics(trades_df: pd.DataFrame, n_bootstrap: int = 1000,
                             seed: int = 42, auto_boost_outliers: bool = True,
                             block_size: int = 1) -> dict:
    """
    Resamples the trade sequence (WITH replacement) `n_bootstrap` times to
    get a DISTRIBUTION of win rate / compounded return, instead of trusting
    the single point estimate from the one historical sequence that
    happened to occur.

    This answers a DIFFERENT question than walk-forward validation:
      - Walk-forward asks "does this strategy generalize to UNSEEN data?"
      - This asks "given the SAME trades that already happened, how much
        would the headline win-rate/return number wobble under a
        different plausible sample/order of those same trades?"
    Neither substitutes for the other — a strategy can pass one and fail
    the other. A tight bootstrap distribution (e.g. win rate 50-65% across
    1000 resamples) means the number is fairly stable; a wide one (e.g.
    20-90%) means the headline number is largely down to which specific
    trades happened to occur, not a repeatable edge.

    OUTLIER AUTO-BOOST: one or two outlier trades (flagged via the
    standard IQR fence: outside Q1-1.5*IQR .. Q3+1.5*IQR) make the
    bootstrap DISTRIBUTION itself noisier to estimate — whether that one
    outlier happens to land in a given resample swings that resample's
    total return a lot, so the percentile estimates (p5/p50/p95) need
    MORE resamples to stabilize than a well-behaved, outlier-free trade
    set would. When outliers are detected, `n_bootstrap` is boosted
    (5x, capped at 20000) automatically — this is about estimation
    precision, not about the outliers being any less "real".

    Returns percentiles (p5/p25/p50/p75/p95/mean) for win_rate and
    compounded total_return_pct, the raw sample arrays (for a histogram
    in the UI), and outlier diagnostics (n_outliers, which trade indices,
    requested vs. actually-used n_bootstrap).
    """
    if trades_df is None or len(trades_df) == 0:
        return {"n_trades": 0, "error": "Tidak ada trade untuk di-bootstrap."}

    # FIX 3d: trade berurutan berkorelasi rezim (3 trade di bull run yang sama
    # bukan 3 observasi independen). block_size > 1 = resample BLOK trade
    # berurutan (diurutkan by exit_date), pola yang sama dengan block
    # bootstrap di meta-model. block_size=1 = perilaku lama (i.i.d.).
    if block_size > 1 and "exit_date" in trades_df.columns:
        trades_df = trades_df.sort_values("exit_date")
    returns_pct = trades_df["return_pct"].values
    returns = returns_pct / 100.0
    n = len(returns)

    n_outliers, outlier_idx = 0, []
    if n >= 4:  # IQR fences aren't meaningful on tiny samples
        q1, q3 = np.percentile(returns_pct, [25, 75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (returns_pct < lower_fence) | (returns_pct > upper_fence)
        n_outliers = int(outlier_mask.sum())
        outlier_idx = np.where(outlier_mask)[0].tolist()

    effective_n_bootstrap = n_bootstrap
    if auto_boost_outliers and n_outliers > 0:
        effective_n_bootstrap = int(min(max(n_bootstrap * 5, 5000), 20000))

    rng = np.random.default_rng(seed)
    win_rates = np.empty(effective_n_bootstrap)
    total_returns = np.empty(effective_n_bootstrap)
    for i in range(effective_n_bootstrap):
        if block_size > 1 and n > block_size:
            n_blocks = int(np.ceil(n / block_size))
            starts = rng.integers(0, n - block_size + 1, n_blocks)
            sample = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        else:
            sample = rng.choice(returns, size=n, replace=True)
        win_rates[i] = (sample > 0).mean() * 100
        total_returns[i] = (np.prod(1 + sample) - 1) * 100

    def _pctiles(arr):
        return {"p5": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)), "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)), "mean": float(arr.mean())}

    return {
        "n_trades": n, "n_bootstrap": effective_n_bootstrap,
        "requested_n_bootstrap": n_bootstrap,
        "n_outliers": n_outliers, "outlier_trade_indices": outlier_idx,
        "win_rate": _pctiles(win_rates),
        "total_return_pct": _pctiles(total_returns),
        "win_rate_samples": win_rates.tolist(),
        "total_return_samples": total_returns.tolist(),
    }


def classify_volatility_regime(df: pd.DataFrame, window: int = 20,
                                trading_days: float = 252.0) -> tuple:
    """Simple realized-volatility regime classifier for the dynamic-weighting
    suggestion — annualized std of daily returns over the recent window."""
    returns = df["Close"].pct_change().dropna().tail(window)
    ann_vol = float(returns.std() * (trading_days ** 0.5)) if len(returns) > 1 else None
    if ann_vol is None:
        return "Tidak diketahui", None
    regime = "Tinggi / Trending" if ann_vol > 0.40 else "Rendah / Sideways"
    return regime, ann_vol


def walk_forward_split(df: pd.DataFrame, n_folds: int = 4, train_ratio: float = 0.7,
                        test_days: int | None = None, min_train_days: int = 100,
                        step_days: int | None = None):
    """
    Yield (train_df, test_df) folds for out-of-sample validation.

    Two modes:
      1. Fixed fold count (default): data split into `n_folds` equal contiguous
         chunks, each internally split train/test by `train_ratio`.
      2. Fixed fold LENGTH (set test_days): expanding-window walk-forward with
         a fixed-size out-of-sample test window of `test_days` calendar rows,
         starting once `min_train_days` of history is available, stepping
         forward by `step_days` (defaults to non-overlapping, i.e. = test_days)
         each fold. Number of folds is then whatever fits in the data — this
         is the standard way real quant walk-forward validation is done,
         since fold *length* (e.g. "test on 60 trading days at a time") is
         usually the meaningful parameter, not an arbitrary fold count.
    """
    n = len(df)

    if test_days is not None:
        step = step_days or test_days
        start = min_train_days
        while start + test_days <= n:
            train_df = df.iloc[:start]
            test_df = df.iloc[start:start + test_days]
            if len(train_df) > 30 and len(test_df) > 5:
                yield train_df, test_df
            start += step
    else:
        fold_size = n // n_folds
        for f in range(n_folds):
            start = f * fold_size
            end = n if f == n_folds - 1 else (f + 1) * fold_size
            chunk = df.iloc[start:end]
            split = int(len(chunk) * train_ratio)
            train_df, test_df = chunk.iloc[:split], chunk.iloc[split:]
            if len(train_df) > 30 and len(test_df) > 10:
                yield train_df, test_df


# ==========================================================================
# ==== SECTION 4.6: BROAD UNIVERSE FETCHING (beyond a manual watchlist) ====
# ==========================================================================
# Dynamically fetch the full list of tickers for a "scan everything" mode,
# instead of being limited to a small hand-picked watchlist. Each fetcher
# has a hardcoded fallback list in case the external source is unreachable
# or changes structure — the screener should degrade gracefully, not crash.

_FALLBACK_IDX = ["BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
                 "ANTM", "ADRO", "PGAS", "INDF", "KLBF", "SMGR", "GOTO", "MDKA",
                 "PTBA", "AKRA", "CPIN", "UNTR"]
_FALLBACK_US = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
                "AMD", "JPM"]
_FALLBACK_CRYPTO_IDR = ["BTC/IDR", "ETH/IDR", "SOL/IDR", "BNB/IDR", "XRP/IDR",
                         "ADA/IDR", "DOGE/IDR"]
_FALLBACK_CRYPTO_USDT = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                          "XRP/USDT", "ADA/USDT", "DOGE/USDT"]

# Static snapshot of long-tenured IDX large-caps — NOT a live pull of the
# official LQ45/IDX30 constituent list (which BEI reviews/rebalances twice a
# year, Feb & Aug, and we have no reliable free real-time source for). This
# exists purely as an optional, fast, "polite to rate limits" pre-filter for
# the Screener universe scan (Section 5.3) — a smaller, well-known set of
# actively-traded names to scan instead of the full 900+ IDX universe. It is
# NOT authoritative for anything index-membership-related; update manually
# if it drifts noticeably from the real LQ45/IDX30 composition.
_BLUECHIP_PRESET_IDX = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BRIS", "TLKM", "ASII", "UNVR", "ICBP",
    "INDF", "ANTM", "ADRO", "PTBA", "ITMG", "PGAS", "MEDC", "SMGR", "INTP",
    "KLBF", "SIDO", "CPIN", "JPFA", "UNTR", "AKRA", "MDKA", "GOTO", "BUKA",
    "EMTK", "TOWR", "TBIG", "EXCL", "ISAT", "AMRT", "MAPI", "ACES",
]


@_cache_data(ttl=86400, show_spinner=False)
def fetch_idx_universe() -> list[str]:
    """
    Full list of IDX-listed tickers (~900+), sourced from a public dataset
    (wildangunawan/Dataset-Saham-IDX on GitHub, sector CSVs aggregated).
    Falls back to a small curated list if the source is unreachable.
    """
    import urllib.request

    try:
        api_url = ("https://api.github.com/repos/wildangunawan/Dataset-Saham-IDX/"
                   "contents/List%20Emiten/Sectors")
        req = urllib.request.Request(api_url, headers={"User-Agent": "quant-dashboard"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            files = json.loads(resp.read().decode("utf-8"))

        tickers = set()
        for f in files:
            if not f["name"].endswith(".csv"):
                continue
            csv_req = urllib.request.Request(f["download_url"],
                                              headers={"User-Agent": "quant-dashboard"})
            with urllib.request.urlopen(csv_req, timeout=15) as csv_resp:
                content = csv_resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                code = row.get("code")
                if code:
                    tickers.add(code.strip().upper())

        if len(tickers) < 100:  # sanity check — source likely malformed
            raise ValueError("Jumlah ticker terlalu sedikit, kemungkinan sumber berubah format")
        return sorted(tickers)
    except Exception:
        return _FALLBACK_IDX


@_cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_universe() -> list[str]:
    """
    Full S&P 500 ticker list. Tries two independent sources before giving up:
        1. Wikipedia, fetched with a real browser User-Agent — pandas.read_html's
           default urllib user agent gets a 403 from Wikipedia/many sites, which
           is the single most common reason this silently fell back before.
        2. A community-maintained GitHub CSV mirror as a second opinion, in
           case Wikipedia's table layout changed or is blocked outright for
           this network.
    Falls back to the small built-in list only if BOTH sources fail.
    """
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

    def _clean(tickers: list) -> list[str]:
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        return [t.replace(".", "-") for t in tickers]  # yfinance format, e.g. BRK.B -> BRK-B

    # ---- source 1: Wikipedia ----
    try:
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                             headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = _clean(df[col].tolist())
        if len(tickers) >= 100:
            return sorted(set(tickers))
    except Exception:
        pass  # try the next source

    # ---- source 2: community-maintained GitHub CSV mirror ----
    try:
        resp = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            headers=headers, timeout=10
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = _clean(df[col].tolist())
        if len(tickers) >= 100:
            return sorted(set(tickers))
    except Exception:
        pass

    return _FALLBACK_US


@_cache_data(ttl=3600, show_spinner=False)
def fetch_crypto_universe(exchange_id: str) -> list[str]:
    """
    Full list of tradeable pairs on the given exchange (via ccxt), filtered
    to the most common quote currency for that exchange (IDR for Indodax,
    USDT elsewhere). Falls back on failure.
    """
    import ccxt

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        markets = exchange.load_markets()
        quote = "IDR" if exchange_id == "indodax" else "USDT"
        symbols = [s for s in markets.keys() if s.endswith(f"/{quote}")]
        if len(symbols) < 5:
            raise ValueError("Jumlah pair terlalu sedikit, kemungkinan quote currency salah")
        return sorted(symbols)
    except Exception:
        return _FALLBACK_CRYPTO_IDR if exchange_id == "indodax" else _FALLBACK_CRYPTO_USDT


def build_meta_features(df: pd.DataFrame, signal_params: dict, horizon_days: int = 15,
                         mc_window: int = 90) -> pd.DataFrame:
    """
    Builds a walk-forward-safe historical feature table for the
    "Kesimpulan" tab's meta-model: for each day T (once enough trailing
    history exists), computes what 3 of the 4 heuristic-aggregation
    components would ACTUALLY have been using only data up to T, plus the
    real future outcome — so a model can be trained/validated against
    them, instead of the heuristic's fixed 0.25 weights that were guessed,
    not fit to evidence.

    Components (each rescaled to roughly [0,1], matching vote_* in the
    Kesimpulan tab):
      - vote_signal: composite_score, rescaled (score+1)/2. Already a
        clean historical time series — no approximation needed.
      - vote_mc: CLOSED-FORM GBM P(price higher in `horizon_days`), from a
        rolling `mc_window`-day mean/std of daily log returns (shifted by
        1 day so day T only sees data strictly before T). This is the
        same GBM assumption simulate_gbm() uses — computed analytically
        instead of by simulation, because actually re-running Monte Carlo
        simulation for every historical day would be far too slow.
      - vote_bt: EXPANDING realized win rate from run_backtest() on this
        same composite_signal, using only trades that had already CLOSED
        by day T — a trade isn't counted in day T's "win rate so far"
        until its exit_date has passed (no lookahead).

    Fundamental score is DELIBERATELY EXCLUDED from this feature table —
    yfinance only exposes the CURRENT snapshot, there's no point-in-time
    historical fundamentals data available, so it can't be validated this
    way. Training against a constant value across the whole window would
    make it a fake feature with a meaningless fitted coefficient. It stays
    a separate, undisputed qualitative input in the UI, same as before —
    this meta-model only replaces the 3 components that CAN be validated.

    target: 1 if Close `horizon_days` trading days ahead > Close today,
    else 0 — NaN for the last `horizon_days` rows where that isn't known
    yet (must never be used as a feature, only as the label to fit/score
    against).
    """
    sig_df = composite_signal(df, **signal_params)
    vote_signal = (sig_df["composite_score"] + 1) / 2

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    roll_mu = log_ret.rolling(mc_window).mean().shift(1)
    roll_sigma = log_ret.rolling(mc_window).std(ddof=1).shift(1)
    from scipy.stats import norm
    drift = (roll_mu - 0.5 * roll_sigma ** 2) * horizon_days
    spread = (roll_sigma * np.sqrt(horizon_days)).replace(0, np.nan)
    z = (drift / spread).replace([np.inf, -np.inf], np.nan)
    vote_mc = pd.Series(norm.cdf(z.values), index=df.index)
    vote_mc[z.isna()] = np.nan

    bt_result = run_backtest(sig_df, signal_col="composite_signal")
    trades = bt_result["trades"]
    vote_bt = pd.Series(np.nan, index=df.index)
    if len(trades) > 0:
        trades_sorted = trades.sort_values("exit_date").copy()
        closed_win = (trades_sorted["return_pct"] > 0).astype(int)
        cum_wr = closed_win.cumsum() / np.arange(1, len(trades_sorted) + 1)
        wr_series = pd.Series(cum_wr.values, index=pd.to_datetime(trades_sorted["exit_date"]))
        wr_series = wr_series[~wr_series.index.duplicated(keep="last")]
        vote_bt = wr_series.reindex(df.index, method="ffill")

    future_close = df["Close"].shift(-horizon_days)
    target = pd.Series(np.where(future_close > df["Close"], 1.0, 0.0), index=df.index)
    target[future_close.isna()] = np.nan

    return pd.DataFrame({"vote_signal": vote_signal, "vote_mc": vote_mc,
                          "vote_bt": vote_bt, "target": target}, index=df.index)


def train_meta_model(df: pd.DataFrame, signal_params: dict, horizon_days: int = 15,
                      mc_window: int = 90, n_folds: int = 4) -> dict:
    """
    Trains a logistic regression meta-model on the 3 validatable
    Kesimpulan-aggregation components (see build_meta_features) to predict
    P(price higher in horizon_days) — a genuinely walk-forward-validated,
    calibration-checked alternative to the fixed 0.25/0.25/0.25/0.25
    heuristic average, using the SAME Brier Skill Score standard as
    walk_forward_ml_signal elsewhere in this file.

    Honest limitations, on purpose, not hidden:
      - Only 3 features (fundamentals excluded — see build_meta_features).
      - Needs a reasonably long price history to have enough valid rows
        after the mc_window/horizon_days warm-up — short histories (e.g.
        "6mo") will likely fail with too few samples; returns
        trained=False with a plain-language reason rather than silently
        producing an unreliable model.
      - This validates whether the AGGREGATION is predictive — it does
        NOT mean the underlying 3 components themselves are individually
        reliable, and a low/negative BSS here should be read as "the
        heuristic weights aren't obviously better than guessing", not as
        proof the opposite direction ("logistic regression IS obviously
        better") — report both and let the person judge.

    Returns dict — either {"trained": False, "reason": str, "n_samples": int}
    or {"trained": True, "n_samples", "n_oos_predictions", "coefficients"
    (dict incl. "intercept"), "calibration" (bss/brier_model/
    brier_baseline/baseline_rate_pct), "predict_fn" (callable taking
    vote_signal, vote_mc, vote_bt -> calibrated probability), "horizon_days"}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    feat = build_meta_features(df, signal_params, horizon_days, mc_window)
    feat_clean = feat.dropna()
    if len(feat_clean) < 60:
        return {"trained": False, "n_samples": len(feat_clean),
                "reason": f"Cuma {len(feat_clean)} baris data valid setelah warm-up period "
                          f"(butuh histori panjang karena tiap komponen — terutama vote_mc "
                          f"dengan window {mc_window} hari — perlu pemanasan) — minimal "
                          f"butuh ~60. Coba histori lebih panjang (10y/max di sidebar)."}

    X = feat_clean[["vote_signal", "vote_mc", "vote_bt"]].values
    y = feat_clean["target"].values
    n = len(feat_clean)
    fold_size = n // (n_folds + 1)
    if fold_size < 20:
        return {"trained": False, "n_samples": n,
                "reason": f"Data terlalu pendek untuk {n_folds} fold walk-forward yang "
                          f"bermakna (tiap fold jadi < 20 baris). Coba histori lebih panjang "
                          f"atau turunkan jumlah fold."}

    oos_proba, oos_actual = [], []
    oos_fold_blocks = []  # list of (proba_array, actual_array) PER FOLD, kept separate so
                           # the block bootstrap below never constructs a "block" that
                           # spans a discontinuous jump in time between two folds' test sets
    for f in range(n_folds):
        train_end = fold_size * (f + 1)
        test_end = min(fold_size * (f + 2), n)
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        fold_proba = model.predict_proba(X_test)[:, 1]
        oos_proba.extend(fold_proba.tolist())
        oos_actual.extend(y_test.tolist())
        oos_fold_blocks.append((fold_proba, y_test))

    if len(oos_proba) < 20:
        return {"trained": False, "n_samples": n,
                "reason": "Nggak cukup prediksi out-of-sample buat dievaluasi kalibrasinya "
                          "dengan bermakna."}

    oos_proba, oos_actual = np.array(oos_proba), np.array(oos_actual)

    # FIX 1c: fit calibrator HANYA di paruh pertama fold walk-forward (waktu
    # lebih awal); BSS + CI bootstrap dihitung HANYA dari paruh kedua.
    n_calib_folds = max(1, len(oos_fold_blocks) // 2)
    calib_p = np.concatenate([fp for fp, _ in oos_fold_blocks[:n_calib_folds]])
    calib_a = np.concatenate([fa for _, fa in oos_fold_blocks[:n_calib_folds]])
    eval_blocks = oos_fold_blocks[n_calib_folds:] or oos_fold_blocks[-1:]
    eval_proba = np.concatenate([fp for fp, _ in eval_blocks])
    eval_actual = np.concatenate([fa for _, fa in eval_blocks])

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calib_p, calib_a)
    calibrated = calibrator.predict(eval_proba)

    brier_model = float(np.mean((calibrated - eval_actual) ** 2))
    baseline_rate = float(eval_actual.mean())
    brier_baseline = float(np.mean((baseline_rate - eval_actual) ** 2))
    bss = (1 - brier_model / brier_baseline) if brier_baseline > 0 else 0.0

    # ---- Confidence interval on BSS via BLOCK bootstrap ----
    # A plain i.i.d. bootstrap (resampling individual rows) badly
    # UNDERSTATES uncertainty here: target = "price higher in
    # horizon_days" is computed for EVERY consecutive day, so day T and
    # day T+1's targets share (horizon_days - 1) of the same future
    # window — massively overlapping/autocorrelated, not independent
    # observations. Treating them as i.i.d. gives falsely narrow CIs
    # (verified empirically: on PURE random-walk price data with zero
    # real structure, an i.i.d. bootstrap here flagged "significant" on
    # ~67% of random seeds — should be ~5-10%). Block bootstrap resamples
    # CONTIGUOUS chunks of length `block_size` (~ horizon_days, the
    # natural autocorrelation length) instead of single rows, so the
    # within-block correlation structure is preserved under resampling —
    # standard remedy for overlapping-window time series inference.
    # Blocks are drawn separately WITHIN each fold (never spanning the
    # time-discontinuity between two folds' test periods).
    block_size = max(int(horizon_days), 5)
    _rng_bss = np.random.default_rng(123)
    _n_oos = len(eval_actual)
    _boot_bss = np.empty(1000)
    for _b in range(1000):
        _bp, _ba = [], []
        for fold_proba, fold_actual in eval_blocks:
            _fn = len(fold_actual)
            if _fn == 0:
                continue
            _n_blocks_needed = int(np.ceil(_fn / block_size))
            _max_start = max(_fn - block_size, 0)
            _starts = _rng_bss.integers(0, _max_start + 1, _n_blocks_needed)
            for _s in _starts:
                _bp.append(fold_proba[_s:_s + block_size])
                _ba.append(fold_actual[_s:_s + block_size])
        _samp_proba = np.concatenate(_bp)[:_n_oos] if _bp else np.array([])
        _samp_actual = np.concatenate(_ba)[:_n_oos] if _ba else np.array([])
        if len(_samp_actual) < 5 or len(np.unique(_samp_actual)) < 2:
            _boot_bss[_b] = np.nan
            continue
        _samp_calibrated = calibrator.predict(_samp_proba)
        _b_model = np.mean((_samp_calibrated - _samp_actual) ** 2)
        _b_rate = _samp_actual.mean()
        _b_baseline = np.mean((_b_rate - _samp_actual) ** 2)
        _boot_bss[_b] = (1 - _b_model / _b_baseline) if _b_baseline > 0 else np.nan
    _boot_bss = _boot_bss[~np.isnan(_boot_bss)]
    if len(_boot_bss) < 100:
        bss_ci = {"p5": None, "p50": None, "p95": None}
        bss_significant = False
    else:
        bss_ci = {"p5": float(np.percentile(_boot_bss, 5)),
                  "p50": float(np.percentile(_boot_bss, 50)),
                  "p95": float(np.percentile(_boot_bss, 95))}
        # "Significant" = the 90% block-bootstrap CI sits entirely above
        # zero. Still a weak bar (~one-sided 95% test) — report the
        # number, don't oversell what it means.
        bss_significant = bss_ci["p5"] > 0

    # Final refit on ALL data for live use — standard practice once
    # walk-forward above has already told us whether to trust the approach;
    # this refit itself is not what validates it.
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X, y)
    coef_dict = dict(zip(["vote_signal", "vote_mc", "vote_bt"], final_model.coef_[0].tolist()))
    coef_dict["intercept"] = float(final_model.intercept_[0])

    def _predict(vote_signal_val, vote_mc_val, vote_bt_val):
        raw = final_model.predict_proba([[vote_signal_val, vote_mc_val, vote_bt_val]])[0, 1]
        return float(calibrator.predict([raw])[0])

    return {
        "trained": True, "n_samples": n, "n_oos_predictions": len(eval_proba),
        "coefficients": coef_dict,
        "calibration": {"bss": float(bss), "brier_model": brier_model,
                         "brier_baseline": brier_baseline,
                         "baseline_rate_pct": baseline_rate * 100,
                         "bss_ci": bss_ci, "bss_significant": bss_significant},
        "predict_fn": _predict, "horizon_days": horizon_days,
    }


def run_aggregate_backtest(tickers: list[str], asset_type: str, scan_kwargs: dict,
                            signal_params: dict, backtest_params: dict,
                            max_workers: int = 8, progress_callback=None,
                            min_turnover: float | None = None) -> dict:
    """
    Runs the SAME signal + backtest configuration across MANY tickers and
    pools every resulting trade into one combined sample \u2014 a more
    statistically defensible way to grow N than loosening entry criteria
    on a single ticker (which trades signal quality for quantity). Tests
    "does this STRATEGY have an edge across a watchlist", not "did this
    one ticker happen to produce decent trades".

    signal_params: kwargs for composite_signal (mr_window, mr_z_entry,
        mom_fast, mom_slow, mr_weight, stochbb_weight, bb_window, bb_std,
        stoch_window, stoch_smooth). Deliberately NO ml_score/ml_weight \u2014
        walk-forward ML training per ticker across a whole watchlist would
        be extremely slow; this tests the rule-based composite signal.
    backtest_params: kwargs for run_backtest (fee_bps, slippage_bps,
        position_size_pct, stop_loss_pct, max_holding_days,
        execution_price, dynamic_slippage).

    Returns: {"trades": pooled DataFrame (extra `symbol` column),
              "per_ticker": DataFrame of per-ticker n_trades/win_rate/avg_return,
              "n_tickers_ok", "n_tickers_failed", "errors": [(ticker, msg), ...]}
    """
    import concurrent.futures

    batch_prices: dict = {}
    if asset_type in ("stock_id", "stock_us"):
        try:
            batch_prices = fetch_stock_batch(tickers, period=scan_kwargs.get("period", "1y"))
        except Exception:
            batch_prices = {}

    all_trades, per_ticker_rows, errors = [], [], []
    completed, total = 0, len(tickers)

    def _one(tkr):
        try:
            df_t = batch_prices.get(tkr)
            if df_t is None:
                df_t = cached_fetch_data(asset_type, tkr, **scan_kwargs)
            if len(df_t) < signal_params.get("mr_window", 20) + 10:
                return ("error", tkr, "Data terlalu pendek")
            if min_turnover is not None:
                _liq_check = compute_liquidity_stats(df_t)
                if _liq_check["avg_turnover"] < min_turnover:
                    return ("error", tkr,
                            f"Turnover Rp{_liq_check['avg_turnover']:,.0f} < batas minimum "
                            f"Rp{min_turnover:,.0f} — dilewati sebelum backtest dijalankan.")
            sig_t = composite_signal(df_t, **signal_params)
            bt_t = run_backtest(sig_t, signal_col="composite_signal", **backtest_params)
            trades_t = bt_t["trades"].copy()
            if len(trades_t) > 0:
                trades_t["symbol"] = tkr
            return ("ok", tkr, trades_t)
        except Exception as e:
            return ("error", tkr, str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, t) for t in tickers]
        for fut in concurrent.futures.as_completed(futures):
            status, tkr, payload = fut.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, total, tkr)
            if status == "ok":
                if len(payload) > 0:
                    all_trades.append(payload)
                    per_ticker_rows.append({
                        "symbol": tkr, "n_trades": len(payload),
                        "win_rate_pct": float((payload["return_pct"] > 0).mean() * 100),
                        "avg_return_pct": float(payload["return_pct"].mean()),
                    })
                else:
                    per_ticker_rows.append({"symbol": tkr, "n_trades": 0,
                                             "win_rate_pct": None, "avg_return_pct": None})
            else:
                errors.append((tkr, payload))

    pooled = (pd.concat(all_trades, ignore_index=True) if all_trades else
              pd.DataFrame(columns=["entry_date", "exit_date", "entry_price", "exit_price",
                                     "return_pct", "exit_reason", "symbol"]))
    return {
        "trades": pooled, "per_ticker": pd.DataFrame(per_ticker_rows),
        "n_tickers_ok": len(tickers) - len(errors), "n_tickers_failed": len(errors),
        "errors": errors,
    }


@_cache_data(ttl=900, show_spinner=False)
def cached_fetch_data(asset_type: str, symbol: str, period: str = "1y",
                       exchange_id: str = "indodax", timeframe: str = "1d",
                       limit: int = 200) -> pd.DataFrame:
    """Cached wrapper around fetch_data — used by scan_universe_parallel so
    a broad-universe scan doesn't re-hit the API for every ticker if called
    more than once within the TTL. (This was previously only defined in the
    old Streamlit UI section, which scan_universe_parallel silently
    depended on via module-level name resolution at call time — a dangling
    reference once that section was split out. Belongs here in the engine
    since it's the engine's own scan function that needs it.)"""
    if asset_type in ("stock_id", "stock_us"):
        return fetch_data(asset_type, symbol, period=period)
    else:
        return fetch_data(asset_type, symbol, exchange_id=exchange_id,
                           timeframe=timeframe, limit=limit)


def scan_universe_parallel(tickers: list[str], asset_type: str, scan_kwargs: dict,
                            mr_window: int, mr_z_entry: float, mom_fast: int,
                            mom_slow: int, mr_weight: float, max_workers: int = 15,
                            progress_callback=None, compute_overall: bool = True,
                            mc_days: int = 15, mc_sims: int = 1000,
                            fee_bps: float = 10, slippage_bps: float = 5,
                            fee_buy_bps: float | None = None,
                            fee_sell_bps: float | None = None,
                            trading_days: float = 252.0,
                            w_signal: float = 0.34, w_mc: float = 0.33,
                            w_bt: float = 0.33, compute_ml: bool = False,
                            ml_weight: float = 0.3, ml_min_train: int = 100,
                            ml_retrain_every: int = 20, ml_model_type: str = "lightgbm",
                            adaptive_ml_weight: bool = True,
                            compute_fundamental: bool = False,
                            w_fund: float = 0.0,
                            min_turnover: float | None = None,
                            compute_accumulation: bool = False) -> tuple[list, list, dict]:
    """
    Concurrent (thread-pool) scan across many tickers — sequential fetching
    of hundreds of symbols via yfinance/ccxt would take many minutes; I/O-bound
    network calls parallelize well with threads. Crypto uses a lower default
    concurrency to stay polite to exchange rate limits.

    When compute_overall=True (default), each ticker also gets a lightweight
    Monte Carlo (fast GBM, not GARCH — too slow to fit per-ticker at scale)
    and a single-period backtest, combined with the composite signal into the
    same weighted "Overall %" used in the Kesimpulan tab.

    When compute_ml=True, each ticker ALSO gets a full walk-forward ML fit
    (same as the Signal tab), blended into the composite score before the
    Overall % is computed. ml_model_type: "lightgbm" (default, fastest),
    "xgboost", or "ensemble" (fits BOTH per ticker — roughly doubles this
    stage's already-heavy per-ticker cost, since it's CPU-bound, not
    network-bound: Python threads don't parallelize CPU work as well as I/O
    (GIL), so this stage runs closer to sequential speed per ticker (~1-10s
    each depending on history length and model_type) even with a thread
    pool. Worker count is capped lower here to avoid oversubscribing CPU
    cores. No benchmark= relative-strength feature is passed here either —
    a per-ticker benchmark fetch would add a network round-trip per ticker
    to a scan that's already looping over dozens/hundreds of them.

    When compute_fundamental=True (stocks only — silently skipped for crypto),
    each ticker also gets a fetch of yfinance .info + rule-based fundamental
    score, blended in as a 4th component. This is a SEPARATE, heavier network
    call than the price history fetch (yfinance's .info endpoint is slower
    and more prone to sparse data / occasional failures than price history),
    so it adds meaningful time per ticker — handled gracefully per-ticker
    (missing/failed fundamentals just fall back to neutral 50%, never crash
    the whole scan).
    """
    import concurrent.futures
    import time as _time

    results, errors = [], []
    total = len(tickers)
    total_w = w_signal + w_mc + w_bt + (w_fund if compute_fundamental else 0.0)
    _YF_DIAGNOSTICS.reset()
    _t_start = _time.monotonic()

    # ---- Batch pre-fetch (stocks only) ----
    # Turns N individually-rate-limited network calls into ~1 batched call —
    # see fetch_stock_batch() docstring. This is what actually fixes "scan
    # is slow with zero CPU usage" (that symptom = time.sleep in the pacing
    # limiter / 429-backoff, not compute). Crypto still fetches per-symbol
    # below (ccxt doesn't expose an equivalent single-call multi-pair OHLCV
    # batch the same way yfinance does) — only stocks benefit here for now.
    _batch_prices: dict = {}
    if asset_type in ("stock_id", "stock_us"):
        try:
            _batch_prices = fetch_stock_batch(tickers, period=scan_kwargs.get("period", "1y"))
        except Exception:
            _batch_prices = {}  # fall through to per-ticker fetch inside _scan_one below
    _t_after_prefetch = _time.monotonic()
    _n_batch_hit = sum(1 for t in tickers if t in _batch_prices)

    def _scan_one(tkr):
        try:
            df_t = _batch_prices.get(tkr)
            if df_t is None:
                df_t = cached_fetch_data(asset_type, tkr, **scan_kwargs)  # fallback: batch miss
            if len(df_t) < mr_window + 10:
                return ("error", tkr, "Data terlalu pendek")

            if min_turnover is not None:
                _liq_check = compute_liquidity_stats(df_t)
                if _liq_check["avg_turnover"] < min_turnover:
                    return ("error", tkr,
                            f"Turnover harian rata-rata Rp{_liq_check['avg_turnover']:,.0f} "
                            f"di bawah batas minimum Rp{min_turnover:,.0f} — dilewati sebelum "
                            f"sinyal/ML dihitung (bukan cuma disaring belakangan).")

            ml_score_t = None
            ml_used = False
            ml_conf_t, ml_conf_detail_t = 1.0, None
            if compute_ml and len(df_t) >= ml_min_train + 50:
                try:
                    ml_score_t, _, _, _, _, _, ml_diag_t = walk_forward_ml_signal(
                        df_t, min_train_days=ml_min_train, retrain_every=ml_retrain_every,
                        model_type=ml_model_type
                    )
                    ml_used = True
                    ml_conf_t, ml_conf_detail_t = ml_calibration_confidence(ml_diag_t)
                except Exception:
                    ml_score_t = None  # fall back to classic-only silently for this ticker

            effective_ml_weight_t = (ml_weight if ml_used else 0.0) * \
                (ml_conf_t if adaptive_ml_weight else 1.0)

            sig_t = composite_signal(
                df_t, mr_window=mr_window, mr_z_entry=mr_z_entry,
                mom_fast=mom_fast, mom_slow=mom_slow, mr_weight=mr_weight,
                ml_score=ml_score_t, ml_weight=effective_ml_weight_t
            )
            latest_t = sig_t.iloc[-1]
            composite_score = float(latest_t["composite_score"])

            row = {
                "Simbol": tkr, "Harga": latest_t["Close"],
                "Sinyal": latest_t["composite_signal"],
                "Score": round(composite_score, 3),
                "Z-score": round(float(latest_t["mr_zscore"]), 2) if pd.notna(latest_t["mr_zscore"]) else None,
                "ML": "✓" if ml_used else ("–" if compute_ml else ""),
                "ML Conf": round(ml_conf_t, 2) if ml_conf_detail_t is not None else None,
            }

            liq_class_t = classify_liquidity(df_t, asset_type)
            row["Turnover"] = round(liq_class_t["avg_turnover"], 0)
            row["Spread~%"] = (round(liq_class_t["consensus_spread_pct"], 2)
                                if liq_class_t["consensus_spread_pct"] is not None else None)
            liq_badge = {"liquid": "✓", "thin": "🚫 Tipis", "hard_illiquid": "⛔ Macet"}
            row["Liquid"] = liq_badge[liq_class_t["tier"]] if liq_class_t["threshold"] is not None else "–"

            if compute_accumulation:
                accum_t = detect_accumulation_signals(df_t)
                row["Accum Score"] = accum_t["accumulation_score"]
                row["Accum Flag"] = "🔎 Ya" if accum_t["flag"] else "–"

            vote_fund = None
            fund_used = False
            if compute_fundamental and asset_type != "crypto":
                try:
                    fund_data = fetch_fundamentals(tkr)
                    f_score, _ = score_fundamentals(fund_data)
                    if f_score is not None:
                        vote_fund = f_score
                        fund_used = True
                except Exception:
                    vote_fund = None
                row["Fund %"] = round(vote_fund * 100, 1) if vote_fund is not None else None
                row["Fund"] = "✓" if fund_used else "–"

            if compute_overall and total_w > 0:
                vote_signal = (composite_score + 1) / 2
                paths = simulate_gbm(df_t["Close"], n_days=mc_days, n_sims=mc_sims)
                mc_summary = summarize_paths(paths)
                vote_mc = float(mc_summary["prob_profit"])

                bt = run_backtest(sig_t, fee_bps=fee_bps, slippage_bps=slippage_bps,
                                   fee_buy_bps=fee_buy_bps, fee_sell_bps=fee_sell_bps,
                                   trading_days=trading_days)
                win_rate = bt["metrics"].get("win_rate_pct")
                n_trades_bt = bt["metrics"].get("n_trades", 0)
                vote_bt = (win_rate / 100) if (win_rate is not None and n_trades_bt >= 3) else 0.5

                weighted_sum = vote_signal * w_signal + vote_mc * w_mc + vote_bt * w_bt
                if compute_fundamental and asset_type != "crypto":
                    weighted_sum += (vote_fund if vote_fund is not None else 0.5) * w_fund
                overall = weighted_sum / total_w
                if overall >= 0.55:
                    verdict = "BUY"
                elif overall <= 0.45:
                    verdict = "SELL"
                else:
                    verdict = "HOLD"

                # Guard stacking: Z-score overbought guard + Liquidity hard-block
                # guard, same pattern as the Kesimpulan tab (Tab 8) — both cap a
                # raw BUY to HOLD independently, and stack in the label if both fire.
                raw_verdict = verdict
                z_val = latest_t["mr_zscore"]
                z_guard_t = raw_verdict == "BUY" and pd.notna(z_val) and z_val >= mr_z_entry
                liq_guard_t = raw_verdict == "BUY" and liq_class_t["tier"] == "hard_illiquid"
                guards_t = (["Z-Guard"] if z_guard_t else []) + (["Liquidity-Guard"] if liq_guard_t else [])
                if guards_t:
                    verdict = f"HOLD ({' + '.join(guards_t)})"

                row["Overall %"] = round(overall * 100, 1)
                row["Verdict"] = verdict
                row["MC %"] = round(vote_mc * 100, 1)
                row["Backtest %"] = round(vote_bt * 100, 1)
                row["N Trades BT"] = n_trades_bt

            return ("ok", tkr, row)
        except Exception as e:
            return ("error", tkr, str(e))

    if compute_ml:
        # CPU-bound stage — don't oversubscribe cores with too many concurrent
        # model fits regardless of what max_workers the caller passed.
        # Ensemble fits TWO models per ticker (LightGBM + XGBoost), so cap
        # concurrency even lower than single-model mode to avoid thrashing.
        workers = min(max_workers, 2 if ml_model_type == "ensemble" else 4)
    elif compute_fundamental:
        # yfinance .info is heavier and more rate-limit-prone than price
        # history — moderate concurrency here to stay polite to Yahoo Finance.
        workers = min(max_workers, 8)
    else:
        workers = 5 if asset_type == "crypto" else max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scan_one, t): t for t in tickers}
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            status, tkr, payload = fut.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, total, tkr)
            if status == "ok":
                results.append(payload)
            else:
                errors.append((tkr, payload))

    _t_end = _time.monotonic()
    _retry_stats = _YF_DIAGNOSTICS.snapshot()
    diagnostics = {
        "n_tickers": total,
        "prefetch_seconds": round(_t_after_prefetch - _t_start, 1),
        "scan_seconds": round(_t_end - _t_after_prefetch, 1),
        "total_seconds": round(_t_end - _t_start, 1),
        "batch_prefetch_hits": _n_batch_hit,
        "batch_prefetch_misses": total - _n_batch_hit if asset_type in ("stock_id", "stock_us") else None,
        "rate_limit_retries": _retry_stats["retry_count"],
        "seconds_spent_in_backoff": _retry_stats["backoff_seconds"],
        "workers_used": workers,
    }
    return results, errors, diagnostics

# ==========================================================================
# ==== SECTION 6: VALIDATION & REALISM ADDITIONS (dari review eksternal) ====
# ==========================================================================
# Fungsi-fungsi baru — tidak mengubah fungsi lama, murni tambahan.
# Dipakai oleh app.py (sudah ke-import otomatis via `from quant_engine import *`).

def signal_ic(df: pd.DataFrame, score: pd.Series, horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    """
    Information Coefficient — metrik inti yang sebelumnya belum ada sama sekali
    (review 3a). Menjawab pertanyaan paling penting secara langsung: "apakah
    score tinggi HARI INI berkorelasi dengan return N hari KE DEPAN?"

    Backtest mengukur satu path trading tertentu; IC mengukur sinyalnya sendiri.
    Aturan praktis: |rank IC| < 0.02 dengan |t| < 2 = sinyal tidak punya edge
    terukur, apa pun kata backtest.

    Returns DataFrame ter-index horizon: {rank_ic, t_stat, n}.
    """
    rows = {}
    for h in horizons:
        fwd = df["Close"].shift(-h) / df["Close"] - 1
        pair = pd.concat([score, fwd], axis=1).dropna()
        n = len(pair)
        if n < 10:
            rows[h] = {"rank_ic": None, "t_stat": None, "n": n}
            continue
        ric = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        t = ric * np.sqrt((n - 2) / max(1e-12, 1 - ric ** 2))
        rows[h] = {"rank_ic": float(ric), "t_stat": float(t), "n": n}
    out = pd.DataFrame(rows).T
    out.index.name = "horizon_days"
    return out


def probabilistic_sharpe_ratio(sharpe: float | None, n_obs: int,
                                skew: float = 0.0, kurt: float = 3.0,
                                bench: float = 0.0) -> float | None:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado) — review 3b.
    P(Sharpe sebenarnya > bench), mengingat estimasi Sharpe dari n_obs
    observasi punya standard error yang tidak nol. Jawaban atas multiple
    testing: GA mengevaluasi ratusan-ribuan kombinasi, jadi "Sharpe tinggi"
    harus didiskon oleh ketidakpastian estimasinya.
    """
    from scipy.stats import norm
    if sharpe is None or n_obs < 3:
        return None
    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / max(n_obs - 1, 1))
    return float(norm.cdf((sharpe - bench) / se)) if se > 0 else 0.0


def expected_max_sharpe_under_null(n_trials: int, n_obs: int) -> float:
    """
    Ekspektasi Sharpe TERTINGGI yang akan muncul murni dari noise setelah
    n_trials percobaan (review 3b) ≈ se × √(2 ln N). Bandingkan best_fitness
    GA dengan angka ini, bukan dengan 0.
    """
    if n_trials < 2 or n_obs < 3:
        return 0.0
    return float(np.sqrt(2 * np.log(n_trials) / max(n_obs - 1, 1)))


def parameter_plateau_test(df: pd.DataFrame, base_params: dict,
                            composite_signal_fn, run_backtest_fn,
                            deltas: dict | None = None,
                            backtest_kwargs: dict | None = None) -> pd.DataFrame:
    """
    Plateau test (review 3c): parameter yang robust duduk di "dataran", bukan
    puncak. Cek Sharpe di tetangga terdekat (mr_window ±5, mr_z_entry ±0.2).
    Kalau performa runtuh di tetangga → spike noise, bukan edge.
    """
    deltas = deltas or {"mr_window": (-5, 0, 5), "mr_z_entry": (-0.2, 0.0, 0.2)}
    backtest_kwargs = backtest_kwargs or {}
    rows = []
    for param, offsets in deltas.items():
        if param not in base_params:
            continue
        for off in offsets:
            params = dict(base_params)
            params[param] = params[param] + off
            if param in ("mr_window", "mom_fast", "mom_slow"):
                params[param] = max(5, int(round(params[param])))
            try:
                sig = composite_signal_fn(df, **params)
                m = run_backtest_fn(sig, **backtest_kwargs)["metrics"]
                sharpe = m.get("sharpe_ratio")
            except Exception:
                sharpe = None
            rows.append({"param": param, "offset": off, "value": params[param],
                          "sharpe": sharpe, "is_base": off == 0})
    return pd.DataFrame(rows)


def apply_regime_gate(sig_df: pd.DataFrame, bench_close: pd.Series,
                       ma_window: int = 200,
                       score_col: str = "composite_score",
                       signal_col: str = "composite_signal") -> pd.DataFrame:
    """
    Market regime gate (review 4a — ROI tertinggi per baris kode). Sinyal
    long-only jauh lebih baik kalau hanya aktif saat pasar mendukung:
    saat benchmark (IHSG/^GSPC/BTC) di BAWAH MA-200, composite score di-clip
    ke ≤ 0 → tidak ada sinyal BUY baru di rezim bearish.

    Menambah kolom `regime_ok` (bool) supaya UI bisa menampilkan status gate.
    """
    out = sig_df.copy()
    bench = bench_close.reindex(out.index).ffill()
    bench_ok = (bench > bench.rolling(ma_window).mean()).fillna(False)
    out[score_col] = out[score_col].where(bench_ok, out[score_col].clip(upper=0))
    sig = pd.Series("HOLD", index=out.index)
    sig[out[score_col] >= 0.5] = "BUY"
    sig[out[score_col] <= -0.5] = "SELL"
    out[signal_col] = sig
    out["regime_ok"] = bench_ok
    return out


def flag_suspicious_prices(df: pd.DataFrame, threshold: float = 0.35) -> pd.Series:
    """
    Data hygiene untuk IDX (review 4f): yfinance kadang punya bad tick /
    corporate action yang salah. Return harian di atas batas ARA tier manapun
    (35%) hampir pasti data error. FLAG untuk cross-check (mis. dengan Twelve
    Data fallback), JANGAN auto-drop.

    Returns Series berisi return mencurigakan ter-index tanggal (kosong = bersih).
    """
    ret = df["Close"].pct_change()
    return ret[ret.abs() > threshold]
# ==========================================================================
# ==== SECTION 7: META-LABELING, PORTFOLIO BACKTEST, IDX REALISM, ==========
# ==== EXTENDED METRICS (lanjutan review: 4b, 4d, 2b, 4e) ==================
# ==========================================================================


# --------------------------------------------------------------------------
# ---- 4b. META-LABELING (López de Prado) ----
# --------------------------------------------------------------------------
# Bukan blending: sinyal rule-based TETAP yang memutuskan entry; model ML
# sekunder hanya menjawab "trade yang diusulkan ini layak diambil atau
# di-skip?" (klasifikasi biner: trade ini profit atau tidak). Ini memisahkan
# tugas dengan bersih — primary model menguasai arah & timing, secondary
# model menguasai filtering — dan biasanya menaikkan win rate tanpa
# menambah sinyal palsu.

def build_meta_label_samples(df: pd.DataFrame, signal_params: dict | None = None,
                              fee_buy_bps: float = 20.0, fee_sell_bps: float = 30.0,
                              slippage_bps: float = 5.0,
                              stop_loss_pct: float | None = None,
                              max_holding_days: int | None = 15) -> pd.DataFrame:
    """
    Bangun dataset trade-level untuk meta-labeling: tiap sinyal BUY historis
    disimulasikan jadi satu trade (entry di Open berikutnya, exit di sinyal
    SELL/EXIT / stop-loss / max holding — aturan yang sama dengan
    run_backtest), lalu dilabeli 1 kalau return net-nya > 0.

    Fitur diambil dari kondisi HARI SINYAL (backward-looking, anti-lookahead):
    semua fitur teknikal compute_ml_features + composite_score + mr_zscore.

    Returns DataFrame: satu baris per trade, kolom = fitur + "label" (0/1) +
    "return_pct" + "signal_date". Kosong kalau tidak ada sinyal BUY sama sekali.
    """
    signal_params = signal_params or {}
    sig = composite_signal(df, **signal_params)
    feats = compute_ml_features(df)
    feats = feats.copy()
    feats["composite_score"] = sig["composite_score"]
    feats["mr_zscore"] = sig["mr_zscore"]

    opens = df["Open"].values
    closes = df["Close"].values
    lows = df["Low"].values
    signals = sig["composite_signal"].values
    n = len(df)
    cost_in = (fee_buy_bps + slippage_bps) / 1e4
    cost_out = (fee_sell_bps + slippage_bps) / 1e4

    rows = []
    i = 0
    while i < n - 1:
        if signals[i] != "BUY":
            i += 1
            continue
        entry_i = i + 1
        entry_px = opens[entry_i] * (1 + cost_in)
        stop = entry_px * (1 - stop_loss_pct) if stop_loss_pct else None
        exit_px, exit_i = None, None
        for j in range(entry_i, n):
            if stop is not None and lows[j] <= stop:
                exit_px, exit_i = stop, j
                break
            if signals[j] in ("SELL", "EXIT") and j + 1 < n:
                exit_px, exit_i = opens[j + 1], j + 1
                break
            if max_holding_days is not None and (j - entry_i) >= max_holding_days:
                exit_px = opens[j + 1] if j + 1 < n else closes[j]
                exit_i = min(j + 1, n - 1)
                break
        if exit_px is None:
            exit_px, exit_i = closes[-1], n - 1
        ret = (exit_px * (1 - cost_out)) / entry_px - 1

        frow = feats.iloc[i]
        if not frow.isna().any():
            rows.append({**frow.to_dict(), "label": int(ret > 0),
                          "return_pct": ret * 100, "signal_date": df.index[i]})
        i = max(exit_i, i + 1)  # tidak ada trade tumpang-tindih (long-only, 1 posisi)

    return pd.DataFrame(rows)


def train_meta_label_model(samples: pd.DataFrame, min_train: int = 20,
                            retrain_every: int = 5,
                            take_threshold: float = 0.5) -> dict:
    """
    Latih model meta-labeling (LightGBM, walk-forward per-trade) di atas
    output build_meta_label_samples.

    Yang dilaporkan (semua out-of-sample, per-trade — bukan per-hari):
      - base_win_rate / base_avg_return: performa SEMUA trade (tanpa filter)
      - filtered_win_rate / filtered_avg_return: hanya trade yang model
        bilang "ambil" (proba >= take_threshold)
      - coverage: fraksi trade yang lolos filter
      - bss: Brier Skill Score probabilitas vs nebak base rate

    CATATAN JUJUR: meta-labeling butuh PULUHAN trade historis untuk bermakna.
    Dengan < 40 sampel, anggap hasilnya indikatif saja — CI-nya sangat lebar.

    Returns dict: {"trained": bool, "reason"?, "n_samples", "oos": {...},
                   "predict_fn" (callable: dict fitur -> proba profit),
                   "feat_cols": list[str]}
    """
    from lightgbm import LGBMClassifier

    drop_cols = {"label", "return_pct", "signal_date"}
    feat_cols = [c for c in samples.columns if c not in drop_cols]
    X = samples[feat_cols].values
    y = samples["label"].values.astype(int)
    rets = samples["return_pct"].values
    n = len(samples)

    if n < min_train + 10:
        return {"trained": False, "n_samples": n,
                "reason": f"Cuma {n} trade historis dari sinyal ini — meta-labeling "
                          f"butuh minimal ~{min_train + 10} (idealnya 40+). Coba histori "
                          f"lebih panjang atau parameter sinyal yang lebih sering menembak."}

    oos_proba = np.full(n, np.nan)
    model, last_train = None, -10**9
    for i in range(min_train, n):
        if model is None or (i - last_train) >= retrain_every:
            if len(np.unique(y[:i])) < 2:
                continue
            model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                    verbosity=-1, random_state=42)
            model.fit(X[:i], y[:i])
            last_train = i
        oos_proba[i] = model.predict_proba(X[[i]])[0, 1]

    mask = ~np.isnan(oos_proba)
    if mask.sum() < 10:
        return {"trained": False, "n_samples": n,
                "reason": "Prediksi out-of-sample terlalu sedikit untuk dievaluasi."}

    p, yy, rr = oos_proba[mask], y[mask], rets[mask]
    base_rate = float(yy.mean())
    brier_model = float(np.mean((p - yy) ** 2))
    brier_base = base_rate * (1 - base_rate)
    bss = (1 - brier_model / brier_base) if brier_base > 1e-9 else 0.0

    take = p >= take_threshold
    oos = {
        "n_oos": int(mask.sum()),
        "base_win_rate": float(base_rate * 100),
        "base_avg_return": float(rr.mean()),
        "coverage": float(take.mean()),
        "filtered_win_rate": float(yy[take].mean() * 100) if take.sum() > 0 else None,
        "filtered_avg_return": float(rr[take].mean()) if take.sum() > 0 else None,
        "skipped_win_rate": float(yy[~take].mean() * 100) if (~take).sum() > 0 else None,
        "bss": float(bss),
        "take_threshold": take_threshold,
    }

    final_model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                  verbosity=-1, random_state=42)
    final_model.fit(X, y)

    def _predict(feat_row: dict) -> float:
        x = np.array([[feat_row.get(c, np.nan) for c in feat_cols]], dtype=float)
        return float(final_model.predict_proba(x)[0, 1])

    return {"trained": True, "n_samples": n, "oos": oos,
            "predict_fn": _predict, "feat_cols": feat_cols}


# --------------------------------------------------------------------------
# ---- 4d. PORTFOLIO-LEVEL BACKTEST ----
# --------------------------------------------------------------------------

def run_portfolio_backtest(signals_by_ticker: dict,
                            initial_capital: float = 100_000_000.0,
                            max_positions: int = 5,
                            position_pct: float = 0.20,
                            fee_buy_bps: float = 20.0, fee_sell_bps: float = 30.0,
                            slippage_bps: float = 5.0,
                            stop_loss_pct: float | None = None,
                            max_holding_days: int | None = None,
                            trading_days: float = 252.0,
                            lot_size: int | None = None,
                            max_turnover_participation: float | None = None) -> dict:
    """
    Backtest level portofolio (review 4d): modal BERSAMA, maksimal
    `max_positions` posisi konkuren, sinyal BUY di-skip kalau slot penuh
    (dihitung & dilaporkan sebagai n_skipped_full_slots). Berbeda dari
    run_aggregate_backtest yang men-pool trade tapi mengasumsikan tiap ticker
    all-in dengan modal sendiri — di sini equity curve mencerminkan apa yang
    benar-benar terjadi kalau kamu menjalankan strategi ini dengan satu rekening.

    signals_by_ticker: {ticker: output composite_signal(df)} — harus punya
        kolom Open/High/Low/Close + composite_signal + composite_score.
    position_pct: fraksi EQUITY saat itu per posisi baru (bukan cash) —
        compounding otomatis terjaga.
    Eksekusi: sinyal dari Close hari T dieksekusi di Open hari T+1 (sama
        seperti run_backtest, anti-lookahead). Kalau ticker tidak punya bar
        di hari eksekusi, order hangus (dokumentasikan, bukan di-ffill).
    Prioritas entry saat slot terbatas: composite_score tertinggi dulu.
    """
    frames = {}
    for t, sdf in signals_by_ticker.items():
        need = {"Open", "High", "Low", "Close", "composite_signal", "composite_score"}
        if not need.issubset(set(sdf.columns)):
            raise ValueError(f"{t}: DataFrame sinyal harus punya kolom {sorted(need)}")
        f = sdf.copy()
        if "Volume" not in f.columns:
            f["Volume"] = np.nan
        frames[t] = f

    all_dates = sorted(set().union(*[set(f.index) for f in frames.values()]))
    if not all_dates:
        return {"error": "tidak ada tanggal sama sekali"}
    pos_in_df = {t: {d: i for i, d in enumerate(f.index)} for t, f in frames.items()}

    cost_buy = (fee_buy_bps + slippage_bps) / 1e4
    cost_sell = (fee_sell_bps + slippage_bps) / 1e4

    cash = float(initial_capital)
    positions: dict = {}
    trades: list = []
    equity_idx, equity_val = [], []
    skipped_full = 0
    pending_entries: list = []
    pending_exits: set = set()

    def _mtm(d):
        v = cash
        for t, p in positions.items():
            i = pos_in_df[t].get(d)
            px = frames[t]["Close"].iloc[i] if i is not None else p["last_px"]
            v += p["units"] * px
        return v

    for d in all_dates:
        for t, p in positions.items():
            i = pos_in_df[t].get(d)
            if i is not None:
                p["last_px"] = frames[t]["Close"].iloc[i]

        # 1) eksekusi exit tertunda di Open hari ini
        for t in list(pending_exits):
            i = pos_in_df[t].get(d)
            if t in positions and i is not None:
                px = frames[t]["Open"].iloc[i]
                p = positions.pop(t)
                reason = p.pop("pending_reason", "signal")
                cash += p["units"] * px * (1 - cost_sell)
                trades.append({"symbol": t, "entry_date": p["entry_date"], "exit_date": d,
                                "entry_price": p["entry_price"], "exit_price": px,
                                "return_pct": (px / p["entry_price"] - 1) * 100,
                                "exit_reason": reason})
        pending_exits = set()

        # 2) eksekusi entry tertunda di Open hari ini, score tertinggi dulu
        slots = max_positions - len(positions)
        for t, _score in sorted(pending_entries, key=lambda x: -x[1]):
            if slots <= 0:
                skipped_full += 1
                continue
            i = pos_in_df[t].get(d)
            if i is None or t in positions:
                continue
            px = frames[t]["Open"].iloc[i]
            if not np.isfinite(px) or px <= 0:
                continue
            alloc = min(cash, _mtm(d) * position_pct)
            if max_turnover_participation is not None:
                to_avg = float((frames[t]["Close"] * frames[t]["Volume"]).iloc[:i + 1].tail(20).mean())
                if np.isfinite(to_avg) and to_avg > 0:
                    alloc = min(alloc, max_turnover_participation * to_avg)
            units = (alloc * (1 - cost_buy)) / px
            if lot_size:
                units = float(np.floor(units / lot_size) * lot_size)
            if units <= 0:
                continue
            cash -= units * px / (1 - cost_buy)
            positions[t] = {"units": units, "entry_price": px, "entry_date": d,
                             "entry_i": i, "last_px": px}
            slots -= 1
        pending_entries = []

        # 3) stop-loss intraday pakai Low hari ini
        if stop_loss_pct is not None:
            for t in list(positions):
                i = pos_in_df[t].get(d)
                if i is None:
                    continue
                p = positions[t]
                stop = p["entry_price"] * (1 - stop_loss_pct)
                if frames[t]["Low"].iloc[i] <= stop:
                    cash += p["units"] * stop * (1 - cost_sell)
                    trades.append({"symbol": t, "entry_date": p["entry_date"], "exit_date": d,
                                    "entry_price": p["entry_price"], "exit_price": stop,
                                    "return_pct": (stop / p["entry_price"] - 1) * 100,
                                    "exit_reason": "stop_loss"})
                    positions.pop(t)

        # 4) generate order dari sinyal hari ini (dieksekusi besok)
        for t, f in frames.items():
            i = pos_in_df[t].get(d)
            if i is None:
                continue
            sig = f["composite_signal"].iloc[i]
            score = float(f["composite_score"].iloc[i])
            if t in positions:
                p = positions[t]
                if sig in ("SELL", "EXIT"):
                    p["pending_reason"] = "signal"
                    pending_exits.add(t)
                elif max_holding_days is not None and (i - p["entry_i"]) >= max_holding_days:
                    p["pending_reason"] = "max_holding_days"
                    pending_exits.add(t)
            elif sig == "BUY":
                pending_entries.append((t, score))

        equity_idx.append(d)
        equity_val.append(_mtm(d))

    # likuidasi sisa posisi di harga terakhir yang diketahui
    for t, p in positions.items():
        px = p["last_px"]
        cash += p["units"] * px * (1 - cost_sell)
        trades.append({"symbol": t, "entry_date": p["entry_date"], "exit_date": equity_idx[-1],
                        "entry_price": p["entry_price"], "exit_price": px,
                        "return_pct": (px / p["entry_price"] - 1) * 100,
                        "exit_reason": "end_of_data"})
    if equity_val:
        equity_val[-1] = cash  # setelah biaya likuidasi akhir

    equity = pd.Series(equity_val, index=pd.DatetimeIndex(equity_idx), name="equity")
    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(equity, trades_df, initial_capital,
                                trading_days=trading_days)
    metrics["n_skipped_full_slots"] = int(skipped_full)
    metrics["max_positions"] = int(max_positions)
    metrics["position_pct"] = float(position_pct)

    # equal-weight buy & hold sebagai pembanding portofolio
    bh_panel = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).reindex(equity.index).ffill()
    bh_norm = bh_panel / bh_panel.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    buy_hold_curve = bh_norm.mean(axis=1, skipna=True) * initial_capital

    return {"equity_curve": equity, "buy_hold_curve": buy_hold_curve,
            "trades": trades_df, "metrics": metrics}


# --------------------------------------------------------------------------
# ---- 2b. IDX REALISM: lot size, tick size, ARB lock ----
# --------------------------------------------------------------------------
# CATATAN: tier ARA/ARB & fraksi tick BEI BERUBAH dari waktu ke waktu
# (terakhir direvisi 2023, dibuat simetris). Angka di bawah adalah default
# yang masuk akal per 2024 — VERIFIKASI ke aturan BEI terkini sebelum
# dipakai untuk keputusan ril, dan override lewat parameter kalau perlu.

def idx_tick_size(price: float) -> float:
    """Fraksi harga (tick) minimum per tier harga. Verifikasi ke BEI."""
    if price < 200:
        return 1.0
    if price < 500:
        return 2.0
    if price < 2000:
        return 5.0
    if price < 5000:
        return 10.0
    return 25.0


def idx_arb_limit_pct(price: float) -> float:
    """Batas Auto Rejection Bawah per tier harga (desimal). Verifikasi ke BEI."""
    if price < 200:
        return 0.25
    if price < 5000:
        return 0.20
    return 0.15


def _is_locked_arb(df: pd.DataFrame, j: int) -> bool:
    """True kalau hari j terkunci ARB: turun sampai batas DAN ditutup di Low
    (tidak ada bid — secara mekanis tidak bisa keluar hari itu)."""
    if j <= 0 or j >= len(df):
        return False
    prev_close = df["Close"].iloc[j - 1]
    close_j = df["Close"].iloc[j]
    low_j = df["Low"].iloc[j]
    if prev_close <= 0:
        return False
    limit = idx_arb_limit_pct(prev_close)
    return (close_j / prev_close - 1) <= -limit + 1e-9 and np.isclose(close_j, low_j)


# --------------------------------------------------------------------------
# ---- 4e. EXTENDED METRICS: Sortino, Calmar, alpha/beta ----
# --------------------------------------------------------------------------

def extended_performance_metrics(equity_curve: pd.Series,
                                  trading_days: float = 252.0,
                                  benchmark_close: pd.Series | None = None) -> dict:
    """
    Metrik tambahan murah (review 4e): Sortino (hanya menghukum volatilitas
    TURUN, bukan naik), Calmar (CAGR / |max drawdown|), dan alpha/beta vs
    benchmark kalau disediakan (regresi return harian; alpha di-anualisasi
    secara aritmetika — aproksimasi standar, bukan exact compounding).
    """
    out = {}
    daily = equity_curve.pct_change().dropna()
    if len(daily) < 5:
        return {"error": "equity curve terlalu pendek"}

    downside = daily[daily < 0]
    dd_std = downside.std(ddof=1)
    out["sortino"] = (float(daily.mean() / dd_std * np.sqrt(trading_days))
                       if len(downside) > 1 and dd_std > 0 else None)

    n_years = len(equity_curve) / trading_days
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else None
    max_dd = float(((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min())
    out["calmar"] = (float(cagr / abs(max_dd))
                      if cagr is not None and max_dd < 0 else None)

    if benchmark_close is not None:
        bench = benchmark_close.reindex(equity_curve.index).ffill().pct_change()
        pair = pd.concat([daily, bench], axis=1, keys=["strat", "bench"]).dropna()
        if len(pair) > 20 and pair["bench"].var() > 0:
            beta = float(pair["strat"].cov(pair["bench"]) / pair["bench"].var())
            alpha_daily = float(pair["strat"].mean() - beta * pair["bench"].mean())
            out["beta"] = beta
            out["alpha_annual_pct"] = alpha_daily * trading_days * 100
            out["benchmark_corr"] = float(pair["strat"].corr(pair["bench"]))
    return out

# ==========================================================================
# ==== SECTION 8: SIGNAL LOG (4c) — forward test otomatis ================
# ==========================================================================
# Dipindah ke sini (bukan app.py, walau instruksi asal panduan taruh di
# app.py) supaya bisa dipanggil headless dari precompute.py tanpa perlu
# menjalankan seluruh Streamlit app -- konsisten dengan arsitektur modul ini
# yang deliberately Streamlit-agnostic. Beda dari Trade Journal (keputusan
# MANUSIA, tetap di app.py) dan Backtest Log (hasil SIMULASI, tetap di
# app.py): ini mencatat keputusan SISTEM (score + verdict hari ini), lalu
# forward return-nya diisi otomatis 5+ hari kemudian. Satu-satunya validasi
# yang tidak bisa dibantah: kalau live IC ≈ 0 padahal backtest bagus ->
# backtest overfit.

SIGNAL_LOG_FILE = "signals_log.csv"
SIGNAL_LOG_COLUMNS = ["logged_at", "log_date", "symbol", "asset_type",
                      "composite_score", "verdict", "close_at_signal", "fwd_return_5d_pct"]


def load_signal_log() -> pd.DataFrame:
    if os.path.exists(SIGNAL_LOG_FILE):
        try:
            return pd.read_csv(SIGNAL_LOG_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)


def log_signal_snapshot(symbol: str, asset_type: str, sig_df: pd.DataFrame):
    """Append 1 baris per (symbol, hari) — dedupe supaya rerun tidak menumpuk."""
    today = pd.Timestamp.now().normalize()
    df = load_signal_log()
    if len(df) > 0:
        dup = (df["symbol"] == symbol) & (pd.to_datetime(df["log_date"]) == today)
        if dup.any():
            return
    latest = sig_df.iloc[-1]
    df = pd.concat([df, pd.DataFrame([{
        "logged_at": pd.Timestamp.now().isoformat(),
        "log_date": str(today.date()),
        "symbol": symbol, "asset_type": asset_type,
        "composite_score": float(latest["composite_score"]),
        "verdict": latest["composite_signal"],
        "close_at_signal": float(latest["Close"]),
        "fwd_return_5d_pct": None,
    }])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)


def update_signal_log_forward_returns() -> pd.DataFrame:
    """Isi fwd_return_5d_pct untuk baris yang umurnya sudah >= 7 hari kalender."""
    df = load_signal_log()
    changed = False
    for idx, row in df[df["fwd_return_5d_pct"].isna()].iterrows():
        log_date = pd.to_datetime(row["log_date"])
        if (pd.Timestamp.now() - log_date).days < 7:
            continue
        try:
            d = cached_fetch_data(row["asset_type"], row["symbol"], period="1mo")
            after = d[d.index > log_date]["Close"]
            if len(after) >= 5:
                df.loc[idx, "fwd_return_5d_pct"] = float(
                    (after.iloc[4] / row["close_at_signal"] - 1) * 100)
                changed = True
        except Exception:
            continue
    if changed:
        df.to_csv(SIGNAL_LOG_FILE, index=False)
    return df
