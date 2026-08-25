"""
advanced_features.py
==========================================================================
The 4 "wow" additions on top of the base engine, kept in their own file
so quant_engine.py stays focused on data + core models:

  1. hrp_weights()              -> Hierarchical Risk Parity portfolio
  2. optimize_signal_params_ga()-> Genetic Algorithm auto-tuner for
                                    composite-signal / backtest params
  3. build_tearsheet_html()     -> self-contained interactive tearsheet
                                    (QuantStats/OpenStatz-style report)
  4. generate_ai_narrative()    -> plain-language summary of a symbol's
                                    combined analysis, for presentations

Only #3 needs plotly, and only #4 optionally needs the `anthropic`
package + an ANTHROPIC_API_KEY — both degrade gracefully if missing,
so this module never hard-blocks a precompute run for one symbol.
==========================================================================
"""

from __future__ import annotations

import os
import random
import time
import numpy as np
import pandas as pd


# ==========================================================================
# ==== 1. HIERARCHICAL RISK PARITY (Lopez de Prado, 2016) ====
# ==========================================================================
# Unlike Min-Variance/Max-Sharpe (which invert the covariance matrix and
# can blow up or go wildly negative when assets are highly correlated or
# the sample is short), HRP never inverts anything. It clusters assets by
# correlation structure, then allocates risk top-down through the cluster
# tree. Slightly less "optimal" in-sample, historically more stable
# out-of-sample — the standard modern complement to sit next to GMV/
# Tangency/Equal-Weight, not a replacement for them.

def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    d = ((1 - corr) / 2.0).clip(lower=0.0)
    return np.sqrt(d)


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Recover the leaf order implied by the linkage tree, so the
    covariance matrix can be read top-to-bottom through the cluster
    hierarchy instead of in arbitrary ticker order."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _get_cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    cov_slice = cov.loc[items, items]
    ivp = 1.0 / np.diag(cov_slice)
    ivp /= ivp.sum()
    w = ivp.reshape(-1, 1)
    return float((w.T @ cov_slice.values @ w).item())


def _get_rec_bisection(cov: pd.DataFrame, sort_ix: list[str]) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [
            i[j:k] for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c0, c1 = c_items[i], c_items[i + 1]
            var0, var1 = _get_cluster_var(cov, c0), _get_cluster_var(cov, c1)
            alpha = 1.0 - var0 / (var0 + var1)
            w[c0] *= alpha
            w[c1] *= (1.0 - alpha)
    return w


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity weights for a set of assets.

    returns: DataFrame of periodic (e.g. daily log) returns, columns =
             tickers, no NaNs (align/dropna before calling — same
             requirement as the existing Markowitz code in Section 5.6).

    Returns a pd.Series of weights indexed by ticker, summing to 1,
    always non-negative (HRP is long-only by construction — this is one
    of its practical advantages over the closed-form GMV/Tangency
    weights, which can go negative and imply short-selling).
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    cov = returns.cov()
    corr = returns.corr()
    dist = _correl_dist(corr)
    link = linkage(squareform(dist.values, checks=False), method="single")
    sort_ix = corr.index[_get_quasi_diag(link)].tolist()
    w = _get_rec_bisection(cov, sort_ix)
    return w.reindex(returns.columns)


def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> dict:
    """
    Pairwise rolling Pearson correlation across a returns panel, on top of
    the usual single-window snapshot (`returns.corr()`, already used at
    the Portfolio tab's covariance-matrix/eigenvalue step).

    WHY THIS MATTERS BEYOND A STATIC SNAPSHOT: a single full-sample
    correlation number hides regime dependence — two assets can average
    a comfortable 0.3 correlation over a year while briefly spiking
    toward 0.8-0.9 during a market-wide selloff, which is exactly the
    moment diversification is needed most and static correlation numbers
    make you think you have it. Tracking correlation through TIME (not
    just its full-sample average) surfaces that kind of regime shift.

    returns: DataFrame of periodic (daily) returns, columns = tickers,
             no NaNs (same requirement as hrp_weights above).
    window:  rolling window length in periods (default 60 trading days
             ≈ ~3 months — long enough for a stable correlation estimate,
             short enough to catch a multi-week regime shift).

    Returns dict: {pairs: {"TICKER_A/TICKER_B": pd.Series of rolling
    correlation indexed like `returns`}, latest: DataFrame (tickers x
    tickers) of the most recent rolling-window correlation — same shape
    as returns.corr() but using only the last `window` periods instead
    of the full sample, so it's directly comparable to see how much the
    "current" correlation structure has drifted from the full-sample one,
    full_sample: DataFrame, the full-sample returns.corr() for reference}.
    """
    cols = list(returns.columns)
    pairs = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            pairs[f"{a}/{b}"] = returns[a].rolling(window).corr(returns[b])

    latest = returns.tail(window).corr()
    full_sample = returns.corr()

    return {"pairs": pairs, "latest": latest, "full_sample": full_sample}


# ==========================================================================
# ==== 2. GENETIC ALGORITHM PARAMETER OPTIMIZER ====
# ==========================================================================
# Auto-tunes composite-signal parameters (mr_window, mr_z_entry, mom_fast,
# mom_slow, mr_weight) against backtest fitness (default: Sharpe ratio,
# penalized for too few trades so it can't "win" by barely trading).
# Plain-numpy GA — no extra dependency (DEAP etc.) needed for a search
# space this small.

PARAM_SPACE = {
    "mr_window":      (10, 40),     # int
    "mr_z_entry":     (1.0, 2.5),   # float
    "mom_fast":       (5, 20),      # int
    "mom_slow":       (20, 60),     # int
    "mr_weight":      (0.2, 0.8),   # float
    "stochbb_weight": (0.0, 0.5),   # float — 0 means GA can choose to not use this leg at all
}


def _random_individual(rng: np.random.Generator) -> dict:
    ind = {}
    for k, (lo, hi) in PARAM_SPACE.items():
        if isinstance(lo, int) and isinstance(hi, int):
            ind[k] = int(rng.integers(lo, hi + 1))
        else:
            ind[k] = float(rng.uniform(lo, hi))
    return ind


def _clip_individual(ind: dict) -> dict:
    out = dict(ind)
    for k, (lo, hi) in PARAM_SPACE.items():
        out[k] = min(max(out[k], lo), hi)
        if isinstance(lo, int) and isinstance(hi, int):
            out[k] = int(round(out[k]))
    if out["mom_slow"] <= out["mom_fast"]:
        out["mom_slow"] = out["mom_fast"] + 5
    return out


def _crossover(a: dict, b: dict, rng: np.random.Generator) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in PARAM_SPACE}


def _mutate(ind: dict, rng: np.random.Generator, rate: float = 0.3) -> dict:
    out = dict(ind)
    for k, (lo, hi) in PARAM_SPACE.items():
        if rng.random() < rate:
            span = (hi - lo) * 0.2
            out[k] = out[k] + rng.uniform(-span, span)
    return _clip_individual(out)


def optimize_signal_params_ga(df: pd.DataFrame, composite_signal_fn, run_backtest_fn,
                               pop_size: int = 24, n_generations: int = 15,
                               seed: int = 42, min_trades: int = 4,
                               train_frac: float = 0.7,
                               progress_callback=None) -> dict:
    """
    Genetic-algorithm search over composite-signal parameters, fitness =
    backtest Sharpe ratio (with a penalty if the resulting strategy trades
    fewer than `min_trades` times over the sample — otherwise the GA can
    "cheat" by finding one lucky rare trade with a huge Sharpe).

    WALK-FORWARD SPLIT (train_frac): the GA only ever SEARCHES using data
    up to `train_frac` of the sample ("train"). The remaining tail
    ("holdout") is never touched during selection — it's evaluated exactly
    ONCE, after the search is over, on the single best individual the GA
    converged to. This matters because a GA searching hundreds of
    parameter combinations against one fixed backtest window WILL find
    something that looks great purely from noise-fitting that window,
    the same way trying enough lottery tickets guarantees a winner. The
    train-fitness number (`best_fitness`) is therefore optimistic by
    construction and should never be reported on its own — `holdout_fitness`
    (and its comparison to `baseline_holdout_fitness`, i.e. how the
    *untouched* default params would have done on that same holdout slice)
    is the only number here that says anything about a real edge.

    Indicators are computed on the FULL `df` for every individual (not on
    a truncated slice) so that rolling windows (mr_window, mom_slow, ...)
    have their normal historical warm-up context right up to the holdout
    boundary — exactly like a live signal that already has price history
    before "today". Only the resulting equity-curve/Sharpe evaluation is
    then restricted to the train or holdout slice. This avoids both
    lookahead (holdout rows never influence which params get selected)
    and an artificial NaN gap at the start of the holdout window.

    composite_signal_fn / run_backtest_fn: pass in quant_engine's
    `composite_signal` / `run_backtest` directly — kept as parameters
    (not imported) so this module has zero import-time dependency on
    quant_engine, and stays reusable/testable standalone.

    Returns dict: {best_params, best_fitness, holdout_fitness,
                    baseline_params, baseline_fitness, baseline_holdout_fitness,
                    improvement, holdout_improvement, split_date,
                    n_train_rows, n_holdout_rows, overfit_gap, history}
    `overfit_gap` = best_fitness - holdout_fitness: large positive gap =
    the GA mostly fit noise in the train window, treat best_params with
    suspicion. `history` is the best TRAIN fitness per generation, for
    plotting convergence (never holdout — the search never sees it).
    """
    rng = np.random.default_rng(seed)

    n = len(df)
    split_idx = int(n * train_frac)
    split_idx = max(20, min(split_idx, n - 20))  # guard: both slices need enough rows
    split_date = df.index[split_idx]

    def _fitness(ind: dict, part: str = "train") -> float:
        try:
            sig_df = composite_signal_fn(
                df, mr_window=ind["mr_window"], mr_z_entry=ind["mr_z_entry"],
                mom_fast=ind["mom_fast"], mom_slow=ind["mom_slow"],
                mr_weight=ind["mr_weight"], stochbb_weight=ind.get("stochbb_weight", 0.0),
            )
            eval_df = sig_df.iloc[:split_idx] if part == "train" else sig_df.iloc[split_idx:]
            result = run_backtest_fn(eval_df, signal_col="composite_signal")
            m = result["metrics"]
            sharpe = m.get("sharpe_ratio")
            n_trades = m.get("n_trades", 0)
            if sharpe is None or np.isnan(sharpe):
                return -10.0
            penalty = 0.0 if n_trades >= min_trades else (min_trades - n_trades) * 0.5
            return float(sharpe) - penalty
        except Exception:
            return -10.0

    baseline_params = {"mr_window": 20, "mr_z_entry": 1.5, "mom_fast": 10,
                        "mom_slow": 30, "mr_weight": 0.5, "stochbb_weight": 0.0}
    baseline_fitness = _fitness(baseline_params, "train")
    baseline_holdout_fitness = _fitness(baseline_params, "holdout")

    population = [_random_individual(rng) for _ in range(pop_size)]
    population[0] = dict(baseline_params)  # seed with the current defaults

    history = []
    best_ind, best_fit = None, -np.inf

    for gen in range(n_generations):
        scored = [(ind, _fitness(ind, "train")) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored[0][1] > best_fit:
            best_ind, best_fit = scored[0]
        history.append(best_fit)

        if progress_callback:
            progress_callback(gen + 1, n_generations, best_fit)

        # elitism: keep top 25%, breed the rest via tournament selection
        survivors = [ind for ind, _ in scored[: max(2, pop_size // 4)]]
        children = list(survivors)
        while len(children) < pop_size:
            p1 = survivors[rng.integers(0, len(survivors))]
            p2 = survivors[rng.integers(0, len(survivors))]
            child = _mutate(_crossover(p1, p2, rng), rng)
            children.append(child)
        population = children

    # Single, one-shot holdout evaluation of the winning individual — this
    # is the number that's actually informative about real edge.
    holdout_fitness = _fitness(best_ind, "holdout")

    return {
        "best_params": best_ind,
        "best_fitness": float(best_fit),
        "holdout_fitness": float(holdout_fitness),
        "baseline_params": baseline_params,
        "baseline_fitness": float(baseline_fitness),
        "baseline_holdout_fitness": float(baseline_holdout_fitness),
        "improvement": float(best_fit - baseline_fitness),
        "holdout_improvement": float(holdout_fitness - baseline_holdout_fitness),
        "overfit_gap": float(best_fit - holdout_fitness),
        "split_date": str(split_date.date()) if hasattr(split_date, "date") else str(split_date),
        "n_train_rows": int(split_idx),
        "n_holdout_rows": int(n - split_idx),
        "history": [float(h) for h in history],
    }


# ==========================================================================
# ==== 3. TEARSHEET (QuantStats/OpenStatz-style, self-contained HTML) ====
# ==========================================================================

def build_tearsheet_html(ticker: str, equity_curve: pd.Series,
                          buy_hold_curve: pd.Series, trades_df: pd.DataFrame,
                          metrics: dict) -> str:
    """
    Build a single self-contained HTML tearsheet (Plotly CDN, no server
    needed to view it) — equity vs buy&hold, drawdown chart, monthly
    returns heatmap, and a metrics table. Meant to be generated ONCE by
    precompute.py per symbol and saved to cache/<ticker>_tearsheet.html;
    the viewer app just embeds the saved file, it never regenerates this
    live (that's the whole point of the precompute split).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    daily_ret = equity_curve.pct_change().dropna()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100

    # ---- Monthly returns heatmap matrix ----
    monthly = (1 + daily_ret).resample("ME").prod() - 1
    monthly.index = pd.to_datetime(monthly.index)
    heat = monthly.to_frame("ret")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    pivot = heat.pivot(index="year", columns="month", values="ret") * 100
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=range(1, 13))

    fig = make_subplots(
        rows=3, cols=1, row_heights=[0.45, 0.25, 0.3],
        subplot_titles=("Equity Curve vs Buy & Hold", "Drawdown (%)",
                         "Monthly Returns Heatmap (%)"),
        vertical_spacing=0.09,
    )
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values,
                              name="Strategy", line=dict(color="#2ca02c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=buy_hold_curve.index, y=buy_hold_curve.values,
                              name="Buy & Hold", line=dict(color="#888", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown",
                              fill="tozeroy", line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Heatmap(z=pivot.values, x=month_labels, y=pivot.index.astype(str),
                              colorscale="RdYlGn", zmid=0, showscale=True,
                              text=np.round(pivot.values, 1), texttemplate="%{text}"),
                  row=3, col=1)
    fig.update_layout(height=950, showlegend=True,
                       title=f"Tearsheet — {ticker}", template="plotly_white")

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _row(label, value):
        return f"<tr><td>{label}</td><td style='text-align:right'><b>{value}</b></td></tr>"

    def _fmt_or_na(v, fmt):
        return fmt.format(v) if v is not None else "N/A"

    rows = "".join([
        _row("Total Return", f"{metrics.get('total_return_pct', float('nan')):.1f}%"),
        _row("CAGR", _fmt_or_na(metrics.get('cagr_pct'), "{:.1f}%")),
        _row("Sharpe Ratio", _fmt_or_na(metrics.get('sharpe_ratio'), "{:.2f}")),
        _row("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.1f}%"),
        _row("Win Rate", _fmt_or_na(metrics.get('win_rate_pct'), "{:.1f}%")),
        _row("Trades", f"{metrics.get('n_trades', 0)}"),
        _row("Buy & Hold Return", f"{metrics.get('buy_hold_return_pct', 0):.1f}%"),
    ])
    metrics_table = f"""
    <table style='border-collapse:collapse;width:340px;font-family:sans-serif;font-size:14px'>
      <tbody>{rows}</tbody>
    </table>
    <style>td {{ padding:6px 12px; border-bottom:1px solid #eee; }}</style>
    """

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Tearsheet — {ticker}</title></head>
    <body style="font-family:sans-serif;max-width:1000px;margin:20px auto">
    <h2>📊 Tearsheet — {ticker}</h2>
    {metrics_table}
    {chart_html}
    </body></html>"""


# ==========================================================================
# ==== 4. AI NARRATIVE SUMMARY ====
# ==========================================================================
# Calls the Anthropic API if ANTHROPIC_API_KEY is set (run ONCE per symbol
# during precompute, never live in the viewer — keeps API cost/latency
# out of the hosted app entirely). Falls back to a template-composed
# Indonesian summary from the same numbers if no key is configured, so
# this feature never blocks a precompute run.

def _template_narrative(ticker: str, signal_row: dict, mc_summary: dict,
                         backtest_metrics: dict, robustness: dict) -> str:
    sig = signal_row.get("composite_signal", "HOLD")
    score = signal_row.get("composite_score", 0.0)
    prob_profit = mc_summary.get("prob_profit", 0.5) * 100
    sharpe = backtest_metrics.get("sharpe_ratio")
    win_rate = backtest_metrics.get("win_rate_pct")
    concentrated = robustness.get("is_concentrated", False)

    parts = [
        f"{ticker} saat ini menunjukkan sinyal composite **{sig}** "
        f"(skor {score:+.2f}).",
        f"Simulasi Monte Carlo memberi probabilitas profit "
        f"~{prob_profit:.0f}% pada horizon simulasi.",
    ]
    if sharpe is not None:
        parts.append(
            f"Backtest historis mencatat Sharpe ratio {sharpe:.2f}"
            + (f" dengan win rate {win_rate:.0f}%." if win_rate is not None else ".")
        )
    if concentrated:
        parts.append(
            "⚠️ Perlu dicatat: sebagian besar return backtest ini berasal dari "
            "satu-dua trade outlier, jadi performa historisnya belum tentu "
            "berulang/robust."
        )
    return " ".join(parts)


def _is_transient_gemini_error(err_str: str) -> bool:
    s = err_str.lower()
    return ("503" in s or "unavailable" in s or "high demand" in s
            or "429" in s or "resource_exhausted" in s or "overloaded" in s)


def _try_gemini(prompt: str, api_key: str) -> tuple[str | None, str | None]:
    """Returns (text, error_message) — exactly one is None. Tries a couple
    of model name candidates in case one is deprecated/renamed (Gemini
    model names churn — e.g. gemini-2.5-flash vs gemini-3.x-flash — and
    this shouldn't require another round of guessing which one your key
    actually has access to). Each candidate also gets a couple of quick
    retries specifically for transient "high demand"/503 errors — common
    on the free tier at peak times, and self-resolves within seconds, so
    it's worth a short retry before moving to the next model or giving up."""
    try:
        from google import genai
    except ImportError:
        return None, "package 'google-genai' belum ke-install (cek requirements.txt)"

    client = genai.Client(api_key=api_key)
    last_err = None
    for model_name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"):
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                text = (getattr(resp, "text", None) or "").strip()
                if text:
                    return text, None
                # No exception, but also no usable text — e.g. blocked by a safety
                # filter (finish_reason != STOP). This is a REAL failure, not a
                # "everything's fine, just empty" — must be reported, not silently
                # dropped (this exact gap was the original bug: silent template
                # fallback with zero indication anything went wrong).
                finish_reason = None
                try:
                    finish_reason = resp.candidates[0].finish_reason
                except Exception:
                    pass
                last_err = f"model '{model_name}' returned empty text (finish_reason={finish_reason})"
                break  # empty-text isn't transient — no point retrying same model
            except Exception as e:
                last_err = f"model '{model_name}': {e}"
                if _is_transient_gemini_error(str(e)) and attempt < 2:
                    time.sleep(2 * (attempt + 1))  # 2s, then 4s
                    continue  # retry SAME model
                break  # non-transient error, or retries exhausted — try next model
    return None, last_err


def _try_groq(prompt: str, api_key: str) -> tuple[str | None, str | None]:
    """Groq runs on its own LPU hardware (Llama/Mixtral/Qwen models, not
    Gemini) — completely separate free-tier quota from Google's, so this
    is a genuinely independent fallback, not just "try Gemini again with
    extra steps". Free tier ~30 req/min, ~1000 req/day, no credit card."""
    try:
        from groq import Groq
    except ImportError:
        return None, "package 'groq' belum ke-install (cek requirements.txt)"
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text, None
        return None, "respons kosong tanpa error eksplisit"
    except Exception as e:
        return None, str(e)


def _try_anthropic(prompt: str, api_key: str) -> tuple[str | None, str | None]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        if text:
            return text, None
        return None, "respons kosong tanpa error eksplisit"
    except Exception as e:
        return None, str(e)


def generate_ai_narrative(ticker: str, signal_row: dict, mc_summary: dict,
                           backtest_metrics: dict, robustness: dict,
                           fundamentals: dict | None = None) -> str:
    """
    Returns a short (3-5 sentence) Indonesian-language narrative summary
    combining Monte Carlo + Signal + Backtest (+ Fundamental, if
    available) for one symbol — meant for the "🎯 Kesimpulan" tab / a
    presentation slide.

    Tries, in order: Gemini (GEMINI_API_KEY — free tier is generous enough
    for on-demand personal use, so this is the default path) -> Anthropic
    (ANTHROPIC_API_KEY, if you'd rather use/have credits for Claude) ->
    a fixed-template fallback (always available, zero cost, but — as the
    name implies — same sentence skeleton every time, just different
    numbers plugged in). EVERY failure along the way (exception OR an
    empty/blocked response with no exception) is recorded and surfaced in
    the final output — no silent drop to template with zero explanation.
    """
    fund_str = f"\nFundamental: {fundamentals}" if fundamentals else ""
    prompt = (
        f"Kamu analis kuantitatif. Tulis ringkasan 3-4 kalimat bahasa "
        f"Indonesia (untuk slide presentasi riset, bukan disclaimer "
        f"panjang) tentang saham/aset {ticker} berdasarkan data berikut:\n"
        f"Sinyal composite: {signal_row}\n"
        f"Monte Carlo: {mc_summary}\n"
        f"Backtest: {backtest_metrics}\n"
        f"Robustness check: {robustness}"
        f"{fund_str}\n"
        f"Netral, faktual, sebutkan angka penting, dan kalau ada tanda "
        f"bahaya (mis. backtest didominasi 1-2 trade, atau win rate "
        f"rendah), sebutkan itu juga. Jangan pakai heading/markdown, "
        f"langsung paragraf. Variasikan gaya bahasa/susunan kalimat "
        f"tiap kali — jangan selalu mulai dengan pola yang sama."
    )

    attempts_tried = []  # for the final error note, if everything fails

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        text, err = _try_gemini(prompt, gemini_key)
        if text:
            return text
        attempts_tried.append(f"Gemini gagal ({err})")

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        text, err = _try_groq(prompt, groq_key)
        if text:
            return text
        attempts_tried.append(f"Groq gagal ({err})")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        text, err = _try_anthropic(prompt, anthropic_key)
        if text:
            return text
        attempts_tried.append(f"Claude gagal ({err})")

    fallback = _template_narrative(ticker, signal_row, mc_summary,
                                    backtest_metrics, robustness)
    if attempts_tried:
        note = " · ".join(attempts_tried)
        return fallback + f"\n\n_(Catatan: {note} — di atas fallback template.)_"
    return fallback
