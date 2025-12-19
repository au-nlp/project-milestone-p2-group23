import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr, spearmanr
from .embed import score_df_by_ideas, score_segments_by_ideas
from .metrics import get_daily_scores

def fetch_daily_prices(
    tickers,
    start="2020-05-01",
    end="2020-06-30",
    price_field="Adj Close",
    calendar_days=True,
    fill_method="ffill",
):
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    yf_end = end + pd.Timedelta(days=1)  # yfinance end is exclusive

    data = yf.download(
        tickers, start=start, end=yf_end, auto_adjust=False, progress=False
    )
    if price_field not in data:
        available = list(data.columns.get_level_values(0).unique())
        raise ValueError(
            f"{price_field} not found. Available price fields: {available}"
        )

    prices = data[price_field].copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    if calendar_days:
        all_days = pd.date_range(start, end, freq="D")
        prices = prices.reindex(all_days)
        if fill_method:
            prices = prices.fillna(method=fill_method)

    prices.index.name = "date"
    prices = prices.reset_index()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices


def fetch_daily_volume(
    tickers,
    start="2020-05-01",
    end="2020-06-30",
    calendar_days=True,
    fill_method="ffill",
):
    """
    Convenience wrapper to pull daily volume with the same semantics as prices.
    """
    return fetch_daily_prices(
        tickers,
        start=start,
        end=end,
        price_field="Volume",
        calendar_days=calendar_days,
        fill_method=fill_method,
    )


# Ssr ftest is a metric to corelate 2 time series. We currently say that the idea scores
# are behind the stock prices and that they can predict them. (lag vs lead)
# TODO: Add lead
def compute_ssr_ftest(daily_df, ticker, idea_score_column, maxlag=10):
    ticker_ret_col = f"{ticker}_ret"
    daily_df[ticker_ret_col] = np.log(daily_df[ticker]).diff()

    gc = daily_df[[ticker_ret_col, idea_score_column]].dropna()

    res = grangercausalitytests(gc[[ticker_ret_col, idea_score_column]], maxlag=maxlag, verbose=False)

    lags = []
    for lag in range(1, maxlag+1):
        lags.append((lag, res[lag][0]['ssr_ftest'][1]))
    
    return lags

# TODO: Add other metrics for correlation

def get_significant_lags(lags, alpha=0.05):
    significant_lags = [lag for lag, p_value in lags if p_value < alpha]
    return significant_lags

# Because ssr_ftest is not that intuitive, we provide a summary function
def ssr_ftest_summary(lags, alpha=0.05):
    n_lags, p_values = zip(*lags)
    plt.stem(n_lags, p_values)
    plt.xlabel("Lags")
    plt.ylabel("p-values")
    plt.title("Lags vs p-values")
    plt.show()

    slags = get_significant_lags(lags, alpha=alpha)

    if len(slags) == 0:
        return "They are not corelated (no significant lags found)."

    best_lag = min(slags, key=lambda lag: next(p for l, p in lags if l == lag))
    
    return f"""They are corelated at lags: {slags}.
    That means, the idea scores have a statistically significant
    predictive power for the stock returns at these lags (p < {alpha}).
    Best lag is {best_lag} with p-value {next(p for l, p in lags if l == best_lag):.5f}(the smaller the better).
    """

# TODO: Make this function more general, to accept multiple metrics
def corelate_idea_with_stock_ssr_ftest(
    df,
    vec_column,
    idea,
    ticker,
    agg="max",
):
    score_df_by_ideas(
        df,
        vec_column=vec_column,
        idea_vecs=idea,
        score_columns=["idea_score_0"],
        device='cuda',
    )
    scores_idea_0 = get_daily_scores(df, "idea_score_0", agg=agg)
    scores_idea_0

    daily_prices = fetch_daily_prices(ticker)
    daily_prices["date"] = pd.to_datetime(daily_prices["date"]).dt.date

    daily_df = pd.merge(
        daily_prices,
        scores_idea_0,
        on="date",
        how="left",
    )
    lags = compute_ssr_ftest(daily_df, ticker, "idea_score_0", maxlag=10)
    #slags = get_significant_lags(lags, alpha=0.05)
    return lags


# ---------- Idea-aware scoring helpers ----------

def score_df_by_ideas_segments(
    df,
    segment_arrays,
    idea_vecs,
    score_columns=None,
    agg="topk_mean",
    top_k=5,
    tau=0.07,
    threshold=0.2,
):
    """
    Attach idea scores to df using per-segment embeddings and idea-aware pooling.

    df must contain a `segment_idx` column that indexes into segment_arrays.
    """
    if score_columns is None:
        if len(idea_vecs) == 1:
            score_columns = ["idea_score"]
        else:
            score_columns = [f"idea_score_{i}" for i in range(len(idea_vecs))]

    segment_arrays = list(segment_arrays)
    segs = [segment_arrays[idx] for idx in df["segment_idx"]]
    scores = score_segments_by_ideas(
        segs,
        idea_vecs,
        agg=agg,
        top_k=top_k,
        tau=tau,
        threshold=threshold,
    )
    for i, col in enumerate(score_columns):
        df[col] = scores[:, i]
    return df


# ---------- Correlation metrics ----------

def pearson_and_spearman(
    daily_df,
    target_column,
    idea_score_column,
    lead_lags=None,
):
    """
    Compute Pearson/Spearman correlations at multiple lead/lag shifts.

    Positive lag means idea_score leads target by `lag` days.
    """
    lead_lags = lead_lags or [0, 1, 2, 3, 5]
    results = []
    for lag in lead_lags:
        shifted_scores = daily_df[idea_score_column].shift(lag)
        aligned = pd.concat(
            [daily_df[target_column], shifted_scores], axis=1
        ).dropna()
        if len(aligned) < 3:
            results.append(
                {
                    "lag": lag,
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "pearson_p": np.nan,
                    "spearman_p": np.nan,
                }
            )
            continue
        # Guard against constant series which make correlation tests undefined.
        if aligned[target_column].nunique() < 2 or aligned[idea_score_column].nunique() < 2:
            pearson, spearman = np.nan, np.nan
            pearson_p, spearman_p = np.nan, np.nan
        else:
            pearson, pearson_p = pearsonr(
                aligned[target_column], aligned[idea_score_column]
            )
            spearman, spearman_p = spearmanr(
                aligned[target_column], aligned[idea_score_column]
            )
        results.append(
            {
                "lag": lag,
                "pearson": pearson,
                "pearson_p": pearson_p,
                "spearman": spearman,
                "spearman_p": spearman_p,
            }
        )
    return results


def general_correlations(
    daily_df,
    ticker,
    idea_score_column,
    maxlag=10,
    lead_lags=None,
    vol_window=5,
    include_volume=True,
):
    """
    Compute a suite of correlation / causal metrics between idea intensity and
    returns plus attention proxies (absolute returns, rolling vol, volume).
    """
    lead_lags = lead_lags or [0, 1, 2, 3, 5]
    ssr = compute_ssr_ftest(daily_df.copy(), ticker, idea_score_column, maxlag=maxlag)

    ticker_ret_col = f"{ticker}_ret"
    ticker_abs_ret_col = f"{ticker}_abs_ret"
    ticker_vol_col = f"{ticker}_vol_{vol_window}d"
    volume_col = f"{ticker}_volume"

    corr = pearson_and_spearman(
        daily_df.copy(),
        target_column=ticker_ret_col,
        idea_score_column=idea_score_column,
        lead_lags=lead_lags,
    )
    attention = {
        "abs_returns": pearson_and_spearman(
            daily_df.copy(),
            target_column=ticker_abs_ret_col,
            idea_score_column=idea_score_column,
            lead_lags=lead_lags,
        ),
        "rolling_volatility": pearson_and_spearman(
            daily_df.copy(),
            target_column=ticker_vol_col,
            idea_score_column=idea_score_column,
            lead_lags=lead_lags,
        ),
    }
    if include_volume and volume_col in daily_df.columns:
        attention["volume"] = pearson_and_spearman(
            daily_df.copy(),
            target_column=volume_col,
            idea_score_column=idea_score_column,
            lead_lags=lead_lags,
        )

    return {
        "ssr_ftest": ssr,
        "correlations": corr,
        "attention": attention,
    }


def correlate_idea_with_stock(
    df,
    segment_arrays,
    idea_vec,
    ticker,
    idea_score_column="idea_score",
    daily_agg="mean",
    segment_agg="topk_mean",
    top_k=5,
    tau=0.07,
    threshold=0.2,
    maxlag=10,
    lead_lags=None,
    vol_window=5,
    include_volume=True,
):
    """
    End-to-end helper:
    - score episodes with idea-aware pooling
    - aggregate to daily intensity
    - compute multiple correlation diagnostics vs returns, absolute returns,
      rolling volatility, and optional volume (market attention proxies)
    """
    df = score_df_by_ideas_segments(
        df.copy(),
        segment_arrays,
        idea_vecs=np.atleast_2d(idea_vec),
        score_columns=[idea_score_column],
        agg=segment_agg,
        top_k=top_k,
        tau=tau,
        threshold=threshold,
    )
    daily_scores = get_daily_scores(df, idea_score_column, agg=daily_agg)
    daily_prices = fetch_daily_prices(ticker)
    daily_prices["date"] = pd.to_datetime(daily_prices["date"]).dt.date
    if include_volume:
        daily_volume = fetch_daily_volume(ticker)
        daily_volume["date"] = pd.to_datetime(daily_volume["date"]).dt.date
        daily_volume = daily_volume.rename(columns={ticker: f"{ticker}_volume"})
        daily_prices = pd.merge(
            daily_prices,
            daily_volume,
            on="date",
            how="left",
        )

    daily_df = pd.merge(
        daily_prices,
        daily_scores,
        on="date",
        how="left",
    )
    daily_df[f"{ticker}_ret"] = np.log(daily_df[ticker]).diff()
    daily_df[f"{ticker}_abs_ret"] = daily_df[f"{ticker}_ret"].abs()
    daily_df[f"{ticker}_vol_{vol_window}d"] = (
        daily_df[f"{ticker}_abs_ret"].rolling(vol_window, min_periods=1).mean()
    )

    metrics = general_correlations(
        daily_df.copy(),
        ticker=ticker,
        idea_score_column=idea_score_column,
        maxlag=maxlag,
        lead_lags=lead_lags,
        vol_window=vol_window,
        include_volume=include_volume,
    )
    metrics["daily_df"] = daily_df
    return metrics


# ---------- Event-study / lead-lag diagnostics ----------

def _bootstrap_ci(values, stat_fn=np.mean, iters=5000, alpha=0.05, random_state=None):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    samples = rng.choice(values, size=(iters, len(values)), replace=True)
    stats = stat_fn(samples, axis=1)
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def event_study_intensity(
    daily_df,
    ticker,
    idea_score_column,
    events,
    window_pre=2,
    window_post=2,
    alpha=0.05,
    bootstrap_iters=2000,
    random_state=None,
):
    """
    Measure how idea intensity behaves around known market events.

    Returns a dict with per-offset means and bootstrap CIs, plus lead/lag
    comparison of returns vs intensity.
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date")
    df[f"{ticker}_ret"] = np.log(df[ticker]).diff()

    offsets = range(-window_pre, window_post + 1)
    intensity_by_offset = {k: [] for k in offsets}
    returns_by_offset = {k: [] for k in offsets}

    for evt in events:
        d = pd.to_datetime(evt).date()
        for offset in offsets:
            dt = (pd.to_datetime(d) + pd.Timedelta(days=offset)).date()
            if dt in df.index:
                intensity_by_offset[offset].append(df.loc[dt, idea_score_column])
                returns_by_offset[offset].append(df.loc[dt, f"{ticker}_ret"])

    summary = []
    for offset in offsets:
        vals = np.array(intensity_by_offset[offset], dtype=float)
        rets = np.array(returns_by_offset[offset], dtype=float)
        ci_int = _bootstrap_ci(vals[~np.isnan(vals)], iters=bootstrap_iters, alpha=alpha, random_state=random_state)
        ci_ret = _bootstrap_ci(rets[~np.isnan(rets)], iters=bootstrap_iters, alpha=alpha, random_state=random_state)
        summary.append(
            {
                "offset": offset,
                "n_events": len(vals),
                "intensity_mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                "intensity_ci": ci_int,
                "return_mean": float(np.nanmean(rets)) if len(rets) else np.nan,
                "return_ci": ci_ret,
            }
        )

    # Lead/lag: compare post-event vs pre-event intensity
    pre_vals = np.concatenate([intensity_by_offset[o] for o in offsets if o < 0]) if window_pre > 0 else np.array([])
    post_vals = np.concatenate([intensity_by_offset[o] for o in offsets if o > 0]) if window_post > 0 else np.array([])
    diff = np.nanmean(post_vals) - np.nanmean(pre_vals) if len(pre_vals) and len(post_vals) else np.nan
    diff_ci = _bootstrap_ci(
        post_vals - np.nanmean(pre_vals) if len(pre_vals) else post_vals,
        iters=bootstrap_iters,
        alpha=alpha,
        random_state=random_state,
    )

    return {
        "per_offset": summary,
        "intensity_change_post_minus_pre": diff,
        "intensity_change_ci": diff_ci,
    }
