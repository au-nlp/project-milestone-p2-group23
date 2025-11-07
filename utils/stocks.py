import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests
from .embed import score_df_by_ideas
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


def compute_ssr_ftest(daily_df, ticker, idea_score_column, maxlag=10):
    ticker_ret_col = f"{ticker}_ret"
    daily_df[ticker_ret_col] = np.log(daily_df[ticker]).diff()

    gc = daily_df[[ticker_ret_col, idea_score_column]].dropna()

    res = grangercausalitytests(gc[[ticker_ret_col, idea_score_column]], maxlag=maxlag, verbose=False)

    lags = []
    for lag in range(1, maxlag+1):
        lags.append((lag, res[lag][0]['ssr_ftest'][1]))
    
    return lags

def get_significant_lags(lags, alpha=0.05):
    significant_lags = [lag for lag, p_value in lags if p_value < alpha]
    return significant_lags

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
