"""Data fetching for the SPY/GLD ratio strategy."""
import yfinance as yf
import pandas as pd

START_DATE = "2005-01-01"
SPY_TICKER = "SPY"
GLD_TICKER = "GLD"
FX_TICKER = "GBPUSD=X"


def load_price_data() -> tuple[pd.Series, pd.DataFrame]:
    """Return the SPY/GLD ratio series and a joined price/FX history DataFrame."""
    spy_close = yf.download(SPY_TICKER, start=START_DATE)["Close"][SPY_TICKER]
    gld_close = yf.download(GLD_TICKER, start=START_DATE)["Close"][GLD_TICKER]

    spy_to_gld_ratio = (spy_close / gld_close).dropna()

    usd_to_gbp = yf.download(FX_TICKER, start=START_DATE)["Close"]
    usd_to_gbp = usd_to_gbp.reindex(spy_to_gld_ratio.index, method="ffill")
    usd_to_gbp = usd_to_gbp.rename(columns={FX_TICKER: "GBPUSD"})

    price_history = pd.concat([spy_close, gld_close, usd_to_gbp], axis=1)
    print(price_history.head())
    return spy_to_gld_ratio, price_history