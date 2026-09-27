"""
data.py
-------
Ticker universe construction and price data helpers.

Responsible for:
  * Scraping/using yfinance to build ticker lists for major indices
    (S&P 500, Hang Seng, Shanghai Composite (manual), FTSE 100, Nikkei 225)
  * Normalizing tickers into yfinance-compatible symbols
  * Downloading and reshaping OHLC / close-price data (flat or MultiIndex
    columns, single or multi-ticker) into consistent pandas objects
  * Building simple in-memory price caches used by the backtest engine
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------------------------------
# HTTP / HTML helpers
# --------------------------------------------------------------------------

def _get_html(url: str, timeout: int = 20) -> str:
    """Fetch a URL's HTML with a browser-like User-Agent (Wikipedia blocks bare requests)."""
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def _lower_str(x) -> str:
    """Robust lowercase/strip that works even if a header is an int/None."""
    return str(x).strip().lower()


# --------------------------------------------------------------------------
# Ticker universes
# --------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """S&P 500 constituents, normalized for yfinance (e.g. BRK.B -> BRK-B)."""
    try:
        tickers = yf.tickers_sp500()
        if tickers:
            return [t.replace(".", "-").strip() for t in tickers]
    except Exception:
        pass  # fall through to HTML fallback

    html = _get_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tbl = pd.read_html(html, match="Symbol")[0]
    return (
        tbl["Symbol"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )


def get_hangseng_tickers() -> list[str]:
    """Hang Seng Index constituents, scraped from Wikipedia."""
    html = _get_html("https://en.wikipedia.org/wiki/Hang_Seng_Index")
    tables = pd.read_html(html)
    headers = {"ticker", "stock code", "stock code(s)", "code"}

    for table in tables:
        cols_lower = {_lower_str(c): c for c in table.columns}
        hit = next((cols_lower[h] for h in headers if h in cols_lower), None)
        if hit is None:
            continue

        codes = table[hit].astype(str).str.extract(r"(\d{4,5})")[0].dropna().astype(str)

        def _fmt(code: str) -> str:
            c = code.strip()
            return (c.zfill(4) if len(c) <= 4 else c) + ".HK"

        out = [_fmt(c) for c in codes.unique()]
        if out:
            return out
    return []


def get_ftse_tickers() -> list[str]:
    """FTSE 100 constituents, scraped from Wikipedia."""
    html = _get_html("https://en.wikipedia.org/wiki/FTSE_100_Index")
    tables = pd.read_html(html)

    for table in tables:
        for cand in ("EPIC", "Ticker", "Ticker symbol"):
            if cand in table.columns:
                raw = table[cand].astype(str).str.strip()
                out = []
                for s in raw:
                    s = s.replace(".", "-")
                    if not s.endswith(".L"):
                        s = s + ".L"
                    out.append(s)
                return [t for t in out if t and t != ".L"]
    return []


def get_nikkei225_tickers() -> list[str]:
    """Nikkei 225 constituents, scraped from Wikipedia (TYO: codes in list items)."""
    from bs4 import BeautifulSoup

    html = _get_html("https://en.wikipedia.org/wiki/Nikkei_225")
    soup = BeautifulSoup(html, "html.parser")

    tickers = set()
    for li in soup.select("li"):
        m = re.search(r"TYO:\s?(\d{4})", li.get_text(" ", strip=True))
        if m:
            tickers.add(m.group(1).zfill(4) + ".T")
    return sorted(tickers)


def get_shanghai_tickers() -> list[str]:
    """
    Shanghai Composite constituents.

    Wikipedia does not maintain a clean, scrape-friendly constituent table for
    this index, so a curated static list of liquid SSE-listed names is used
    instead. Update this list periodically if you need it to track the live
    index membership exactly.
    """
    return [
        "601398.SS", "601288.SS", "601939.SS", "600519.SS", "600941.SS",
        "601988.SS", "601857.SS", "600036.SS", "601628.SS", "601318.SS",
        "600938.SS", "600900.SS", "601088.SS", "601658.SS", "601328.SS",
        "601728.SS", "600028.SS", "601166.SS", "601138.SS", "601899.SS",
        "601998.SS", "600000.SS", "688981.SS", "600030.SS", "601319.SS",
        "600276.SS", "601601.SS", "688041.SS", "601211.SS", "601816.SS",
        "601818.SS", "601668.SS", "600919.SS", "601919.SS", "601127.SS",
        "688256.SS", "600690.SS", "600016.SS", "603288.SS", "600809.SS",
        "688235.SS", "603259.SS", "600104.SS", "601985.SS", "601766.SS",
        "601225.SS", "600406.SS", "603993.SS", "600025.SS", "600887.SS",
        "601336.SS", "600050.SS", "601066.SS", "600760.SS", "601633.SS",
        "601229.SS", "605499.SS", "600019.SS", "600031.SS", "601688.SS",
        "601881.SS", "601169.SS", "603501.SS", "600660.SS", "600150.SS",
        "600999.SS", "601009.SS", "600547.SS", "601898.SS", "601995.SS",
        "600015.SS", "600018.SS", "601006.SS", "601390.SS", "601800.SS",
        "600926.SS", "601012.SS", "601888.SS", "600905.SS", "688111.SS",
        "688506.SS", "600886.SS", "601111.SS", "600436.SS", "600989.SS",
        "600585.SS", "688012.SS", "601600.SS", "600415.SS", "601989.SS",
        "688271.SS", "601186.SS", "600011.SS", "600188.SS", "603019.SS",
        "601727.SS", "600346.SS", "600893.SS", "600048.SS", "601916.SS",
        "601100.SS", "601825.SS", "688008.SS", "600029.SS", "600418.SS",
        "600111.SS", "600795.SS", "688036.SS", "603296.SS", "601838.SS",
        "603195.SS", "601669.SS", "601698.SS", "600010.SS", "600039.SS",
        "601868.SS", "603986.SS", "601077.SS", "600438.SS", "600600.SS",
        "600115.SS", "600674.SS", "600009.SS", "600183.SS", "600160.SS",
        "600958.SS", "601788.SS", "601689.SS", "603392.SS", "600023.SS",
        "601018.SS", "600377.SS", "601360.SS", "600489.SS", "600362.SS",
        "601901.SS", "603799.SS", "688347.SS", "603893.SS", "600089.SS",
        "601238.SS", "600570.SS", "600196.SS", "688396.SS", "600803.SS",
        "601607.SS", "600027.SS", "600066.SS", "600584.SS", "600845.SS",
        "600221.SS", "900945.SS", "601618.SS", "600703.SS", "600741.SS",
        "900926.SS", "600372.SS", "600875.SS", "603260.SS", "601456.SS",
        "688223.SS", "601991.SS", "601298.SS", "688047.SS", "601021.SS",
        "601377.SS", "601059.SS", "601136.SS", "600236.SS", "600482.SS",
        "688009.SS", "688126.SS", "600350.SS", "688303.SS", "688187.SS",
        "600176.SS", "688082.SS", "605117.SS", "601877.SS", "601872.SS",
        "601878.SS", "601162.SS", "600085.SS", "603369.SS", "600061.SS",
        "688521.SS", "600522.SS", "601117.SS", "600487.SS", "603806.SS",
        "600256.SS", "601777.SS", "600562.SS", "688188.SS", "601108.SS",
        "601198.SS", "600901.SS", "600398.SS", "603129.SS", "688331.SS",
        "688375.SS", "688777.SS", "601958.SS", "688520.SS", "600925.SS",
        "603606.SS", "601865.SS", "688180.SS", "600079.SS", "601799.SS",
        "601598.SS", "601963.SS", "688336.SS", "688297.SS", "603833.SS",
        "600177.SS", "600879.SS", "600885.SS", "600378.SS", "600118.SS",
        "600352.SS", "688065.SS", "688472.SS", "688385.SS", "688122.SS",
        "601665.SS", "688538.SS", "600521.SS", "603087.SS", "688469.SS",
        "600109.SS", "600549.SS", "603568.SS", "600580.SS", "601990.SS",
        "600298.SS", "600871.SS", "688192.SS", "601696.SS", "600685.SS",
        "600737.SS", "900934.SS", "688425.SS", "600511.SS", "601966.SS",
        "688037.SS", "603658.SS", "603236.SS", "603179.SS", "900905.SS",
        "600170.SS", "600655.SS", "900937.SS", "600131.SS",
        "600612.SS", "603300.SS", "688017.SS", "688615.SS", "601156.SS",
        "601001.SS", "688018.SS", "600312.SS", "600726.SS", "600380.SS",
        "600007.SS", "600977.SS", "601606.SS", "601118.SS",
        "601608.SS", "603565.SS", "600057.SS", "600779.SS", "600884.SS",
        "603589.SS", "688709.SS", "603816.SS", "600120.SS", "600072.SS",
        "688726.SS", "601019.SS", "603256.SS", "600323.SS", "600820.SS",
        "601016.SS", "603688.SS", "600062.SS", "603650.SS", "688248.SS",
        "601187.SS", "603290.SS", "603233.SS", "601106.SS", "601333.SS",
        "688200.SS", "600499.SS", "600320.SS", "900947.SS", "688387.SS",
        "601965.SS",
    ]


DEFAULT_UNIVERSE_BUILDERS = {
    "SP500": get_sp500_tickers,
    "HangSeng": get_hangseng_tickers,
    "Shanghai": get_shanghai_tickers,
    "FTSE100": get_ftse_tickers,
    "Nikkei225": get_nikkei225_tickers,
}

GLOBAL_MACRO_TICKERS = [
    "QQQ", "IWM", "VXX", "UVXY",
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG",
    "XLE", "OIH", "ARKK", "EEM", "EFA",
    "TQQQ", "SQQQ", "FXI", "KWEB", "EWZ", "INDA",
    "BTC-USD", "ETH-USD", "XAUUSD=X", "GC=F", "SI=F", "CL=F", "NG=F", "BZ=F",
]

INDEX_BASKET = ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "FXI", "EWU", "EWG", "EWQ", "EWZ"]


def build_ticker_universe(include_global: bool = True) -> dict[str, list[str]]:
    """Build the full {region: [tickers]} universe used by the strategy."""
    universe = {name: builder() for name, builder in DEFAULT_UNIVERSE_BUILDERS.items()}
    if include_global:
        universe["Global"] = list(GLOBAL_MACRO_TICKERS)
    return universe


# --------------------------------------------------------------------------
# Price series reshaping helpers
# --------------------------------------------------------------------------

def _as_series(x) -> pd.Series:
    """Force a 1-D Series of prices from a yfinance result."""
    if isinstance(x, pd.Series):
        return x.dropna()
    if hasattr(x, "shape"):  # DataFrame-like
        return x.iloc[:, 0].dropna()
    raise TypeError(f"Unexpected price type: {type(x)}")


def get_close_series(df: pd.DataFrame) -> pd.Series:
    """Return a 1-D Series of Close/Adj Close from a yfinance download result."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        for field in ("Adj Close", "Close"):
            for lvl in range(df.columns.nlevels):
                try:
                    s = df.xs(field, axis=1, level=lvl)
                    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
                except KeyError:
                    continue
        raise KeyError("No Close/Adj Close in MultiIndex columns")

    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if "Close" in df.columns:
        return df["Close"]
    raise KeyError("No Close/Adj Close in columns")


def get_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['Open','High','Low','Close'].

    Works for yfinance outputs with either flat columns or MultiIndex
    columns. If multiple tickers are present in a MultiIndex, the first
    ticker's OHLC is used (this function is meant to be called per-ticker).
    """
    wanted = ["Open", "High", "Low", "Close"]

    if df is None or df.empty:
        return pd.DataFrame(columns=wanted)

    if not isinstance(df.columns, pd.MultiIndex):
        out = {f: (df[f] if f in df.columns else pd.Series(index=df.index, dtype=float)) for f in wanted}
        return pd.DataFrame(out)[wanted]

    out = {}
    for f in wanted:
        try:
            sub = df.xs(f, axis=1, level=-1)
            if isinstance(sub, pd.DataFrame):
                out[f] = sub.iloc[:, 0] if sub.shape[1] else pd.Series(index=df.index, dtype=float)
            else:
                out[f] = pd.Series(sub)
        except KeyError:
            out[f] = pd.Series(index=df.index, dtype=float)

    res = pd.DataFrame(out)[wanted]
    for f in wanted:
        res[f] = pd.to_numeric(res[f], errors="coerce")
    return res


def calc_volatility(ticker: str, period: str = "20y") -> float | None:
    """Annualized volatility of daily returns over `period`, or None on failure."""
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        col = "Adj Close" if "Adj Close" in data.columns else "Close" if "Close" in data.columns else None
        if col is None:
            return None
        returns = data[col].pct_change().dropna()
        vol = returns.std() * (252 ** 0.5)
        if isinstance(vol, pd.Series):
            vol = vol.item()
        return float(vol) if pd.notna(vol) else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Price caches used by the backtest engine
# --------------------------------------------------------------------------

def build_close_cache(tickers, first_day, last_day, bdays) -> dict[str, pd.Series]:
    """Adjusted close-only cache for MTM/plots. Returns {ticker: Series} reindexed to bdays."""
    px = {}
    for t in tickers:
        p = yf.download(t, start=first_day, end=last_day + pd.Timedelta(days=1),
                         interval="1d", auto_adjust=True, progress=False)
        s = get_close_series(p)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        px[t] = s.reindex(bdays).ffill()
    return px


def build_ohlc_cache(tickers, first_day, last_day, bdays) -> dict[str, pd.DataFrame]:
    """Non-adjusted OHLC cache (needed for true intraday stop-loss checks)."""
    cache = {}
    for t in tickers:
        p = yf.download(t, start=first_day, end=last_day + pd.Timedelta(days=1),
                         interval="1d", auto_adjust=False, progress=False)
        o = get_ohlc(p)
        o.index = pd.to_datetime(o.index).tz_localize(None).normalize()
        cache[t] = o.reindex(bdays).ffill()
    return cache


def px_close(px: dict, t: str, d) -> float | None:
    s = px.get(t)
    if s is None:
        return None
    v = s.get(d, np.nan)
    return float(v) if pd.notna(v) else None
