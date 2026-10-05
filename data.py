"""Market data download, cleaning, and provenance manifest."""
from __future__ import annotations
import json, importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

DEFAULT_TICKERS={
"ASML.AS":"Technology","SAP.DE":"Technology","SIE.DE":"Industrials","ALV.DE":"Financials",
"DTE.DE":"Telecommunications","BAS.DE":"Materials","ADS.DE":"Consumer discretionary","AIR.PA":"Industrials",
"MC.PA":"Consumer discretionary","OR.PA":"Consumer staples","SAN.PA":"Healthcare","BNP.PA":"Financials",
"SU.PA":"Industrials","AI.PA":"Industrials","ENEL.MI":"Utilities","ENI.MI":"Energy",
"ISP.MI":"Financials","UCG.MI":"Financials","BMW.DE":"Consumer discretionary","RWE.DE":"Utilities"
}


def download_adjusted_prices(start="2015-01-01", end="2026-10-06", tickers=None, out_dir="data/processed"):
    try: import yfinance as yf
    except ImportError as exc: raise RuntimeError("yfinance is required; install requirements.txt") from exc
    tickers=tickers or DEFAULT_TICKERS; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    raw=yf.download(list(tickers),start=start,end=end,auto_adjust=True,progress=False,group_by="column",threads=True)
    if raw.empty: raise RuntimeError("No price data returned. Check network, tickers, and provider availability.")
    if isinstance(raw.columns,pd.MultiIndex):
        field="Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.get_level_values(0)[0]
        prices=raw[field].copy()
    else: prices=raw[["Close"]].rename(columns={"Close":list(tickers)[0]})
    prices=prices.loc[:,[c for c in tickers if c in prices.columns]].sort_index()
    missing={c:int(prices[c].isna().sum()) for c in prices.columns}
    # Retain only assets with >=95% coverage on the requested period; record exclusions.
    coverage=prices.notna().mean(); keep=coverage[coverage>=.95].index.tolist(); excluded=coverage[coverage<.95].to_dict()
    prices=prices[keep].dropna(how="all")
    if prices.shape[1]<4: raise RuntimeError(f"Only {prices.shape[1]} assets passed coverage; refusing a weak universe.")
    prices.to_csv(out/"adjusted_prices.csv",index_label="Date")
    manifest={"retrieved_utc":datetime.now(timezone.utc).isoformat(),"provider":"Yahoo Finance via yfinance",
      "requested_start":start,"requested_end_exclusive":end,"price_field":"auto_adjust=True Close (vendor adjusted series)",
      "frequency":"daily trading observations","tickers_requested":list(tickers),"sectors":tickers,
      "tickers_retained":keep,"missing_observations_before_filter":missing,"coverage_fraction":coverage.to_dict(),
      "excluded_low_coverage":excluded,"retained_date_start":str(prices.index.min()),"retained_date_end":str(prices.index.max()),
      "n_rows":int(len(prices)),"n_assets":int(prices.shape[1]),"limitations":["Current-name universe has survivorship bias.","Universe is restricted to EUR-denominated listings to avoid unmodelled FX exposure; this does not eliminate issuer domicile or index survivorship concerns.","Vendor adjusted prices may be revised.","Provider may not expose delisted names or full corporate-action history."],
      "versions":{p:importlib.metadata.version(p) for p in ["yfinance","pandas"]}}
    (out/"data_manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
    return prices,manifest


def download_market_benchmark(start="2015-01-01", end="2026-10-06", ticker="EXSA.DE", out_dir="data/processed"):
    """Download the EUR-denominated iShares STOXX Europe 600 ETF used as CAPM proxy."""
    try: import yfinance as yf
    except ImportError as exc: raise RuntimeError("yfinance is required; install requirements.txt") from exc
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    raw=yf.download(ticker,start=start,end=end,auto_adjust=True,progress=False,group_by="column",threads=False)
    if raw.empty: raise RuntimeError(f"No benchmark data returned for {ticker}.")
    if isinstance(raw.columns,pd.MultiIndex):
        field="Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.get_level_values(0)[0]
        px=raw[field].squeeze()
    else:
        px=raw["Close"] if "Close" in raw else raw.iloc[:,0]
    px=pd.Series(px,name=ticker).dropna().sort_index()
    px.to_csv(out/"market_adjusted_prices.csv",index_label="Date")
    r=px.pct_change().dropna().rename("market_return")
    r.to_csv(out/"market_returns.csv",index_label="Date")
    manifest={"ticker":ticker,"provider":"Yahoo Finance via yfinance","price_field":"auto_adjust=True Close",
              "currency":"EUR","benchmark":"iShares STOXX Europe 600 UCITS ETF (DE), EXSA",
              "requested_start":start,"requested_end_exclusive":end,"realized_start":str(px.index.min()),
              "realized_end":str(px.index.max()),"n_rows":int(len(px)),
              "limitations":["ETF returns are a tradable proxy rather than the index itself and include fund-level implementation effects.",
                              "The benchmark is not a risk-free asset and should not be interpreted as one.",
                              "Vendor data may be revised."]}
    (out/"market_manifest.json").write_text(json.dumps(manifest,indent=2))
    return r,manifest
