"""Distribution diagnostics and transparent volatility-regime labeling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, normaltest, skew, kurtosis

def distribution_summary(returns: pd.Series) -> dict:
    """Descriptive statistics and normality tests; p-values are not proof of a model."""
    x=pd.Series(returns).dropna().astype(float)
    if len(x)<8: raise ValueError("At least eight observations are required for normality diagnostics")
    jb=jarque_bera(x); nt=normaltest(x)
    return {"n":int(len(x)),"mean":float(x.mean()),"variance":float(x.var(ddof=1)),
            "skewness":float(skew(x,bias=False)),"excess_kurtosis":float(kurtosis(x,fisher=True,bias=False)),
            "q01":float(x.quantile(.01)),"q05":float(x.quantile(.05)),"median":float(x.median()),
            "q95":float(x.quantile(.95)),"q99":float(x.quantile(.99)),
            "jarque_bera_stat":float(jb.statistic),"jarque_bera_p":float(jb.pvalue),
            "dagostino_pearson_stat":float(nt.statistic),"dagostino_pearson_p":float(nt.pvalue)}

def volatility_regimes(returns: pd.Series, window: int=63, high_quantile: float=.80, low_quantile: float=.20) -> pd.DataFrame:
    """Classify high/low volatility using expanding, lagged quantile thresholds.

    At each date, thresholds are estimated from prior rolling-volatility observations
    only, preventing future information from defining the regime label.
    """
    x=pd.Series(returns).dropna().astype(float)
    rv=x.rolling(window,min_periods=window).std()*np.sqrt(252)
    high=[]; low=[]
    for i,v in enumerate(rv):
        hist=rv.iloc[:i].dropna()
        high.append(hist.quantile(high_quantile) if len(hist)>=window else np.nan)
        low.append(hist.quantile(low_quantile) if len(hist)>=window else np.nan)
    out=pd.DataFrame({"return":x,"rolling_volatility":rv,"high_threshold":high,"low_threshold":low})
    out["regime"]="normal"
    out.loc[out.rolling_volatility>out.high_threshold,"regime"]="high_volatility"
    out.loc[out.rolling_volatility<out.low_threshold,"regime"]="low_volatility"
    out.loc[out.rolling_volatility.isna() | out.high_threshold.isna() | out.low_threshold.isna(),"regime"]="unclassified"
    return out
