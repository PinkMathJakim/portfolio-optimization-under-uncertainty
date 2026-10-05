"""Portfolio risk measures. VaR and ES are reported as positive loss magnitudes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

def portfolio_volatility(weights, covariance) -> float:
    w=np.asarray(weights,dtype=float); s=np.asarray(covariance,dtype=float)
    return float(np.sqrt(max(w @ s @ w, 0.0)))

def risk_contributions(weights, covariance) -> pd.Series:
    """Euler contributions to portfolio volatility; sum equals portfolio volatility."""
    w=np.asarray(weights,dtype=float); s=np.asarray(covariance,dtype=float)
    vol=portfolio_volatility(w,s)
    if vol <= 0: return pd.Series(np.zeros(len(w)))
    return pd.Series(w * (s @ w) / vol)

def max_drawdown(returns) -> float:
    x=pd.Series(returns).dropna()
    if x.empty: return float("nan")
    wealth=(1+x).cumprod().to_numpy()
    # Include initial capital as a peak so a first-period loss is counted.
    wealth_with_initial=np.concatenate(([1.0], wealth))
    peak=np.maximum.accumulate(wealth_with_initial)
    return float(np.min(wealth_with_initial/peak-1))

def historical_var(returns, confidence: float=0.95) -> float:
    x=pd.Series(returns).dropna()
    if not 0 < confidence < 1: raise ValueError("confidence must be in (0,1)")
    return float(max(0.0, -x.quantile(1-confidence)))

def historical_cvar(returns, confidence: float=0.95) -> float:
    x=pd.Series(returns).dropna()
    q=x.quantile(1-confidence)
    tail=x[x <= q]
    return float(max(0.0, -tail.mean()))

def parametric_var(returns, confidence: float=0.95) -> float:
    x=pd.Series(returns).dropna()
    return float(max(0.0, -(x.mean()+norm.ppf(1-confidence)*x.std(ddof=1))))

def sortino_ratio(returns, risk_free_daily: float=0.0, periods: int=252) -> float:
    x=pd.Series(returns).dropna(); excess=x-risk_free_daily
    downside=np.sqrt(np.mean(np.minimum(excess.to_numpy(),0.0)**2))
    return float(np.sqrt(periods)*excess.mean()/downside) if downside>0 else float("nan")

def performance_metrics(returns, risk_free_daily: float=0.0, periods: int=252) -> dict:
    x=pd.Series(returns).dropna()
    if x.empty: return {}
    ann=(1+x).prod()**(periods/len(x))-1 if (1+x).gt(0).all() else float("nan")
    vol=x.std(ddof=1)*np.sqrt(periods)
    sharpe=np.sqrt(periods)*(x.mean()-risk_free_daily)/x.std(ddof=1) if x.std(ddof=1)>0 else float("nan")
    mdd=max_drawdown(x)
    return {"annual_return":float(ann),"annual_volatility":float(vol),"sharpe":float(sharpe),
            "sortino":sortino_ratio(x,risk_free_daily,periods),"max_drawdown":mdd,
            "calmar":float(ann/abs(mdd)) if mdd<0 else float("nan"),
            "historical_var_95":historical_var(x),"historical_cvar_95":historical_cvar(x),
            "parametric_var_95":parametric_var(x),"mean_period_return":float(x.mean()),"n_obs":int(len(x))}
