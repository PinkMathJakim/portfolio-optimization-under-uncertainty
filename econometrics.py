"""CAPM regression and paired strategy-difference inference."""
from __future__ import annotations
import pandas as pd
import statsmodels.api as sm

def capm_regression(portfolio_returns: pd.Series, market_returns: pd.Series, risk_free: pd.Series | float=0.0):
    data=pd.concat([portfolio_returns.rename("p"),market_returns.rename("m")],axis=1).dropna()
    if isinstance(risk_free,pd.Series): rf=risk_free.reindex(data.index).fillna(0.0)
    else: rf=float(risk_free)
    y=data["p"]-rf; x=data["m"]-rf if not isinstance(rf,float) else data["m"]-rf
    return sm.OLS(y,sm.add_constant(x.rename("market_excess"))).fit(cov_type="HAC",cov_kwds={"maxlags":5})

def hac_mean_difference(a: pd.Series, b: pd.Series, maxlags: int=5) -> dict:
    d=pd.concat([a.rename("a"),b.rename("b")],axis=1).dropna(); diff=d.a-d.b
    design=pd.DataFrame({"constant":1.0},index=diff.index)
    fit=sm.OLS(diff,design).fit(cov_type="HAC",cov_kwds={"maxlags":maxlags})
    return {"mean_difference":float(fit.params.iloc[0]),"hac_se":float(fit.bse.iloc[0]),"t_stat":float(fit.tvalues.iloc[0]),
            "p_value":float(fit.pvalues.iloc[0]),"n_obs":int(len(diff))}
