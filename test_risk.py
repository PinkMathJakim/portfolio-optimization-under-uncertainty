import numpy as np, pandas as pd
from src.risk import portfolio_volatility, risk_contributions, max_drawdown, historical_var, historical_cvar
def test_vol_and_risk_contributions():
 cov=np.array([[.04,.01],[.01,.09]]); w=np.array([.5,.5]); vol=portfolio_volatility(w,cov)
 rc=risk_contributions(w,cov)
 assert np.isclose(rc.sum(),vol)
def test_drawdown_var_cvar():
 r=pd.Series([.1,-.2,.05,-.1])
 assert max_drawdown(r)<0
 assert historical_var(r)>=0 and historical_cvar(r)>=historical_var(r)
def test_first_day_loss_counts_as_drawdown():
    assert np.isclose(max_drawdown(pd.Series([-0.10, 0.02])), -0.10)
