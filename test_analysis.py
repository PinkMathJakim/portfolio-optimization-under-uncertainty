import numpy as np, pandas as pd
from src.analysis import distribution_summary, volatility_regimes

def test_distribution_summary_and_regime_labels():
    rng=np.random.default_rng(42)
    r=pd.Series(rng.normal(0,.01,500),index=pd.bdate_range('2020-01-01',periods=500))
    summary=distribution_summary(r)
    assert summary['n']==500 and 0 <= summary['jarque_bera_p'] <= 1
    regimes=volatility_regimes(r,window=30)
    assert set(regimes['regime'].unique()).issubset({'unclassified','normal','high_volatility','low_volatility'})
