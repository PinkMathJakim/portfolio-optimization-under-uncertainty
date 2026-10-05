import numpy as np, pandas as pd
from src.backtest import rolling_backtest
def test_backtest_starts_after_training_and_has_no_future_dates():
 idx=pd.bdate_range('2018-01-01',periods=300); r=pd.DataFrame(np.random.default_rng(1).normal(.0002,.01,(300,4)),index=idx,columns=list('ABCD'))
 out,weights=rolling_backtest(r,'equal_weight',train_window=100,rebalance_every=20,max_weight=.5,transaction_cost_bps=5)
 assert out.index.min()>idx[99] and len(out)>0 and len(weights)>0
 assert all(abs(sum(x.values())-1)<1e-6 for x in weights['weights'])

def test_buy_and_hold_weights_drift_between_rebalances():
    idx=pd.bdate_range('2020-01-01',periods=5)
    r=pd.DataFrame([[0,0],[0,0],[0.10,0.0],[0.10,0.0],[0.0,0.0]],index=idx,columns=['A','B'])
    out,_=rolling_backtest(r,'equal_weight',train_window=2,rebalance_every=10,max_weight=.6,transaction_cost_bps=0)
    # After the first +10% A return, A's weight drifts from 50% to 0.55/1.05.
    # The second +10% A return therefore earns 5.2381%, not a fixed 5%.
    expected=(1.05)*(1+0.10*(0.55/1.05))
    assert np.isclose((1+out.iloc[0])*(1+out.iloc[1]),expected)
