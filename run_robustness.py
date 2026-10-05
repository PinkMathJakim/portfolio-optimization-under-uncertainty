"""Run a pre-specified sensitivity grid; may take several minutes."""
import itertools, json
from pathlib import Path
import pandas as pd
from src.backtest import rolling_backtest
from src.risk import performance_metrics
r=pd.read_csv('data/processed/simple_returns.csv',index_col=0,parse_dates=True).dropna()
rows=[]
for strategy,window,rebalance,cap,cov,mu,cost in itertools.product(['equal_weight','min_variance','max_sharpe','risk_parity'],[504,756,1008],[21,63,252],[.10,.20],['sample','ledoit_wolf'],['historical_mean','shrinkage_mean'],[0,10,25]):
 try:
  out,_=rolling_backtest(r,strategy,window,rebalance,cap,cov,mu,cost)
  rows.append({'strategy':strategy,'window':window,'rebalance':rebalance,'max_weight':cap,'covariance':cov,'expected_return':mu,'cost_bps':cost,**performance_metrics(out)})
 except (ValueError,RuntimeError) as exc:
  rows.append({'strategy':strategy,'window':window,'rebalance':rebalance,'max_weight':cap,'covariance':cov,'expected_return':mu,'cost_bps':cost,'error':str(exc)})
Path('report').mkdir(exist_ok=True); pd.DataFrame(rows).to_csv('report/robustness_grid.csv',index=False)
print(f'Wrote {len(rows)} configurations, including failures, to report/robustness_grid.csv')
