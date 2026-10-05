"""Download data, run walk-forward strategies, produce figures and summary."""
import argparse, json
from pathlib import Path
import pandas as pd
from src.data import download_adjusted_prices, download_market_benchmark
from src.returns import simple_returns, log_returns
from src.backtest import rolling_backtest
from src.risk import performance_metrics
from src.visualization import save_standard_figures, save_diagnostic_figures
from src.analysis import distribution_summary, volatility_regimes
from src.econometrics import hac_mean_difference

def main():
    p=argparse.ArgumentParser(); p.add_argument('--start',default='2015-01-01'); p.add_argument('--end',default='2026-10-06'); p.add_argument('--train-window',type=int,default=756); p.add_argument('--rebalance',type=int,default=63); p.add_argument('--max-weight',type=float,default=.10); p.add_argument('--cost-bps',type=float,default=10); args=p.parse_args()
    prices,manifest=download_adjusted_prices(args.start,args.end)
    market,market_manifest=download_market_benchmark(args.start,args.end)
    rets=simple_returns(prices).dropna(how='all'); log_returns(prices).to_csv('data/processed/log_returns.csv',index_label='Date'); rets.to_csv('data/processed/simple_returns.csv',index_label='Date')
    all_returns={}; all_weights={}
    for name in ['equal_weight','min_variance','max_sharpe','risk_parity']:
        r,w=rolling_backtest(rets,name,args.train_window,args.rebalance,args.max_weight,'ledoit_wolf' if name=='risk_parity' else 'sample','historical_mean',args.cost_bps)
        all_returns[name]=r; all_weights[name]=w
        w.to_json(f'data/processed/weights_{name}.json',orient='records',date_format='iso')
    strat=pd.concat(all_returns,axis=1).dropna(how='all'); strat.to_csv('data/processed/oos_strategy_returns.csv',index_label='Date')
    metrics={k:performance_metrics(v) for k,v in all_returns.items()}
    pd.DataFrame(metrics).T.to_csv('report/results_summary.csv')
    save_standard_figures(prices,rets,strat)
    save_diagnostic_figures(rets,strat,all_weights)
    dist=pd.DataFrame({c:distribution_summary(rets[c]) for c in rets.columns}).T
    dist.to_csv('report/asset_distribution_diagnostics.csv')
    ew_regime=volatility_regimes(rets.mean(axis=1))
    ew_regime.to_csv('report/equal_weight_volatility_regimes.csv',index_label='Date')
    comparisons={k:hac_mean_difference(all_returns[k],all_returns['equal_weight']) for k in all_returns if k!='equal_weight'}
    from src.econometrics import capm_regression
    capm={k:{'alpha_daily':float(capm_regression(v,market).params.iloc[0]),'beta':float(capm_regression(v,market).params.iloc[1]),'alpha_t':float(capm_regression(v,market).tvalues.iloc[0]),'alpha_pvalue':float(capm_regression(v,market).pvalues.iloc[0]),'r_squared':float(capm_regression(v,market).rsquared)} for k,v in all_returns.items()}
    Path('report').mkdir(exist_ok=True); Path('report/paired_hac_tests.json').write_text(json.dumps(comparisons,indent=2)); Path('report/capm_results.json').write_text(json.dumps(capm,indent=2)); Path('data/processed/market_manifest.json').write_text(json.dumps(market_manifest,indent=2))
    print('Pipeline completed. Inspect report/results_summary.csv, report/paired_hac_tests.json, data/processed/data_manifest.json and figures/.')
    print(pd.DataFrame(metrics).T.round(4).to_string())
if __name__=='__main__': main()
