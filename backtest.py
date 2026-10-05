"""Point-in-time rolling backtest with explicit rebalancing and turnover costs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .covariance import estimate_covariance
from .optimization import equal_weight, minimum_variance, max_sharpe, risk_parity

def rolling_backtest(returns: pd.DataFrame, strategy: str="equal_weight", train_window: int=756,
                     rebalance_every: int=63, max_weight: float=.10, covariance_method: str="sample",
                     expected_return_method: str="historical_mean", transaction_cost_bps: float=10,
                     risk_free_daily: float=0.0) -> tuple[pd.Series,pd.DataFrame]:
    """Point-in-time walk-forward backtest with daily buy-and-hold drift.

    Parameters are estimated only from observations strictly before each test block.
    Target weights are established at the rebalance date, then holdings drift with
    realized returns until the next rebalance. Turnover is measured on pre-trade
    weights, and transaction costs are charged on the rebalance return observation.
    """
    x=returns.dropna(how="any").copy()
    if len(x)<=train_window: raise ValueError("Insufficient observations for training window and test")
    names=x.columns; previous=np.zeros(len(names)); gross=[]; net=[]; records=[]; dates=[]
    for start in range(train_window,len(x),rebalance_every):
        train=x.iloc[start-train_window:start]; test=x.iloc[start:start+rebalance_every]
        if test.empty: break
        cov=estimate_covariance(train,covariance_method).to_numpy()
        mu=train.mean().to_numpy()
        if expected_return_method=="shrinkage_mean": mu=0.5*mu+0.5*np.repeat(mu.mean(),len(mu))
        elif expected_return_method!="historical_mean": raise ValueError("Unknown expected return method")
        if strategy=="equal_weight": w=equal_weight(len(names),max_weight)
        elif strategy=="min_variance": w=minimum_variance(cov,max_weight)
        elif strategy=="max_sharpe": w=max_sharpe(mu,cov,risk_free_daily,max_weight)
        elif strategy=="risk_parity": w=risk_parity(cov,max_weight)
        else: raise ValueError(f"Unknown strategy: {strategy}")
        turnover=float(np.abs(w-previous).sum()); cost=transaction_cost_bps/10000*turnover
        # Daily portfolio return under buy-and-hold drift. Trade into target weights
        # at the beginning of the block; the first day's return is earned on target weights.
        current=w.copy(); block=[]; block_net=[]
        for j, (_, row) in enumerate(test.iterrows()):
            daily=float(row.to_numpy()@current)
            net_daily=daily-cost if j==0 else daily
            block.append(daily); block_net.append(net_daily)
            wealth_weights=current*(1.0+row.to_numpy())
            denom=float(wealth_weights.sum())
            if denom <= 0: raise RuntimeError("Non-positive post-return portfolio value encountered")
            current=wealth_weights/denom
        gross.extend(block); net.extend(block_net); dates.extend(test.index.tolist())
        records.append({"rebalance_date":test.index[0],"turnover":turnover,"transaction_cost":cost,
                        "weights":dict(zip(names,w.tolist())),"training_start":train.index[0],"training_end":train.index[-1]})
        previous=current
    idx=pd.DatetimeIndex(dates)
    return pd.Series(net,index=idx,name=strategy),pd.DataFrame(records)
