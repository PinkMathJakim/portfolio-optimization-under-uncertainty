"""Publication-oriented plotting helpers."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def save_standard_figures(prices, returns, strategy_returns=None, out_dir="figures"):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); sns.set_theme(style="whitegrid",context="talk")
    wealth=(1+returns.fillna(0)).cumprod()
    fig,ax=plt.subplots(figsize=(12,6)); (wealth/wealth.iloc[0]).plot(ax=ax,lw=1.2); ax.set(title="Cumulative total-return index (rebased)",ylabel="Growth of 1",xlabel="Date"); fig.tight_layout(); fig.savefig(out/"asset_cumulative_returns.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,9)); sns.heatmap(returns.corr(),cmap="vlag",center=0,vmin=-1,vmax=1,ax=ax); ax.set_title("Daily return correlation"); fig.tight_layout(); fig.savefig(out/"correlation_matrix.png",dpi=200); plt.close(fig)
    if strategy_returns is not None:
        fig,ax=plt.subplots(figsize=(12,6)); ((1+strategy_returns.fillna(0)).cumprod()).plot(ax=ax); ax.set(title="Out-of-sample wealth paths",ylabel="Growth of 1"); fig.tight_layout(); fig.savefig(out/"oos_performance.png",dpi=200); plt.close(fig)
        wealth=(1+strategy_returns.fillna(0)).cumprod(); dd=wealth/wealth.cummax()-1
        fig,ax=plt.subplots(figsize=(12,5)); dd.plot(ax=ax); ax.set(title="Out-of-sample drawdowns",ylabel="Drawdown"); fig.tight_layout(); fig.savefig(out/"drawdowns.png",dpi=200); plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,6)); strategy_returns.plot(kind="hist",bins=70,alpha=.55,density=True,ax=ax); ax.set(title="Out-of-sample daily return distributions",xlabel="Daily return"); fig.tight_layout(); fig.savefig(out/"return_distributions.png",dpi=200); plt.close(fig)

def save_diagnostic_figures(returns, strategy_returns, weight_logs, out_dir="figures"):
    """Generate distribution, covariance, rolling-risk, turnover, and frontier diagnostics."""
    from scipy import stats
    import numpy as np
    import pandas as pd
    from .covariance import estimate_covariance
    from .optimization import efficient_frontier, equal_weight, minimum_variance, max_sharpe, risk_parity
    from .risk import risk_contributions
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    clean=returns.dropna(how="any")
    cov=estimate_covariance(clean,"sample")
    eig=np.linalg.eigvalsh(cov.to_numpy())
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(np.arange(1,len(eig)+1),eig[::-1],marker="o"); ax.set_yscale("log"); ax.set(title="Sample covariance eigenvalue spectrum",xlabel="Eigenvalue rank",ylabel="Eigenvalue (log scale)"); fig.tight_layout(); fig.savefig(out/"covariance_eigenvalues.png",dpi=200); plt.close(fig)
    portfolio=clean.mean(axis=1)
    fig,ax=plt.subplots(figsize=(7,7)); stats.probplot(portfolio,dist="norm",plot=ax); ax.set_title("Q–Q plot: equal-weight daily returns (exploratory) "); fig.tight_layout(); fig.savefig(out/"qq_plot.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); (portfolio.rolling(63).std()*np.sqrt(252)).plot(ax=ax); ax.set(title="Rolling 63-day annualized volatility",ylabel="Annualized volatility"); fig.tight_layout(); fig.savefig(out/"rolling_volatility.png",dpi=200); plt.close(fig)
    rolling_mean=portfolio.rolling(252).mean()*252; rolling_std=portfolio.rolling(252).std()*np.sqrt(252)
    fig,ax=plt.subplots(figsize=(12,5)); (rolling_mean/rolling_std).plot(ax=ax); ax.set(title="Rolling one-year Sharpe proxy (zero risk-free rate)",ylabel="Sharpe ratio"); fig.tight_layout(); fig.savefig(out/"rolling_sharpe.png",dpi=200); plt.close(fig)
    if strategy_returns is not None:
        fig,ax=plt.subplots(figsize=(12,5)); (strategy_returns.rolling(63).std()*np.sqrt(252)).plot(ax=ax); ax.set(title="Rolling strategy volatility",ylabel="Annualized volatility"); fig.tight_layout(); fig.savefig(out/"strategy_rolling_volatility.png",dpi=200); plt.close(fig)
    if weight_logs:
        fig,ax=plt.subplots(figsize=(12,5))
        for name,log in weight_logs.items():
            if not log.empty: ax.plot(pd.to_datetime(log.rebalance_date),log.turnover,label=name)
        ax.set(title="Gross absolute turnover at rebalance",ylabel="Turnover",xlabel="Rebalance date"); ax.legend(); fig.tight_layout(); fig.savefig(out/"turnover.png",dpi=200); plt.close(fig)
    # Exploratory frontier uses full sample and must never be described as out-of-sample evidence.
    try:
        mu=clean.mean().to_numpy(); c=cov.to_numpy(); cap=max(.10,1/len(mu))
        frontier=efficient_frontier(mu,c,cap,40)
        if not frontier.empty:
            fig,ax=plt.subplots(figsize=(9,6)); ax.plot(frontier.volatility,frontier['return'],label='Feasible frontier')
            for name,w in [('Equal weight',equal_weight(len(mu))),('Min variance',minimum_variance(c,cap)),('Max Sharpe',max_sharpe(mu,c,0,cap)),('Risk parity',risk_parity(c,cap))]:
                ax.scatter(np.sqrt(w@c@w),w@mu,s=55,label=name)
            ax.set(title='In-sample Markowitz frontier (descriptive only)',xlabel='Daily volatility',ylabel='Expected daily return'); ax.legend(); fig.tight_layout(); fig.savefig(out/'efficient_frontier.png',dpi=200); plt.close(fig)
    except Exception as exc:
        (out/'efficient_frontier_error.txt').write_text(str(exc))
