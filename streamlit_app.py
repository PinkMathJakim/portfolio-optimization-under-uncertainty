"""Interactive educational dashboard for portfolio research; not investment advice."""
from pathlib import Path
import streamlit as st
import pandas as pd
from src.backtest import rolling_backtest
from src.risk import performance_metrics
from src.covariance import estimate_covariance
from src.optimization import efficient_frontier, equal_weight, minimum_variance, max_sharpe, risk_parity
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Optimization Under Uncertainty", layout="wide")
st.title("Portfolio Optimization Under Uncertainty")
st.caption("Educational research application only — not investment advice. Historical and simulated performance do not guarantee future results.")
path=Path("data/processed/simple_returns.csv")
if not path.exists():
    st.warning("No local market data found. Run `python scripts/run_pipeline.py` first. The app does not fabricate a dataset.")
    st.stop()
returns=pd.read_csv(path,index_col=0,parse_dates=True).dropna(how="any")
with st.sidebar:
    st.header("Research specification")
    strategy_label=st.selectbox("Strategy",["Equal Weight","Minimum Variance","Maximum Sharpe","Risk Parity"])
    strategy={"Equal Weight":"equal_weight","Minimum Variance":"min_variance","Maximum Sharpe":"max_sharpe","Risk Parity":"risk_parity"}[strategy_label]
    rebalance_label=st.selectbox("Rebalancing",["Monthly","Quarterly","Annual"],index=1)
    rebalance={"Monthly":21,"Quarterly":63,"Annual":252}[rebalance_label]
    cap=st.select_slider("Maximum asset weight",options=[.05,.10,.20],value=.10,format_func=lambda x:f"{x:.0%}")
    covariance=st.selectbox("Covariance estimator",["Sample","Ledoit–Wolf"])
    covariance_method={"Sample":"sample","Ledoit–Wolf":"ledoit_wolf"}[covariance]
    window_years=st.selectbox("Estimation window",[2,3,4],index=1)
    costs=st.select_slider("One-way transaction cost",options=[0,5,10,25],value=10,format_func=lambda x:f"{x} bps")
    run=st.button("Run selected backtest",type="primary")

if not run:
    st.info("Choose research settings in the sidebar and click **Run selected backtest**. The first run may take time because the rolling optimizer is solved at each rebalance.")
    st.stop()
train_window=window_years*252
try:
    strategy_returns,weight_log=rolling_backtest(returns,strategy,train_window,rebalance,cap,covariance_method,"historical_mean",costs)
except Exception as exc:
    st.error(f"Backtest could not run with this configuration: {exc}")
    st.stop()
metrics=performance_metrics(strategy_returns)
cols=st.columns(4)
for col,(label,key,fmt) in zip(cols,[("Annualized return","annual_return","{:.2%}"),("Annualized volatility","annual_volatility","{:.2%}"),("Sharpe ratio","sharpe","{:.2f}"),("Maximum drawdown","max_drawdown","{:.2%}")]):
    col.metric(label,fmt.format(metrics.get(key,float('nan'))))
cols=st.columns(4)
for col,(label,key,fmt) in zip(cols,[("Sortino ratio","sortino","{:.2f}"),("Historical VaR 95%","historical_var_95","{:.2%}"),("Historical CVaR 95%","historical_cvar_95","{:.2%}"),("Calmar ratio","calmar","{:.2f}")]):
    col.metric(label,fmt.format(metrics.get(key,float('nan'))))
left,right=st.columns(2)
with left:
    st.subheader("Out-of-sample wealth")
    st.line_chart((1+strategy_returns).cumprod().rename("Growth of 1"))
with right:
    st.subheader("Drawdown")
    wealth=(1+strategy_returns).cumprod(); st.line_chart((wealth/wealth.cummax()-1).rename("Drawdown"))
st.subheader("Latest target weights")
if not weight_log.empty:
    weights=weight_log.iloc[-1]["weights"]
    st.bar_chart(pd.Series(weights).sort_values(ascending=False).rename("Weight"))
    st.caption(f"Latest rebalance: {weight_log.iloc[-1]['rebalance_date']}; turnover: {weight_log.iloc[-1]['turnover']:.2%}; estimated transaction cost: {weight_log.iloc[-1]['transaction_cost']:.4%} of portfolio value.")
st.subheader("Efficient frontier (training sample at latest rebalance)")
try:
    last_date=pd.Timestamp(weight_log.iloc[-1]["rebalance_date"])
    hist=returns.loc[returns.index<last_date].tail(train_window)
    cov=estimate_covariance(hist,covariance_method).to_numpy(); mu=hist.mean().to_numpy()
    frontier=efficient_frontier(mu,cov,cap,35)
    if not frontier.empty:
        fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(frontier.volatility,frontier["return"],label="Feasible frontier")
        ax.set_xlabel("Daily volatility"); ax.set_ylabel("Expected daily return"); ax.legend(); ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)
except Exception as exc:
    st.warning(f"Efficient frontier unavailable for this specification: {exc}")
st.subheader("Metrics and configuration")
st.dataframe(pd.DataFrame({"Value":metrics}),use_container_width=True)
st.caption(f"Training window: {window_years} years; rebalance interval: {rebalance_label}; covariance: {covariance}; cap: {cap:.0%}; cost: {costs} bps. The backtest is a simplified research model; weights are held fixed within each rebalance block, so it approximates rather than fully models daily holdings drift and execution.")
