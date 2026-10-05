"""Long-only portfolio construction with explicit feasibility checks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .risk import portfolio_volatility, risk_contributions

def _validate_inputs(mu, cov, max_weight=1.0):
    mu=np.asarray(mu,dtype=float); cov=np.asarray(cov,dtype=float)
    if cov.shape != (len(mu),len(mu)): raise ValueError("mu and covariance dimensions differ")
    if len(mu)*max_weight < 1-1e-10: raise ValueError("Infeasible max weight: N * max_weight < 1")
    cov=(cov+cov.T)/2 + np.eye(len(mu))*1e-10
    return mu,cov

def equal_weight(n: int, max_weight: float | None=None) -> np.ndarray:
    if n < 1: raise ValueError("n must be positive")
    w=np.repeat(1/n,n)
    if max_weight is not None and w.max()>max_weight+1e-12: raise ValueError("Equal weights violate cap")
    return w

def minimum_variance(cov, max_weight=0.10) -> np.ndarray:
    cov=np.asarray(cov,dtype=float); n=cov.shape[0]
    if n*max_weight < 1-1e-10: raise ValueError("Infeasible weight cap")
    x0=np.repeat(1/n,n)
    res=minimize(lambda w: float(w@cov@w),x0,method="SLSQP",bounds=[(0,max_weight)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"maxiter":2000,"ftol":1e-12})
    if not res.success: raise RuntimeError(f"Minimum-variance optimization failed: {res.message}")
    return _normalize_checked(res.x,max_weight)

def max_sharpe(mu, cov, risk_free=0.0, max_weight=0.10) -> np.ndarray:
    mu,cov=_validate_inputs(mu,cov,max_weight); n=len(mu); x0=np.repeat(1/n,n)
    def objective(w):
        vol=portfolio_volatility(w,cov)
        return -((w@mu-risk_free)/vol) if vol>1e-12 else 1e6
    res=minimize(objective,x0,method="SLSQP",bounds=[(0,max_weight)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"maxiter":3000,"ftol":1e-10})
    if not res.success: raise RuntimeError(f"Max-Sharpe optimization failed: {res.message}")
    return _normalize_checked(res.x,max_weight)

def risk_parity(cov, max_weight=0.10) -> np.ndarray:
    """Minimize dispersion of normalized Euler volatility contributions."""
    cov=np.asarray(cov,dtype=float); n=cov.shape[0]
    if n*max_weight < 1-1e-10: raise ValueError("Infeasible weight cap")
    x0=np.repeat(1/n,n)
    def objective(w):
        rc=risk_contributions(w,cov).to_numpy(); total=rc.sum()
        if total<=0: return 1e6
        share=rc/total
        return float(np.sum((share-1/n)**2))
    res=minimize(objective,x0,method="SLSQP",bounds=[(1e-8,max_weight)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"maxiter":3000,"ftol":1e-13})
    if not res.success: raise RuntimeError(f"Risk-parity optimization failed: {res.message}")
    return _normalize_checked(res.x,max_weight)

def _normalize_checked(w, cap):
    w=np.clip(np.asarray(w,dtype=float),0,cap); w=w/w.sum()
    if abs(w.sum()-1)>1e-8 or w.min() < -1e-9 or w.max()>cap+1e-7: raise RuntimeError("Optimizer returned infeasible weights")
    return w

def efficient_frontier(mu, cov, max_weight=0.10, points=40):
    mu,cov=_validate_inputs(mu,cov,max_weight); n=len(mu); lo=float(minimum_variance(cov,max_weight)@mu)
    hi=float(max(mu))
    out=[]
    for target in np.linspace(lo,hi,points):
        res=minimize(lambda w:float(w@cov@w),np.repeat(1/n,n),method="SLSQP",
          bounds=[(0,max_weight)]*n,
          constraints=[{"type":"eq","fun":lambda w:w.sum()-1},{"type":"ineq","fun":lambda w:float(w@mu-target)}],
          options={"maxiter":1500,"ftol":1e-10})
        if res.success: out.append({"return":float(res.x@mu),"volatility":portfolio_volatility(res.x,cov)})
    return pd.DataFrame(out).drop_duplicates().sort_values("volatility")
