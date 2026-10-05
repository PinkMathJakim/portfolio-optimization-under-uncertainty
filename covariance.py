"""Covariance estimation and diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def estimate_covariance(returns: pd.DataFrame, method: str = "sample", annualize: bool = False, periods: int = 252) -> pd.DataFrame:
    """Estimate covariance using pairwise-complete sample covariance or Ledoit-Wolf.

    Ledoit-Wolf requires a complete rectangular matrix; caller must explicitly align
    complete observations to avoid implicit, asset-specific sample sizes.
    """
    x = returns.dropna(how="any")
    if len(x) < 2:
        raise ValueError("At least two complete observations are required.")
    if method == "sample":
        cov = x.cov().to_numpy()
    elif method in {"ledoit_wolf", "lw"}:
        cov = LedoitWolf().fit(x.to_numpy()).covariance_
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    if annualize: cov = cov * periods
    return pd.DataFrame(cov, index=x.columns, columns=x.columns)

def covariance_diagnostics(cov: pd.DataFrame) -> dict:
    """Return eigenvalues, condition number, and positive-definiteness diagnostics."""
    a = (cov.to_numpy() + cov.to_numpy().T) / 2
    eig = np.linalg.eigvalsh(a)
    pos = eig[eig > 1e-12]
    return {"min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max()),
            "condition_number": float(np.linalg.cond(a)), "positive_definite": bool(eig.min() > 0),
            "eigenvalues": eig.tolist()}
