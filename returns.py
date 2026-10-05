"""Return transformations and annualization utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple close-to-close returns; missing price pairs remain missing."""
    if (prices.dropna(how="all") <= 0).any().any():
        raise ValueError("Prices must be positive.")
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute continuously compounded log returns."""
    if (prices.dropna(how="all") <= 0).any().any():
        raise ValueError("Prices must be positive.")
    return np.log(prices / prices.shift(1))

def annualized_volatility(returns, periods_per_year: int = 252) -> float:
    """Annualize sample standard deviation of periodic simple returns."""
    x = pd.Series(returns).dropna()
    return float(x.std(ddof=1) * np.sqrt(periods_per_year))

def annualized_return(returns, periods_per_year: int = 252) -> float:
    """Geometric annualized return from periodic simple returns."""
    x = pd.Series(returns).dropna()
    if len(x) == 0 or (1 + x).le(0).any():
        return float("nan")
    return float(np.prod(1 + x) ** (periods_per_year / len(x)) - 1)
