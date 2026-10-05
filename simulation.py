"""Monte Carlo geometric Brownian motion baseline."""
from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_gbm_paths(s0: float, mu: float, sigma: float, years: float=1.0, steps_per_year: int=252,
                       n_paths: int=5000, seed: int=42) -> np.ndarray:
    if s0<=0 or sigma<0 or years<=0 or steps_per_year<1 or n_paths<1: raise ValueError("Invalid GBM parameters")
    rng=np.random.default_rng(seed); n_steps=int(years*steps_per_year)
    z=rng.standard_normal((n_steps,n_paths))
    increments=(mu-0.5*sigma**2)/steps_per_year + sigma/np.sqrt(steps_per_year)*z
    log_paths=np.vstack([np.zeros(n_paths),np.cumsum(increments,axis=0)])
    return s0*np.exp(log_paths)

def summarize_terminal_wealth(paths: np.ndarray, target_return: float=0.10) -> dict:
    terminal=np.asarray(paths)[-1,:]; initial=np.asarray(paths)[0,0]
    return {"mean_terminal_wealth":float(terminal.mean()),"median_terminal_wealth":float(np.median(terminal)),
            "probability_loss":float(np.mean(terminal<initial)),
            "probability_target":float(np.mean(terminal>=initial*(1+target_return))),
            "terminal_q05":float(np.quantile(terminal,.05)),"terminal_q95":float(np.quantile(terminal,.95))}
