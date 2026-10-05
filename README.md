# Portfolio Optimization Under Uncertainty

**Research question.** Does sophisticated portfolio optimization outperform equal weighting out of sample after estimation uncertainty, downside risk, turnover costs, and changing market conditions are accounted for?

> **Research status:** this repository provides the reproducible research pipeline and an explicitly provisional methodology report. No empirical performance conclusions are pre-filled. Run the pipeline against downloaded market data before interpreting or publishing results.

## Abstract
This project compares equal weighting, global minimum variance, maximum-Sharpe, and equal-risk-contribution portfolios on a diversified European equity universe. It emphasizes point-in-time rolling evaluation, shrinkage covariance estimation, realistic long-only constraints, turnover-based costs, downside risk, regime analysis, and inference that respects time-series dependence. The design treats equal weighting as a serious benchmark and reports both favorable and unfavorable outcomes without selecting the test period based on results.

## Research design
- Daily adjusted close prices, converted to a common reporting currency where data availability permits; see data caveats below.
- Training window: 756 trading days (approximately three years); quarterly rebalancing by default.
- Long-only portfolios; fully invested; default maximum single-name weight 10% where feasible.
- Four core strategies: equal weight, global minimum variance, maximum Sharpe, and equal risk contribution.
- Covariance estimates: sample and Ledoit–Wolf shrinkage. Expected returns: historical mean and shrinkage toward cross-sectional mean; sensitivity reported rather than assumed away.
- Transaction cost convention: one-way cost rate multiplied by the sum of absolute weight changes. The resulting turnover convention is **gross absolute traded notional**; a 100% replacement can therefore yield turnover above 100%. Costs are deducted at rebalance.
- Primary evaluation is walk-forward, not optimized on the test sample.

## Repository map
```text
src/                  reusable research code
notebooks/            ten reproducible, ordered research notebooks
tests/                mathematical and backtest unit tests
data/                 data policy; downloaded data are not committed
figures/              generated figures
report/               report PDF and source
app/                  Streamlit research dashboard
scripts/              pipeline entry points
```

## Install
Python 3.11+ is recommended.
```bash
git clone <your-repository-url>
cd portfolio-optimization-under-uncertainty
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce
```bash
python scripts/run_pipeline.py --start 2015-01-01 --end 2026-10-06
pytest -q
streamlit run app/streamlit_app.py
```
The first pipeline run downloads prices using `yfinance`. Network availability, Yahoo Finance terms, ticker mapping, and historical corporate-action adjustments can affect results. The script writes a manifest with request parameters, retrieval timestamp, data coverage, and package versions. For a reproducible rerun, archive the manifest and data snapshot locally; do not assume a third-party API is immutable.

## Data universe and limitations
The default universe contains 20 liquid European large-cap names listed in EUR across Germany, France, Italy and the Netherlands: ASML.AS, SAP.DE, SIE.DE, ALV.DE, DTE.DE, BAS.DE, ADS.DE, AIR.PA, MC.PA, OR.PA, SAN.PA, BNP.PA, SU.PA, AI.PA, ENEL.MI, ENI.MI, ISP.MI, UCG.MI, BMW.DE and RWE.DE. Restricting the investable listings to EUR avoids silently mixing local-currency returns with an unmodelled FX component. The list is still a **current-name convenience universe**, not a survivorship-bias-free historical constituent universe; delisted and failed firms may be absent. Dividends/splits are reflected only to the extent the vendor's adjusted-price field is accurate. The CAPM benchmark is the EUR-listed EXSA ETF, an iShares STOXX Europe 600 UCITS ETF proxy for the broad European equity market.

The default benchmark is the STOXX Europe 600 proxy (`^STOXX` availability must be checked before use); if it cannot be downloaded, the CAPM regression will be skipped rather than silently substituting an unrelated benchmark. A broad-market ETF or properly sourced factor series may be supplied by the researcher. Risk-free rate defaults to zero only as a documented fallback and is varied in robustness tests; it is not presented as a historically accurate European cash return.

## Methods and interpretation
Simple returns are used for portfolio aggregation and realized backtests. Log returns are provided for distribution analysis. Annualization uses 252 trading days. Historical VaR is the positive loss quantile; CVaR/Expected Shortfall is the average loss in the tail beyond that quantile. Maximum drawdown is computed from compounded wealth. The GBM module is a controlled benchmark model, not a claim that returns are Gaussian or volatility constant.

The project reports Sharpe ratios but does not treat raw Sharpe rankings as statistically decisive. Strategy comparisons use paired daily return differences and a Newey–West/HAC standard error for mean differences; inference remains sensitive to overlapping positions, strategy selection, and the short effective sample. Multiple-comparison concerns and backtest overfitting are discussed in the report.

## Expected output files
- `data/processed/adjusted_prices.csv`
- `data/processed/simple_returns.csv`
- `data/processed/data_manifest.json`
- `figures/*.png`
- `report/results_summary.csv`
- `report/quantitative_portfolio_report.pdf`

If no data can be retrieved, the pipeline exits with a clear error and does not synthesize market observations. Results and figures are created only from actual retrieved or researcher-supplied data.

## References
- Markowitz, H. (1952). “Portfolio Selection.” *The Journal of Finance*, 7(1), 77–91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
- Sharpe, W. F. (1964). “Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk.” *The Journal of Finance*, 19(3), 425–442. https://doi.org/10.1111/j.1540-6261.1964.tb02865.x
- Ledoit, O., & Wolf, M. (2004). “Honey, I Shrunk the Sample Covariance Matrix.” *The Journal of Portfolio Management*, 30(4), 110–119. https://doi.org/10.3905/jpm.2004.110
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). “Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?” *The Review of Financial Studies*, 22(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075
- Newey, W. K., & West, K. D. (1987). “A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.” *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610

## Disclaimer
Educational research only. Not investment advice. Backtested or simulated performance does not predict future returns.
