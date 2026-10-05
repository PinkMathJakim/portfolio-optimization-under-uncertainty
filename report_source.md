# Portfolio Optimization Under Uncertainty
## Comparative empirical design for European equities

### Abstract
This study is designed to test whether sophisticated portfolio construction improves out-of-sample risk-adjusted performance relative to equal weighting after estimation uncertainty, downside risk, trading costs and changing market conditions are considered. Four long-only strategies are compared using a point-in-time rolling evaluation. The empirical findings section must be populated only after the pipeline has downloaded and validated the actual data. No performance ranking is asserted in this pre-analysis version.

### 1. Motivation and research question
Mean–variance optimization formalizes the trade-off between expected return and variance, but it requires estimates of expected returns and covariances. Estimation error can be amplified by optimization, leading to concentrated or unstable allocations. Equal weighting is a useful benchmark because it avoids estimating expected returns and covariances for portfolio construction. This project tests the net empirical value of complexity rather than assuming it.

### 2. Mathematical framework
Let \(R_t\in\mathbb{R}^N\) be the vector of simple asset returns and \(w\in\mathbb{R}^N\) a fully invested weight vector. The portfolio return is \(R_{p,t}=w^\top R_t\), expected return is \(w^\top\mu\), and variance is \(w^\top\Sigma w\). Simple returns aggregate linearly across assets for a fixed beginning-of-period weight vector; log returns add through time for a single asset but do not aggregate linearly across assets. Long-only constraints are \(0\leq w_i\leq c\) and \(\sum_iw_i=1\).

Global minimum variance minimizes \(w^\top\Sigma w\). Maximum Sharpe maximizes \((w^\top\mu-r_f)/\sqrt{w^\top\Sigma w}\). Risk parity targets approximately equal normalized Euler contributions, where \(RC_i=w_i(\Sigma w)_i/\sigma_p\) and \(\sum_iRC_i=\sigma_p\) for a positive-volatility portfolio. Risk parity does not imply equal capital weights or minimum variance.

### 3. Data and sample construction
The implementation downloads daily adjusted close series from Yahoo Finance through yfinance. The proposed current-name universe spans technology, financials, healthcare, energy, industrials, consumer goods, utilities and telecommunications. A coverage threshold is applied and all exclusions are recorded. Requested and realized date ranges, ticker coverage, missing observations, package versions and limitations are stored in a JSON manifest.

The data design is intentionally candid about its shortcomings. A current-name list does not reconstruct historical index membership and therefore suffers survivorship bias. The default universe is restricted to EUR-denominated listings, avoiding an unmodelled FX conversion. This is a deliberate scope constraint rather than evidence that FX risk is irrelevant; a broader European universe should convert each series into a common base currency with aligned FX data. The current-name universe still suffers survivorship bias. Vendor adjusted prices can be revised and are not an archival point-in-time database.

### 4. Estimation and portfolio construction
Sample covariance is compared with Ledoit–Wolf shrinkage. Shrinkage trades some model bias for lower estimation variance and often improves numerical conditioning. It cannot eliminate structural breaks or incorrect universe design. Expected returns are estimated from training-window historical means, with an alternative shrinkage toward the cross-sectional mean. Historical means are noisy, particularly at daily frequency; maximum-Sharpe weights should therefore be interpreted as highly estimation-sensitive.

The optimization is long-only and capped at a configurable maximum weight. Feasibility requires \(Nc\geq1\); for 20 assets a 5% cap is feasible only if all 20 assets are retained and all weights are exactly at the cap. Any excluded asset or tighter universe makes this constraint infeasible. Optimizer failures are surfaced rather than silently replaced with arbitrary weights.

### 5. Backtest protocol
At each rebalance date, only returns strictly before the test block are used for parameter estimation. The default training window is 756 daily observations and the test block is 63 observations. The strategy holds the target weights within each block; the implementation documents that this is an approximation to daily drifted weights. Turnover is the gross sum of absolute target-weight changes, and transaction costs are charged at the start of each block. Results should be stress-tested across cost assumptions and rebalancing frequencies.

The backtest explicitly simulates daily buy-and-hold weight drift between rebalances. A stronger production implementation would additionally model cash, market holidays, spreads, market impact and partial fills. The framework remains an academic backtest rather than an execution simulator.

### 6. Risk measures and distributional analysis
Annualized volatility scales daily sample standard deviation by \(\sqrt{252}\). Sharpe uses excess mean return divided by standard deviation; Sortino replaces standard deviation with downside deviation. Maximum drawdown is the largest peak-to-trough wealth decline. Historical VaR is the empirical loss quantile, and Expected Shortfall averages losses beyond the threshold. Parametric Gaussian VaR is included as a model-dependent comparison, not as the preferred tail-risk estimate.

Skewness, kurtosis, quantiles, histograms, Q–Q plots and normality tests should be examined jointly. Rejection of normality does not identify a unique alternative distribution; failure to reject does not establish Gaussian returns. Large samples make tests sensitive to economically small deviations, while small samples have low power against tail risk.

### 7. Monte Carlo baseline
The geometric Brownian motion benchmark evolves \(S_{t+\Delta t}=S_t\exp[(\mu-\sigma^2/2)\Delta t+\sigma\sqrt{\Delta t}Z]\), where \(Z\sim N(0,1)\). Simulations summarize terminal wealth, loss probability, target probability and quantiles. GBM assumes constant drift and volatility and Gaussian independent increments. It excludes jumps, stochastic volatility, changing correlations, liquidity shocks and parameter uncertainty; its output is illustrative conditional on inputs, not a forecast guarantee.

### 8. Econometric inference
A CAPM regression relates portfolio excess return to market excess return. HAC standard errors are used to allow heteroskedasticity and serial dependence in residuals. The benchmark series must be documented and currency-aligned; the pipeline deliberately skips the regression if a validated benchmark is absent. Alpha is an intercept conditional on the model, not proof of skill or causality. Fama–French extensions require a documented factor dataset with matching currency and frequency.

Strategy comparison uses paired return differences and HAC standard errors. Such tests are still vulnerable to multiple testing, strategy selection, regime dependence, and the fact that the test sample is finite. Sharpe-ratio comparison can use specialized methods such as the Jobson–Korkie/Memmel correction or bootstrap procedures as an extension; ordinary t-tests on raw Sharpe ratios are not treated as definitive.

### 9. Results and robustness
This report template intentionally contains no invented empirical results. Run `python scripts/run_pipeline.py`, inspect the data manifest and generated metrics, then insert results with sample dates and confidence intervals. Robustness should vary the universe, estimation window, rebalance schedule, transaction costs, weight cap, covariance estimator, expected-return estimator, sample period and risk-free assumption. Report every pre-specified configuration, including optimizer failures and poor outcomes. Do not choose the headline strategy based on test-period performance.

### 10. Limitations and extensions
Main limitations are current-name survivorship bias, vendor-adjusted price uncertainty, simplified transaction costs, absence of historical constituents, the ETF proxy used for CAPM, and the lack of explicit market-impact/partial-fill modelling. Extensions include a point-in-time historical constituent universe, FX-inclusive non-EUR listings, total-return index benchmarks, volatility targeting, Black–Litterman or Bayesian return estimates, block bootstrap inference, Deflated Sharpe Ratio, Probability of Backtest Overfitting, regime-switching covariance models, and richer execution-cost models.

### References
Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91. https://doi.org/10.2307/2975974

Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. *The Journal of Finance*, 19(3), 425–442. https://doi.org/10.1111/j.1540-6261.1964.tb02865.x

Ledoit, O., & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *The Journal of Portfolio Management*, 30(4), 110–119. https://doi.org/10.3905/jpm.2004.110

Maillard, S., Roncalli, T., & Teiletche, J. (2010). On the Properties of Equally-Weighted Risk Contributions Portfolios. *The Journal of Portfolio Management*, 36(4), 60–70. https://doi.org/10.3905/jpm.2010.36.4.060

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy? *The Review of Financial Studies*, 22(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075

Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610

### 11. Pre-specified reporting protocol
For every strategy, report the geometric annualized return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, historical 95% VaR, historical 95% Expected Shortfall, turnover and net return. Metrics must be calculated on identical out-of-sample dates. If a strategy fails to optimize at a rebalance, report the failure count and investigate it; do not silently drop only failed dates. The primary output table should state the number of daily observations and the exact sample period.

### 12. Downside risk and tail interpretation
VaR is a quantile threshold: at the 95% confidence level, a one-day loss greater than the reported VaR should occur approximately 5% of the time under the empirical distribution, subject to sampling error. VaR does not describe the severity of losses beyond the threshold. Expected Shortfall estimates the average loss conditional on being in the tail, although finite samples and discrete quantiles complicate exact conditional interpretation. Historical measures preserve observed skewness and kurtosis but cannot reliably estimate losses more extreme than the sample contains. Parametric Gaussian VaR is sensitive to the normality assumption and may materially understate tail losses when returns are heavy-tailed.

### 13. Regime definition and interpretation
Regimes are defined mechanically using trailing realized volatility and thresholds estimated from past observations only. A high-volatility label means the rolling estimate exceeds its past 80th percentile; a low-volatility label means it is below its past 20th percentile. This avoids assigning crisis labels after observing portfolio outcomes. It does not imply that all high-volatility periods are crises or that volatility alone captures macroeconomic regimes. The same strategy may be exposed to different sectors, currencies, and factor risks across regimes, so regime-conditioned averages should be described as conditional associations rather than causal effects.

### 14. Statistical uncertainty and multiplicity
Annualized metrics are point estimates. A difference in annualized return or Sharpe ratio is not necessarily statistically distinguishable from zero. HAC inference for paired mean daily-return differences is one transparent first pass, but the estimated mean is often small relative to return noise. A bootstrap should preserve serial dependence, for example by resampling blocks rather than individual daily observations. Testing many universes, windows, costs, caps and estimators inflates the chance of selecting a favorable result by chance. The robustness grid must therefore be treated as a sensitivity map, not as a menu from which to select the best-looking specification. The final paper should identify a primary specification before discussing alternative specifications.

### 15. Research reproducibility checklist
Before submission, archive the retrieval manifest and the exact local price snapshot; record Python and dependency versions; rerun all tests; execute each notebook from a clean kernel in numerical order; regenerate figures and the PDF; confirm that no result in the report predates the current data snapshot; and manually inspect every generated chart and table. Record the Git commit hash used for the final analysis. A clean run on a fresh environment is stronger evidence of reproducibility than a notebook that works only in an already-populated session.
