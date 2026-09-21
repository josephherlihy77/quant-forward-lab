# DOWNSIDE_V1 — Research Specification

Status: RESEARCH ONLY — NOT FROZEN / NOT LIVE
Created: 2026-09-21

## Purpose
Develop an independent model that ranks liquid U.S.-listed equities by downside risk. This model is not the inverse of BALANCED_FORWARD_V1 and must be validated independently.

## Research target
Primary target: rank stocks most likely to fall into the worst decile of 1-month forward excess returns versus SPY.
Secondary horizons: 3 months.
Secondary outcome diagnostics: absolute return, maximum adverse excursion, maximum favorable excursion, and drawdown.

No short trade is implied by a bearish ranking. The output is research information for manual decisions.

## Universe / integrity
Use the same integrity-clean research universe philosophy as BALANCED_FORWARD_V1:
- U.S.-listed equities
- Price >= $5
- Median daily dollar volume >= $5M
- Adequate history
- Minimum monthly trading-day requirement
- Corporate-action / suspicious-return firewall
- Exclude obvious ETFs, warrants, rights, units, preferreds, depositary instruments, notes and bonds where identifiable
- Fail closed on materially incomplete data

## Candidate feature families
Features are hypotheses to test, not predetermined winners:
1. Relative weakness: 1m, 3m, 6m and 12m returns relative to SPY / cross-section.
2. Trend breakdown: distance below medium- and long-horizon moving averages and recent breakdown behavior.
3. Downside semivolatility: realized volatility using negative-return observations.
4. Volatility expansion: short-horizon volatility relative to longer-horizon volatility.
5. Drawdown state: distance from trailing highs and drawdown acceleration.
6. Liquidity deterioration: changes in dollar volume / trading activity.
7. Market sensitivity: beta and downside beta.
8. Market regime context: broad-market trend / volatility state, kept distinct from stock-level bearishness.

All features must be lagged so information available after the prediction timestamp cannot enter the score.

## Baselines
DOWNSIDE_V1 must beat simple baselines before promotion:
- Negative 6-1 momentum
- Negative 12-1 momentum
- Largest trailing drawdown
- Highest downside volatility
- Random / unconditional worst-decile rate

## Validation
Use chronological train / validation / test or walk-forward evaluation. Never random-shuffle time-series observations.

Primary metrics:
- Precision among the predicted worst decile
- Recall of realized worst-decile stocks
- Average forward excess return of the bearish-ranked basket
- Spread between bearish-ranked and non-bearish groups
- ROC-AUC where appropriate
- Brier score / calibration if probabilities are produced
- Maximum adverse and favorable excursion

Evaluate stability across market regimes and rolling windows. Include turnover and realistic costs before interpreting a short implementation.

## Anti-overfitting rules
- Do not use Snapshot #001 outcomes to tune DOWNSIDE_V1.
- Do not alter BALANCED_FORWARD_V1.
- Keep an untouched final forward period.
- Log every tested specification.
- Prefer simple models / factors unless complexity produces stable out-of-sample improvement.
- Freeze the model definition before any live downside snapshot.

## Promotion gate
DOWNSIDE_V1 can become a forward candidate only if:
1. It improves materially over simple downside baselines out of sample.
2. Performance is not concentrated in one brief regime.
3. Calibration / ranking remains useful in rolling windows.
4. Integrity and leakage audits pass.
5. The exact model/configuration is frozen and hashed.

## Eventual dashboard output
For each security:
- Upside rank from the frozen long model (when applicable)
- Downside-risk rank / score from DOWNSIDE_V1
- Regime context
- Evidence / factor breakdown
- Status: Bullish / Neutral / Bearish only after thresholds are frozen and validated

Execution remains MANUAL. No brokerage connection or automated trading.
