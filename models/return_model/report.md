# Return model gbm-20260918-c97181eb

Trained 2026-09-25T15:53:40+00:00 on 33312 article-symbol samples (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD). Label: same-session close vs first bar after publication > +0.25%. All numbers below are **out of sample** (test segment).

| Segment | Samples | Period | Up rate |
|---|---:|---|---:|
| train | 19790 | 2023-11-25 → 2025-09-01 | 0.399 |
| calibrate | 6588 | 2025-09-05 → 2026-03-06 | 0.395 |
| test | 6663 | 2026-03-10 → 2026-09-17 | 0.411 |

| Metric | Model | Baseline |
|---|---:|---:|
| ROC AUC (vs sentiment-only logistic) | 0.5242 | 0.4793 |
| ROC AUC, one article per symbol-day (n=1251) | 0.5322 | 0.5 |
| ROC AUC, tradeable articles only (n=2593) | 0.5081 | 0.5 |
| ROC AUC, tradeable, one per symbol-day (n=883) | 0.5071 | 0.5 |
| Brier score (vs always predicting the train up-rate) | 0.2416 | 0.2422 |
| Log loss (vs train up-rate) | 0.6764 | 0.6775 |

Mean forward return, all test samples -0.008%, tradeable +0.013%. Predicted prob_up quantiles on tradeable articles (1/10/50/90/99%): 0.330, 0.361, 0.393, 0.434, 0.479

## What the router's gate would have let through (tradeable articles)

Chosen on the calibration segment:

| prob_up ≥ | Samples | Share | Hit rate | Mean fwd return | Symbol-days |
|---:|---:|---:|---:|---:|---:|
| 0.40 | 953 | 33.4% | 0.380 | -0.033% | 344 |
| 0.45 | 90 | 3.2% | 0.300 | -0.236% | 40 |
| 0.50 | 2 | 0.1% | 0.500 | -0.470% | 2 |
| 0.55 | 0 | 0.0% | — | — | 0 |
| 0.60 | 0 | 0.0% | — | — | 0 |

Confirmed on the test segment:

| prob_up ≥ | Samples | Share | Hit rate | Mean fwd return | Symbol-days |
|---:|---:|---:|---:|---:|---:|
| 0.40 | 1052 | 40.6% | 0.368 | +0.021% | 381 |
| 0.45 | 116 | 4.5% | 0.586 | +0.312% | 49 |
| 0.50 | 12 | 0.5% | 0.750 | +0.838% | 6 |
| 0.55 | 0 | 0.0% | — | — | 0 |
| 0.60 | 0 | 0.0% | — | — | 0 |

## Calibration (tradeable test articles, predicted-probability deciles)

| Mean predicted | Observed up rate | n |
|---:|---:|---:|
| 0.345 | 0.358 | 260 |
| 0.367 | 0.432 | 259 |
| 0.376 | 0.355 | 259 |
| 0.383 | 0.332 | 259 |
| 0.389 | 0.338 | 260 |
| 0.397 | 0.331 | 260 |
| 0.404 | 0.313 | 259 |
| 0.413 | 0.355 | 259 |
| 0.426 | 0.293 | 259 |
| 0.454 | 0.523 | 260 |

## Permutation importance (drop in test AUC)

- `rsi14`: +0.0175
- `mom20`: +0.0144
- `mom5`: +0.0094
- `regular_hours`: +0.0040
- `sent_pos`: +0.0010
- `n_symbols`: +0.0006
- `sent_neu`: +0.0006
- `sent_neg`: -0.0008
- `atr14_pct`: -0.0010
- `sent_net`: -0.0016
- `gap1`: -0.0024
- `gk20`: -0.0027
- `rv20`: -0.0036
- `vol_z20`: -0.0060
- `dist_sma20`: -0.0083
