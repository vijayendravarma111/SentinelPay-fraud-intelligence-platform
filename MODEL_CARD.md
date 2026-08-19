# SentinelPay Model Card — Hackathon Version

## Evaluation

The included 100,000-row synthetic dataset contains 4.471% fraud. The data is ordered by transaction time.

The final evaluation uses a **future holdout that was not used for model selection or calibration**.

Current final future-test metrics:

- Accuracy: **97.09%**
- Balanced accuracy: **79.84%**
- ROC-AUC: **95.30%**
- PR-AUC: **75.77%**
- F1: **68.08%**
- Precision: **77.63%**
- Recall: **60.63%**
- False-positive rate: **0.94%**
- Brier score: **0.0223**

These are prototype results on synthetic data. They are not production-bank performance claims.

## Why accuracy is not the headline

With fraud detection, a high accuracy number can be misleading because genuine transactions dominate the population. SentinelPay therefore shows PR-AUC, recall, precision and false-positive rate alongside accuracy.

## Model selection

The project evaluates Logistic Regression, Random Forest, and XGBoost on a chronological validation segment. The selected model is the model with the strongest validation PR-AUC/F1/ROC-AUC combination. A dedicated later calibration segment is then used for sigmoid probability calibration. The final future test is untouched.

## Decision architecture

`Calibrated ML probability + independent evidence families → operational risk → decision`

Evidence families include:

- amount anomaly
- time anomaly
- velocity
- geographic inconsistency
- new-device behavior

No single business rule is allowed to declare fraud solely because an amount is large. Strong decisions require multiple independent signals or a very strong calibrated model plus supporting evidence.

## Cold-start policy

If a card has no earlier transaction in the available history, SentinelPay does not invent a personal baseline. It explicitly reports `COLD START` and relies on transaction-level and population-level evidence.
