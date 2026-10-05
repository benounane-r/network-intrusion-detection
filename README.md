# Network Intrusion Detection

A demonstration of ML-based anomaly detection in network traffic — from simulated baselines to real-world benchmark evaluation.

## What This Project Does

- `generate_data.py` — generates a simulated dataset of attack and normal network traffic
- `train_model.py` — trains a Random Forest classifier with stratified split and evaluates with full metrics
- `visualise.py` — generates feature importance chart and precision-recall curve
- `compare_models.py` — compares Random Forest vs XGBoost on simulated data
- `nsl_kdd_model.py` — trains and compares both models on the real NSL-KDD benchmark dataset

## How It Works

The model is trained on network traffic data containing both attack and normal connections. It uses the Random Forest algorithm — an ensemble of 100 decision trees that vote together to classify each connection. A stratified split ensures the attack/normal ratio is preserved across train and test sets. The model produces a confusion matrix, feature importance analysis, and precision-recall curve.

## Data and Evaluation

**Simulated dataset:**
- 1,000 normal + 200 attack connections
- Stratified 80/20 train/test split
- No duplicate rows, no leakage — encoding applied after splitting
- No feature scaling applied — tree-based models are scale-invariant

**NSL-KDD benchmark dataset:**

| | Train | Test |
|---|---|---|
| Rows | 125,973 | 22,544 |
| Normal | 53.5% | 43.1% |
| Attack | 46.5% | 56.9% |

The dataset originally contained 23 attack types. Some classes were severely underrepresented — the spy class had only 2 examples — making multi-class classification unreliable. A binary approach (normal vs attack) was used instead.

Data was split by file — `KDDTrain+.txt` and `KDDTest+.txt` are predefined splits in the benchmark. This simulates real deployment where training happens on historical data and testing on future data.

No duplicate rows were found in either the train or test set. Encoding was applied separately to train and test sets — no leakage.

## Key Results

**Simulated data — Random Forest (stratified split):**

| Metric | Value |
|---|---|
| Accuracy | 0.95 |
| Attack Recall | 0.75 |
| Precision-Recall AP | 0.95 |

The feature importance analysis showed that `num_connections` was the strongest attack indicator (49.8%), followed by `packet_size` (31.6%) and `duration` (18.6%). This suggests that monitoring connection rate is more effective than packet size or duration alone.

The precision-recall curve (AP=0.95) shows the model maintains perfect precision up to 75% recall. Beyond that, catching more attacks requires accepting more false alarms — a threshold security engineers can tune based on operational requirements.

**NSL-KDD real benchmark — RF vs XGBoost:**

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 0.77 | 0.80 |
| Attack Recall | 0.61 | 0.67 |
| Macro F1 | 0.77 | 0.80 |

XGBoost outperformed Random Forest across all metrics on real data. The performance drop compared to simulated data (95% → 80% accuracy) demonstrates the challenge of real-world network traffic classification, where attack patterns overlap with legitimate traffic.

The test set contains 56.9% attacks vs 46.5% in training — an intentional distribution shift simulating real deployment where attack patterns change over time. This explains the lower recall compared to balanced datasets.

## Why This Matters

False negatives — real attacks classified as normal — are the most dangerous outcome in a security context. This project demonstrates both the capability and limitations of ML-based intrusion detection across simulated and real-world data, and highlights why recall is a more meaningful metric than accuracy alone in defence-critical systems.

## How To Run

**Simulated data:**
1. `python generate_data.py` — generate dataset
2. `python train_model.py` — train and evaluate
3. `python visualise.py` — generate charts
4. `python compare_models.py` — compare RF vs XGBoost

**Real NSL-KDD data:**
1. Download `KDDTrain+.txt` and `KDDTest+.txt` from [Kaggle NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd)
2. Place both files in the project folder
3. `python nsl_kdd_model.py` — train and compare both models