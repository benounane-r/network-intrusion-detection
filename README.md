# Network Intrusion Detection

A demonstration of model training to detect anomalies in network data.

## What This Project Does

- `generate_data.py` — generates a dataset containing both attack and normal network traffic and combines them into one dataset
- `train_model.py` — trains the model on 80% of the dataset and tests it on the remaining 20%
- `visualise.py` — generates a feature importance chart showing which features the model relied on most

## How It Works

The model is trained on simulated network traffic containing both attack and normal data. It uses the Random Forest algorithm — an ensemble of 100 decision trees that vote together to classify each connection. 80% of the data is used for training and 20% for testing. The model then produces a confusion matrix showing true and false positives and negatives, as well as a feature importance analysis showing which features were most useful for identifying attack traffic.

## Key Results

The model produced 196 true negatives and 35 true positives out of 43 real attacks, achieving an attack recall of 0.81 (81%). It was unable to detect 8 real attacks, classifying them as normal — in a real network environment, these would represent undetected breaches. It also flagged 1 normal connection as an attack, which may cause unnecessary delays but is not inherently dangerous.

The feature importance analysis showed that `num_connections` was the strongest attack indicator (49.8%), followed by `packet_size` (31.6%) and `duration` (18.6%). This suggests that monitoring connection rate is more effective than packet inspection alone.

Overall, while the model correctly identified 81% of attacks, these results highlight that a single model is not sufficient for a fully secure network — additional filtering layers would be needed in a production environment.

## How To Run

1. Run `python generate_data.py` to generate the dataset
2. Run `python train_model.py` to train and evaluate the model
3. Run `python visualise.py` to generate the feature importance chart

## Why This Matters

False negatives — real attacks classified as normal — are the most dangerous outcome in a security context. This project demonstrates both the capability and the limitations of ML-based intrusion detection, and highlights why recall is a more meaningful metric than accuracy alone in defence-critical systems.