# Early Sepsis Detection using Deep Learning

Sepsis prediction using ICU time-series clinical data from the PhysioNet Computing in Cardiology Challenge 2019.

## Overview
Sepsis is a life-threatening condition where every hour of delayed treatment increases mortality risk. This project builds a deep learning model that looks at **10 hours of ICU patient data** and predicts whether the patient will develop sepsis in the next hour.

## Dataset
- **Source:** PhysioNet Computing in Cardiology Challenge 2019
- **Size:** 20,336 ICU patients
- **Format:** Hourly readings of vital signs and lab values per patient
- **Class imbalance:** ~1:53 ratio of sepsis to non-sepsis cases

## Project Pipeline
1. `psv_to_df.ipynb` — Load all patient PSV files and combine into one DataFrame
2. `feature_engineering.ipynb` — Preprocess data, handle missing values, create sliding windows
3. `feature_selection.ipynb` — Analyze feature correlations, remove redundant features
4. `train_model.ipynb` — Build, train and evaluate LSTM and GRU models

## Key Modifications from Original
- Changed from `keras` to explicit `tensorflow.keras` imports
- Improved missing data handling using **ffill + bfill** instead of backfill only
- Dropped features with near-100% missing data (EtCO2, Bilirubin_direct, Fibrinogen, TroponinI)
- Added **GRU model** for comparison with LSTM baseline
- Added **F1 Score** and **Precision-Recall AUC** metrics for better evaluation on imbalanced data

## Model Architecture
Two parallel networks merged into one output:
- **Model 1 (Bidirectional LSTM/GRU):** Takes 10-hour sequences of continuous vital signs (HR, MAP, O2Sat, SBP, Resp)
- **Model 2 (Dense Network):** Takes median values of sparse lab measurements
- Both merged and passed through a softmax layer for binary classification

## Preprocessing Techniques
- **Forward-fill + Backward-fill** for continuous vital signs with less than 15% missing data
- **Median imputation** for sparse lab values with more than 15% missing data
- **Standardization** using training set mean and std only
- **Class weight balancing** to handle 1:53 sepsis to non-sepsis ratio
- **Masking layer** using π as placeholder for remaining NaN values
- **Dropped 100% missing feature** EtCO2 and other near-100% missing lab values

## Missing Data Summary
| Variable | Percent Missing |
|----------|----------------|
| HR | 10% |
| MAP | 12% |
| O2Sat | 13% |
| SBP | 15% |
| Resp | 15% |
| DBP | 31% |
| Temp | 66% |
| Glucose | 83% |
| EtCO2 | 100% — dropped |
| Bilirubin_direct | 99% — dropped |
| Fibrinogen | 99% — dropped |
| TroponinI | 99% — dropped |

## Results
- **ROC AUC:** ~0.76 on test set
- Early stopping used to prevent overfitting (patience=5)

## Requirements
tensorflow
pandas
numpy
scikit-learn
matplotlib
seaborn

## How to Run
1. Download the PhysioNet 2019 dataset from https://physionet.org/content/challenge-2019/1.0.0/
2. Place the training data in a folder called `training` inside the project directory
3. Run notebooks in order:
   - `psv_to_df.ipynb`
   - `feature_engineering.ipynb`
   - `feature_selection.ipynb`
   - `train_model.ipynb`

## References
- PhysioNet Challenge 2019: https://physionet.org/content/challenge-2019/1.0.0/
- Original repository: https://github.com/nerajbobra/sepsis-prediction