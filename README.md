
# A ML Classification Demo (Titanic-like)

A machine learning classification demo built with Python, pandas, scikit-learn, Matplotlib, and Seaborn.

This project generates a synthetic Titanic-like dataset, trains two classic classifiers (Logistic Regression and Random Forest), evaluates them with a confusion matrix and classification report, and saves the outputs to Google Drive when run in Google Colab.

---

## Overview

This repository demonstrates a minimal end-to-end machine learning workflow:

- Generate or load tabular data.
- Perform basic preprocessing.
- Split the data into training and testing sets.
- Train and evaluate two classifiers.
- Save the dataset and evaluation figures for later review.

The dataset used in this project is synthetic, so no external data download is required.

---

## Features

- Synthetic Titanic-like dataset with 1,000 samples.
- Basic feature engineering with categorical encoding.
- Two classification models:
  - Logistic Regression
  - Random Forest
- Evaluation metrics:
  - Accuracy
  - Classification Report
  - Confusion Matrix
- Saves the generated CSV file and plots to Google Drive when executed in Colab.

---

## Project Structure

```text
ml-sample-demo/
├── titanic_ml_demo.py
├── README.md
├── requirements.txt
├── synthetic_titanic_sample.cvs
└── images/
    ├── confusion_matrix_logistic_regression.png
    └── confusion_matrix_random_forest.png
```
## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
