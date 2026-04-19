
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
## Example Output

![/imageA](/images/confusion_matrix_logistic_regression.png)

![/imageB](/images/confusion_matrix_random_forest.png)

## How to Run in Google Colab
If you are running this project in Google Colab, first mount Google Drive:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
Then run the script. The output files will be saved to:
```bash
/MyDrive/ml-sample-demo/
```
Generated files:
- `synthetic_titanic_sample.csv`
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_random_forest.png`

## How to run locally
```bash
git clone https://github.com/sncarenyang/ml-sample-demo.git
cd ml-sample-demo

python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
python titanic_ml_demo.py
```

## Model Workflow
**1. Data Generation**
A synthetic dataset is created with these columns:
- `Age`
- `Sex`
-  `Pclass`
-  `Survived`
  
**2. Preprocessing**
- `Sex` is converted into a numeric feature using label encoding.
- Features and labels are split into `X` and `y`.

**3. Training**
Two models are trained and evaluated:
- Logistic Regression
- Random Forest
  
**4. Evaluation**
The script prints:
- Dataset shape 
- A preview of the data
- Accuracy
- Classification report
It also saves confusion matrix plots for both models.

## Results
In a typical run, both models produce reasonable baseline results on the synthetic dataset.  
Because the dataset is generated artificially, the scores are meant for demonstration rather than benchmarking.

##  License & Disclaimer
- The code is released under the MIT License.
- This is a demo project for learning and portfolio purposes.
- The synthetic dataset does not represent the real Titanic dataset.
- The goal is to demonstrate a standard ML workflow in a clean, reproducible way.
  





