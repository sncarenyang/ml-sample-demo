#  ML Sample Demo (Titanic-style Classification)

This project demonstrates a complete machine learning workflow using a synthetic Titanic-style dataset. The demo built with Python, pandas, scikit-learn, Matplotlib, and Seaborn.

## 🧠 Overview

This demo covers the full pipeline:

- Data generation  
- Data preprocessing  
- Model training  
- Model evaluation and comparison  

The dataset used in this project is synthetic, so no external data download is required.

## 📊 Features

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

## 🏗️ Project Structure

```text
ml-sample-demo/
├── titanic_ml_demo.py
├── README.md
├── requirements.txt
├── synthetic_titanic_sample.cvs
└── images/
    ├── logistic_regression_data.png
    ├── random_forest_data.png
    ├── confusion_matrix_logistic_regression.png
    └── confusion_matrix_random_forest.png
```

## 🧰 Tech Stack

- Python  
- NumPy  
- Pandas  
- Scikit-learn
- Matplotlib



## 🔍 How to Run in Google Colab
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

## 🔍 How to run locally
```bash
git clone https://github.com/sncarenyang/ml-sample-demo.git
cd ml-sample-demo

python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
python titanic_ml_demo.py
```

## 🚀 Model Workflow

**1. Data Generation**
The workflow begins by generating a synthetic dataset inspired by the Titanic survival problem. 

a synthetic dataset is created with 1000 rows × 4 columns first:
- `Age`
- `Sex`
-  `Pclass`
-  `Survived`
  
and then the Sex column is coverted into a numerica feature using label encoding to generate the 5th column:
-  `Sex_encoded`

A **1000 raws * 5 columns** dataset is completed. This dataset includes both numerical and categorical features, making it suitable for supervised learning tasks

**2. Training**
Two models are trained and evaluated:
- Logistic Regression
- Random Forest
  
**3. Evaluation**
The script prints:
- Dataset shape 
- A preview of the data
- Accuracy
- Classification report
It also saves confusion matrix plots for both models.

## 🔬 Results

In a typical run, both models produce reasonable baseline results on the synthetic dataset.  
Because the dataset is generated artificially, the scores are meant for demonstration rather than benchmarking.

📈 Logistic Regression
![Logistic Regression data](/images/logistic_regression_data.png)

![Logistic Regression](/images/confusion_matrix_logistic_regression.png)

**Performance:**
- Accuracy: 0.635  
- Strong recall for "Survived" (0.98)  
- Very poor recall for "Not Survived" (0.09)
- The model tends to classify most samples as "Survived"

---

📈  Random Forest
![Random_forest data](/images/random_forest_data.png)
![Random Forest](/images/confusion_matrix_random_forest.png)

**Performance:**
- Accuracy: 0.555  
- More balanced recall across classes  
- Lower overall performance compared to Logistic Regress
  
## 🧠 Key Insights

- Logistic Regression achieves higher accuracy  
- However, it is **heavily biased toward the "Survived" class**  
- Random Forest provides a more balanced classification  

 ⚠️ **Accuracy alone is not sufficient to evaluate model performance**

---

## 📑 Classification Report Summary

### Logistic Regression
- Precision (Not Survived): 0.78  
- Recall (Not Survived): 0.09 ❗  
- Recall (Survived): 0.98  

👉 Indicates strong imbalance in predictions  

---

### Random Forest
- Precision (Not Survived): 0.43  
- Recall (Not Survived): 0.46  
- Recall (Survived): 0.61  

👉 More stable across classes  

---

## 🎯 Conclusion

This project demonstrates that:

- Model selection affects not only accuracy but also class behavior  
- Evaluation metrics such as **recall and precision are critical**  
- Balanced performance may be preferable depending on application  

---

## 💡 Highlights
 - End-to-end ML pipeline
 - Model comparison (Logistic Regression vs Random Forest)
 - Clear demonstration of model bias
 - Practical understanding of evaluation metric

## 👩‍💻 Author
Shi-Ning Caren Yang


## 🌐 License & Disclaimer
- The code is released under the MIT License.
- This is a demo project for learning and portfolio purposes.
- The synthetic dataset does not represent the real Titanic dataset.
- The goal is to demonstrate a standard ML workflow in a clean, reproducible way.
  





