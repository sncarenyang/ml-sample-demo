# A ML classification demo on the Titanic dataset(using scikit-learn)


from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess():
    np.random.seed(42)

    N = 1000
    age = np.random.normal(30, 15, N)
    sex = np.random.choice(["male", "female"], N)
    pclass = np.random.choice([1, 2, 3], N)

    survived_logic = ((age < 40) & (sex == "female")) | (age < 20)
    survived = np.random.binomial(1, 0.5 + 0.2 * survived_logic.astype(float), N)

    df = pd.DataFrame({
        "Age": age,
        "Sex": sex,
        "Pclass": pclass,
        "Survived": survived
    })

    le = LabelEncoder()
    df["Sex_encoded"] = le.fit_transform(df["Sex"])

    X = df[["Age", "Sex_encoded", "Pclass"]]
    y = df["Survived"]

    return X, y, df

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, save_dir):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    filename = f"{save_dir}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    save_dir = "/content/drive/MyDrive/ml-sample-demo"
    os.makedirs(save_dir, exist_ok=True)

    X, y, df = load_and_preprocess()

    print("Dataset shape:", X.shape)
    print("\nDataset info:")
    print(df.head())

    csv_path = f"{save_dir}/synthetic_titanic_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    train_and_evaluate(X_train, X_test, y_train, y_test, "Logistic Regression", lr, save_dir)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate(X_train, X_test, y_train, y_test, "Random Forest", rf, save_dir)

    print(f"\nAll files saved to Google Drive folder: {save_dir}")

    #####download output to my local machine######
    #from google.colab import files
    #files.download('synthetic_titanic_sample.csv')
    #files.download('confusion_matrix_logistic_regression.png')
    #files.download('confusion_matrix_random_forest.png')


if __name__ == "__main__":
    main()
