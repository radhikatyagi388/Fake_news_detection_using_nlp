from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from text_utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"

def load_data():
    fake = pd.read_csv(FAKE_PATH)
    true = pd.read_csv(TRUE_PATH)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    df = df.drop_duplicates()
    df = df.dropna(subset=["text"])

    # Subject is intentionally NOT used as a feature.
    # Only title + text are used.
    df["content"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    )
    df["content"] = df["content"].apply(clean_text)

    df = df[df["content"].str.split().str.len() >= 15]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )

def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Linear SVM": LinearSVC(random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Passive Aggressive": PassiveAggressiveClassifier(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=30, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
    }

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("tfidf", build_vectorizer()),
        ("model", model)
    ])

    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    pred = pipe.predict(X_test)
    predict_time = time.perf_counter() - start

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "Training Time (sec)": train_time,
        "Prediction Time (sec)": predict_time,
    }

    return pipe, result, pred

def main():
    print("Loading and cleaning ISOT dataset...")
    df = load_data()

    print(f"Rows after cleaning: {len(df):,}")
    print(df["label"].value_counts().rename(index={0: "Fake", 1: "Real"}))

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    results = []
    models = build_models()

    best_pipe = None
    best_f1 = -1
    best_name = None

    for name, model in models.items():
        print(f"\nTraining: {name}")
        pipe, result, pred = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results.append(result)

        print(pd.Series(result).to_string())

        if result["F1"] > best_f1:
            best_f1 = result["F1"]
            best_pipe = pipe
            best_name = name

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # Final pipeline:
    # Logistic Regression is used when confidence/probabilities are required.
    # We select the best of the six for the comparison report, but the app
    # requires predict_proba, so Logistic Regression is used as the deployable model.
    final_model = Pipeline([
        ("tfidf", build_vectorizer()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    final_model.fit(X_train, y_train)

    test_pred = final_model.predict(X_test)
    report = classification_report(
        y_test, test_pred, target_names=["Fake", "Real"], output_dict=True
    )

    cm = confusion_matrix(y_test, test_pred).tolist()

    cv_scores = cross_val_score(
        final_model, X_train, y_train, cv=5, scoring="f1"
    )

    joblib.dump(final_model, MODEL_DIR / "fake_news_pipeline.pkl")

    metadata = {
        "deployable_model": "Logistic Regression",
        "best_comparison_model": best_name,
        "best_comparison_f1": float(best_f1),
        "final_f1": float(f1_score(y_test, test_pred)),
        "final_accuracy": float(accuracy_score(y_test, test_pred)),
        "final_precision": float(precision_score(y_test, test_pred)),
        "final_recall": float(recall_score(y_test, test_pred)),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "confusion_matrix": cm,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n===== FINAL RESULT =====")
    print(results_df.to_string(index=False))
    print("\nFinal Logistic Regression metrics:")
    print(json.dumps(metadata, indent=2))
    print("\nSaved:")
    print(MODEL_DIR / "fake_news_pipeline.pkl")
    print(REPORT_DIR / "model_comparison.csv")
    print(REPORT_DIR / "metrics.json")

if __name__ == "__main__":
    main()
