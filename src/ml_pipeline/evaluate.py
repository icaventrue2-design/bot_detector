import json
from pathlib import Path
import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["evaluate"]

    threshold = params["bot_threshold"]

    test_df = pd.read_csv("data/processed/test.csv", encoding="utf-8")
    X_test = test_df["clean_text"].fillna("")
    y_true = test_df["label"].astype(int)

    pipeline = joblib.load("models/baseline_tfidf_logreg.pkl")

    # Получаем вероятности класса 1 (Бот)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_probs)), 4),
        "confusion_matrix": {
            "true_negatives_human": int(tn),
            "false_positives_bot": int(fp),
            "false_negatives_human": int(fn),
            "true_positives_bot": int(tp),
        },
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "evaluation.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("Результаты оценки на тестовой выборке:")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()