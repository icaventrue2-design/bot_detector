from pathlib import Path
import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["train"]

    train_df = pd.read_csv("data/processed/train.csv", encoding="utf-8")
    X_train = train_df["clean_text"].fillna("")
    y_train = train_df["label"].astype(int)

    # Создаем единый конвейер: TF-IDF векторизация + Логистическая регрессия
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=params["max_features"],
                    ngram_range=tuple(params["ngram_range"]),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=params["c_reg"],
                    max_iter=params["max_iter"],
                    class_weight="balanced",
                    random_state=params["random_state"],
                ),
            ),
        ]
    )

    print("Обучение модели TF-IDF + LogisticRegression...")
    pipeline.fit(X_train, y_train)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "baseline_tfidf_logreg.pkl"

    joblib.dump(pipeline, model_path)
    print(f"Модель успешно сохранена в {model_path}")


if __name__ == "__main__":
    main()