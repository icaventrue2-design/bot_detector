import html
import re
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def clean_comment_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Декодируем HTML-сущности (например, &amp;, &#39;)
    text = html.unescape(text)
    # Приводим к нижнему регистру
    text = text.lower()
    # Заменяем URL-адреса на маркер url_token (наличие ссылки — важный признак спам-бота)
    text = re.sub(r"https?://\S+|www\.\S+", " url_token ", text)
    # Удаляем HTML-теги (например, <br />, <a href...>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Оставляем буквы (латиница + кириллица), цифры и базовые токены
    text = re.sub(r"[^a-zа-яё0-9_]+", " ", text)
    # Убираем множественные пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["preprocess"]

    df_raw = pd.read_csv("data/raw/train_raw.csv", encoding="utf-8")
    feedback_path = Path("data/raw/verified_feedback.csv")

    # Если пользователи уже верифицировали комментарии через дашборд, добавляем их в выборку
    if feedback_path.exists():
        df_feedback = pd.read_csv(feedback_path, encoding="utf-8")
        if not df_feedback.empty:
            print(f"Добавлено {len(df_feedback)} верифицированных записей из обратной связи.")
            df_raw = pd.concat([df_raw, df_feedback], ignore_index=True)
            df_raw = df_raw.drop_duplicates(subset=["comment_text"], keep="last")

    # Удаляем пропуски и очищаем текст (ФТ-1)
    df_raw = df_raw.dropna(subset=["comment_text", "label"]).copy()
    df_raw["clean_text"] = df_raw["comment_text"].apply(clean_comment_text)

    # Фильтруем пустые или слишком короткие строки после очистки
    df_raw = df_raw[df_raw["clean_text"].str.len() >= params["min_text_length"]].copy()
    df_raw["label"] = df_raw["label"].astype(int)

    # Стратифицированное разбиение на обучающую и тестовую выборки
    train_df, test_df = train_test_split(
        df_raw,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df_raw["label"],
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")
    print(f"Предобработка завершена. Train: {len(train_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()