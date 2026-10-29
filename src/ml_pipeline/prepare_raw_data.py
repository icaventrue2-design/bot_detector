import io
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import ssl
import certifi

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
RAW_DIR = Path("data/raw")
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def download_and_merge_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание датасета YouTube Spam Collection из {UCI_ZIP_URL}...")

    response = urllib.request.urlopen(UCI_ZIP_URL)
    zip_bytes = io.BytesIO(response.read())

    frames = []
    with zipfile.ZipFile(zip_bytes) as zf:
        for filename in zf.namelist():
            if filename.endswith(".csv") and not filename.startswith("__MACOSX"):
                # Извлекаем название видео/темы из имени файла (например, Youtube01-Psy.csv -> Psy)
                topic_name = Path(filename).stem.split("-")[-1]
                with zf.open(filename) as f:
                    df_part = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
                    df_part["topic_id"] = topic_name
                    frames.append(df_part)

    df_full = pd.concat(frames, ignore_index=True)

    # Приводим названия столбцов к архитектуре нашей системы
    df_full = df_full.rename(
        columns={
            "COMMENT_ID": "comment_id",
            "AUTHOR": "author_id",
            "DATE": "created_at",
            "CONTENT": "comment_text",
            "CLASS": "label",  # 1 - Бот/Спам, 0 - Реальный пользователь
        }
    )

    # Оставляем целевые колонки
    cols = ["comment_id", "author_id", "topic_id", "created_at", "comment_text", "label"]
    df_full = df_full[cols]

    raw_train_path = RAW_DIR / "train_raw.csv"
    df_full.to_csv(raw_train_path, index=False, encoding="utf-8")
    print(f"Стартовый датасет сохранен в {raw_train_path} (всего записей: {len(df_full)})")
    print(f"Распределение классов:\n{df_full['label'].value_counts()}")

    # Инициализируем файл для ручной верификации (дообучения с дашборда), если его еще нет
    feedback_path = RAW_DIR / "verified_feedback.csv"
    if not feedback_path.exists():
        pd.DataFrame(columns=cols).to_csv(feedback_path, index=False, encoding="utf-8")
        print(f"Создан шаблон файла обратной связи: {feedback_path}")


if __name__ == "__main__":
    download_and_merge_dataset()