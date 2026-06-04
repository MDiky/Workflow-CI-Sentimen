import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def latih_model_ci():
    print("Memulai proses CI: Melatih model...")
    df = pd.read_csv("dataset_shopee_ready.csv")
    X = df['teks_bersih'].fillna('')
    y = df['sentimen']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    print("Pelatihan selesai.")

    artifact_path = "model_artifact"
    if os.path.exists(artifact_path):
        shutil.rmtree(artifact_path)

    mlflow.sklearn.save_model(pipeline, artifact_path)
    print(f"Artefak model berhasil disimpan di folder: {artifact_path}")


if __name__ == "__main__":
    latih_model_ci()