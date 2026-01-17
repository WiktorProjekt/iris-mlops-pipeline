import os
import joblib
import json
import logging
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train():
    # 1. Wczytanie danych
    logger.info("Wczytywanie zbioru danych Iris...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Podział danych zakończony. Próbki treningowe: {len(X_train)}, testowe: {len(X_test)}")

    # 3. Trening modelu
    logger.info("Rozpoczynanie treningu modelu Regresji Logistycznej...")
    model = LogisticRegression(max_iter=200, solver='lbfgs')
    model.fit(X_train, y_train)

    # 4. Ewaluacja
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Model wytrenowany. Uzyskana dokładność (Accuracy): {accuracy:.4f}")

    # 5. Zapis metryk do pliku (ważne dla CI/CD!)
    metrics = {"accuracy": accuracy}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # 6. Zapis modelu (Artefakt)
    if not os.path.exists("model"):
        os.makedirs("model")

    model_path = "model/model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model zapisano w ścieżce: {model_path}")


if __name__ == "__main__":
    train()