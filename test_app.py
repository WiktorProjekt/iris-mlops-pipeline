from fastapi.testclient import TestClient
from app import app

# Inicjalizacja klienta testowego, który symuluje przeglądarkę
client = TestClient(app)


def test_read_root():
    """
    Test 1: Sprawdzenie dostępności API (Health Check).
    Cel: Upewnienie się, że serwer startuje i odpowiada na żądania.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "System działa poprawnie"


def test_predict_iris_valid():
    """
    Test 2: Sprawdzenie poprawności predykcji dla danych poprawnych.
    Cel: Weryfikacja, czy model potrafi przetworzyć poprawny wektor
    """
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    # Oczekujemy, że odpowiedź zawiera klucz
    assert "prediction" in response.json()
    # Oczekujemy, że wynik jest tekstem
    assert isinstance(response.json()["prediction"], str)


def test_predict_iris_invalid_data():
    """
    Test 3: Sprawdzenie walidacji danych wejściowych.
    Cel: Upewnienie się, że system nie ulegnie awarii po otrzymaniu błędnych danych.
    """
    payload = {
        "sepal_length": "BŁĄD",  # Przesłanie tekstu zamiast liczby
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)

    # Oczekujemy kodu 422 (Unprocessable Entity) - błąd walidacji
    assert response.status_code == 422
