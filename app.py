import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np


# 1. Definicja schematu danych wejściowych (Data Validation)
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# 2. Inicjalizacja aplikacji FastAPI
app = FastAPI(
    title="Iris Model API",
    description="API do klasyfikacji gatunków irysów w procesie MLOps",
    version="1.0.0"
)

# 3. Wczytanie wytrenowanego modelu przy starcie aplikacji
# Model jest ładowany raz, aby nie obciążać dysku przy każdym zapytaniu
try:
    model = joblib.load("model/model.pkl")
    class_names = ["Setosa", "Versicolor", "Virginica"]
except Exception as e:
    print(f"Błąd ładowania modelu: {e}")
    model = None


# 4. Definicja endpointu głównego (Health Check)
@app.get("/")
def read_root():
    return {"status": "System działa poprawnie", "model_loaded": model is not None}


# 5. Definicja endpointu predykcyjnego
@app.post("/predict")
def predict(request: IrisRequest):
    if not model:
        return {"error": "Model nie jest dostępny. Uruchom skrypt trenujący."}

    # Przygotowanie danych do formatu akceptowanego przez Scikit-Learn (tablica 2D)
    data = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    # Wykonanie predykcji
    prediction_index = model.predict(data)[0]
    prediction_name = class_names[prediction_index]

    return {
        "prediction": prediction_name,
        "input_data": request.model_dump()
    }


# Kod uruchamiający serwer (tylko przy uruchomieniu lokalnym)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
