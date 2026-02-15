# Wybór obrazu bazowego
FROM python:3.11-slim

# Ustawienie katalogu roboczego wewnątrz kontenera
WORKDIR /app

# Kopiowanie pliku z zależnościami
COPY requirements.txt .

# Instalacja bibliotek
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie całego kodu źródłowego do kontenera
COPY . .

RUN python train_model.py

# Utworzenie katalogu na model (jeśli nie istnieje)
RUN mkdir -p model

# Zmienna środowiskowa dla Pythona
ENV PYTHONUNBUFFERED=1

# Komenda startowa
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]