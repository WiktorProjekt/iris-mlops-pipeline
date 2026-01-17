#1. Wybór obrazu bazowego
FROM python:3.9-slim

#2. Ustawienie katalogu roboczego wewnątrz kontenera
WORKDIR /app

#3. Kopiowanie pliku z zależnościami
COPY requirements.txt .

#4. Instalacja bibliotek
RUN pip install --no-cache-dir -r requirements.txt

#5. Kopiowanie całego kodu źródłowego do kontenera
COPY src/ src/

#6. Utworzenie katalogu na model (jeśli nie istnieje)
RUN mkdir -p model

#7. Zmienna środowiskowa dla Pythona
ENV PYTHONUNBUFFERED=1

#8. Komenda startowa
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]