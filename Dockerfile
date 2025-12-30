# Temel imaj
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Gereksinimleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Gradio portu
EXPOSE 7860

# Çalıştırma
CMD ["python", "app.py"]
