FROM python:3.13-slim

WORKDIR /app

# Copiar o requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos do projeto para o contêiner
# (Isso vai incluir o app.py, o modelo_aapl_lstm.keras e o scaler_aapl.gz)
COPY . .

# Expor a porta que o FastAPI vai rodar
EXPOSE 8000

# Comando para iniciar o servidor do Uvicorn dentro do Docker
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]