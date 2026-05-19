from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Inicializando a aplicação
app = FastAPI(title="API de Previsão de Ações - LSTM", description="Tech Challenge - Fase 4")

# Carregando o modelo e o scaler na memória ao iniciar a API
try:
    model = load_model('modelo_aapl_lstm.keras')
    scaler = joblib.load('scaler_aapl.gz')
    print("Modelo e Scaler carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar arquivos do modelo: {e}")

# Definindo o formato de entrada que a API espera do usuário
class StockData(BaseModel):
    # A API espera receber uma lista de 60 preços de fechamento (float)
    historico_precos: list[float]  

@app.post("/prever")
def prever_preco(dados: StockData):
    precos = dados.historico_precos
    
    # Validando o input da janela de 60 dias exigido
    if len(precos) != 60:
        raise HTTPException(
            status_code=400, 
            detail=f"Por favor, forneça exatamente 60 dias de histórico. Você enviou {len(precos)}."
        )
    
    try:
        # Transformando a lista em um array numpy no formato vertical
        precos_array = np.array(precos).reshape(-1, 1)
        
        # Normalizando os dados usando scaler do treinamento
        precos_scaled = scaler.transform(precos_array)
        
        # Formatando para o tensor do LSTM (1 amostra, 60 timesteps, 1 feature)
        X_input = np.reshape(precos_scaled, (1, 60, 1))
        
        # previsão
        previsao_scaled = model.predict(X_input)
        
        # Revertendo a normalização para obter o valor em dólares
        previsao_usd = scaler.inverse_transform(previsao_scaled)
        
        return {
            "status": "sucesso",
            "previsao_proximo_dia_usd": round(float(previsao_usd[0][0]), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a previsão: {str(e)}")

# uvicorn app:app --reload

# http://127.0.0.1:8000/docs
