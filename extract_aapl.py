import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math
import joblib

# (Extração -> Pré-processamento -> Treinamento)

# Apple
symbol = 'AAPL'

# Data início e fim
start_date = '2018-01-01'
end_date = '2026-05-18' 

# Usando a função download para obter os dados
print(f"Baixando dados para {symbol}...")
df = yf.download(symbol, start=start_date, end=end_date)

# Print primeira e última linha
print(df.head())
print(df.tail())

# Plotando o histórico de fechamento para visualização inicial
plt.figure(figsize=(14, 6))
plt.plot(df['Close'], label='Preço de Fechamento (USD)')
plt.title(f'Histórico de Preço de Fechamento - {symbol}')
plt.xlabel('Data')
plt.ylabel('Preço (USD)')
plt.legend()
# plt.show()

#Pré-processamento

# 1. Isolando os valores de Fechamento (Close)
dataset = df['Close'].values 

# 2. Normalizando os dados para a escala entre 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 3. Criando as janelas temporais (60 dias de histórico para prever o próximo)
window_size = 60
X = []
y = []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0]) # Pega do dia (i-60) até o dia anterior a (i)
    y.append(scaled_data[i, 0])               # Pega o dia (i) como resposta (target)

# Convertendo para arrays do numpy
X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Dados de Treino (80%) e Teste (20%)
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Formato dos dados de Treino (X): {X_train.shape}")
print(f"Formato dos dados de Teste (X): {X_test.shape}")

#Treinamento

# 1. Construção da Arquitetura do Modelo
model = Sequential()

# Primeira camada LSTM (return_sequences=True)
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2)) # Desliga 20% dos neurônios aleatoriamente para evitar overfitting

# Segunda camada LSTM (return_sequences=False)
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Camada saida
model.add(Dense(units=1))

# Compilando o modelo com o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(optimizer='adam', loss='mean_squared_error')

# 2. Treinamento do Modelo
print("Iniciando o treinamento do modelo LSTM...")
# epochs=20 (vai passar pelos dados 20 vezes) e batch_size=32 (atualiza os pesos a cada 32 amostras)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

print("Treinamento concluído com sucesso!")

# Avaliação

# 1. Gerando as previsões com os dados de teste
print("Gerando previsões com o modelo LSTM...")
predictions = model.predict(X_test)

# 2. Revertendo a normalização para voltar à escala original (Dólares)
# As previsões do modelo
predictions_usd = scaler.inverse_transform(predictions)

y_test_usd = scaler.inverse_transform(y_test.reshape(-1, 1))

# 3. Calculando as Métricas Exigidas
mae = mean_absolute_error(y_test_usd, predictions_usd)
rmse = math.sqrt(mean_squared_error(y_test_usd, predictions_usd))
mape = mean_absolute_percentage_error(y_test_usd, predictions_usd)

print("\n--- Métricas de Avaliação ---")
print(f"MAE (Erro Absoluto Médio): ${mae:.2f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): ${rmse:.2f}")
print(f"MAPE (Erro Percentual Absoluto Médio): {mape * 100:.2f}%")

# 4. Plotando o Gráfico Comparativo: Preço Real vs Previsão
plt.figure(figsize=(14, 6))
plt.plot(y_test_usd, color='blue', label='Preço Real da Apple (Teste)')
plt.plot(predictions_usd, color='red', label='Previsão do Modelo LSTM')
plt.title('Comparação: Preço Real vs Previsão (AAPL)')
plt.xlabel('Dias (Período de Teste)')
plt.ylabel('Preço de Fechamento (USD)')
plt.legend()
plt.show()

# Save modelo

# 1. Salvando o modelo treinado em Keras
model.save('modelo_aapl_lstm.keras')
print("Modelo LSTM salvo com sucesso como 'modelo_aapl_lstm.keras'!")

# 2. Salvando o scaler (normalizador)
joblib.dump(scaler, 'scaler_aapl.gz')
print("Scaler salvo com sucesso como 'scaler_aapl.gz'!")
