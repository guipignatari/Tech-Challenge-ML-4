# Tech Challenge - Fase 04 | Machine Learning Engineering

## 📌 Sobre o Projeto
Este projeto é a entrega final do Tech Challenge da Fase 4, focado na criação de um modelo preditivo utilizando redes neurais de Deep Learning. O objetivo principal é prever o valor de fechamento das ações da Apple (AAPL) com base em dados históricos, utilizando a arquitetura Long Short-Term Memory (LSTM). 

Além da construção e treinamento do modelo, o projeto contempla o desenvolvimento completo da pipeline de produção: desde a coleta de dados até o deploy do modelo em uma API RESTful utilizando FastAPI, conteinerizada via Docker.

## 🚀 Tecnologias Utilizadas
* **Linguagem:** Python
* **Coleta de Dados:** `yfinance`
* **Machine Learning/Deep Learning:** `TensorFlow` / `Keras` (LSTM), `scikit-learn`
* **Manipulação e Visualização:** `pandas`, `numpy`, `matplotlib`
* **API:** `FastAPI`, `Uvicorn`, `pydantic`
* **Deploy:** Docker

## 📁 Estrutura do Repositório
* `extract_aapl.py`: Script responsável pela coleta dos dados no Yahoo Finance, pré-processamento, normalização, treinamento do modelo LSTM e cálculo das métricas de avaliação (MAE, RMSE, MAPE).
* `app.py`: Código-fonte da API desenvolvida em FastAPI para servir o modelo em produção.
* `modelo_aapl_lstm.keras`: O modelo neural treinado e exportado para inferência.
* `scaler_aapl.gz`: O objeto normalizador (MinMaxScaler) salvo, necessário para processar novas entradas do usuário da mesma forma que os dados de treino.
* `requirements.txt`: Documento listando todas as dependências e versões exatas do projeto.
* `Dockerfile`: Arquivo de instruções para a criação da imagem do contêiner Docker.

## ⚙️ Como Executar o Projeto

### Pré-requisitos
* **Docker** (ex: Docker Desktop) instalado e rodando em sua máquina.
* **Git** para clonar o repositório.

### Passo a Passo (Deploy com Docker)

1.  **Clone o repositório:**
    ```bash
    git clone git@github.com:guipignatari/Tech-Challenge-ML-4.git
    ```

2.  **Construa a imagem Docker:**
    No terminal, dentro da pasta do projeto, execute o comando abaixo para construir a imagem base da aplicação. Esse passo instalará todas as dependências isoladamente.
    ```bash
    docker build -t api-lstm-apple .
    ```

3.  **Execute o contêiner:**
    Suba a aplicação mapeando a porta 8000 local para a porta do contêiner.
    ```bash
    docker run -p 8000:8000 api-lstm-apple
    ```

4.  **Acesse a API:**
    Com o servidor ativo, abra o navegador e acesse a documentação interativa gerada automaticamente pelo Swagger UI:
    👉 **http://127.0.0.1:8000/docs**

## 🧪 Como Testar a API
A API foi desenvolvida para permitir que o usuário forneça uma sequência de dados históricos e receba a previsão futura. 

Na interface do Swagger:
1. Expanda a rota verde `POST /prever`.
2. Clique em **Try it out**.
3. A API exige uma janela de exatos 60 dias de preços para realizar o cálculo. Substitua o conteúdo do *Request body* pelo exemplo JSON abaixo e clique em **Execute**:

**Exemplo de Payload de Entrada:**
```json
{
  "historico_precos": [
    180.5, 181.2, 182.0, 181.5, 180.8, 179.5, 178.2, 177.0, 176.5, 175.8,
    174.2, 175.0, 176.2, 177.5, 178.0, 179.2, 180.5, 181.8, 182.5, 183.0,
    184.2, 185.0, 184.5, 183.8, 182.5, 181.0, 180.2, 179.5, 178.8, 177.5,
    176.0, 175.2, 174.5, 175.8, 176.5, 177.2, 178.0, 179.5, 180.2, 181.5,
    182.8, 183.5, 184.0, 185.2, 186.0, 185.5, 184.8, 183.5, 182.0, 181.2,
    180.5, 179.8, 178.5, 177.0, 176.2, 175.5, 176.8, 177.5, 178.2, 179.0
  ]
}
```

A API processará o tensor, reverterá a normalização e retornará um JSON com o valor do preço de fechamento previsto (em dólares) para o próximo dia útil.

<img width="1403" height="161" alt="image" src="https://github.com/user-attachments/assets/604a9b06-cbaf-4c7c-90c0-6d92f074b025" />

## 👤 Autor

LinkedIn: [linkedin.com/in/guilhermepignatari](https://linkedin.com/in/guilhermepignatari)
GitHub: [github.com/guipignatari](https://github.com/guipignatari)
