# ============================================================
# PROJETO DE PREVISÃO DE PREÇOS - PETR4.SA
# Modelos: Auto-ARIMA (se disponível) / ARIMA, Random Forest e LSTM
# Treino: até 2023 | Teste: 2024 | Produção: 2025
# Observação: lógica original mantida 100%, apenas reorganizado
# ============================================================

import os
import random
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Tentativa de carregar o auto_arima (pmdarima)
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

# ------------------------------------------------------------
# 1. BAIXAR OS DADOS
# ------------------------------------------------------------
print("Baixando dados da PETR4...")
df = yf.download("PETR4.SA", start="2010-01-01", end="2025-12-31", progress=False)

if df.empty:
    raise ValueError("Erro: dados retornaram vazios.")

# Normalização de colunas
if "Close" not in df.columns and "Adj Close" in df.columns:
    df["Close"] = df["Adj Close"]

df = df[["Close", "Volume"]].dropna()
df.index = pd.to_datetime(df.index)

# ------------------------------------------------------------
# 2. DEFINIR INTERVALOS
# ------------------------------------------------------------
DATA_TREINO_INICIO = "2010-01-01"
DATA_TREINO_FIM    = "2023-12-31"

DATA_TESTE_INICIO  = "2024-01-01"
DATA_TESTE_FIM     = "2024-12-31"

DATA_PROD_INICIO   = "2025-01-01"
DATA_PROD_FIM      = "2025-12-31"

# Separação dos datasets
train = df.loc[DATA_TREINO_INICIO:DATA_TREINO_FIM].copy()
test  = df.loc[DATA_TESTE_INICIO:DATA_TESTE_FIM].copy()
prod  = df.loc[DATA_PROD_INICIO:DATA_PROD_FIM].copy()

print(f"Registros — Total: {len(df)} | Treino: {len(train)} | Teste: {len(test)} | Produção: {len(prod)}")

# ------------------------------------------------------------
# 3. FUNÇÃO DE AVALIAÇÃO
# ------------------------------------------------------------
def avaliar(real, prev):
    min_len = min(len(real), len(prev))
    real = np.array(real)[:min_len]
    prev = np.array(prev)[:min_len]
    return {
        "rmse": mean_squared_error(real, prev) ** 0.5,
        "mape": mean_absolute_percentage_error(real, prev),
        "mae": mean_absolute_error(real, prev),
        "n": min_len
    }

# ------------------------------------------------------------
# 4. MODELO ARIMA
# ------------------------------------------------------------
print("\n== ARIMA ==")
y_train = train["Close"]

if PMDARIMA_AVAILABLE:
    print("Rodando auto_arima...")
    arima_auto = auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )
    arima_order = arima_auto.order
    print("Order encontrado:", arima_order)

    arima_pred_2024 = arima_auto.predict(n_periods=len(test))
    arima_pred_2025 = arima_auto.predict(n_periods=len(prod))

else:
    print("pmdarima indisponível. Usando fallback ARIMA(5,1,2)")
    arima_model = ARIMA(y_train, order=(5,1,2)).fit()
    arima_pred_2024 = arima_model.forecast(steps=len(test))
    arima_pred_2025 = arima_model.forecast(steps=len(prod))

arima_pred_2024 = np.array(arima_pred_2024)[:len(test)]
arima_pred_2025 = np.array(arima_pred_2025)[:len(prod)]

# ------------------------------------------------------------
# 5. MODELO RANDOM FOREST
# ------------------------------------------------------------
print("\n== Random Forest ==")

rf_df = df.copy()

# Features + Lags
rf_df["Return"] = rf_df["Close"].pct_change()
rf_df["MMA_7"]  = rf_df["Close"].rolling(7).mean()
rf_df["MMA_21"] = rf_df["Close"].rolling(21).mean()
rf_df["Vol_21"] = rf_df["Return"].rolling(21).std()

for lag in [1,2,3,5,7,10]:
    rf_df[f"lag_{lag}"] = rf_df["Close"].shift(lag)

# RSI
delta = rf_df["Close"].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
ema_up = up.rolling(14).mean()
ema_down = down.rolling(14).mean()
rf_df["RSI_14"] = 100 - (100 / (1 + (ema_up / (ema_down + 1e-9))))

rf_df.dropna(inplace=True)

rf_train = rf_df.loc[DATA_TREINO_INICIO:DATA_TREINO_FIM]
rf_test  = rf_df.loc[DATA_TESTE_INICIO:DATA_TESTE_FIM]
rf_prod  = rf_df.loc[DATA_PROD_INICIO:DATA_PROD_FIM]

features = [c for c in rf_df.columns if c != "Close"]

X_train = rf_train[features]
X_test  = rf_test[features]
X_prod  = rf_prod[features]
y_train_rf = rf_train["Close"]
y_test_rf  = rf_test["Close"]

rf_model = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=SEED)
rf_model.fit(X_train, y_train_rf)

rf_pred_2024 = rf_model.predict(X_test)
rf_pred_2025 = rf_model.predict(X_prod)

# ------------------------------------------------------------
# 6. MODELO LSTM
# ------------------------------------------------------------
print("\n== LSTM ==")

seq_len = 60
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[["Close"]].values)

X_all, y_all, date_all = [], [], []

for i in range(seq_len, len(scaled_close)):
    X_all.append(scaled_close[i-seq_len:i])
    y_all.append(scaled_close[i][0])
    date_all.append(df.index[i])

X_all = np.array(X_all)
y_all = np.array(y_all)
date_all = np.array(date_all)

seq_df = pd.DataFrame({"date": date_all})
seq_df["is_train"] = seq_df["date"].between(DATA_TREINO_INICIO, DATA_TREINO_FIM)
seq_df["is_test"]  = seq_df["date"].between(DATA_TESTE_INICIO, DATA_TESTE_FIM)
seq_df["is_prod"]  = seq_df["date"].between(DATA_PROD_INICIO, DATA_PROD_FIM)

idx_train_seq = seq_df[seq_df["is_train"]].index.values
idx_test_seq  = seq_df[seq_df["is_test"]].index.values
idx_prod_seq  = seq_df[seq_df["is_prod"]].index.values

X_train_lstm = X_all[idx_train_seq]
y_train_lstm = y_all[idx_train_seq]
X_test_lstm  = X_all[idx_test_seq]
y_test_lstm  = y_all[idx_test_seq]
X_prod_lstm  = X_all[idx_prod_seq]

print(f"LSTM samples => treino: {len(X_train_lstm)}, teste: {len(X_test_lstm)}, prod: {len(X_prod_lstm)}")

lstm_pred_2024 = np.array([])
lstm_pred_2025 = np.array([])

if len(X_train_lstm) > 0:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_lstm, y_train_lstm, epochs=15, batch_size=32, verbose=1)

    if len(X_test_lstm) > 0:
        lstm_pred_2024 = scaler.inverse_transform(model.predict(X_test_lstm)).flatten()

    if len(X_prod_lstm) > 0:
        lstm_pred_2025 = scaler.inverse_transform(model.predict(X_prod_lstm)).flatten()

# ------------------------------------------------------------
# 7. CONSTRUIR SÉRIES PARA 2025
# ------------------------------------------------------------
arima_series_2025 = pd.Series(arima_pred_2025, index=prod.index)
rf_series_2025    = pd.Series(rf_pred_2025, index=prod.index)
lstm_series_2025  = pd.Series(lstm_pred_2025, index=date_all[idx_prod_seq])

# ------------------------------------------------------------
# 8. PLOT (e SALVAR FIGURA)
# ------------------------------------------------------------
plt.figure(figsize=(15,7))

if len(prod) > 0:
    plt.plot(prod.index, prod["Close"], label="Real (2025)", color="black", linewidth=2)

plt.plot(arima_series_2025.index, arima_series_2025.values, label="ARIMA 2025")
plt.plot(rf_series_2025.index, rf_series_2025.values, label="Random Forest 2025")
plt.plot(lstm_series_2025.index, lstm_series_2025.values, label="LSTM 2025")

plt.title("Previsões 2025 — PETR4 (Treino até 2023)")
plt.xlabel("Data"); plt.ylabel("Preço (R$)")
plt.legend(); plt.grid()

# === SALVAR IMAGEM ===
plt.savefig("predict_2025_petr4.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 9. TABELA 2025 COM VARIAÇÃO %
# ------------------------------------------------------------
df_2025 = pd.DataFrame(index=prod.index)
df_2025["Real"]  = prod["Close"]
df_2025["ARIMA"] = arima_series_2025
df_2025["RF"]    = rf_series_2025
df_2025["LSTM"]  = lstm_series_2025

df_2025["Var_ARIMA_%"] = (df_2025["ARIMA"] / df_2025["Real"] - 1) * 100
df_2025["Var_RF_%"]    = (df_2025["RF"] / df_2025["Real"] - 1) * 100
df_2025["Var_LSTM_%"]  = (df_2025["LSTM"] / df_2025["Real"] - 1) * 100

print(df_2025.head())

# ------------------------------------------------------------
# 10. EXPORTAR EXCEL
# ------------------------------------------------------------
df_2025.to_excel("petr4_predict_2025.xlsx")

print("\nFim do script.")
