# Previsões de PETR4 — Comparação ARIMA vs Random Forest vs LSTM  
Treino até 2023 • Teste 2024 • Previsões para 2025

Este mini projeto teve como objetivo **comparar diferentes modelos de previsão** ao tentar antecipar o preço diário da PETR4.  
O foco não é fornecer recomendações de investimento, mas **testar como algoritmos diferentes se comportam diante de uma série altamente volátil**.

---

## Modelos utilizados
| Modelo | Tipo | Pontos fortes | Limitações no contexto |
|-------|------|----------------|-------------------------|
| **ARIMA** | Estatístico | Séries estáveis e previsíveis | Não captura volatilidade, perde amplitude |
| **Random Forest Regressor** | Machine Learning | Dados tabulares, muitas variáveis | Sem memória temporal, suaviza demais |
| **LSTM (Long Short-Term Memory)** | Deep Learning | Dependência temporal e volatilidade | Exige mais dados e processamento |

---

## 📅 Janela de dados

- **Treino:** 2010 a 2023  
- **Teste:** 2024  
- **Produção (previsões):** 2025  

---

## Gráfico — Previsões 2025  
A imagem abaixo é salva automaticamente pelo script como:


<img width="842" height="737" alt="image" src="https://github.com/user-attachments/assets/11b4ec6b-60c7-476f-bf16-54172c470dc1" />


---

## Principais conclusões

### **ARIMA: tendência linear irrelevante**
O ARIMA **não conseguiu capturar a volatilidade da PETR4**, gerando:

- previsão crescente e suave, completamente descasada da realidade  
- erros mensais acima de **+30%** em vários momentos  
- tendência linear, sem amplitude  

➡️ Bom para: energia, inflação, séries estáveis e não voláteis

---

### **Random Forest: acertou a forma, errou a escala**
O modelo conseguiu capturar a **direção geral**, mas o preço previsto ficou:

- muito próximo de zero  
- sem amplitude  
- suavizado demais  

➡️ Bom para: churn, propensão de compra, crédito, modelos tabulares

---

### **LSTM: modelo mais consistente**
A LSTM conseguiu:

- acompanhar o sobe-e-desce real  
- manter amplitude coerente  
- capturar volatilidade  
- apresentar erros entre **–1,2% e +2,1%** na maioria dos meses  

➡️ Excelente para séries complexas, como: ações, cripto, tráfego, sinais biomédicos

### **Tecnologias**
- pandas
- numpy
- scikit-learn
- tensorflow
- yfinance
- matplotlib
- seaborn
- pmdarima


