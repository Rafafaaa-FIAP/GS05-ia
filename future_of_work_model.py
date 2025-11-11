# future_of_work_model_cv.py
# ---------------------------------------------------------------
# Projeto: O Futuro do Trabalho - Predição de Prosperidade Regional
# Autor: Rafael Cristofali
# Descrição: Aplicação de Machine Learning com validação cruzada e regularização
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Carregar dados reais (California Housing)
# ---------------------------------------------------------------
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

print("✅ Dataset carregado com sucesso!")
print("Número de amostras:", X.shape[0])
print("Número de atributos:", X.shape[1])
print()

# ---------------------------------------------------------------
# 2. Divisão em treino e teste
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------
# 3. Normalização
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Modelos de Regressão
# ---------------------------------------------------------------
models = {
    "Linear": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1)
}

results = {}

# ---------------------------------------------------------------
# 5. Treinamento, Validação Cruzada e Avaliação
# ---------------------------------------------------------------
for name, model in models.items():
    # Validação cruzada (k-fold = 5)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results[name] = {
        "R² teste": r2,
        "MSE teste": mse,
        "R² CV (médio)": cv_scores.mean()
    }

# ---------------------------------------------------------------
# 6. Resultados numéricos
# ---------------------------------------------------------------
df_results = pd.DataFrame(results).T
print("📊 Resultados de desempenho:\n")
print(df_results)
print()

# ---------------------------------------------------------------
# 7. Visualização: Comparação de desempenho
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.bar(df_results.index, df_results["R² teste"], color=['#4CAF50', '#2196F3', '#FF9800'])
plt.title("Comparação de desempenho entre modelos (R² no teste)")
plt.ylabel("R²")
plt.show()

# ---------------------------------------------------------------
# 8. Visualização: Previsões do melhor modelo
# ---------------------------------------------------------------
best_model_name = df_results["R² teste"].idxmax()
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title(f"Real vs Predito ({best_model_name})")
plt.xlabel("Valores reais")
plt.ylabel("Valores preditos")
plt.show()

print(f"🏆 Melhor modelo: {best_model_name}")
