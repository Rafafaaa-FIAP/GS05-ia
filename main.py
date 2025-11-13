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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1)
}

results = {}

# ---------------------------------------------------------------
# 5. Treinamento, Validação Cruzada e Avaliação
# ---------------------------------------------------------------
for name, model in models.items():
    print(f"Treinando modelo: {name} ...")
    
    # Validação cruzada
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Treinamento
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results[name] = {
        "Modelo": model,
        "R² teste": r2,
        "MSE teste": mse,
        "R² CV (médio)": cv_scores.mean(),
        "Predições": y_pred
    }

# ---------------------------------------------------------------
# 6. Resultados numéricos
# ---------------------------------------------------------------
df_results = pd.DataFrame(results).T.drop(columns=["Modelo", "Predições"])
print("\n📊 Resultados de desempenho:\n")
print(df_results.round(4))
print()

# ---------------------------------------------------------------
# 7. Visualização: Comparação geral dos modelos
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.bar(df_results.index, df_results["R² teste"], color=['#4CAF50', '#2196F3', '#FF9800'])
plt.title("Comparação de desempenho entre modelos (R² no teste)")
plt.ylabel("R²")
plt.show()

# ---------------------------------------------------------------
# 8. Visualização: Real vs Predito (todos os modelos)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, info) in zip(axes, results.items()):
    y_pred = info["Predições"]
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    ax.set_title(name)
    ax.set_xlabel("Valores reais")
    ax.set_ylabel("Valores preditos")

plt.suptitle("Comparação: Valores Reais vs Preditos para cada modelo", fontsize=14)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 9. Identificar e exibir o melhor modelo
# ---------------------------------------------------------------
best_model_name = df_results["R² teste"].idxmax()
best_model_info = results[best_model_name]
best_r2 = best_model_info["R² teste"]
best_mse = best_model_info["MSE teste"]
best_cv = best_model_info["R² CV (médio)"]

print("🏆 MELHOR MODELO ENCONTRADO 🏆")
print(f"Modelo: {best_model_name}")
print(f"R² no teste: {best_r2:.4f}")
print(f"MSE no teste: {best_mse:.4f}")
print(f"R² médio (validação cruzada): {best_cv:.4f}")
print()

# ---------------------------------------------------------------
# 10. Visualização: Gráfico do melhor modelo
# ---------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_model_info["Predições"], alpha=0.6, color="#4CAF50")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title(f"Real vs Predito — {best_model_name}")
plt.xlabel("Valores reais")
plt.ylabel("Valores preditos")
plt.show()
