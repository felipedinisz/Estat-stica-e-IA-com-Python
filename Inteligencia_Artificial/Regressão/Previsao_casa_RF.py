import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from ferramentas.utils import salvar_grafico 
import matplotlib.pyplot as plt


housing = fetch_california_housing()
rmse_rf_list = []
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

print("Treinando o Random Forest Regressor...")
n_estimators_range = range(1, 102, 10)

for n in n_estimators_range: 
    rf = RandomForestRegressor(n_estimators=n, random_state=41)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    rmse_rf_list.append(rmse_rf)
    print(f"\nRMSE do Random Forest: {rmse_rf:.4f} com n={n}")


array_rmse_rf = np.array(rmse_rf_list)
maior_rmse = array_rmse_rf.max()
menor_rmse = array_rmse_rf.min()
print(f"Maior rmse: {maior_rmse:.4f}. Menor rmse: {menor_rmse:.4f}")

plt.figure(figsize=(10,6), dpi=150)
plt.plot(n_estimators_range, rmse_rf_list, marker='o', linestyle='--')
plt.title('RMSE do Modelo vs. Número de Árvores na Floresta')
plt.xlabel('Número de Árvores')
plt.ylabel('RMSE')
plt.xticks(n_estimators_range)
plt.grid(True)

melhor_indice = np.argmin(rmse_rf_list)
melhor_n = n_estimators_range[melhor_indice]
melhor_rmse = rmse_rf_list[melhor_indice]

plt.annotate(
    f'Menor Erro\nRMSE: {melhor_rmse:.4f}\nEstimators: {melhor_n}',
    xy=(melhor_n, melhor_rmse),
    xytext=(melhor_n + 10, melhor_rmse + 0.01),
    arrowprops=dict(facecolor='black', shrink=0.05)
)



salvar_grafico('n_vs_RMSE.pdf')
plt.show()