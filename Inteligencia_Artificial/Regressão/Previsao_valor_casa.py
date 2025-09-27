import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from ferramentas.utils import salvar_grafico_interativo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()


df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target


# --- Criação do Gráfico de Dispersão ---
fig = px.scatter(
    df, 
    x='AveRooms', 
    y='MedHouseVal', 
    title='Preço da Casa vs. Número Médio de Quartos (Interativo)',
    labels={'AveRooms': 'Número Médio de Quartos', 'MedHouseVal': 'Preço Mediano ($100k)'}
)


salvar_grafico_interativo(fig, "gráfico_dispersão_casa.html")

# --- Dividir os dados --- 
X = df.drop("MedHouseVal", axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Divisão concluída:")
print(f"Formato dos dados de treino: {X_train.shape}")
print(f"Formato dos dados de teste: {X_test.shape}")

# --- Treinar o Modelo ---

lr = LinearRegression()

lr.fit(X_train, y_train)

print("Modelo de regressão Linear treinado com sucesso!")

# --- Avaliação ---

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"O Erro Quadrático Médio (MSE) do modelo foi: {mse:.4f}")
print(f"A Raiz do Erro Quadrático Médio (RMSE) foi: {rmse:.4f}")