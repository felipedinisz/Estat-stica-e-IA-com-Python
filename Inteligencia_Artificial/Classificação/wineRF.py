from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ferramentas.utils import salvar_grafico

wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acuracia_rf = accuracy_score(y_test, y_pred_rf)

print(f"A acurácia do Random Forest foi: {acuracia_rf:.4f}")

importancias = rf.feature_importances_
nomes_features = wine.feature_names

df_importancias = pd.DataFrame({
    'Feature': nomes_features,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

print("--- Importância de Cada Feature ---")
print(df_importancias)

plt.figure(figsize=(10,8), dpi=100)
sns.barplot(x='Importancia', y="Feature", data=df_importancias)
plt.title('Importância de Cada Característica na Previsão do Vinho')
plt.xlabel("Importância Relativa")
plt.ylabel("Característica")
salvar_grafico('ImportanciaRF.pdf')
plt.show()