from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ferramentas.utils import salvar_grafico



wine = load_wine()
lista_acuracia = []
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Formato das features de treino: {X_train.shape}")
print(f"Formato das features de teste: {X_test.shape}")


# Treinar modelo para diferentes valores de K 
k_valores = range(1, 11)
for k in k_valores: 
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred) 
    lista_acuracia.append(acuracia)
    print(f"Acurácia com k={k}: {acuracia:.4f}")


# Criar um array de lista_acuracia

array_acuracia = np.array(lista_acuracia)
maior_acuracia = array_acuracia.max()
menor_acuracia = array_acuracia.min()
print(f"A maior acurácia foi {maior_acuracia:.4f}. A menor foi {menor_acuracia:.4f}")


# Gráfico para visualização


plt.figure(figsize=(10,6), dpi=150)
plt.plot(k_valores, lista_acuracia, marker='o', linestyle='--')
plt.title('Acurácia do modelo KNN vs. Valor de K')
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.xticks(k_valores)
plt.grid(True)

melhor_indice = np.argmax(lista_acuracia)
melhor_k = k_valores[melhor_indice]
melhor_acuracia = lista_acuracia[melhor_indice]

plt.annotate(
    f'Melhor K: {melhor_k}\nAcurácia: {melhor_acuracia:.4f}',
    xy=(melhor_k, melhor_acuracia),
    xytext=(melhor_k + 1, melhor_acuracia - 0.01),
    arrowprops=dict(facecolor='black', shrink=0.05)
)

min_acuracia = min(lista_acuracia)
plt.ylim(min_acuracia - 0.01, 0.81)

plt.yticks(np.arange(min_acuracia - 0.01, 0.81, 0.01))
salvar_grafico('K_vs_Acurácia.pdf')
plt.show()