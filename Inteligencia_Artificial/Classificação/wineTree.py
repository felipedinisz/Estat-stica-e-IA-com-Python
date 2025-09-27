from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ferramentas.utils import salvar_grafico
    
wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Árvore completa ---
tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acuracia_tree = accuracy_score(y_test, y_pred_tree)

print(f"A acurácia da Árvore de Decisão foi: {acuracia_tree:.4f}")

# --- Árvore Podada ---

tree_podada = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_podada.fit(X_train, y_train)
y_pred_podada = tree_podada.predict(X_test)
acuracia_podada = accuracy_score(y_test, y_pred_podada)
print(f"Acurácia da Árvore com max_depth=3: {acuracia_podada:.4f}")

# Gráfico de visualização
print("Gerando a visualização da Árvore de Decisão...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

plot_tree(
    tree,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    ax=axes[0]
)

axes[0].set_title("Árvore de Decisão completa")

plot_tree(
    tree_podada,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    ax=axes[1]
)

axes[1].set_title("Árvore de Decisão podada")

plt.tight_layout()

salvar_grafico('fluxograma_arvores.pdf')
plt.show()