from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()

print(f"Formato dos dados (features): {iris.data.shape}")
print("Primeiras 5 linhas das features: ")
print(iris.data[:5])

print(f"\nFormato do target (respostas): {iris.target.shape}")
print("Primeiras 5 respostas:")
print(iris.target[:5])

print(f"\nNomes das features: {iris.feature_names}")
print(f"Nomes do target (espécies): {iris.target_names}")

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(f"Formato de X_train (features de treino): {X_train.shape}")
print(f"Formato de y_train (respostas de treino): {Y_train.shape}")
print(f"Formato de X_test (features de teste): {X_test.shape}")
print(f"Formato de y_test (respostas de teste): {Y_test.shape}")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, Y_train)

print("Modelo KNN treinado com sucesso!")

Y_pred = knn.predict(X_test)
print(f"Previsões do modelo: {Y_pred}")
print(f"Respostas corretas: {Y_test}")

acuracia = accuracy_score(Y_test,Y_pred)
print(f"\n A acurácia do modelo foi: {acuracia:.2f}")