import pandas as pd
import os 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

path_csv = './csv/Electrical Fault detection'
full_path_classData = os.path.join(path_csv, 'classData.csv')
df = pd.read_csv(full_path_classData)
df_falhas = df[df.sum(axis=1) > 0]



def decodificar_falha(row):
    if row['G'] == 0 and row['C'] == 0 and row['B'] == 0 and row['A'] == 0:
        return 'Sem_Falha'
    
    nome_falha = ''
    if row['A'] == 1:
        nome_falha += 'A'
    if row['B'] == 1:
        nome_falha += 'B'
    if row['C'] == 1:
        nome_falha += 'C'
    if row['G'] == 1:
        nome_falha += 'G'

    return f'Falha_{nome_falha}'


df['Tipo_Falha'] = df.apply(decodificar_falha, axis=1)


X = df.iloc[:, :6]
y = df['Tipo_Falha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

steps = [
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
]

pipeline = Pipeline(steps=steps)

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, verbose=1)

print("Iniciando a busca pelos melhores parâmetros com Grid Search...")
grid_search.fit(X_train, y_train)


y_pred = grid_search.predict(X_test)


acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia final do modelo otimizado: {acuracia:.2%}")


print("\n--- Relatório de Classificação Detalhado ---")
print(classification_report(y_test, y_pred))


print(grid_search.best_params_)

prob_final = grid_search.predict_proba(X_test)

df_prob_finais = pd.DataFrame(prob_final, columns=grid_search.classes_)


print(df_prob_finais.head())