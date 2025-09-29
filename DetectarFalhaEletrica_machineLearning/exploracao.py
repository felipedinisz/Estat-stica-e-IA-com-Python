import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from ferramentas.utils import salvar_grafico

path_csv = './csv/Electrical Fault detection'
full_path_classData = os.path.join(path_csv, 'classData.csv')
df = pd.read_csv(full_path_classData)
df_falhas = df[df.sum(axis=1) > 0]



def calc_prob_condicional(event, cond):
    total_condicao = df_falhas[cond].sum()
    if total_condicao == 0:
        return 0

    ambos_ocorreram = df_falhas[(df_falhas[event] == 1) & (df_falhas[cond] == 1)].shape[0]
    return ambos_ocorreram / total_condicao


tipos_de_falha = ['A', 'B', 'C', 'G']


matriz_prob = pd.DataFrame(index=tipos_de_falha, columns=tipos_de_falha, dtype=float)

for cond in tipos_de_falha:
    for event in tipos_de_falha:
        prob = calc_prob_condicional(event, cond)
        matriz_prob.loc[cond, event] = prob

print("--- Matriz de Probabilidade P(Evento | Condição) ---")
print(matriz_prob)

plt.figure(figsize=(8,6))
sns.heatmap(matriz_prob, annot=True, cmap='viridis', fmt='.2%')
plt.title('Matriz de Probabilidade Condicional de Falhas')
plt.xlabel("Evento")
plt.ylabel("Condição")
salvar_grafico('prob_matrix.pdf')