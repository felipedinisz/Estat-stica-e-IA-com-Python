import numpy as np

array_de_notas = np.array([7.8, 10.0, 5.2, 6.9, 2.2])

media = array_de_notas.mean()

maior_nota = array_de_notas.max()
menor_nota = array_de_notas.min()

desvio_padrao = array_de_notas.std()

print(f"Array Original: {array_de_notas}")
print("--- Estatísticas ---")
print(f"Média da turma: {media:.2f}")
print(f"Maior nota: {maior_nota}")
print(f"Menor nota: {menor_nota}")
print(f"Desvio Padrão: {desvio_padrao:.2f}")