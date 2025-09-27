import numpy as np

lista_de_notas = [10, 9.2, 4.9, 6.2]
print(f'Isso é uma lista de notas: {lista_de_notas}')

array_de_notas = np.array(lista_de_notas)
print(f'Isso é um array de notas: {array_de_notas}')

print(f'Tipo de lista: {type(lista_de_notas)}')
print(f'Tipo de array: {type(array_de_notas)}')

notas_com_bonus = array_de_notas + 1.5

print(f"Resultado com array NumPy: {notas_com_bonus}")