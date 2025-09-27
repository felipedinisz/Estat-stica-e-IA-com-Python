import pandas as pd

import io

dados_csv = """nome,nota,curso,semestre
Maria,7.8,Engenharia,3
Felipe,10.0,Computação,5
Aurélio,5.0,Física,2
Ana,8.5,Engenharia,5
Bruno,4.2,Computação,1
Carla,9.5,Química,8
"""

df = pd.read_csv(io.StringIO(dados_csv))

print(df)
print("-"*50)
nomes = df['nome']

print(nomes)

print(f'\nTipo do objeto: {type(nomes)}')

print("-"*50)

colunas_desejadas = ['nome', 'nota']

nomes_e_notas = df[colunas_desejadas]
# nomes_e_nptas = df[['nome', 'nota']]

print(nomes_e_notas)
print(f'\nTipo do objeto: {type(nomes_e_notas)}')