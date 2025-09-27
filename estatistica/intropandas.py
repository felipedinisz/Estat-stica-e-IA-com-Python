import pandas as pd

dados_alunos = {
    'nome' : ['Maria', 'Felipe', 'Aurélio', 'Ana', 'Bruno'],
    'nota' : [7.8, 10.0, 5.0, 8.5, 4.2]
}

df = pd.DataFrame(dados_alunos)
print(df)
