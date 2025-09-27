import pandas as pd
import io

dados_csv = """nome,nota,curso,semestre
Maria,7.8,Engenharia,3
Felipe,10.0,Computação,5
Aurélio,5.0,Física,2
Ana,8.5,Engenharia,5
Bruno,4.2,Computação,1
"""

df = pd.read_csv(io.StringIO(dados_csv))

mascara_engenharia = df['curso'] == "Engenharia"
print(mascara_engenharia)

alunos_de_engenharia = df[mascara_engenharia]

print(alunos_de_engenharia)

alunos_aprovados_computacao = df[(df['curso'] == 'Computação') & (df['nota'] > 8.0)]

print(alunos_aprovados_computacao)