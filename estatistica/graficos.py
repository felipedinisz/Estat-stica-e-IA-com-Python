import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt


dados_csv = """nome,nota,curso,semestre
Maria,7.8,Engenharia,3
Felipe,10.0,Computação,5
Aurélio,5.0,Física,2
Ana,8.5,Engenharia,5
Bruno,4.2,Computação,1
"""

df = pd.read_csv(io.StringIO(dados_csv))

sns.set_theme(style='whitegrid')

sns.barplot(x='nome', y='nota', data=df)

plt.show()