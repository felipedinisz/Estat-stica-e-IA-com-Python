import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ferramentas.utils import salvar_csv, salvar_grafico
url = 'https://gist.githubusercontent.com/designernatan/27da044c6dc823f7ac7fe3a01f4513ed/raw/d15b5c7d7a5efb38750b16ec935fc126ec9a6e79/vgsales.csv'

df_games = pd.read_csv(url)
df_limpo = df_games.dropna() # Removida linhas com valores nulos

salvar_csv(df_limpo, 'Venda_jogos')

vendas_por_genero_ordenado = df_limpo.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)


print("--- Total de Vendas Globais por Gênero (em milhões) ---")
print(vendas_por_genero_ordenado)


mascara_1Bi = vendas_por_genero_ordenado > 1000

generos_1bi = vendas_por_genero_ordenado[mascara_1Bi]

print(" \n --- Gêneros com mais de 1 bilhão em vendas --- \n")
print(generos_1bi)
"""
# --- Gráfico Pizza ---
top_5 = vendas_por_genero_ordenado.head(5)
outros = pd.Series([vendas_por_genero_ordenado.iloc[5:].sum()], index=['Outros'])
dados_pizza = pd.concat([top_5, outros])

print("Gerando o gráfico de pizza...")

plt.figure(figsize=(10,8))

plt.pie(dados_pizza, labels=dados_pizza.index, autopct='%1.1f%%', startangle=140)

plt.title("Porcentagem de Vendas Globais por Gênero de Jogo")

plt.axis("equal")
plt.show()
# --- Criação do Gráfico ---

print("Gerando gráfico...")
plt.figure(figsize=(12, 6))

cores = ['red' if venda > 1000 else 'grey' for venda in vendas_por_genero_ordenado]

sns.barplot(x=vendas_por_genero_ordenado.index, y=vendas_por_genero_ordenado.values, palette=cores)

plt.title('Total de Vendas Globais por Gênero de Jogo')
plt.xlabel('Gênero')
plt.ylabel('Vendas Globais (em milhões)')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()

plt.show()
"""

# --- Exibição de ambos simultaneamente

top_5 = vendas_por_genero_ordenado.head(5)
outros = pd.Series([vendas_por_genero_ordenado.iloc[5:].sum()], index=['Outros'])
dados_pizza = pd.concat([top_5, outros])

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
cores = ['red' if venda > 1000 else 'grey' for venda in vendas_por_genero_ordenado]

sns.barplot(x=vendas_por_genero_ordenado.index, 
            y=vendas_por_genero_ordenado.values,
            hue=vendas_por_genero_ordenado.index, 
            palette=cores, 
            legend=False,
            ax=axes[0]
)
axes[0].set_title("Total de Vendas Globais por Gênero")
axes[0].set_xlabel("Gênero")
axes[0].set_ylabel("Vendas Globais (em milhões)")
axes[0].tick_params(axis='x', rotation=45)

axes[1].pie(dados_pizza, labels=dados_pizza.index, autopct='%1.1f%%', startangle=140)
axes[1].set_title('Porcentagem de Vendas por Gênero')
axes[1].axis('equal')

plt.tight_layout()

salvar_grafico("Dashboard_vendas.pdf")
plt.show()