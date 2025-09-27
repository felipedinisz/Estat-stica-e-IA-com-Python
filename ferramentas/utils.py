import os
import matplotlib.pyplot as plt
import pandas as pd



def verificacao(path): 
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Pasta '{path}' criada com sucesso.")   


def salvar_grafico(nome_arquivo):
    path = './graficos'
    verificacao(path)
    caminho_completo = os.path.join(path, nome_arquivo)

    plt.savefig(caminho_completo, dpi=150, bbox_inches='tight')
    print(f"Gráfico salvo em: {caminho_completo}")

def salvar_csv(df: pd.DataFrame, nome_arquivo, index=False): 
    path = './csv'
    verificacao(path)
    caminho_completo = os.path.join(path, nome_arquivo)
    df.to_csv(caminho_completo, index=index)
    print(f"CSV salvo em: {caminho_completo}")