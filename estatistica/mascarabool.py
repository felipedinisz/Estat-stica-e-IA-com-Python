import numpy as np

array_de_notas = np.array([7.8, 10.0, 5.2, 6.9, 2.2])
nota_corte = 7.0

mascara_aprovados = array_de_notas >= nota_corte
print(f'A máscara booleana é: {mascara_aprovados}')

notas_dos_aprovados = array_de_notas[mascara_aprovados]
print(f"Apenas as notas dos aprovados: {notas_dos_aprovados}")

media_dos_aprovados = notas_dos_aprovados.mean()
print(f"A média dos aprovados é: {media_dos_aprovados:.2f}")