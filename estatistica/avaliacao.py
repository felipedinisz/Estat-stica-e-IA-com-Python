alunos = [
    {"nome": "Maria", "nota": 7.8 },
    {"nome": "Felipe", "nota": 10},
    {"nome": "Aurélio", "nota": 5}
]

nota_corte = 7.0

somatorio = 0

for aluno in alunos:
    nome_aluno = aluno["nome"]
    nota_aluno = aluno["nota"]
    somatorio += nota_aluno
    print(f'{nome_aluno} tirou {nota_aluno}.')
    if (nota_aluno >= nota_corte):
        print(f'{nome_aluno} foi aprovado.')
    else:
        print(f'{nome_aluno} foi reprovado.')

media = somatorio / len(alunos)

print(f'A média da turma foi {media:.2f}.')