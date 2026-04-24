import os
import csv
import cv2
from database import VectorDatabase
import numpy as np
from collections import Counter

db = VectorDatabase()
dados = db.get_all_vectors()

print(f"Total de vetores: {len(dados)}")


# menu simples
print("\nEscolha uma opção:")
print("1 - Ver primeiros 100 valores de TODOS os IDs")
print("2 - Comparar dois IDs")
print("3 - Ver primeiros 1000 valores de um ID específico")
print("4 - Encontrar IDs mais similares a um ID")
print("5 - Ver posições mais significativas de um ID")
print("6 - Gerar CSV com top 5 posições de cada vetor")
print("7 - Ver top classificações de um ID")

opcao = input("Opção: ")


# mostrar todos
if opcao == "1":

    for rid, oid, arquivo, data_hora, vec, _ in dados:
        print(f"\nOBJECT_ID: {oid}")
        print(f"Tamanho: {len(vec)}")
        print(f"Primeiros 100 valores:\n{vec[:100]}")
        print("-" * 40)


# comparar
elif opcao == "2":

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    id1 = int(input("Digite o primeiro ID: "))
    id2 = int(input("Digite o segundo ID: "))

    vec1 = None
    vec2 = None

    for rid, oid, arquivo, data_hora, vec, _ in dados:
        if oid == id1:
            vec1 = vec
        elif oid == id2:
            vec2 = vec

    if vec1 is None or vec2 is None:
        print("Um dos IDs não foi encontrado.")
    else:
        print(f"\nID {id1} vs ID {id2}")

        print(f"\nPrimeiros valores ID {id1}: {vec1[:10]}")
        print(f"Primeiros valores ID {id2}: {vec2[:10]}")

        score = cosine_similarity(vec1, vec2)

        print(f"\nSimilaridade: {score:.4f}")


# mostrar um específico
elif opcao == "3":

    id_busca = int(input("Digite o ID: "))

    vec_encontrado = None

    for rid, oid, arquivo, data_hora, vec, _ in dados:
        if oid == id_busca:
            vec_encontrado = vec
            break

    if vec_encontrado is None:
        print("ID não encontrado.")
    else:
        print(f"\nID: {id_busca}")
        print(f"Tamanho: {len(vec_encontrado)}")

        limite = min(1000, len(vec_encontrado))
        print(f"Primeiros {limite} valores:\n{vec_encontrado[:limite]}")


# similares
elif opcao == "4":

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    id_base = int(input("Digite o ID base: "))
    top_n = int(input("Quantos resultados deseja ver? "))

    vec_base = None

    for rid, oid, arquivo, data_hora, vec, _ in dados:
        if oid == id_base:
            vec_base = vec
            break

    if vec_base is None:
        print("ID não encontrado.")
    else:
        resultados = []

        for rid, oid, arquivo, data_hora, vec, _ in dados:
            if oid != id_base:
                score = cosine_similarity(vec_base, vec)
                resultados.append((oid, score))

        resultados.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} mais similares ao ID {id_base}:\n")

        for oid, score in resultados[:top_n]:
            print(f"ID: {oid} | Similaridade: {score:.4f}")


# posições importantes
elif opcao == "5":

    id_busca = int(input("Digite o ID: "))
    top_k = int(input("Quantas posições mais importantes deseja ver? "))

    vec_encontrado = None

    for rid, oid, arquivo, data_hora, vec, _ in dados:
        if oid == id_busca:
            vec_encontrado = vec
            break

    if vec_encontrado is None:
        print("ID não encontrado.")
    else:
        indices = np.argsort(np.abs(vec_encontrado))[::-1]

        print(f"\nTop {top_k} posições mais significativas do ID {id_busca}:\n")

        for n, i in enumerate(indices[:top_k]):
            print(f"{n:3} - x[{i:3}] = {vec_encontrado[i]:7.4f}")

elif opcao == "6":

    top_k = 5
    arquivo_saida = "saida_ids.csv"

    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f, delimiter=';')

        writer.writerow([
            "id_registro",
            "object_id",
            "arquivo",
            "data_hora",
            "pos1", "val1",
            "pos2", "val2",
            "pos3", "val3",
            "pos4", "val4",
            "pos5", "val5",
            "modulo_top5"
        ])

        for rid, oid, arquivo, data_hora, vec, _ in dados:

            indices = np.argsort(np.abs(vec))[::-1]
            posicoes = indices[:top_k]
            valores = [float(vec[i]) for i in posicoes]

            modulo = float(np.linalg.norm(valores))

            linha = [rid, oid, arquivo, data_hora]

            for p, v in zip(posicoes, valores):
                linha.append(int(p))
                linha.append(v)

            linha.append(modulo)

            writer.writerow(linha)

    print("Arquivo 'saida_ids.csv' gerado com sucesso!")

elif opcao == "7":

    object_id_busca = int(input("Digite o OBJECT_ID: "))
    top_k = 10

    resultados = []

    for rid, oid, arquivo, data_hora, vec, classes in dados:

        if oid != object_id_busca:
            continue

        for classe, score in classes:

            if classe is None:
                continue

            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            resultados.append((classe, score, arquivo))

    if not resultados:
        print("Nenhum dado válido encontrado para esse ID.")
    else:
        # remove duplicatas (mesmo arquivo, classe e score)
        resultados = list(set(resultados))
        resultados.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_k} classificações para ID {object_id_busca}:\n")

        for i, (classe, score, arquivo) in enumerate(resultados[:top_k]):
            print(f"{i+1:2} - {classe:15} | {score:.4f} | {arquivo}")

        from collections import Counter

        classes_lista = [c for c, _, _ in resultados]

        print("\nFrequência das classes:")
        print(Counter(classes_lista))

else:
    print("Opção inválida.")