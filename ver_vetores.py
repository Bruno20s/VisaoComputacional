from database import VectorDatabase
import numpy as np

db = VectorDatabase()
dados = db.get_all_vectors()

print(f"Total de vetores: {len(dados)}")


# menu simples
print("\nEscolha uma opção:")
print("1 - Ver primeiros 100 valores de TODOS os IDs")
print("2 - Comparar dois IDs")
print("3 - Ver primeiros 1000 valores de um ID específico")
print("4 - Encontrar IDs mais similares a um ID")

opcao = input("Opção: ")


# mostrar todos
if opcao == "1":

    for oid, vec in dados:
        print(f"\nID: {oid}")
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

    for oid, vec in dados:
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

    for oid, vec in dados:
        if oid == id_busca:
            vec_encontrado = vec
            break

    if vec_encontrado is None:
        print("ID não encontrado.")
    else:
        print(f"\nID: {id_busca}")
        print(f"Tamanho: {len(vec_encontrado)}")

        # garante que não estoura se tiver menos de 1000
        limite = min(1000, len(vec_encontrado))

        print(f"Primeiros {limite} valores:\n{vec_encontrado[:limite]}")
        
elif opcao == "4":

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    id_base = int(input("Digite o ID base: "))
    top_n = int(input("Quantos resultados deseja ver? "))

    vec_base = None

    # encontrar vetor base
    for oid, vec in dados:
        if oid == id_base:
            vec_base = vec
            break

    if vec_base is None:
        print("ID não encontrado.")
    else:
        resultados = []

        for oid, vec in dados:
            if oid != id_base:
                score = cosine_similarity(vec_base, vec)
                resultados.append((oid, score))

        # ordenar do maior pro menor
        resultados.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} mais similares ao ID {id_base}:\n")

        for oid, score in resultados[:top_n]:
            print(f"ID: {oid} | Similaridade: {score:.4f}")


else:
    print("Opção inválida.")