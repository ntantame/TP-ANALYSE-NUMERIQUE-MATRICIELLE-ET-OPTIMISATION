import numpy as np

def decomposition_LU(A):
   
    # Copie de A pour ne pas modifier la matrice originale
    Y = A.copy().astype(float)
    n = len(Y)
    
    for k in range(n - 1):
        for i in range(k + 1, n):
            Y[i, k] = Y[i, k] / Y[k, k]
            
            for j in range(k + 1, n):
                Y[i, j] = Y[i, j] - Y[i, k] * Y[k, j]
    
    return Y

def saisir_matrice():
    
    # Permet à l'utilisateur de saisir une matrice
    
    n = int(input("Entrez la taille de la matrice carrée (n × n) : "))
    
    print(f"\nEntrez les éléments de la matrice {n}×{n} :")
    A = []
    
    for i in range(n):
        ligne = []
        for j in range(n):
            valeur = float(input(f"A[{i+1}][{j+1}] = "))
            ligne.append(valeur)
        A.append(ligne)
    
    return np.array(A)

# Configuration pour l'affichage avec 2 décimales
np.set_printoptions(precision=2, suppress=True)

# Programme principal
print("=== Décomposition LU ===\n")

A = saisir_matrice()

print("\nMatrice A saisie :")
print(A)

Y = decomposition_LU(A)

print("\nMatrice Y (contient L et U) :")
print(Y)

# Extraction de L et U
L = np.tril(Y, -1) + np.eye(len(Y))
U = np.triu(Y)

print("\nMatrice L (triangulaire inférieure) :")
print(L)

print("\nMatrice U (triangulaire supérieure) :")
print(U)

# Vérification : L * U devrait donner A
produit = np.dot(L, U)
print("\nVérification L × U :")
print(produit)

print("\nDifférence A - L×U :")
print(A - produit)

print("\nVérification réussie :", np.allclose(A, produit))
