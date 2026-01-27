import numpy as np

def decomposition_cholesky(A):
    
    n = len(A)
    B = np.zeros((n, n))
    
    for j in range(n):
        s = 0
        for k in range(j):
            s = s + B[j, k]**2
        
        B[j, j] = np.sqrt(A[j, j] - s)
        
        for i in range(j + 1, n):
            s = 0
            for k in range(j):
                s = s + B[i, k] * B[j, k]
            
            B[i, j] = (A[i, j] - s) / B[j, j]
    
    return B

def saisir_matrice():
    
    #Permet à l'utilisateur de saisir une matrice
    
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
print("=== Décomposition de Cholesky ===\n")

A = saisir_matrice()

print("\nMatrice A saisie :")
print(A)

try:
    B = decomposition_cholesky(A)
    
    print("\nMatrice B (triangulaire inférieure) :")
    print(B)
    
    # Vérification : B * B^T devrait donner A
    produit = np.dot(B, B.T)
    print("\nVérification B × B^T :")
    print(produit)
    
    print("\nDifférence A - B×B^T :")
    print(A - produit)
    
    print("\nVérification réussie :", np.allclose(A, produit))
    
except ValueError as e:
    print(f"\nErreur : {e}")
    print("La matrice doit être symétrique et définie positive pour la décomposition de Cholesky.")
