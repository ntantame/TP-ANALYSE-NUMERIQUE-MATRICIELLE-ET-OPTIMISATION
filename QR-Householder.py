import numpy as np

def decomposition_householder_QR(A):
    
    R = A.copy().astype(float)
    m, n = R.shape
    
    for k in range(n):
        # Extraction du vecteur v
        v = R[k:m, k].copy()
        
        # Calcul de alpha
        norme_v = np.linalg.norm(v)
        
        if v[0] >= 0:
            alpha = -norme_v
        else:
            alpha = norme_v
        
        # Vérification pour éviter division par zéro
        if abs(norme_v) < 1e-10:
            continue
        
        # Calcul de beta
        beta = alpha**2 - alpha * v[0]
        
        # Vérification pour éviter division par zéro
        if abs(beta) < 1e-10:
            continue
        
        # Modification de v[0]
        v[0] = v[0] - alpha
        
        # Application de la transformation de Householder à R
        for j in range(k, n):
            # Extraction de la colonne j de k à m
            colonne = R[k:m, j].copy()
            
            # Calcul de gamma
            gamma = (1.0 / beta) * np.dot(v, colonne)
            
            # Mise à jour de la colonne
            R[k:m, j] = colonne - gamma * v
    
    # Nettoyage des -0.00
    R[np.abs(R) < 1e-10] = 0
    
    return R

def saisir_matrice():
    """
    Permet à l'utilisateur de saisir une matrice
    """
    m = int(input("Entrez le nombre de lignes (m) : "))
    n = int(input("Entrez le nombre de colonnes (n) : "))
    
    print(f"\nEntrez les éléments de la matrice {m}×{n} :")
    A = []
    
    for i in range(m):
        ligne = []
        for j in range(n):
            valeur = float(input(f"A[{i+1}][{j+1}] = "))
            ligne.append(valeur)
        A.append(ligne)
    
    return np.array(A)

# Configuration pour l'affichage avec 2 décimales
np.set_printoptions(precision=2, suppress=True)

# Programme principal
print("=== Décomposition QR par Householder ===\n")

A = saisir_matrice()

print("\nMatrice A saisie :")
print(A)

# Appel de la fonction pour obtenir R
R = decomposition_householder_QR(A)

print("\nMatrice R (triangulaire supérieure) :")
print(R)

# Détermination de Q dans le programme principal
# Q = A × R^(-1) pour les matrices carrées
# Ou utiliser la pseudo-inverse pour les matrices rectangulaires
m, n = A.shape

if m == n:
    # Matrice carrée : Q = A × R^(-1)
    Q = np.dot(A, np.linalg.inv(R))
else:
    # Matrice rectangulaire : utiliser la pseudo-inverse ou numpy
    Q, _ = np.linalg.qr(A)

Q[np.abs(Q) < 1e-10] = 0

print("\nMatrice Q (orthogonale) :")
print(Q)

# Vérifications
print("\nVérification Q × R :")
produit = np.dot(Q, R)
print(produit)

print("\nDifférence A - Q×R :")
print(A - produit)

print("\nVérification Q^T × Q (doit être I) :")
print(np.dot(Q.T, Q))

print("\nDécomposition réussie :", np.allclose(A, produit))