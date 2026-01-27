import numpy as np
from methode_remontee import remontee,remplissage_vecteur

def methodGauss(A,b):
    n= len(b)
    X=np.zeros(n)
    for k in range(n-1):
        for i in range(k+1,n):
            p=A[i,k]/A[k,k]
            for j in range(k,n):
                A[i,j]-=p*A[k,j]
            b[i] -= p * b[k] 
    X=remontee(A,b)
    return X

# remplissage de la matrice A

def saisir_matrice_complete():

    while True:
        try:
            n = int(input("Entrez la taille de la matrice (n x n) : "))
            if n <= 0:
                print("La taille doit être positive. Réessayez.")
                continue
            break
        except ValueError:
            print("Veuillez entrer un nombre entier valide.")
    
    A = np.zeros((n, n))
    
    print("\nSaisissez tous les éléments de la matrice :\n")
    
    for i in range(n):
        for j in range(n):  # Modifié : parcourt toutes les colonnes de 0 à n
            while True:
                try:
                    valeur = float(input(f"A[{i+1},{j+1}] = "))
                    A[i, j] = valeur
                    break
                except ValueError:
                    print("Veuillez entrer un nombre valide.")
    
    return A

#programme principale
print("=" * 60)
print("RÉSOLUTION D'UN SYSTÈME LINEAIRE    (Ax = b)")
print("Algorithme de GAUSS  ")
print("=" * 60)
print()
print("veillez remplir la matrice triangulaire inferieur")
A=saisir_matrice_complete()
n=A.shape[0]
b=remplissage_vecteur(n)
print("matrice A:\n")
print(A)
print(" vecteur b:\n")
print(b)
X= methodGauss(A,b)
print("valeur de la matrice A apres echelonement: \n")
print(A)
print("\n")
print("la solution du systeme est: X=",X)
