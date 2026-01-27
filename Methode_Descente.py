import numpy as np

def descente(A,b):
    n=len(b)
    X=np.zeros(n)
    for i in range(n):
        s=0
        for j in range(i):
            s+=A[i,j]*X[j]
        X[i]=(b[i]-s)/A[i,i]
    return X
    
#remplissage de la matrice A 

def saisir_matrice():
   
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
    
    print("\nSaisissez les éléments de la matrice triangulaire inférieure :")
    print("(Les éléments au-dessus de la diagonale seront automatiquement mis à 0)\n")
    
    for i in range(n):
        for j in range(i + 1):
            while True:
                try:
                    valeur = float(input(f"A[{i+1},{j+1}] = "))
                    if j == i and valeur == 0:
                        print("L'élément diagonal ne peut pas être 0. Réessayez.")
                        continue
                    A[i, j] = valeur
                    break
                except ValueError:
                    print("Veuillez entrer un nombre valide.")
    
    return A

#remplissage du vecteur b
def saisir_vecteur(n):
    
    b = np.zeros(n)
    
    print("\nSaisissez les éléments du vecteur b :")
    for i in range(n):
        while True:
            try:
                b[i] = float(input(f"b[{i+1}] = "))
                break
            except ValueError:
                print("Veuillez entrer un nombre valide.")
    
    return b

#programme principale
print("=" * 60)
print("RÉSOLUTION D'UN SYSTÈME TRIANGULAIRE INFÉRIEURE (Ax = b)")
print("Algorithme de descente (forward substitution)")
print("=" * 60)
print()
print("veillez remplir la matrice triangulaire inferieur")
A=saisir_matrice()
n=A.shape[0]
b=saisir_vecteur(n)
print("matrice A:\n")
print(A)
print(" vecteur b:\n")
print(b)
X=descente(A,b)
print("la solution du systeme est: X=",X)
