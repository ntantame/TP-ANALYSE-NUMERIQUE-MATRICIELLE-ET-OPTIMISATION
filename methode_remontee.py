import numpy as np

def remontee(A,b):
    n= len(b)
    X=np.zeros(n)
    for i in range(n-1,-1,-1):
        s=0
        for j in range(i+1,n):
            s+=A[i,j]*X[j]
        X[i]=(b[i]-s)/A[i,i]
    return X

# remplissage de la matrice triangulaire superieure
def saisir_matrice_tringulaire_superieur():
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
    
    print("\nSaisissez les éléments de la matrice triangulaire supérieure :")
    print("(Les éléments en-dessous de la diagonale seront automatiquement mis à 0)\n")
    
    for i in range(n):
        for j in range(i, n):
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

# remplissage du vecteur b
def remplissage_vecteur(n):
    
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


# -----------------------------------------------------
# PROGRAMME PRINCIPAL — s'exécute uniquement si lancé
# directement, pas quand importé dans un autre fichier
# -----------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("RÉSOLUTION D'UN SYSTÈME TRIANGULAIRE SUPERIEURE (Ax = b)")
    print("Algorithme de remontee (backward substitution)")
    print("=" * 60)
    print()

    print("veillez remplir la matrice triangulaire inferieur")
    A = saisir_matrice_tringulaire_superieur()
    n = A.shape[0]
    b = remplissage_vecteur(n)

    print("matrice A:\n")
    print(A)
    print(" vecteur b:\n")
    print(b)

    X = remontee(A, b)
    print("la solution du systeme est: X=", X)
