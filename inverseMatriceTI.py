import numpy as np 
def inverse_Matrice_Triangulaire_Inferieur(A):
    n=A.shape[0]
    B=np.zeros((n,n))
    for i in range(n):
        B[i,i]=1/A[i,i]
        for j in range(i):
            s=0
            for k in range(j+1,i+1):
                s+=B[i,k]*A[k,j]
            B[i,j]=-s/A[j,j]
    return B

#remplisaage de la matrice A
def remplissage_matrice():
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
        for j in range(i + 1):  # Seulement jusqu'à la diagonale
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

# programme principale

print("=" * 60)
print("CALCUL DE L'INVERSE D'UNE MATRICE TRIANGULAIRE INFÉRIEURE")
print("=" * 60)
print()

print("veillez remplir la matrice triangulaire inferieur")
A=remplissage_matrice()
print(" ci-dessous la matriece triangulaire inferieur A:\n")
print(A)
print("\n")
B=inverse_Matrice_Triangulaire_Inferieur(A)
print("matrice inverse de A :\n")
print(B)
print(" verification du resultat")
print("A*B=\n", A@B)  
