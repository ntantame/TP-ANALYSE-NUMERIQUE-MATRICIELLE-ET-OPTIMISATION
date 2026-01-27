import numpy as np

def descente(T, r):
   
    n = len(r)
    y = np.zeros(n)
    
    for i in range(n):
        s = 0
        for j in range(i):
            s = s + T[i, j] * y[j]
        y[i] = (r[i] - s) / T[i, i]
    
    return y

def methode_gauss_seidel(A, b, x0, epsilon, kmax):
   
    k = 1
    x = x0.copy()
    r = b - np.dot(A, x)
    T = np.tril(A)  # Partie triangulaire inférieure de A
    n = len(x)
    
    print(f"Itération {0} : ||r|| = {np.linalg.norm(r):.10f}")
    print(f"x = {x}")
    print()
    
    while (np.linalg.norm(r) >= epsilon) and (k < kmax):
        # Résolution de Ty = r par descente
        y = descente(T, r)
        
        # Mise à jour de x
        x = x + y
        
        # Calcul du nouveau résidu
        r = b - np.dot(A, x)
        
        print(f"Itération {k} : ||r|| = {np.linalg.norm(r):.10f}")
        print(f"x = {x}")
        print()
        
        k = k + 1
    
    if k >= kmax:
        print(f"Attention : Nombre maximum d'itérations ({kmax}) atteint!")
    else:
        print(f"Convergence atteinte en {k-1} itérations.")
    
    return x

def saisir_matrice(nom="A"):
    
    # Permet à l'utilisateur de saisir une matrice
    
    n = int(input(f"Entrez la taille de la matrice {nom} (n×n) : "))
    
    print(f"\nEntrez les éléments de la matrice {nom} {n}×{n} :")
    matrice = []
    
    for i in range(n):
        ligne = []
        for j in range(n):
            valeur = float(input(f"{nom}[{i+1}][{j+1}] = "))
            ligne.append(valeur)
        matrice.append(ligne)
    
    return np.array(matrice)

def saisir_vecteur(nom="b", n=None):
 
     #Permet à l'utilisateur de saisir un vecteur
  
    if n is None:
        n = int(input(f"Entrez la taille du vecteur {nom} : "))
    
    print(f"\nEntrez les éléments du vecteur {nom} :")
    vecteur = []
    
    for i in range(n):
        valeur = float(input(f"{nom}[{i+1}] = "))
        vecteur.append(valeur)
    
    return np.array(vecteur)

# Configuration pour l'affichage avec 4 décimales
np.set_printoptions(precision=4, suppress=True)

# Programme principal
print("=== Méthode de Gauss-Seidel ===\n")

A = saisir_matrice("A")
n = len(A)

b = saisir_vecteur("b", n)

print("\nVecteur initial x0 :")
x0 = saisir_vecteur("x0", n)

epsilon = float(input("\nEntrez la tolérance (epsilon) : "))
kmax = int(input("Entrez le nombre maximum d'itérations (kmax) : "))

print("\n" + "="*60)
print("Début des itérations")
print("="*60 + "\n")

print("Matrice A :")
print(A)
print("\nVecteur b :")
print(b)
print("\nVecteur initial x0 :")
print(x0)
print(f"\nTolérance epsilon = {epsilon}")
print(f"Nombre maximum d'itérations kmax = {kmax}")
print()

# Vérification : la matrice T = tril(A) doit avoir une diagonale non nulle
T = np.tril(A)
print("Matrice T (partie triangulaire inférieure de A) :")
print(T)
print()

diag_T = np.diag(T)
if np.any(np.abs(diag_T) < 1e-10):
    print("ERREUR : La matrice T a des éléments diagonaux nuls!")
    print("La méthode de Gauss-Seidel ne peut pas être appliquée.")
else:
    # Résolution
    x = methode_gauss_seidel(A, b, x0, epsilon, kmax)
    
    print("\n" + "="*60)
    print("Résultat")
    print("="*60)
    print("\nSolution approchée x :")
    print(x)
    
    # Vérification
    r_final = b - np.dot(A, x)
    print(f"\nRésidu final ||b - Ax|| = {np.linalg.norm(r_final):.10f}")
    
    # Comparaison avec la solution exacte
    x_exact = np.linalg.solve(A, b)
    print("\nSolution exacte (numpy) :")
    print(x_exact)
    print(f"\nErreur ||x - x_exact|| = {np.linalg.norm(x - x_exact):.10f}")
