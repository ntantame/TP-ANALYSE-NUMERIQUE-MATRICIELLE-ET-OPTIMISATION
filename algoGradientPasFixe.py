import numpy as np

def gradient_pas_fixe(A, b, x0, alpha, epsilon, kmax=1000):
    
    x = x0.copy()
    r = b - np.dot(A, x)
    k = 0
    
    print(f"Itération {k} : ||r|| = {np.linalg.norm(r):.10f}")
    print(f"x = {x}")
    print()
    
    while np.linalg.norm(r) >= epsilon:
        # Mise à jour de x
        x = x + alpha * r
        
        # Calcul du nouveau résidu
        r = b - np.dot(A, x)
        
        k = k + 1
        print(f"Itération {k} : ||r|| = {np.linalg.norm(r):.10f}")
        print(f"x = {x}")
        print()
        
        # Sécurité pour éviter une boucle infinie
        if k >= kmax:
            print(f"Attention : Nombre maximum d'itérations ({kmax}) atteint!")
            break
    
    if k < kmax:
        print(f"Convergence atteinte en {k} itérations.")
    
    return x

def saisir_matrice(nom="A"):
    
    #Permet à l'utilisateur de saisir une matrice
    
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
print("=== Méthode du Gradient à Pas Fixe ===\n")

A = saisir_matrice("A")
n = len(A)

b = saisir_vecteur("b", n)

print("\nVecteur initial x0 :")
x0 = saisir_vecteur("x0", n)

# Calcul de la plus grande valeur propre de A
valeurs_propres = np.linalg.eigvals(A)
lambda_max = np.max(np.real(valeurs_propres))

print(f"\nPlus grande valeur propre de A : lambda_max = {lambda_max:.6f}")
print(f"Pour la convergence, choisir 0 < alpha < {2/lambda_max:.6f}")

alpha = float(input("\nEntrez le pas alpha : "))

# Vérification de la condition de convergence
if alpha <= 0 or alpha >= 2/lambda_max:
    print(f"\nATTENTION : alpha doit être dans l'intervalle (0, {2/lambda_max:.6f}) pour garantir la convergence!")
    reponse = input("Voulez-vous continuer quand même ? (o/n) : ")
    if reponse.lower() != 'o':
        exit()

epsilon = float(input("Entrez la tolérance (epsilon) : "))
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
print(f"\nPas alpha = {alpha}")
print(f"Tolérance epsilon = {epsilon}")
print(f"Nombre maximum d'itérations kmax = {kmax}")
print()

# Résolution
x = gradient_pas_fixe(A, b, x0, alpha, epsilon, kmax)

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