import numpy as np

def gradient_pas_optimal(A, b, x0, epsilon, kmax=1000):
    
    x = x0.copy()
    r = b - np.dot(A, x)
    
    # Calcul du pas optimal alpha
    Ar = np.dot(A, r)
    alpha = np.dot(r, r) / np.dot(Ar, r)
    
    k = 0
    
    print(f"Itération {k} : ||r|| = {np.linalg.norm(r):.10f}, alpha = {alpha:.10f}")
    print(f"x = {x}")
    print()
    
    while np.linalg.norm(r) >= epsilon:
        # Mise à jour de x
        x = x + alpha * r
        
        # Calcul du nouveau résidu
        r = b - np.dot(A, x)
        
        # Calcul du nouveau pas optimal alpha
        Ar = np.dot(A, r)
        denominateur = np.dot(Ar, r)
        
        # Vérification pour éviter division par zéro
        if abs(denominateur) < 1e-15:
            print("Division par zéro détectée. Arrêt de l'algorithme.")
            break
        
        alpha = np.dot(r, r) / denominateur
        
        k = k + 1
        print(f"Itération {k} : ||r|| = {np.linalg.norm(r):.10f}, alpha = {alpha:.10f}")
        print(f"x = {x}")
        print()
        
        # Sécurité pour éviter une boucle infinie
        if k >= kmax:
            print(f"Attention : Nombre maximum d'itérations ({kmax}) atteint!")
            break
    
    if k < kmax and np.linalg.norm(r) < epsilon:
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
print("=== Méthode du Gradient à Pas Optimal ===\n")

A = saisir_matrice("A")
n = len(A)

# Vérification que A est symétrique
if not np.allclose(A, A.T):
    print("\nATTENTION : La matrice A n'est pas symétrique!")
    print("La méthode du gradient à pas optimal nécessite une matrice symétrique définie positive.")
    reponse = input("Voulez-vous continuer quand même ? (o/n) : ")
    if reponse.lower() != 'o':
        exit()

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

# Résolution
x = gradient_pas_optimal(A, b, x0, epsilon, kmax)

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
