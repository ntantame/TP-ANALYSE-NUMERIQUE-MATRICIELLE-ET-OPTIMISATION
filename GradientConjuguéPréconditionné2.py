import numpy as np

def gradient_conjugue_preconditionne_v2(A, b, M, x0, epsilon, kmax=1000):

    x = x0.copy()
    
    # Calcul du résidu initial : r = b - Ax
    r = b - np.dot(A, x)
    
    # Résoudre Mq = r
    q = np.linalg.solve(M, r)
    
    # Initialisation de p
    p = q.copy()
    
    k = 0
    
    print(f"Itération {k} : ||r|| = {np.linalg.norm(r):.10f}")
    print(f"x = {x}")
    print()
    
    while np.linalg.norm(r) >= epsilon:
        # Calcul de c = q.r
        c = np.dot(q, r)
        
        # Calcul de y = Ap
        y = np.dot(A, p)
        
        # Calcul du pas alpha
        denominateur = np.dot(y, p)
        
        # Vérification pour éviter division par zéro
        if abs(denominateur) < 1e-15:
            print("Division par zéro détectée. Arrêt de l'algorithme.")
            break
        
        alpha = c / denominateur
        
        # Mise à jour de x
        x = x + alpha * p
        
        # Mise à jour du résidu r
        r = r - alpha * y
        
        # Résoudre Mq = r
        q = np.linalg.solve(M, r)
        
        # Calcul de beta
        beta = np.dot(q, r) / c
        
        # Mise à jour de la direction de recherche p
        p = q + beta * p
        
        k = k + 1
        print(f"Itération {k} : ||r|| = {np.linalg.norm(r):.10f}, alpha = {alpha:.10f}, beta = {beta:.10f}")
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
print("=== Méthode du Gradient Conjugué Préconditionné (Version 2) ===\n")

A = saisir_matrice("A")
n = len(A)

# Vérification que A est symétrique
if not np.allclose(A, A.T):
    print("\nATTENTION : La matrice A n'est pas symétrique!")
    reponse = input("Voulez-vous continuer quand même ? (o/n) : ")
    if reponse.lower() != 'o':
        exit()

b = saisir_vecteur("b", n)

print("\nMatrice de préconditionnement M :")
print("1. Jacobi : M = D (diagonale de A)")
print("2. SSOR : M = (D + L) * D^(-1) * (D + L^T)")
print("3. Cholesky incomplet")
print("4. Identité (pas de préconditionnement)")
print("5. Matrice personnalisée")
choix = int(input("Choisissez une option : "))

if choix == 1:
    # Jacobi : M = D (diagonale de A)
    M = np.diag(np.diag(A))
    print("\nMatrice M (Jacobi - Diagonale de A) :")
    print(M)
elif choix == 2:
    # SSOR
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    D_inv = np.linalg.inv(D)
    M = np.dot(np.dot(D + L, D_inv), (D + L.T))
    print("\nMatrice M (SSOR) :")
    print(M)
elif choix == 3:
    # Cholesky incomplet (approximation simple : on prend la partie triangulaire inférieure)
    L_inc = np.tril(A)
    M = np.dot(L_inc, L_inc.T)
    print("\nMatrice M (Cholesky incomplet approximé) :")
    print(M)
elif choix == 4:
    # Identité (pas de préconditionnement)
    M = np.eye(n)
    print("\nMatrice M (Identité - pas de préconditionnement) :")
    print(M)
else:
    M = saisir_matrice("M")

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
print("\nMatrice M :")
print(M)
print("\nVecteur initial x0 :")
print(x0)
print(f"\nTolérance epsilon = {epsilon}")
print(f"Nombre maximum d'itérations kmax = {kmax}")
print()

# Résolution
x = gradient_conjugue_preconditionne_v2(A, b, M, x0, epsilon, kmax)

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
