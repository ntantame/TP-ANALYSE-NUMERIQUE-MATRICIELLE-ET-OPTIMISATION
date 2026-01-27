import numpy as np

def gradient_conjugue_preconditionne(A, b, E, x0_tilde, epsilon, kmax=1000):
    
    x_tilde = x0_tilde.copy()
    
    # Calcul de E^(-1) et E^(-T)
    E_inv = np.linalg.inv(E)
    E_invT = E_inv.T
    
    # Calcul du résidu initial : r_tilde = E^(-1)*b - E^(-1)*A*E^(-T)*x_tilde
    r_tilde = np.dot(E_inv, b) - np.dot(np.dot(E_inv, A), np.dot(E_invT, x_tilde))
    
    # Initialisation de p_tilde
    p_tilde = r_tilde.copy()
    
    k = 0
    
    print(f"Itération {k} : ||r_tilde|| = {np.linalg.norm(r_tilde):.10f}")
    print(f"x_tilde = {x_tilde}")
    print()
    
    while np.linalg.norm(r_tilde) >= epsilon:
        # Calcul de c_tilde = r_tilde . r_tilde
        c_tilde = np.dot(r_tilde, r_tilde)
        
        # Calcul de y_tilde = E^(-1)*A*E^(-T)*p_tilde
        y_tilde = np.dot(np.dot(E_inv, A), np.dot(E_invT, p_tilde))
        
        # Calcul du pas alpha
        denominateur = np.dot(y_tilde, p_tilde)
        
        # Vérification pour éviter division par zéro
        if abs(denominateur) < 1e-15:
            print("Division par zéro détectée. Arrêt de l'algorithme.")
            break
        
        alpha = c_tilde / denominateur
        
        # Mise à jour de x_tilde
        x_tilde = x_tilde + alpha * p_tilde
        
        # Mise à jour du résidu r_tilde
        r_tilde = r_tilde - alpha * y_tilde
        
        # Calcul de beta
        beta = np.dot(r_tilde, r_tilde) / c_tilde
        
        # Mise à jour de la direction de recherche p_tilde
        p_tilde = r_tilde + beta * p_tilde
        
        k = k + 1
        print(f"Itération {k} : ||r_tilde|| = {np.linalg.norm(r_tilde):.10f}, alpha = {alpha:.10f}, beta = {beta:.10f}")
        print(f"x_tilde = {x_tilde}")
        print()
        
        # Sécurité pour éviter une boucle infinie
        if k >= kmax:
            print(f"Attention : Nombre maximum d'itérations ({kmax}) atteint!")
            break
    
    if k < kmax and np.linalg.norm(r_tilde) < epsilon:
        print(f"Convergence atteinte en {k} itérations.")
    
    # Retour à l'espace original : x = E^(-T)*x_tilde
    x = np.dot(E_invT, x_tilde)
    
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
print("=== Méthode du Gradient Conjugué Préconditionné ===\n")

A = saisir_matrice("A")
n = len(A)

# Vérification que A est symétrique
if not np.allclose(A, A.T):
    print("\nATTENTION : La matrice A n'est pas symétrique!")
    reponse = input("Voulez-vous continuer quand même ? (o/n) : ")
    if reponse.lower() != 'o':
        exit()

b = saisir_vecteur("b", n)

print("\nMatrice de préconditionnement E :")
print("1. Cholesky : E tel que A = E*E^T")
print("2. Jacobi : E = sqrt(D) où D est la diagonale de A")
print("3. Matrice personnalisée")
choix = int(input("Choisissez une option : "))

if choix == 1:
    # Cholesky : A = E*E^T, donc E = chol(A)
    try:
        E = np.linalg.cholesky(A)
        print("\nMatrice E (Cholesky de A) :")
        print(E)
    except np.linalg.LinAlgError:
        print("Erreur : La matrice A n'est pas définie positive, impossible d'utiliser Cholesky.")
        exit()
elif choix == 2:
    # Jacobi : E = sqrt(D)
    D = np.diag(np.diag(A))
    E = np.sqrt(D)
    print("\nMatrice E (racine de la diagonale de A) :")
    print(E)
else:
    E = saisir_matrice("E")

print("\nVecteur initial x0_tilde (dans l'espace préconditionné) :")
x0_tilde = saisir_vecteur("x0_tilde", n)

epsilon = float(input("\nEntrez la tolérance (epsilon) : "))
kmax = int(input("Entrez le nombre maximum d'itérations (kmax) : "))

print("\n" + "="*60)
print("Début des itérations")
print("="*60 + "\n")

print("Matrice A :")
print(A)
print("\nVecteur b :")
print(b)
print("\nMatrice E :")
print(E)
print("\nVecteur initial x0_tilde :")
print(x0_tilde)
print(f"\nTolérance epsilon = {epsilon}")
print(f"Nombre maximum d'itérations kmax = {kmax}")
print()

# Résolution
x = gradient_conjugue_preconditionne(A, b, E, x0_tilde, epsilon, kmax)

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
