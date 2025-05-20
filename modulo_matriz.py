import numpy as np

def rbf_multicuadrica(r, c):
    return np.sqrt(r**2 + c)

def rbf_multicuadrica_inversa(r, c):
    return 1.0 / np.sqrt(r**2 + c)

def calcular_distancia(X, Y):
    puntos = np.column_stack((X.ravel(), Y.ravel()))
    dist = np.linalg.norm(puntos[:, None, :] - puntos[None, :, :], axis=2)
    return dist

def construir_sistema(X, Y, Z, c, tipo='multicuadrica'):
    dist = calcular_distancia(X, Y)
    if tipo == 'multicuadrica':
        A = rbf_multicuadrica(dist, c)
    elif tipo == 'inversa':
        A = rbf_multicuadrica_inversa(dist, c)
    else:
        raise ValueError("Tipo de RBF no válido.")
    b = Z.ravel()
    coef = np.linalg.solve(A, b)
    return coef, A