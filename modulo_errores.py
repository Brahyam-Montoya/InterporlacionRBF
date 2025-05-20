import numpy as np
import matplotlib.pyplot as plt
from modulo_matriz import calcular_distancia, rbf_multicuadrica, rbf_multicuadrica_inversa
from modulo_puntos import funcion_original

def interpolar(X_new, Y_new, X, Y, coef, c, tipo='multicuadrica'):
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    puntos = np.column_stack((X_flat, Y_flat))
    puntos_nuevos = np.column_stack((X_new.ravel(), Y_new.ravel()))
    dist = np.linalg.norm(puntos_nuevos[:, None, :] - puntos[None, :, :], axis=2)

    if tipo == 'multicuadrica':
        phi = rbf_multicuadrica(dist, c)
    elif tipo == 'inversa':
        phi = rbf_multicuadrica_inversa(dist, c)
    else:
        raise ValueError("Tipo de RBF no válido.")
    
    Z_interp = phi @ coef
    return Z_interp.reshape(X_new.shape)

def graficar_superficie(X, Y, Z, titulo):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(titulo)
    plt.show()

def graficar_errores(X, Y, Z_real, Z_interp, titulo):
    error = np.abs(Z_real - Z_interp)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, error, cmap='inferno')
    ax.set_title(titulo)
    plt.show()