import numpy as np

def funcion_original(x, y):
    return x * y + 1 / (2 * x) + 1 / (2 * y)

def generar_puntos(n, m, xmin=-3, xmax=3, ymin=-1, ymax=6):
    x_vals = np.linspace(xmin, xmax, n)
    y_vals = np.linspace(ymin, ymax, m)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = funcion_original(X, Y)
    return X, Y, Z, x_vals, y_vals