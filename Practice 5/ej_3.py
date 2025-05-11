"""
Considera una imagen de tamaño 101x101. A partir de las máscaras gaussianas centradas que has generado en los ejercicios
1 y 2, muestra en una misma gráfica el perfil de la línea central de cada máscara (en rojo la del ejercicio 1 y en azul,
la del 2).

Comenta las gráficas obtenidas. ¿Puedes explicar por qué la transformada de Fourier de la segunda gaussiana se parece
más a la transformada de un filtro media que a la transformada de un filtro gaussiano? ¿Puedes dar alguna recomendación
a la hora de diseñar un filtro gaussiano?
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

size = 101
MASK_SIZES = [5, 31]
SIGMA = 5


def perfil_linea_central(size, MASK_SIZE, SIGMA):
    # Convolución en el espacio
    mascara = np.outer(scipy.signal.windows.gaussian(MASK_SIZE, SIGMA), scipy.signal.windows.gaussian(MASK_SIZE, SIGMA))
    mascara /= np.sum(mascara)  # Máscara normalizada

    # Ampliamos la máscara con ceros para que tenga el mismo tamaño que la imagen
    mascara_centrada = np.zeros((size, size))
    fila_i = size // 2 - MASK_SIZE // 2
    col_i = size // 2 - MASK_SIZE // 2
    mascara_centrada[fila_i:fila_i + MASK_SIZE, col_i:col_i + MASK_SIZE] = mascara

    perfil = mascara_centrada[size//2]
    return perfil


perfil_5 = perfil_linea_central(size, MASK_SIZES[0], SIGMA)
perfil_31 = perfil_linea_central(size, MASK_SIZES[1], SIGMA)

# Visulización de resultados
fig, axs = plt.subplots(1, 1, layout="constrained")
axs.plot(perfil_5, 'b-', label='5')
axs.plot(perfil_31, 'r-', label='31')

plt.show()
