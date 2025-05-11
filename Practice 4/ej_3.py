"""
En el ejemplo 3 de teoría se puede ver que cuando una máscara de convolución 2D es separable, obtenemos el mismo
resultado aplicando dos convoluciones con vectores unidimensionales.

En este ejercicio debes aplicar el filtro media, tanto con máscaras bidimensionales como unidimensionales de tamaños
impares de 3 a 21 (ambos incluidos), sobre una imagen cualquiera, por ejemplo, boat.512.tiff.

Debes mostrar un resultado similar al del enunciado.

Como puedes ver, aparece la imagen utilizada en la prueba y dos gráficas: la línea roja muestra el tiempo empleado en la
convolución bidimensional para cada tamaño de máscara, mientras que la línea azul muestra el tiempo empleado por las
convoluciones unidimensionales.

Para evitar fluctuaciones extrañas en las gráficas, debes repetir cada convolución un cierto número de veces y
considerar su tiempo de ejecución como el promedio de todas las repeticiones. En la gráfica anterior, cada convolución
se repitió 10 veces.

Comenta las gráficas obtenidas. Céntrate en la diferencia de tiempos de ejecución entre la convolución 2D y las dos
convoluciones 1D.
"""
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

STD_DEV = 3
# Definir los tamaños de las máscaras
mask_sizes = list(range(3, 22, 2))

imagen = ski.io.imread("images/boat.512.tiff")
imagen = ski.util.img_as_float(imagen)

# Listas para almacenar los tiempos de ejecución
tiempos_2D = []
tiempos_1D = []

# Repetir cada convolución 10 veces para obtener un promedio
repeticiones = 10

for mask_size in mask_sizes:
    # Generar la máscara 1D
    vector = scipy.signal.windows.gaussian(mask_size, STD_DEV)
    vector /= np.sum(vector)

    vectorH = vector.reshape(1, mask_size)
    vectorV = vector.reshape(mask_size, 1)

    # Generar la máscara 2D
    matriz = vectorV @ vectorH

    # Medir el tiempo de ejecución para la convolución 2D
    inicio_2D = time.time()
    for x in range(repeticiones):
        res_convol2D = scipy.ndimage.convolve(imagen, matriz)
    fin_2D = time.time()

    # Medir el tiempo de ejecución para la convolución 1D
    inicio_1D = time.time()
    for x in range(repeticiones):
        resH = scipy.ndimage.convolve(imagen, vectorH)
        res1D = scipy.ndimage.convolve(resH, vectorV)
    fin_1D = time.time()

    tiempo_promedio_2D = (fin_2D - inicio_2D) / repeticiones
    tiempos_2D.append(tiempo_promedio_2D)

    tiempo_promedio_1D = (fin_1D - inicio_1D) / repeticiones
    tiempos_1D.append(tiempo_promedio_1D)

tiempos_1D_float = [float(tiempo) for tiempo in tiempos_1D]  # Convierte los strings a float
tiempos_2D_float = [float(tiempo) for tiempo in tiempos_2D]  # Convierte los strings a float
tiempos2D = np.mean(tiempos_2D_float)   # Calcula la media
tiempos1D = np.mean(tiempos_1D_float)   # Calcula la media

print("¿Obtenemos el mismo resultado?", np.allclose(res_convol2D, res1D))
print(f"Tiempo empleado con máscara  2D: {tiempos2D:0.9f}")
print(f"Tiempo empleado con máscaras 1D: {tiempos1D:0.9f}")
print(f"Factor 2D/1D: {tiempos2D / tiempos1D:0.2f}")

# Graficar los tiempos de ejecución
fig, axs = plt.subplots(1, 2, layout='constrained')

# Mostrar la imagen original
axs[0].imshow(imagen, cmap=plt.cm.gray)
axs[0].axis('off')

# Graficar los tiempos de ejecución de las convoluciones
axs[1].plot(mask_sizes, tiempos_2D, 'r-', label='Convolución 2D')
axs[1].plot(mask_sizes, tiempos_1D, 'b-', label='Convolución 1D')

plt.show()
