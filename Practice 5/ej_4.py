"""
El ejemplo 3 de teoría permite comprobar el teorema de la convolución, es decir, que la convolución en el espacio es
equivalente a un producto punto a punto en las frecuencias.

En este ejercicio debes aplicar el filtro media, trabajando tanto en el espacio como en las frecuencias, con tamaños
impares de 3 a 21 (ambos incluidos), sobre una imagen cualquiera, por ejemplo, boat.511.tiff.

Debes mostrar un resultado similar al del enunciado.

Como puedes ver, aparece la imagen utilizada en la prueba y dos gráficas: la línea roja muestra el tiempo empleado por
la convolución en el espacio para cada tamaño de máscara, mientras que la línea azul muestra el tiempo empleado cuando
trabajamos en las frecuencias.

Para evitar fluctuaciones extrañas en las gráficas, debes repetir cada convolución un cierto número de veces y
considerar su tiempo de ejecución como el promedio de todas las repeticiones. En la gráfica anterior, cada convolución
se repitió 10 veces.

Comenta las gráficas obtenidas. Céntrate en la diferencia de tiempos de ejecución cuando trabajamos en el espacio o en
las frecuencias.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft as fft
import time

# Definir los tamaños de las máscaras
mask_sizes = list(range(3, 22, 2))

imagen = ski.io.imread("images/boat.511.tiff")
imagen = ski.util.img_as_float(imagen)

# Listas para almacenar los tiempos de ejecución
tiempos_espacio = []
tiempos_frecuencia = []

# Repetir cada convolución 10 veces para obtener un promedio
repeticiones = 10

for mask_size in mask_sizes:
    # Convolución en el espacio
    mascara = np.ones((mask_size, mask_size))  # Máscara de NxN toda con 1
    mascara /= np.sum(mascara)  # Máscara normalizada

    # Medir el tiempo de ejecución para la convolución en el espacio
    inicio_espacio = time.time()
    for x in range(repeticiones):
        res_convol = scipy.ndimage.convolve(imagen, mascara, mode="wrap")
    fin_espacio = time.time()

    tiempo_promedio_espacio = (fin_espacio - inicio_espacio) / repeticiones
    tiempos_espacio.append(tiempo_promedio_espacio)

    # Ampliamos la máscara con ceros para que tenga el mismo tamaño que la imagen
    mascara_centrada = np.zeros(imagen.shape)
    fila_i = imagen.shape[0] // 2 - mask_size // 2
    col_i = imagen.shape[1] // 2 - mask_size // 2
    mascara_centrada[fila_i:fila_i + mask_size, col_i:col_i + mask_size] = mascara

    # Pasamos imagen y máscara a las frecuencias
    FTimagen = fft.fft2(imagen)
    mascara_en_origen = fft.ifftshift(mascara_centrada)
    FTmascara = fft.fft2(mascara_en_origen)

    # Medir el tiempo de ejecución para la convolución en las frecuencias
    inicio_frecuencia = time.time()
    for x in range(repeticiones):
        FTimagen_filtrada = FTimagen * FTmascara  # Producto punto a punto
        res_filtro_FT = fft.ifft2(FTimagen_filtrada)
    fin_frecuencia = time.time()

    tiempo_promedio_frecuencia = (fin_frecuencia - inicio_frecuencia) / repeticiones
    tiempos_frecuencia.append(tiempo_promedio_frecuencia)


# Recuperamos resultado en el espacio
res_filtro_FT = fft.ifft2(FTimagen_filtrada)
res_filtro_real = np.real(res_filtro_FT)
res_filtro_imag = np.imag(res_filtro_FT)
if not np.allclose(res_filtro_imag, np.zeros(imagen.shape)):
    print("Warning. Algo no está yendo bien!!!")

# Comparamos los dos resultados
print("¿Obtenemos el mismo resultado?", np.allclose(res_convol, res_filtro_FT))

# Visulización de resultados
fig, axs = plt.subplots(1, 2, layout="constrained")

# Mostrar la imagen original
axs[0].imshow(imagen, cmap=plt.cm.gray)
axs[0].axis('off')

plt.plot(mask_sizes, tiempos_espacio, 'r-', label='Espacio')
plt.plot(mask_sizes, tiempos_frecuencia, 'b-', label='Frecuencia')
plt.show()
