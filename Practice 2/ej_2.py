"""
Queremos visualizar la misma imagen que has elegido en el ejercicio anterior, pero ahora en niveles de gris. Para ello,
utiliza la funciónrgb2gray para obtener una imagen en niveles de gris para cada una de las 7 imágenes en color
visualizadas en el ejercicio anterior. Muestra las 7 imágenes en niveles de gris junto con sus correspondientes
histogramas y comenta los resultados obtenidos.
"""

import skimage as ski
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# Crear una figura con 7 filas y 2 columnas
fig, axs = plt.subplots(7, 2, layout="constrained")

# Lista de nombres de las imágenes
nombres = ["img_orig", "img_clara", "img_clara_2", "img_eq", "img_eq_adapt_2", "img_eq_adapt_4",
           "img_eq_adapt_8"]

for n, imagen in enumerate(nombres):
    # Cargar la imagen original
    img_original = ski.io.imread(f'images/{imagen}.png')

    # Convertir la imagen a niveles de gris
    img_gray = rgb2gray(img_original)
    img_gray = ski.util.img_as_ubyte(img_gray)

    # Calcular el histograma de la imagen en niveles de gri
    h_gray, c_gray = ski.exposure.histogram(img_gray)

    axs[n, 0].imshow(img_gray, cmap=plt.cm.gray)

    # Mostrar el histograma en la segunda columna de la fila actual
    axs[n, 1].bar(c_gray, h_gray, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
