"""
Considera las imágenes monedas1.png, monedas2.png y monedas3.png de la práctica anterior.

Escribe un programa que, tomando como entrada dichas imágenes, genere resultados similares a los del enunciado.

Para ello, debes umbralizar la imagen mediante algún método de cálculo automático del umbral, utilizar operaciones de
morfología matemática para eliminar las regiones pequeñas (ruido), rellenar huecos, eliminar regiones que no sean
redondas, etc. Después, basándote en las propiedades de las regiones, debes generar un resultado que muestre en rojo el
área aproximada de las monedas de un euro y en verde, el área de las de 10 céntimos. El procesamiento que apliques debe
ser exactamente el mismo para las tres imágenes.

En la memoria debes indicar qué método de cálculo automático del umbral has utilizado y qué valores ha proporcionado
para cada una de las imágenes, así como qué secuencia de operaciones de morfología matemática has utilizado. Por último,
explica qué criterio has utilizado para diferenciar las monedas de un euro de las de 10 céntimos.

Ayuda: En la tercera imagen te puede resultar difícil separar las 3 monedas que se están tocando entre sí. Para
solucionar este problema, te proponemos que hagas una erosión con un disco lo suficientemente grande como para que en
las monedas pequeñas permanezca tan solo una mínima semilla central. Seguidamente, haz una dilatación con un disco de un
tamaño ligeramente inferior al usado en la erosión. De este modo, las monedas recuperarán un tamaño similar al original,
pero sin volver a conectarse entre sí.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

monedas1 = ski.io.imread("images/monedas1.png")
monedas2 = ski.io.imread("images/monedas2.png")
monedas3 = ski.io.imread("images/monedas3.png")


def draw_circle(image):
    umbral = ski.filters.threshold_otsu(image)
    img_umbralizada = image > umbral    # Valor True a los píxeles cuya intensidad es mayor que el umbral calculado

    img_umbralizada = ski.morphology.remove_small_objects(img_umbralizada)
    img_umbralizada = ski.morphology.remove_small_holes(img_umbralizada)

    erosion = ski.morphology.disk(18)
    dilatacion = ski.morphology.disk(16)
    img_erosion = ski.morphology.erosion(img_umbralizada, footprint=erosion)
    img_dilatacion = ski.morphology.dilation(img_erosion, footprint=dilatacion)
    img_cierre = ski.morphology.binary_closing(img_dilatacion)

    img_etiquetada = ski.morphology.label(img_cierre)
    props = ski.measure.regionprops(img_etiquetada)

    img_monedas = np.zeros(image.shape + (3,))
    for p in props:
        if p.eccentricity < 0.75:
            if p.area > 1200:
                img_monedas[img_etiquetada == p.label] = [1, 0, 0]
            elif p.area < 1200:
                img_monedas[img_etiquetada == p.label] = [0, 1, 0]

    return img_monedas, umbral


resultados = [monedas1, monedas2, monedas3]
nombres = ["Monedas 1", "Monedas 2", "Monedas 3"]

# Lista para almacenar los umbrales de cada imagen
umbrales = []

fig, axs = plt.subplots(nrows=len(resultados), ncols=2, layout="constrained")
fig.suptitle("Rellenos de las monedas detectadas", fontsize=24)

for i, imagen in enumerate(resultados):
    img_circulos, umbral = draw_circle(imagen)
    umbrales.append(umbral)

    axs[i, 0].imshow(resultados[i], cmap='gray')
    axs[i, 0].set_title(f"Monedas original {i+1}", fontsize=16)

    axs[i, 1].imshow(img_circulos, cmap='gray')
    axs[i, 1].set_title(nombres[i], fontsize=16)

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()

# Imprimir los umbrales calculados para cada imagen
print("Umbrales calculados:", umbrales)



