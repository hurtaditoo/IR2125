"""
Considera las imágenes monedas1.png, monedas2.png y monedas3.png. Te puede ser útil la siguiente información:
*las imágenes han sido capturadas utilizando una resolución de 50 puntos por pulgada (una pulgada equivale a 2,54 cm).
*el diámetro de una moneda de euro es de 23 mm y el de una moneda de 10 céntimos es de 19,5 mm.

Escribe un programa que para cada una de dichas imágenes genere un resultado similar al del enunciado, que se
corresponde con la imagen monedas1.png.

Como ves, en el resultado se muestran los contornos de las monedas detectadas. La monedas de un euro se muestran en un
color y las de 10 céntimos en otro.

Para solucionar este ejercicio puedes utilizar cualquier preprocesado que consideres oportuno de los que hemos estudiado
(filtros para eliminar ruido, detectores de bordes, etc.). Finalmente, debes utilizar la transformada de Hough para
círculos para la detección de las monedas. Puedes ajustar todos los parámetros necesarios como consideres oportuno, pero
debes utilizar los mismos valores para todas las imágenes. No se aceptan soluciones que ajusten los parámetros en
función de la imagen que se esté procesando.

Comenta en la memoria los pasos de preprocesado que has utilizado, los parámetros utilizados en cada caso y los
resultados finales obtenidos.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Leer la imagen y crear mapa de bordes usando el gradiente de Sobel
monedas1 = ski.io.imread("images/monedas1.png")
monedas2 = ski.io.imread("images/monedas2.png")
monedas3 = ski.io.imread("images/monedas3.png")


def detector_contornos(image):
    mapa_bordes = ski.feature.canny(image, sigma=3)

    # Transformada de Hough para círculos
    radios_posibles = np.arange(16, 25, 1)  # Buscará círculos con radios entre 8 y 14 de 1 en 1
    hough_res = ski.transform.hough_circle(mapa_bordes, radios_posibles)
    accums, cx, cy, radii = ski.transform.hough_circle_peaks(hough_res, radios_posibles, min_xdistance=10,
                                                             min_ydistance=10,
                                                             threshold=hough_res.max() / 2)  # El threshold no debería ser necesario. Se supone que es el valor por defecto

    # Dibujar los círculos en la imagen resultado
    resultado = np.zeros(image.shape + (3,))
    for fila, col, radio in zip(cy, cx, radii):
        if radio > 21:
            circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=image.shape)  # Dibuja un círculo
            resultado[circy, circx] = [1, 0, 0]  # Moneda de 1 euro en rojo
        else:
            circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=image.shape)  # Dibuja un círculo
            resultado[circy, circx] = [0, 1, 0]  # Moneda de 10 céntimos en verde

    return resultado


resultados = [monedas1, monedas2, monedas3]
nombres = ["Monedas 1", "Monedas 2", "Monedas 3"]

fig, axs = plt.subplots(nrows=1, ncols=len(resultados), layout="constrained")
fig.suptitle("Contornos de las monedas detectadas", fontsize=24)

for i in range(len(resultados)):
    axs[i].imshow(detector_contornos(resultados[i]), cmap='gray')
    axs[i].set_title(nombres[i], fontsize=16)

for ax in axs:
    ax.set_axis_off()
plt.show()
