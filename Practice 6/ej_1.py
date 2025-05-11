"""
A partir de la imagen cuadros.png, genera tres nuevas imágenes añadiendo a la imagen original ruido gaussiano con
diferentes varianzas: 0.001, 0.0015 y 0.0025. Para cada una de las 4 imágenes (la original y las tres ruidosas) realiza
los siguientes procesos:

-Sobel:
    *Obtén el gradiente de la imagen con el operador de Sobel
    *Genera el mapa de bordes con la función apply_hysteresis_threshold
    *Refina el mapa de bordes para que el grosor de los bordes sea de un solo píxel (mira cómo se usa la función thin en
    los ejemplos de teoría) - Resultado 1
    *Obtén los segmentos correspondientes a sus contornos utilizando la transformada de Hough probabilística y
    progresiva - Resultado 2

-Canny:
    *Obtén el mapa de bordes con el detector de Canny - Resultado 3
    *Obtén los segmentos correspondientes a sus contornos utilizando la transformada de Hough probabilística y
    progresiva - Resultado 4

Para cada una de las 4 imágenes debes mostrar los 4 resultados que se indican más arriba.

Para las funciones apply_hysteresis_threshold, canny y probabilistic_hough_line puedes especificar los parámetros que
consideres más adecuados, pero debes utilizar los mismos parámetros (o el mismo criterio) para todas las imágenes.
Es decir, si un parámetro toma el valor 10, siempre debe tomar ese valor. Alternativamente, puede tomar un porcentaje de
un valor concreto (por ejemplo, el 10 % del gradiente máximo). En ese caso, el porcentaje siempre debe ser el 10% y no
puede variar de una imagen a otra.

Comenta todos los resultados obtenidos así como el significado de los parámetros que hayas tenido que especificar en
cada una de las funciones indicadas en el párrafo anterior.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

image = ski.io.imread("images/cuadros.png")

# Añadir ruido Gaussiano
gaussian_values = (0.001,
                   0.0015,
                   0.0025)

gaussian_noise = [image]
for i in range(len(gaussian_values)):
    img_noise = ski.util.random_noise(image, mode="gaussian", var=gaussian_values[i])
    gaussian_noise.append(img_noise)


def sobel(image):
    gradiente = ski.filters.sobel(image)

    maximo = gradiente.max()
    low = maximo * 0.1
    high = maximo * 0.2
    mapa_bordes = ski.filters.apply_hysteresis_threshold(gradiente, low, high)

    mapa_bordes = ski.morphology.thin(mapa_bordes)  # Reduce el grosor de los bordes a un solo píxel
    resultado1 = mapa_bordes

    # Transformada de Hough Probabilística y Progresiva
    segmentos = ski.transform.probabilistic_hough_line(mapa_bordes, threshold=10, line_length=5, line_gap=3)
    resultado2 = segmentos

    return [resultado1, resultado2]


def canny(image):
    mapa_bordes = ski.feature.canny(image, sigma=3)
    resultado3 = mapa_bordes

    # Transformada de Hough Probabilística y Progresiva
    segmentos = ski.transform.probabilistic_hough_line(mapa_bordes, threshold=16, line_length=8,
                                                       line_gap=6)  # threshold=10, line_length=5, line_gap=3
    resultado4 = segmentos

    return [resultado3, resultado4]


def mostrar(img_fila1, img_fila2, msg_fila1, msg_fila2, titulo):
    fig, ax = plt.subplots(nrows=2, ncols=len(img_fila1), layout="constrained")
    fig.suptitle(titulo, fontsize=24)
    for i in range(len(img_fila1)):
        if i % 2 == 0:
            ax[0, i].imshow(img_fila1[i], cmap='gray')
        else:
            ax[0, i].imshow(np.zeros(img_fila1[0].shape), cmap='gray')
            for segmento in img_fila1[i]:
                p0, p1 = segmento
                ax[0, i].plot((p0[0], p1[0]), (p0[1], p1[1]), color='r')  # Dibujar segmento
        ax[0, i].set_title(msg_fila1[i], fontsize=16)

        if i % 2 == 0:
            ax[1, i].imshow(img_fila2[i], cmap='gray')
        else:
            ax[1, i].imshow(np.zeros(img_fila2[0].shape), cmap='gray')
            for segmento in img_fila2[i]:
                p0, p1 = segmento
                ax[1, i].plot((p0[0], p1[0]), (p0[1], p1[1]), color='r')  # Dibujar segmento
        ax[1, i].set_title(msg_fila2[i], fontsize=16)
    for a in ax.ravel():
        a.set_axis_off()
    plt.show()


for idx, img_noise in enumerate(gaussian_noise):
    if idx == 0: continue
    # Resultados para Sobel
    sobel_results = sobel(img_noise)
    sobel_msg = ["Mapa de bordes (Sobel)", "Hough (Sobel)"]

    # Resultados para Canny
    canny_results = canny(img_noise)
    canny_msg = ["Mapa de bordes (Canny)", "Hough (Canny)"]

    tablas = [sobel_results[0], sobel_results[1], canny_results[0], canny_results[1]]
    for x in tablas:
        print(x)

    # Mostrar resultados
    mostrar(sobel_results, canny_results, sobel_msg, canny_msg,
            f"Imagen con ruido gaussiano {gaussian_values[idx - 1]}")
