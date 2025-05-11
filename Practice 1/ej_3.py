"""
Considera la imagen mapas.png. A partir de ella, genera imágenes jpg con calidades del 100%, 75% y 15%. Calcula los
errores cometidos para cada banda de cada imagen al codificar las imágenes en jpg.

NOTA. En los programas de ejemplo tienes funciones para calcular el máximo error absoluto, el error medio y el error
cuadrático medio.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

imagenOriginal = ski.io.imread("images/mapas.png")
ski.io.imsave("mapas15.jpg", imagenOriginal, quality=15)
imag15 = ski.io.imread("mapas15.jpg")
ski.io.imsave("mapas75.jpg", imagenOriginal, quality=75)
imag75 = ski.io.imread("mapas75.jpg")
ski.io.imsave("mapas100.jpg", imagenOriginal, quality=100)
imag100 = ski.io.imread("mapas100.jpg")

imagenOriginal = ski.util.img_as_float(imagenOriginal)  # Valores flotantes en el rango [0,1]
imagen15 = ski.util.img_as_float(imag15)
imagen75 = ski.util.img_as_float(imag75)
imagen100 = ski.util.img_as_float(imag100)


def errorCuadráticoMedioBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.sum(np.power(mo2 - mo1, 2), None) / m1.size


def errorMedioBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.sum(abs(mo2 - mo1), None) / m1.size


def maximoErrorAbsolutoBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.max(np.abs(mo2 - mo1))


bandas = ["Roja", "Verde", "Azul"]
imagenes = [imagen15, imagen75, imagen100]
nombres = ['de 15', 'de 75', 'de 100']

fig, axs = plt.subplots(4, 3, layout="constrained")
axs[0][0].imshow(imagenOriginal)

for i in range(len(imagenes)):
    print(f"---Imagen {nombres[i]} ---")
    for nBanda, nombre in enumerate(bandas):
        print(f"Banda {nombre}")
        print(f"   Máximo error: {maximoErrorAbsolutoBanda(imagenOriginal[:, :, nBanda], imagenes[i][:, :, nBanda])}")
        print(f"   Error medio: {errorMedioBanda(imagenOriginal[:, :, nBanda], imagenes[i][:, :, nBanda])}")
        print(
            f"   Error cuadrático medio: {errorCuadráticoMedioBanda(imagenOriginal[:, :, nBanda], imagenes[i][:, :, nBanda])}")

    errores = []
    for nBanda in range(3):
        errorBanda = abs(imagenes[i][:, :, nBanda] - imagenOriginal[:, :, nBanda])
        errores.append(errorBanda)
    errorGlobal = np.stack(errores, axis=-1)

    axs[i+1][0].imshow(imagenes[i])
    axs[i+1][1].imshow(errorGlobal/errorGlobal.max())  # Piensa cómo mejorar la visualización del error

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
