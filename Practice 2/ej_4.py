"""
A partir de la imagen en color del ejercicio 1, mejora su contraste de las siguientes formas:

*Aplicando el proceso de ecualización a cada banda de color. En este caso, debes crear una nueva función utilizando el
decorador adapt_rgb con el parámetro each_channel, como se propuso en el ejercicio 1. Si ya lo hiciste así en ese
ejercicio, solo tienes que reutilizar tu código.

*Aplicando también el proceso de ecualización y creando una nueva función utilizando el decorador adapt_rgb, pero ahora
con el parámetro hsv_value.

Muestra la imagen original y las dos imágenes ecualizadas. ¿Qué resultado te parece más adecuado? Para entender lo que
hace el parámetro hsv_value, acaba de leer el ejemplo Adapting gray-scale filters to RGB images.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value


@adapt_rgb(each_channel)
def equalization_each(image):
    return ski.exposure.equalize_hist(image)


@adapt_rgb(each_channel)
def equalization_adaptativa(image, value):
    return ski.exposure.equalize_adapthist(image, kernel_size=value)


@adapt_rgb(hsv_value)
def equalization_hsv(image):
    return ski.exposure.equalize_hist(image)


@adapt_rgb(hsv_value)
def equalization_adaptativa_hsv(image, value):
    return ski.exposure.equalize_adapthist(image, kernel_size=value)


def rgb(image, axis, fila):
    banda = ["Roja", "Verde", "Azul"]
    for nBanda, color in enumerate(banda):
        h_orig, c_orig = ski.exposure.histogram(image[:, :, nBanda])

        axis[fila, nBanda + 1].bar(c_orig, h_orig, 1.1)


img_original = ski.io.imread("images/calle.png")
h_orig, c_orig = ski.exposure.histogram(img_original)

img_real = ski.util.img_as_float(img_original) * 255

img_eq = equalization_each(img_real / 255)
img_eq = ski.util.img_as_ubyte(img_eq)
h_eq, c_eq = ski.exposure.histogram(img_eq)

img_eq_adapt_8 = equalization_adaptativa(img_real / 255, 2)
img_eq_adapt_8 = ski.util.img_as_ubyte(img_eq_adapt_8)
h_eq_adapt_8, c_eq_adapt_8 = ski.exposure.histogram(img_eq_adapt_8)

img_eq_hsv = equalization_hsv(img_real / 255)
img_eq_hsv = ski.util.img_as_ubyte(img_eq_hsv)
h_eq_hsv, c_eq_hsv = ski.exposure.histogram(img_eq_hsv)

img_eq_adapt_8_hsv = equalization_adaptativa_hsv(img_real / 255, 2)
img_eq_adapt_8_hsv = ski.util.img_as_ubyte(img_eq_adapt_8_hsv)
h_eq_adapt_2_hsv, c_eq_adapt_2_hsv = ski.exposure.histogram(img_eq_adapt_8_hsv)

fig, axs = plt.subplots(5, 4, layout="constrained")
imagenes = [img_original, img_eq, img_eq_adapt_8, img_eq_hsv, img_eq_adapt_8_hsv]

for n, imagen in enumerate(imagenes):
    axs[n, 0].imshow(imagen, cmap=plt.cm.gray)
    rgb(imagen, axs, n)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 4):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
    axs_lineal[i + 2].set_xticks([0, 64, 128, 192, 255])
    axs_lineal[i + 3].set_xticks([0, 64, 128, 192, 255])
plt.show()
