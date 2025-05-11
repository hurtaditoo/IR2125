"""
Elige una de las dos imágenes oscuras que te proporcionamos: calle.png o templo.png. Como puedes observar, se trata de
imágenes en color. También puedes utilizar cualquier otra imagen oscura que tengas.

Intenta mejorar el contraste de la imagen elegida de las siguientes formas:

*Usando las dos funciones de aclarado que hemos estudiado en la asignatura.
*Mediante la ecualización del histograma.
*Mediante la ecualización adaptativa del histograma. En este caso, prueba los valores 2, 4 y 8 para el parámetro
kernel_size.

Ten en cuenta que las funciones de ecualización del histograma solo admiten imágenes en niveles de gris. Para aplicarlas
sobre imágenes en color tienes dos opciones:

*Programar tú explícitamente que cada función se aplique sobre cada canal de color R, G, B independientemente.
*Crear una nueva función utilizando el decorador adapt_rgb con el parámetro each_channel. Puedes leer la documentación
del ejemplo Adapting gray-scale filters to RGB images. No es necesario leer el ejemplo completo, basta con la parte
inicial.

Como resultado final debes visualizar las 7 imágenes en color (original, aclarada 1, aclarada 2, ecualizada,  ecualizada
adaptativa 2, ecualizada adaptativa 4 y ecualizada adaptativa 8) y, junto a cada una de ellas, los histogramas de sus
correspondientes canales R, G, B. No olvides comentar en la memoria los resultados obtenidos, tanto desde el punto de
vista de la mejora del contraste en cada caso, como de la influencia del parámetro kernel_size en la ecualización
adaptativa.

¿Consideras que tiene sentido hacer una ecualización de histograma en una imagen en color aplicando el proceso a cada
banda por separado?
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color.adapt_rgb import adapt_rgb, each_channel


@adapt_rgb(each_channel)
def equalization_each(image):
    return ski.exposure.equalize_hist(image)


@adapt_rgb(each_channel)
def equalization_adaptativa(image, value):
    return ski.exposure.equalize_adapthist(image, kernel_size=value)


def rgb(image, axis, fila):
    banda = ["Roja", "Verde", "Azul"]
    for nBanda, color in enumerate(banda):
        h_orig, c_orig = ski.exposure.histogram(image[:, :, nBanda])

        axis[fila, nBanda + 1].bar(c_orig, h_orig, 1.1)


img_original = ski.io.imread("images/calle.png")
h_orig, c_orig = ski.exposure.histogram(img_original)

img_real = ski.util.img_as_float(img_original) * 255

img_clara = np.sqrt(255 * img_real)
img_clara = ski.util.img_as_ubyte(img_clara / 255)
h_clara, c_clara = ski.exposure.histogram(img_clara)

img_clara_2 = np.cbrt(255 ** 2 * img_real)
img_clara_2 = ski.util.img_as_ubyte(img_clara_2 / 255)
h_clara_2, c_clara_2 = ski.exposure.histogram(img_clara_2)

img_eq = equalization_each(img_real / 255)
img_eq = ski.util.img_as_ubyte(img_eq)
h_eq, c_eq = ski.exposure.histogram(img_eq)

img_eq_adapt_2 = equalization_adaptativa(img_real / 255, 2)
img_eq_adapt_2 = ski.util.img_as_ubyte(img_eq_adapt_2)
h_eq_adapt_2, c_eq_adapt_2 = ski.exposure.histogram(img_eq_adapt_2)

img_eq_adapt_4 = equalization_adaptativa(img_real / 255, 4)
img_eq_adapt_4 = ski.util.img_as_ubyte(img_eq_adapt_4)
h_eq_adapt_4, c_eq_adapt_4 = ski.exposure.histogram(img_eq_adapt_4)

img_eq_adapt_8 = equalization_adaptativa(img_real / 255, 8)
img_eq_adapt_8 = ski.util.img_as_ubyte(img_eq_adapt_8)
h_eq_adapt_8, c_eq_adapt_8 = ski.exposure.histogram(img_eq_adapt_8)

fig, axs = plt.subplots(7, 4, layout="constrained")
imagenes = [img_original, img_clara, img_clara_2, img_eq, img_eq_adapt_2, img_eq_adapt_4, img_eq_adapt_8]
nombres = ["img_orig", "img_clara", "img_clara_2", "img_eq", "img_eq_adapt_2", "img_eq_adapt_4",
           "img_eq_adapt_8"]

for n, imagen in enumerate(imagenes):
    axs[n, 0].imshow(imagen, cmap=plt.cm.gray)
    rgb(imagen, axs, n)
    img = f'images/{nombres[n]}.png'
    ski.io.imsave(fname=img, arr=imagen)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 4):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
    axs_lineal[i + 2].set_xticks([0, 64, 128, 192, 255])
    axs_lineal[i + 3].set_xticks([0, 64, 128, 192, 255])

plt.show()
