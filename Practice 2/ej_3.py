"""
Vamos a repetir el ejercicio anterior, pero con un enfoque distinto. A partir de la imagen original en color, obtén una
versión en niveles de gris mediante la función rgb2gray. Trata de mejorar el contraste de esta imagen en niveles de gris
empleando las mismas funciones de aclarado y ecualización que has usado en el ejercicio 1. Muestra las 7 imágenes
obtenidas junto con sus correspondientes histogramas y comenta los resultados obtenidos.

Los resultados obtenidos en este caso son parecidos a los obtenidos en el ejercicio 2, pero ¿son exactamente iguales?
Razona tu respuesta.

¿Qué proceso te parece más acertado para obtener una imagen en niveles de gris en la que hayamos mejorado el contraste?
¿El utilizado en los ejercicios 1 y 2 o el utilizado en el ejercicio 3? Justifica tu respuesta.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

images = []

img_original = ski.io.imread("images/calle.png")
img_original = rgb2gray(img_original)
images.append(img_original)

img_real_gray = ski.util.img_as_float(img_original) * 255

img_clara = np.sqrt(255 * img_real_gray)
img_clara = ski.util.img_as_ubyte(img_clara / 255)
images.append(img_clara)

img_clara_2 = np.cbrt(255 ** 2 * img_real_gray)
img_clara_2 = ski.util.img_as_ubyte(img_clara_2 / 255)
images.append(img_clara_2)

img_eq = ski.exposure.equalize_hist(img_real_gray / 255)
img_eq = ski.util.img_as_ubyte(img_eq)
images.append(img_eq)

img_eq_adapt_2 = ski.exposure.equalize_adapthist(img_real_gray / 255, 2)
img_eq_adapt_2 = ski.util.img_as_ubyte(img_eq_adapt_2)
images.append(img_eq_adapt_2)

img_eq_adapt_4 = ski.exposure.equalize_adapthist(img_real_gray / 255, 4)
img_eq_adapt_4 = ski.util.img_as_ubyte(img_eq_adapt_4)
images.append(img_eq_adapt_4)

img_eq_adapt_8 = ski.exposure.equalize_adapthist(img_real_gray / 255, 8)
img_eq_adapt_8 = ski.util.img_as_ubyte(img_eq_adapt_8)
images.append(img_eq_adapt_8)

fig, axs = plt.subplots(7, 2, layout="constrained")

for i, image in enumerate(images):
    axs[i, 0].imshow(image, cmap=plt.cm.gray)
    h_gray, c_gray = ski.exposure.histogram(image)
    axs[i, 1].bar(c_gray, h_gray, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
plt.show()
