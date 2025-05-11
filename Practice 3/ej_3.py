"""
Considera las imágenes de Einstein, de Marilyn o alguna otra imagen tuya. Define sobre una de ellas una malla inicial y
una modificación de la misma para deformar alguna característica de la imagen. Usa tu imaginación para crear una
deformación divertida: ojos con rasgos orientales, agrandar la nariz, alargar el mentón o ¡todo a la vez!

Cuando tengas definidas las dos mallas, crea un objeto de la clase PiecewiseAffineTransform, estima los parámetros para
realizar la transformación a trozos desde la malla original hasta la malla transformada y aplica dicha transformación a
la imagen mediante la función warp.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/Albert_Einstein.jpg")

src = [[0, 0],
       [0, 1066],
       [377, 403],  # Ojo izquierdo
       [530, 406],  # Ojo derecho
       [480, 524],  # nariz
       [572, 533],  # moflete derecho
       [326, 545],  # moflete izquierdo
       [369, 348],  # ceja izqda
       [554, 369],  # ceja dcha
       [391, 601],  # labios izqda
       [511, 600],  # labios dcha
       [466, 711],  # barbilla
       [459, 283],  # frente mitad
       [357, 685],  # barbilla izqda
       [554, 646],  # barbilla dcha
       [799, 0],
       [799, 1066]]

dst = src
src = np.array(src)
dst = np.array(dst)

dst[4] = [531, 516]  # Nariz alargada

tform = ski.transform.PiecewiseAffineTransform()
tform.estimate(src, dst)
img_t = ski.transform.warp(img_original, inverse_map=tform.inverse)

fig, axs = plt.subplots(1, 2, layout="constrained")
axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[0].plot(src[:, 0], src[:, 1], '.r')
axs[1].imshow(img_t, cmap=plt.cm.gray)

axs[0].set_axis_off()
axs[1].set_axis_off()
plt.show()
