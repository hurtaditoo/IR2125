"""
Considera las imágenes banderaIrlanda.jpg y banderaItalia.jpg.

El programa debe centrar la imagen más pequeña dentro de la mayor, con independencia de las imágenes usadas y de su
tamaño. Es decir, para un par de imágenes cualesquiera, el programa debe mostrar una nueva imagen con la imagen pequeña
en vertical y centrada dentro de la mayor.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

grande = ski.io.imread("images/banderaItalia.jpg")
pequenya = ski.io.imread("images/banderaIrlanda.jpg")

if pequenya.size > grande.size:
    pequenya, grande = grande, pequenya

pequenya = np.rot90(pequenya)

inicio_x = (grande.shape[0] - pequenya.shape[0]) // 2
inicio_y = (grande.shape[1] - pequenya.shape[1]) // 2

nueva = grande.copy()
nueva[inicio_x:inicio_x + pequenya.shape[0], inicio_y:inicio_y + pequenya.shape[1], :] = pequenya

plt.imshow(nueva)
plt.axis('off')
plt.show()
