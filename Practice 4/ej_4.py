"""
Considera la imagen borrosa.png, a la que llamaremos I. Aplica sobre ella un filtro gaussiano con sigma=3. Llamaremos a
la imagen filtrada F. Obtén una nueva imagen, R, calculada de la siguiente forma:

R = I + alpha * (I - F)

Utiliza un valor de alpha entre 0.5 y 1.5, por ejemplo, 1.0.

Muestra como resultado las imágenes I y R. Comenta en detalle los resultados obtenidos.
"""

import skimage as ski
import matplotlib.pyplot as plt

I = ski.io.imread("images/borrosa.png")
I = ski.util.img_as_float(I)

# Aplicar filtro gaussiano con sigma=3
F = ski.filters.gaussian(I, sigma=3)

# Definir el valor de alpha entre 0.5 y 1.5
alpha = 1.0

# Calcular la nueva imagen R
R = I + alpha * (I - F)

# Mostrar las imágenes original y resultante
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(I, cmap=plt.cm.gray)
axs[0].set_title('Imagen Original (I)')
axs[0].axis('off')

axs[1].imshow(R, cmap=plt.cm.gray)
axs[1].set_title('Imagen Resultante (R)')
axs[1].axis('off')

plt.show()
