"""
En los programas de ejemplo hemos utilizado la función random_noise para añadir ruido a una imagen en niveles de gris.
Sin embargo, la función se puede utilizar para añadir ruido a cualquier tipo de imagen. Considera una imagen en color,
por ejemplo, ojo_azul.png. Utiliza la función random_noise para añadirle varias cantidades de ruido sal y pimienta y de
ruido gaussiano. Muestra los resultados obtenidos de forma similar a como aparecen en el ejemplo 1 de teoría. Comenta
los resultados obtenidos.
"""

import skimage as ski
import matplotlib.pyplot as plt
import math

imagen = ski.io.imread("images/ojo_azul.png")

imagen = ski.util.img_as_float(imagen)

# Añadir ruido sal y pimienta
sp_values = (0.01,  # 1%
             0.05,  # 5%
             0.10)  # 10%
sp_noise = []
for i in range(len(sp_values)):
    img_noise = ski.util.random_noise(imagen, mode="s&p", amount=sp_values[i])
    sp_noise.append(img_noise)

# Añadir ruido Gaussiano
gaussian_values = (0.001,  # sigma = 0.032
                   0.005,  # sigma = 0.071
                   0.010)  # sigma = 0.1
gaussian_noise = []
for i in range(len(gaussian_values)):
    img_noise = ski.util.random_noise(imagen, mode="gaussian", var=gaussian_values[i])
    gaussian_noise.append(img_noise)


# Mostrar Resultados
def mostrar_por_filtro(titulo, imagen, sp_values, gaussian_values, sp_images, gaussian_images):
    fig, axs = plt.subplots(2, len(sp_values) + 1, layout="constrained")
    fig.suptitle(titulo, size=24)
    axs[0, 0].imshow(imagen)
    axs[0, 0].set_title("Original")

    for i in range(len(sp_values)):
        axs[0, i + 1].imshow(sp_images[i])
        axs[0, i + 1].set_title(f"Ruido S&P {sp_values[i] * 100:0.0f}%")

    for i in range(len(gaussian_values)):
        axs[1, i + 1].imshow(gaussian_images[i])
        axs[1, i + 1].set_title(f"Ruido Gaussiano $\\sigma$ = {math.sqrt(gaussian_values[i]):0.2f}")

    ax = axs.ravel()
    for a in ax:
        a.set_axis_off()
    plt.show()


mostrar_por_filtro("Imágenes con ruido", imagen, sp_values, gaussian_values, sp_noise, gaussian_noise)
