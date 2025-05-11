"""
A partir de la imagen flecha_transparente.png genera cuatro imágenes como las que se muestran a continuación. Puedes
usar el método para transponer una matriz así como operaciones para invertir sus filas o columnas.
A partir de estas cuatro imágenes, genera un gif animado y visualízalo con un navegador.

El hecho de que la imagen inicial sea transparente hace que el gif animado no resulte adecuado. Por lo tanto, añade a tu
programa una línea de código para eliminar el canal de transparencia de la imagen original. Ajusta los parámetros del
gif para que parezca que la flecha gira sobre su centro de manera indefinida.
"""

import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

flecha = ski.io.imread("images/flecha_transparente.png")[:, :, 0:3] # Si con 3 coge el transparente, hacemos que no
# pille le 3


def rotar(flecha, nveces=3):
    rotadas = [flecha]
    imagen = flecha  # Esto se hace así para que sobre la imagen original, se rote 3 veces más y cada vez sea 90º mas
    # que el anterior

    for i in range(nveces):
        imagen = np.rot90(imagen, k=-1)
        rotadas.append(imagen)

    return rotadas


secuencia = np.stack(rotar(flecha), axis=0)

ski.io.imsave("images/flecha.gif", secuencia, loop=0, fps=4)
# Abre el archivo gif desde un navegador...

