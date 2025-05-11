"""
Busca en internet la imagen de una bandera en la que aparezcan, al menos, tres colores distintos sobre un cielo
mayormente azul y descárgala.

Antes de programar nada, describe el contenido que esperarías encontrar en cada uno de los canales de color R, G, B
de dicha imagen si se visualizase cada canal por separado.

Escribe un programa que visualice la imagen en color y cada canal por separado. Comprueba si el contenido de cada canal
coincide o no con el que habías previsto inicialmente.
"""

import skimage as ski
import matplotlib.pyplot as plt

imagen_en_color = ski.io.imread("images/banderaGuinea.jpg")

plano_rojo = imagen_en_color[:, :, 0]
plano_verde = imagen_en_color[:, :, 1]
plano_azul = imagen_en_color[:, :, 2]

fig, axs = plt.subplots(2, 3, layout="constrained")
axs[0, 1].imshow(imagen_en_color)
axs[1, 0].imshow(plano_rojo, cmap=plt.cm.gray)
axs[1, 1].imshow(plano_verde, cmap=plt.cm.gray)
axs[1, 2].imshow(plano_azul, cmap=plt.cm.gray)

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()

"""
Canal Rojo (R): Los elementos de la bandera que contienen rojo serán más notorios. Es probable que el resto de la 
bandera aparezca más oscura o incluso negra, ya que no contiene mucha información en el canal rojo.

Canal Verde (G): Los elementos que contienen verde serán más evidentes, como la franja verde. Los detalles en rojo, 
amarillo y azul tenderán a aparecer más oscuros, ya que estos tonos no están representados en el canal.

Canal Azul (B): El fondo azul será la característica más destacada, mientras que el resto de elementos tenderán a 
aparecer más oscuros o casi negros.


Resultados -> 
    Canal Rojo (R): Esperaba que el elemento rojo de la bandera fuese el más visible, y aunque es muy visible, el color 
    amarillo destaca más.
    
    Canal Verde (G): En este caso, pasa igual que en el rojo, el amarillo destaca aun más que el verde.
    
    Canal Azul (B): Tal y como habíamos predicho el azul destaca más.
"""