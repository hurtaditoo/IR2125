"""
Considera la misma imagen del ejercicio anterior. A partir de ella, genera y visualiza las siguientes transformaciones
afines (debes mostrar, en cada caso, la matriz de transformación utilizada y el resultado de aplicarla sobre la imagen
que se indica):

1. Sobre la imagen original, un escalado de 0.5 en X y 0.75 en Y.
2. Sobre el resultado anterior, un desplazamiento de 64 píxeles en X y 0 en Y.
3. Sobre el resultado anterior, una rotación de 15 grados.
4. Sobre el resultado anterior, una inclinación de -10 grados tanto en X como en Y.
5. A partir de la clase AffineTransform, crea una transformación que realice simultáneamente las cuatro transformaciones
anteriores. Aplícala sobre la imagen original. ¿Obtienes el mismo resultado que en el punto 4? Justifica tu respuesta.
6. A partir de las matrices de transformación que has empleado en los cuatro primeros puntos, crea tu propia matriz de
transformación multiplicando las anteriores en el orden adecuado. Aplica la transformación que define dicha matriz a la
imagen original. ¿Obtienes el mismo resultado del punto 4, del punto 5 o ninguno de ellos? Razona tu respuesta.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/lena256.pgm")

# No puedo usar esto 'FACTOR_ESCALADO = 1.5', porque son valores diferentes en x y en y
ta = ski.transform.AffineTransform(scale=(0.5, 0.75))
mscalado = ta.params
# Se hace una copia de la original para realizar la transformación
img_escalada = ski.transform.warp(img_original.copy(), ta.inverse)

DISTANCIA_TRASLACION = (64, 0)
ta = ski.transform.AffineTransform(translation=DISTANCIA_TRASLACION)
mtranslacion = ta.params
img_trasladada = ski.transform.warp(img_escalada, ta.inverse)

ANGULO_ROTACION_EN_GRADOS = 15
ta = ski.transform.AffineTransform(rotation=np.radians(ANGULO_ROTACION_EN_GRADOS))
mrotacion = ta.params
img_rotada = ski.transform.warp(img_trasladada, ta.inverse)

ANGULO_INCLINACION_EN_GRADOS = (-10)
ta = ski.transform.AffineTransform(shear=np.radians(ANGULO_INCLINACION_EN_GRADOS))
mshear = ta.params
img_inclinada = ski.transform.warp(img_rotada, ta.inverse)

ta_simul = ski.transform.AffineTransform(scale=(0.5, 0.75), translation=DISTANCIA_TRASLACION,
                                         rotation=np.radians(ANGULO_ROTACION_EN_GRADOS),
                                         shear=ANGULO_INCLINACION_EN_GRADOS)
img_simultanea = ski.transform.warp(img_original.copy(), ta_simul.inverse)

matriz_total = mshear @ mrotacion @ mtranslacion @ mscalado
ta_propia = ski.transform.AffineTransform(matriz_total)
img_matriz = ski.transform.warp(img_original.copy(), ta_propia.inverse)

fig, axs = plt.subplots(7, 1, layout="constrained")
images = [img_original, img_escalada, img_trasladada, img_rotada, img_inclinada, img_simultanea, img_matriz]
nombres = ['escalado con factor 0.5 en "x" y 0.75 en "y"', 'traslación con distancia (64, 0)',
           'rotación con ángulo de 15º', 'inclinación con ángulo de -10', 'todas las transformaciones simultáneamente',
           'transformación propia']
matrices = [None, mscalado, mtranslacion, mrotacion, mshear, ta_simul.params, ta_propia.params]

for i, image in enumerate(images):
    if 0 < i <= len(images):
        print(f'Matriz de {nombres[i-1]}:\n{matrices[i]}')
    axs[i].imshow(images[i], cmap=plt.cm.gray)
    axs[i].set_axis_off()

plt.show()

