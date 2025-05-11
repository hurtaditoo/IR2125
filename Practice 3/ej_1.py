"""
Considera una imagen en niveles de gris, por ejemplo, lena256.pgm. Utiliza la función rotate para rotar la imagen N
grados: rotate(img, N). Si lees detenidamente la documentación de dicha función, podrás comprobar que el ángulo se mide
en grados y el giro se realiza en sentido antihorario desde el centro de la imagen.

Queremos obtener una transformación euclídea que realice exactamente la misma transformación. Para ello, deberás generar
una matriz de transformación que combine los siguientes pasos:

*Trasladar el centro de la imagen hasta la posición (0, 0).
*Rotar la imagen el ángulo deseado.
*Trasladar el centro de la imagen hasta su posición original.

Una vez que hayas obtenido la transformación deseada, aplícala a la imagen utilizando la función warp. Si todo ha ido
bien, deberías obtener exactamente el mismo resultado que en el caso anterior

Al crear la matriz de transformación, deberás tener en cuenta que, aunque la función rotate considera el giro en sentido
antihorario, las matrices de rotación lo consideran en sentido horario. Además, la función rotate mide el ángulo en
grados y para las matrices tendrás que expresarlo en radianes.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/lena256.pgm")

img_girada = ski.transform.rotate(img_original, 45)

# Definir el ángulo de rotación en grados
angle = 45

# Convertir el ángulo de grados a radianes
angle_rad = np.radians(360 - angle)

translation = np.array([img_original.shape[0] / 2, img_original.shape[1] / 2])

transf_despl = ski.transform.EuclideanTransform(translation=(-translation))
transf_rot = ski.transform.EuclideanTransform(rotation=angle_rad)
transf_despl_2 = ski.transform.EuclideanTransform(translation=translation)

# Combinar las transformaciones en una sola matriz
matriz_transformacion = transf_despl_2.params @ transf_rot.params @ transf_despl.params
transf_final = ski.transform.EuclideanTransform(matriz_transformacion)

img_transf_euclidea = ski.transform.warp(img_original, transf_final.inverse, order=3)

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[1].imshow(img_girada, cmap=plt.cm.gray)
axs[2].imshow(img_transf_euclidea, cmap=plt.cm.gray)

plt.show()
