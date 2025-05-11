"""
El ejemplo 3 de teoría prueba el teorema de la convolución para una máscara media de tamaño 7x7. Modifica una copia de
dicho programa para utilizar una máscara gaussiana de 31x31 con sigma=5.

Comenta los resultados obtenidos. Céntrate en las diferencias encontradas entre la transformada de Fourier de la máscara
media y la máscara gaussiana.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft as fft

MASK_SIZE = 31
SIGMA = 5

imagen = ski.io.imread("images/boat.511.tiff")
imagen = ski.util.img_as_float(imagen)

# Convolución en el espacio
mascara = np.outer(scipy.signal.windows.gaussian(MASK_SIZE, SIGMA), scipy.signal.windows.gaussian(MASK_SIZE, SIGMA))
# Máscara de NxN con sigma 5

mascara /= np.sum(mascara)  # Máscara normalizada
res_convol = scipy.ndimage.convolve(imagen, mascara, mode="wrap")

# Ampliamos la máscara con ceros para que tenga el mismo tamaño que la imagen
mascara_centrada = np.zeros(imagen.shape)
fila_i = imagen.shape[0] // 2 - MASK_SIZE // 2
col_i = imagen.shape[1] // 2 - MASK_SIZE // 2
mascara_centrada[fila_i:fila_i + MASK_SIZE, col_i:col_i + MASK_SIZE] = mascara

# Pasamos imagen y máscara a las frecuencias
FTimagen = fft.fft2(imagen)
mascara_en_origen = fft.ifftshift(mascara_centrada)
FTmascara = fft.fft2(mascara_en_origen)

# Convolución en la frecuancia
FTimagen_filtrada = FTimagen * FTmascara  # Producto punto a punto

# Recuperamos resultado en el espacio
res_filtro_FT = fft.ifft2(FTimagen_filtrada)
res_filtro_real = np.real(res_filtro_FT)
res_filtro_imag = np.imag(res_filtro_FT)
if not np.allclose(res_filtro_imag, np.zeros(imagen.shape)):
    print("Warning. Algo no está yendo bien!!!")

# Comparamos los dos resultados
print("¿Obtenemos el mismo resultado?", np.allclose(res_convol, res_filtro_FT))

# Visulización de resultados
fig, axs = plt.subplots(2, 4, layout="constrained")
axs[0, 0].imshow(imagen, cmap=plt.cm.gray)
axs[0, 1].imshow(mascara_centrada, cmap=plt.cm.gray)
axs[0, 3].imshow(res_convol, cmap=plt.cm.gray)

magnitud_imagen = fft.fftshift(np.log(np.absolute(FTimagen) + 1))
magnitud_mascara = fft.fftshift(np.log(np.absolute(FTmascara) + 1))
magnitud_producto = fft.fftshift(np.log(np.absolute(FTimagen_filtrada) + 1))
axs[1, 0].imshow(magnitud_imagen, cmap=plt.cm.gray)
axs[1, 1].imshow(magnitud_mascara, cmap=plt.cm.gray)
axs[1, 2].imshow(magnitud_producto, cmap=plt.cm.gray)
axs[1, 3].imshow(res_filtro_real, cmap=plt.cm.gray)

for a in axs.ravel():
    a.set_axis_off()
plt.show()
