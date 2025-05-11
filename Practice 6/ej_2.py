"""
Considera la imagen cuadros.png. Añádele una pequeña cantidad de ruido gaussiano (var = 0.001). Aplica sobre ella todos
los detectores de esquinas estudiados (Kitchen & Rosenfeld, Foerstner, Moravec, Harris y Fast). No es necesario que
muestres la respuesta de cada detector, basta con mostrar los resultados tras la detección de picos (las esquinas
detectadas). Ajusta el valor del umbral utilizado en cada caso para que las esquinas detectadas se aproximen lo máximo
posible a las esperadas. Ten en cuenta que el proceso de añadir ruido es aleatorio, por lo que puedes obtener resultados
ligeramente distintos en cada ejecución de tu programa.

Una vez que para la imagen ruidosa obtengas unos resultados satisfactorios, repite todo el proceso tras girar la imagen
ruidosa 0º, 22.5º, 45º, 67.5º y 90º desde el centro de la misma (función rotate) y comenta los resultados obtenidos.
¿Dirías que todos los métodos son invariantes a la rotación o hay algunos que se ven más afectados que otros? ¿Cuáles?
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

image1 = ski.io.imread("images/cuadros.png")  # Probar también con sintética
image1 = ski.util.img_as_float(image1)
image1 = ski.util.random_noise(image1, mode="gaussian", var=0.001)

images = []
angles = [0, 22.5, 45, 67.5, 90]
for angle in angles:
    images.append(ski.transform.rotate(image1, angle=angle))


def ajustar_01(imagen):
    maxi = imagen.max()
    mini = imagen.min()
    return (imagen - mini) / (maxi - mini)


def filtrar(image, nombres_filtros):
    images = [image]
    for nf in nombres_filtros:
        if nf == "moravec":
            img = my_moravec(image)
        else:
            if nf == "fast":
                param = ", 9"
            else:
                param = ""
            img = eval("ski.feature.corner_" + nf + "(image" + param + ")")
            if nf == "foerstner":
                img = img[0]
            elif nf == "kitchen_rosenfeld":
                img = np.abs(img)
        images.append(img)
    return images


def detectar_picos(images, umbral):
    resultados = [images[0]]
    for i in range(1, len(images)):
        img = ski.feature.corner_peaks(images[i], indices=False, min_distance=10, threshold_rel=umbral[i-1])
        resultados.append(img)
    return resultados


def my_moravec(cimage, window_size=1):
    rows = cimage.shape[0]
    cols = cimage.shape[1]
    out = np.zeros(cimage.shape)
    for r in range(2 * window_size, rows - 2 * window_size):
        for c in range(2 * window_size, cols - 2 * window_size):
            min_msum = 1E100
            for br in range(r - window_size, r + window_size + 1):
                for bc in range(c - window_size, c + window_size + 1):
                    if br != r or bc != c:  #### En scikit-image aquí aparece un AND !!!!
                        msum = 0
                        for mr in range(- window_size, window_size + 1):
                            for mc in range(- window_size, window_size + 1):
                                t = cimage[r + mr, c + mc] - cimage[br + mr, bc + mc]
                                msum += t * t
                        min_msum = min(msum, min_msum)

            out[r, c] = min_msum
    return out


def mostrar(titulo, resultados1, nombres):
    fig, ax = plt.subplots(nrows=1, ncols=len(resultados1), layout="constrained")
    fig.suptitle(titulo, fontsize=24)
    for i in range(len(resultados1)):
        ax[i].imshow(resultados1[i], cmap='gray')
        ax[i].set_title(nombres[i], fontsize=16)
    for a in ax.ravel():
        a.set_axis_off()
    plt.show()


filtros = ["kitchen_rosenfeld", "foerstner", "moravec", "harris", "fast"]
nombres = ["0º", "22.5º", "45º", "67.5º", "90º"]

umbral = [0.85, 0.05, 0.07, 0.005, 0.045]
for id, image in enumerate(images):
    images1 = filtrar(image, filtros)
    picos1 = detectar_picos(images1, umbral)
    mostrar(f"Imagen ruidosa rotada {nombres[id]} (umbrales usados = {umbral})", picos1, [nombres[id]] + filtros)

# kitchen_rosenfeld = ninguno termina de ir bien, foerstner = 0.05, moravec = 0.07, harris = 0.005, fast = 0.05
