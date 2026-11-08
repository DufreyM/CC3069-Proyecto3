# Análisis de la solución secuencial y diseño paralelo

Pasos 1, 3 y 4 del enunciado. La solución base es el Laboratorio 3 de CC3084 Data Science
(reconocimiento del alfabeto ASL con CNN), formado por dos notebooks:

| Parte | Función |
|---|---|
| `preprocesamiento.ipynb` | Lista las 87,000 imágenes, toma 650 por clase, asigna etiquetas, divide 70/15/15, carga y normaliza cada imagen y guarda los tensores en `.npy`. |
| `modelado.ipynb` | Carga los tensores, los envuelve en `Dataset`/`DataLoader`, define las CNN, las entrena 15 épocas, evalúa en test y compara con otros modelos. |

Este proyecto conserva lo que mide rendimiento (preparación y entrenamiento de las dos CNN) y deja
fuera la red fully-connected, el Random Forest, el data augmentation y las fotos propias.

## 1. Operaciones del programa

| No. | Operación identificada | Descripción de su funcionamiento |
|---|---|---|
| 1 | Inventario de archivos | Recorre las 29 carpetas de clase y lista los 87,000 JPG. |
| 2 | Muestreo estratificado | Toma 650 imágenes por clase con semilla 42 (18,850 en total). |
| 3 | Asignación de etiquetas | Convierte cada clase en un entero de 0 a 28. |
| 4 | División train/val/test | Separa 13,195 / 2,827 / 2,828 imágenes de forma estratificada. |
| 5 | Lectura y decodificación | `cv2.imread` lee el archivo del disco y descomprime el JPG a una matriz 200×200×3. |
| 6 | Conversión de color | Reordena los canales de BGR a RGB. |
| 7 | Redimensionado | Interpola de 200×200 a 64×64 píxeles. |
| 8 | Normalización | Convierte a `float32` y divide entre 255 para dejar valores en [0, 1]. |
| 9 | Construcción del tensor | Apila las imágenes en un arreglo (N, 64, 64, 3). |
| 10 | Persistencia | Guarda los tensores en `.npy` para no repetir la preparación. |
| 11 | Formato para PyTorch | Transpone a (N, 3, 64, 64) y arma lotes de 64 con `DataLoader`. |
| 12 | Propagación hacia adelante | Convoluciones, ReLU, BatchNorm, MaxPool, Dropout y capas densas producen 29 puntajes por imagen. |
| 13 | Función de pérdida | Entropía cruzada entre los puntajes y la etiqueta real. |
| 14 | Retropropagación | Calcula el gradiente de la pérdida respecto a cada peso. |
| 15 | Actualización de pesos | Adam ajusta todos los parámetros con tasa 0.001. |
| 16 | Validación | Al final de cada época, propaga el set de validación sin calcular gradientes. |
| 17 | Evaluación | Predice el set de prueba y calcula accuracy y F1 macro. |

## 2. Operaciones repetitivas

Valores para la muestra de 18,850 imágenes y 15 épocas. Los MAC (multiplicación + suma) por imagen
se contaron sobre las capas convolucionales y densas de cada modelo.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Lectura y decodificación JPG (5) | Un archivo de unos 13 KB → 200×200×3 = 120,000 valores | 18,850 (87,000 con el dataset completo) |
| Conversión de color (6) | 40,000 píxeles por imagen | 18,850 |
| Redimensionado (7) | 120,000 valores de entrada → 12,288 de salida | 18,850 |
| Normalización (8) | 12,288 valores por imagen | 18,850 imágenes = 231.6 millones de divisiones |
| Paso de entrenamiento (12-15) | Lote de 64 imágenes de 3×64×64 | 207 lotes × 15 épocas = 3,105 |
| Convolución, CNN básica (12) | 18.8 millones de MAC por imagen (92.1 % del total de 20.4 M) | 13,195 imágenes × 15 épocas = 197,925 propagaciones, más su retropropagación |
| Convolución, CNN profunda (12) | 116.8 millones de MAC por imagen (98.2 % del total de 118.9 M) | 197,925 propagaciones, más su retropropagación |
| Actualización Adam (15) | 1,628,893 parámetros (básica) o 2,244,477 (profunda) | 3,105 por modelo |
| Validación (16) | 2,827 imágenes en 45 lotes | 15 por modelo |

Solo en propagación hacia adelante, la CNN profunda hace unos 1.57 billones de MAC por época
(13,195 × 118.9 M); la retropropagación cuesta aproximadamente el doble.

## 3. Oportunidades de paralelización

| Operación | ¿Paralelizable? | Dónde | Justificación |
|---|---|---|---|
| Inventario, muestreo y división (1-4) | No conviene | CPU, 1 proceso | Se ejecutan una vez y tardan poco frente al resto. |
| Lectura, decodificación, color, redimensionado y normalización (5-8) | Sí | CPU, varios procesos | Cada imagen es independiente (paralelismo de datos). La lectura toca el disco y la decodificación JPG es lógica secuencial con muchas ramas; ninguna de las dos aprovecha bien miles de hilos de GPU. |
| Construcción del tensor (9) | Sí, sin costo extra | CPU, memoria compartida | Cada proceso escribe su bloque directo en el tensor final, así no hay que copiar ni serializar resultados. |
| Propagación, pérdida, retropropagación y Adam (12-15) | Sí | GPU con CUDA | Son operaciones de álgebra lineal sobre miles de valores independientes: cada píxel de salida de una convolución se calcula con la misma fórmula sobre datos distintos (SIMT). Es el 92-98 % del cómputo. |
| Validación y evaluación (16-17) | Sí | GPU con CUDA | Es la misma propagación hacia adelante, sin gradientes. |
| Ciclo de épocas | No | Secuencial | Cada época necesita los pesos de la anterior; solo se paraleliza dentro de cada paso. |

## 4. Diseño del algoritmo paralelo

### Fase de CPU: preparación de imágenes

1. El proceso principal crea un tensor vacío de (N, 64, 64, 3) en memoria compartida.
2. La lista de rutas se divide en bloques de 256 imágenes contiguas.
3. Un grupo de P procesos (`ProcessPoolExecutor`) toma bloques a medida que se desocupa, lo que reparte la carga aunque el i9-13900HX mezcle núcleos rápidos (P) y eficientes (E).
4. Cada proceso aplica las operaciones 5-8 y escribe cada imagen en su posición; el orden final es idéntico al secuencial, y se comprueba con una huella SHA-256 del tensor.
5. Se usan procesos y no hilos para que el GIL de Python no serialice el trabajo. OpenCV y OpenBLAS se limitan a 1 hilo por proceso para no sobresuscribir los núcleos.

### Fase de GPU: entrenamiento

Los tensores ya preparados se envían al GPU por lotes de 64 imágenes (`pin_memory` + copia asíncrona)
o completos a la VRAM en el modo optimizado. PyTorch delega cada capa a cuDNN, que elige la
configuración de hilos y bloques. Para mostrar ese reparto explícitamente se implementó la primera
convolución de la CNN profunda como kernel propio en CUDA Python (Numba, `src/conv_cuda.py`):

| Nivel | Asignación |
|---|---|
| Hilo | Calcula un píxel de salida (i, j) de un filtro k para una imagen n: 27 MAC (3 canales × 3×3) más el sesgo y ReLU. |
| Bloque | 16×16 = 256 hilos que cubren un parche de 16×16 píxeles de la misma imagen y el mismo filtro. Los 27 pesos del filtro se cargan una sola vez en memoria compartida. |
| Grid | (⌈64/16⌉, ⌈64/16⌉, N·K) = (4, 4, N·32). Los ejes x, y recorren la imagen y el eje z cada pareja (imagen, filtro). Un lote de 64 imágenes lanza 4×4×2,048 = 32,768 bloques, es decir 8.4 millones de hilos. |

Como cada hilo escribe una salida distinta, no hay condiciones de carrera ni se necesitan
operaciones atómicas; la única barrera (`syncthreads`) asegura que los pesos estén en memoria
compartida antes de usarlos.
