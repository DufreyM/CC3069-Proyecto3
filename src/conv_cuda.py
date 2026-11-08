"""Primera capa convolucional de la CNN profunda (3→32 canales, filtro 3x3, padding 1) + ReLU.

Tres versiones con el mismo algoritmo, para mostrar explícitamente el diseño en hilos, bloques y grid
que en el entrenamiento resuelve cuDNN de forma interna:
  conv_cpu_secuencial  bucles compilados con Numba en un solo hilo
  conv_cpu_paralela    el mismo bucle repartido entre núcleos con prange
  conv_cuda            kernel CUDA: un hilo calcula un píxel de salida de un canal de una imagen

Diseño del kernel:
  bloque = 16x16 hilos (256), cada uno es un píxel (x, y) de salida
  grid   = (ceil(W/16), ceil(H/16), N*K): los ejes x, y recorren la imagen y el eje z la pareja
           (imagen, filtro). Con N=64, K=32 y 64x64 son 4x4x2048 bloques = 8.4 millones de hilos.
  Los 27 pesos del filtro de cada bloque se copian a memoria compartida una sola vez, porque
  los 256 hilos del bloque usan el mismo filtro.
"""
import math

import numpy as np
from numba import cuda, njit, prange

BLOQUE = (16, 16)


@njit(cache=True)
def conv_cpu_secuencial(x, w, b, salida):
    n_img, canales, alto, ancho = x.shape
    filtros = w.shape[0]
    for n in range(n_img):
        for k in range(filtros):
            for i in range(alto):
                for j in range(ancho):
                    suma = b[k]
                    for c in range(canales):
                        for di in range(3):
                            for dj in range(3):
                                fi, fj = i + di - 1, j + dj - 1
                                if 0 <= fi < alto and 0 <= fj < ancho:
                                    suma += x[n, c, fi, fj] * w[k, c, di, dj]
                    salida[n, k, i, j] = suma if suma > 0 else 0.0


@njit(parallel=True, cache=True)
def conv_cpu_paralela(x, w, b, salida):
    n_img, canales, alto, ancho = x.shape
    filtros = w.shape[0]
    for tarea in prange(n_img * filtros):
        n, k = tarea // filtros, tarea % filtros
        for i in range(alto):
            for j in range(ancho):
                suma = b[k]
                for c in range(canales):
                    for di in range(3):
                        for dj in range(3):
                            fi, fj = i + di - 1, j + dj - 1
                            if 0 <= fi < alto and 0 <= fj < ancho:
                                suma += x[n, c, fi, fj] * w[k, c, di, dj]
                salida[n, k, i, j] = suma if suma > 0 else 0.0


@cuda.jit
def _kernel_conv(x, w, b, salida):
    j, i = cuda.grid(2)
    nk = cuda.blockIdx.z
    filtros = w.shape[0]
    n, k = nk // filtros, nk % filtros
    canales, alto, ancho = x.shape[1], x.shape[2], x.shape[3]

    pesos = cuda.shared.array(27, dtype=np.float32)
    hilo = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    if hilo < canales * 9:
        pesos[hilo] = w[k, hilo // 9, (hilo % 9) // 3, hilo % 3]
    cuda.syncthreads()

    if i < alto and j < ancho:
        suma = b[k]
        for c in range(canales):
            for di in range(3):
                for dj in range(3):
                    fi, fj = i + di - 1, j + dj - 1
                    if 0 <= fi < alto and 0 <= fj < ancho:
                        suma += x[n, c, fi, fj] * pesos[c * 9 + di * 3 + dj]
        salida[n, k, i, j] = suma if suma > 0 else 0.0


def configuracion(forma_x: tuple, filtros: int) -> tuple[tuple, tuple]:
    """Devuelve (grid, bloque) para un lote de forma (N, C, H, W)."""
    n_img, _, alto, ancho = forma_x
    grid = (math.ceil(ancho / BLOQUE[0]), math.ceil(alto / BLOQUE[1]), n_img * filtros)
    return grid, BLOQUE


def conv_cuda(x_gpu, w_gpu, b_gpu, salida_gpu) -> None:
    """Lanza el kernel sobre arreglos que ya están en el GPU (no sincroniza)."""
    grid, bloque = configuracion(x_gpu.shape, w_gpu.shape[0])
    _kernel_conv[grid, bloque](x_gpu, w_gpu, b_gpu, salida_gpu)
