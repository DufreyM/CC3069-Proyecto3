"""Mide la primera convolución en CPU secuencial, CPU paralela, kernel CUDA propio y cuDNN.

Usa imágenes reales ya preparadas y los pesos iniciales de la CNN profunda. Antes de medir,
comprueba que las cuatro versiones den el mismo resultado.

Uso: python -m src.medir_kernel --imagenes 512 --repeticiones 5
"""
import argparse
import csv
import time

import numpy as np
from numba import cuda, get_num_threads

from src.config import DIR_CACHE, DIR_RESULTADOS, IMG_SIZE, SEMILLA
from src.conv_cuda import configuracion, conv_cpu_paralela, conv_cpu_secuencial, conv_cuda


def compilar_kernel() -> None:
    """Compila el kernel antes de importar PyTorch.

    PyTorch trae su propio nvJitLink de CUDA 12.8 con el mismo nombre de DLL que el de Numba
    (12.9); si PyTorch se carga primero, Windows reutiliza esa DLL y el enlace del kernel falla.
    """
    x = cuda.to_device(np.zeros((1, 3, 4, 4), np.float32))
    w = cuda.to_device(np.zeros((1, 3, 3, 3), np.float32))
    b = cuda.to_device(np.zeros(1, np.float32))
    conv_cuda(x, w, b, cuda.device_array((1, 1, 4, 4), np.float32))
    cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenes", type=int, default=512)
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()

    compilar_kernel()
    import torch

    from src.modelos import CNNProfunda

    torch.manual_seed(SEMILLA)
    capa = CNNProfunda(29, IMG_SIZE).block1[0]
    w = capa.weight.detach().numpy().astype(np.float32).copy()
    b = capa.bias.detach().numpy().astype(np.float32).copy()
    X = np.load(DIR_CACHE / "X_train.npy", mmap_mode="r")[: args.imagenes]
    x = np.ascontiguousarray(X.transpose(0, 3, 1, 2), dtype=np.float32)
    forma_salida = (x.shape[0], w.shape[0], x.shape[2], x.shape[3])
    grid, bloque = configuracion(x.shape, w.shape[0])
    print(f"Entrada {x.shape} | salida {forma_salida} | grid {grid} | bloque {bloque} | "
          f"hilos {np.prod(grid) * np.prod(bloque):,} | núcleos Numba {get_num_threads()}", flush=True)

    referencia = torch.relu(capa(torch.from_numpy(x))).detach().numpy()
    salida = np.empty(forma_salida, dtype=np.float32)

    def cpu_secuencial():
        conv_cpu_secuencial(x, w, b, salida)
        return salida

    def cpu_paralela():
        conv_cpu_paralela(x, w, b, salida)
        return salida

    w_gpu, b_gpu = cuda.to_device(w), cuda.to_device(b)
    salida_gpu = cuda.device_array(forma_salida, dtype=np.float32)
    tiempos_kernel = {}

    def kernel_con_transferencias():
        """Incluye copiar la entrada al GPU y traer la salida, y mide el kernel aparte con eventos."""
        x_gpu = cuda.to_device(x)
        inicio, fin = cuda.event(), cuda.event()
        inicio.record()
        conv_cuda(x_gpu, w_gpu, b_gpu, salida_gpu)
        fin.record()
        fin.synchronize()
        tiempos_kernel["ultimo"] = cuda.event_elapsed_time(inicio, fin) / 1000
        return salida_gpu.copy_to_host()

    capa_gpu = capa.to("cuda")
    torch.backends.cudnn.benchmark = False

    def cudnn():
        with torch.inference_mode():
            r = torch.relu(capa_gpu(torch.from_numpy(x).to("cuda"))).cpu().numpy()
        return r

    versiones = {
        "cpu_secuencial": cpu_secuencial,
        "cpu_paralela": cpu_paralela,
        "cuda_kernel_propio": kernel_con_transferencias,
        "cudnn_pytorch": cudnn,
    }
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(DIR_RESULTADOS / "tiempos_kernel.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["version", "imagenes", "repeticion", "segundos", "segundos_solo_kernel", "error_max"])
        for nombre, funcion in versiones.items():
            resultado = funcion()  # calentamiento: compila Numba y carga kernels
            if nombre.startswith("cuda") or nombre.startswith("cudnn"):
                cuda.synchronize()
                torch.cuda.synchronize()
            error = float(np.abs(resultado - referencia).max())
            if error > 1e-4:
                raise RuntimeError(f"{nombre} difiere de PyTorch: error máximo {error}")
            for repeticion in range(1, args.repeticiones + 1):
                inicio = time.perf_counter()
                funcion()
                segundos = time.perf_counter() - inicio
                solo_kernel = tiempos_kernel["ultimo"] if nombre == "cuda_kernel_propio" else ""
                escritor.writerow([nombre, x.shape[0], repeticion, round(segundos, 5),
                                   round(solo_kernel, 6) if solo_kernel != "" else "", f"{error:.2e}"])
                archivo.flush()
                print(f"{nombre} rep={repeticion}: {segundos:.4f} s"
                      + (f" (kernel {solo_kernel * 1000:.2f} ms)" if solo_kernel != "" else ""), flush=True)


if __name__ == "__main__":
    main()
