"""Mide el entrenamiento por modo y agrega las filas a resultados/tiempos_entrenamiento.csv.

Requiere correr antes src.medir_preparacion, que deja los tensores en caché.

Uso: python -m src.medir_entrenamiento --modos cuda cpu_nhilos --modelos profunda --repeticiones 3
"""
import argparse
import csv
import platform
import random
import time
from dataclasses import asdict

import numpy as np
import torch

from src.config import DIR_CACHE, DIR_LOGS, DIR_MONITOREO, DIR_RESULTADOS, EPOCAS, IMG_SIZE, SEMILLA
from src.entrenamiento import MODOS, entrenar
from src.modelos import MODELOS
from src.monitor import Monitor, guardar_nvidia_smi


def cargar_datos() -> dict:
    """Carga los tensores que dejó la fase de preparación."""
    return {
        nombre: (np.load(DIR_CACHE / f"X_{nombre}.npy"), np.load(DIR_CACHE / f"y_{nombre}.npy"))
        for nombre in ("train", "val", "test")
    }


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modos", nargs="+", choices=MODOS, default=list(MODOS))
    parser.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--epocas", type=int, default=EPOCAS)
    parser.add_argument("--desglosar", action="store_true", help="Separa lote, transferencia y cómputo")
    parser.add_argument("--salida", default="tiempos_entrenamiento.csv")
    args = parser.parse_args()

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos()
    clases = len(np.unique(datos["train"][1]))
    hilos_defecto = torch.get_num_threads()
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | Python {platform.python_version()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no disponible'}", flush=True)
    print(f"Hilos por defecto de PyTorch: {hilos_defecto}", flush=True)
    print({n: X.shape for n, (X, _) in datos.items()}, flush=True)

    ruta = DIR_RESULTADOS / args.salida
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if nuevo:
            escritor.writerow([
                "modo", "modelo", "repeticion", "epocas", "hilos", "imagenes", "seg_total", "seg_entrenamiento",
                "seg_validacion", "seg_primera_epoca", "seg_lote", "seg_transferencia", "seg_computo", "accuracy_test", "f1_test",
                "vram_pico_gb", "desglosado",
            ])
        for modo in args.modos:
            hilos = 1 if modo == "cpu_1hilo" else hilos_defecto
            torch.set_num_threads(hilos)
            for nombre in args.modelos:
                sufijo = "_desglose" if args.desglosar else ""
                evidencia = DIR_LOGS / f"nvidia_smi_{modo}_{nombre}{sufijo}.txt"

                def al_terminar_epoca(epoca: int) -> None:
                    if epoca == 0 and args.desglosar is False:
                        guardar_nvidia_smi(evidencia)

                with Monitor(DIR_MONITOREO / f"entrenamiento_{modo}_{nombre}{sufijo}.csv"):
                    for repeticion in range(1, args.repeticiones + 1):
                        fijar_semilla(SEMILLA + repeticion)
                        modelo = MODELOS[nombre](clases, IMG_SIZE)
                        inicio = time.perf_counter()
                        r = entrenar(modelo, datos, modo, args.epocas, args.desglosar, al_terminar_epoca)
                        total = time.perf_counter() - inicio
                        escritor.writerow([
                            modo, nombre, repeticion, args.epocas, hilos, len(datos["train"][1]), round(total, 3),
                            *(round(v, 4) for v in asdict(r).values()), args.desglosar,
                        ])
                        archivo.flush()
                        print(
                            f"{modo} {nombre} rep={repeticion}: {total:.1f} s | train {r.seg_entrenamiento:.1f} s | "
                            f"acc {r.accuracy_test:.4f} | f1 {r.f1_test:.4f} | VRAM {r.vram_pico_gb:.2f} GB",
                            flush=True,
                        )
                        del modelo
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
