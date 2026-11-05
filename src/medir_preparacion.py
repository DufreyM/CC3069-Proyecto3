"""Mide la preparación de imágenes en secuencial y con varios procesos, y guarda los tensores.

Uso: python -m src.medir_preparacion --repeticiones 5
"""
import argparse
import csv
import hashlib
import time

import numpy as np

from src.config import DIR_CACHE, DIR_LOGS, DIR_MONITOREO, DIR_RESULTADOS
from src.monitor import Monitor
from src.preparacion_paralela import preparar_paralelo
from src.preparacion_secuencial import preparar_secuencial


def calentar_disco(rutas: list[str]) -> None:
    """Lee todos los archivos una vez para que ninguna corrida pague la descarga de OneDrive."""
    for ruta in rutas:
        with open(ruta, "rb") as archivo:
            archivo.read()


def guardar_tensores(df) -> None:
    """Guarda los tensores por partición para que el entrenamiento no toque los JPG."""
    from src.datos import dividir

    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    rutas = df["path"].tolist()
    X = preparar_paralelo(rutas, procesos=8)
    posiciones = {ruta: i for i, ruta in enumerate(rutas)}
    for nombre, parte in dividir(df).items():
        indices = [posiciones[ruta] for ruta in parte["path"]]
        np.save(DIR_CACHE / f"X_{nombre}.npy", X[indices])
        np.save(DIR_CACHE / f"y_{nombre}.npy", parte["label_id"].to_numpy())


def main() -> None:
    # Importación diferida: en Windows cada proceso hijo reimporta este módulo, y pandas y
    # sklearn sumarían segundos de arranque que no tienen relación con las imágenes.
    from src.datos import listar_imagenes

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--procesos", type=int, nargs="+", default=[2, 4, 8, 12, 16, 24, 32])
    parser.add_argument("--completo", action="store_true", help="Mide también las 87,000 imágenes")
    parser.add_argument("--reps-completo", type=int, default=3)
    args = parser.parse_args()

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    muestra = listar_imagenes(muestra=True)
    conjuntos = {"muestra": muestra["path"].tolist()}
    if args.completo:
        conjuntos["completo"] = listar_imagenes(muestra=False)["path"].tolist()

    salida = DIR_RESULTADOS / "tiempos_preparacion.csv"
    with open(salida, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["conjunto", "imagenes", "procesos", "repeticion", "segundos"])
        for nombre, rutas in conjuntos.items():
            print(f"[{nombre}] calentando disco con {len(rutas)} archivos", flush=True)
            calentar_disco(rutas)
            # Se compara una huella y no el tensor: con 87,000 imágenes guardar otra copia ocupa 4 GB.
            referencia = None
            for procesos in [1, *args.procesos]:
                with Monitor(DIR_MONITOREO / f"preparacion_{nombre}_p{procesos}.csv"):
                    for repeticion in range(1, (args.reps_completo if nombre == "completo" else args.repeticiones) + 1):
                        inicio = time.perf_counter()
                        if procesos == 1:
                            X = preparar_secuencial(rutas)
                        else:
                            X = preparar_paralelo(rutas, procesos)
                        segundos = time.perf_counter() - inicio
                        huella = hashlib.sha256(X.data).hexdigest()
                        if referencia is None:
                            referencia = huella
                        elif huella != referencia:
                            raise RuntimeError(f"Resultado distinto con {procesos} procesos")
                        escritor.writerow([nombre, len(rutas), procesos, repeticion, round(segundos, 4)])
                        archivo.flush()
                        print(f"[{nombre}] procesos={procesos} rep={repeticion}: {segundos:.2f} s", flush=True)
                        del X

    guardar_tensores(muestra)
    print(f"Tensores guardados en {DIR_CACHE}", flush=True)


if __name__ == "__main__":
    main()
