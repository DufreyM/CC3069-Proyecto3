"""Fase de preparación paralela: varios procesos de CPU decodifican las imágenes.

Se usan procesos porque cada imagen es independiente y así no compiten por el GIL. Cada
proceso escribe directo en memoria compartida para no serializar los resultados de vuelta.
"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

from src.config import IMG_SIZE
from src.preparacion_secuencial import preparar_imagen

_memoria = None
_destino = None


def _iniciar_proceso(nombre: str, forma: tuple[int, ...]) -> None:
    """Conecta el proceso con el tensor compartido."""
    global _memoria, _destino
    cv2.setNumThreads(1)
    _memoria = SharedMemory(name=nombre)
    _destino = np.ndarray(forma, dtype=np.float32, buffer=_memoria.buf)


def _procesar_bloque(bloque: tuple[int, list[str]]) -> int:
    """Prepara un bloque de rutas contiguas y lo escribe en su posición del tensor."""
    inicio, rutas = bloque
    for i, ruta in enumerate(rutas):
        _destino[inicio + i] = preparar_imagen(ruta)
    return len(rutas)


def preparar_paralelo(rutas: list[str], procesos: int, tam_bloque: int = 256) -> np.ndarray:
    """Construye el mismo tensor que la versión secuencial y conserva el orden de las rutas."""
    forma = (len(rutas), IMG_SIZE, IMG_SIZE, 3)
    memoria = SharedMemory(create=True, size=int(np.prod(forma)) * 4)
    try:
        bloques = [(i, rutas[i:i + tam_bloque]) for i in range(0, len(rutas), tam_bloque)]
        with ProcessPoolExecutor(procesos, initializer=_iniciar_proceso, initargs=(memoria.name, forma)) as grupo:
            procesadas = sum(grupo.map(_procesar_bloque, bloques))
        if procesadas != len(rutas):
            raise RuntimeError(f"Se procesaron {procesadas} de {len(rutas)} imágenes")
        compartido = np.ndarray(forma, dtype=np.float32, buffer=memoria.buf)
        resultado = compartido.copy()
        del compartido
        return resultado
    finally:
        memoria.close()
        memoria.unlink()
