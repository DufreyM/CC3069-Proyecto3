"""Fase de preparación original: un solo proceso recorre las imágenes una por una."""
import cv2
import numpy as np

from src.config import IMG_SIZE

# OpenCV usa hilos internos por defecto; con uno solo la línea base es realmente secuencial.
cv2.setNumThreads(1)


def preparar_imagen(ruta: str) -> np.ndarray:
    """Lee el JPG, pasa de BGR a RGB, redimensiona a 64x64 y normaliza a [0, 1]."""
    img = cv2.imread(ruta)
    if img is None:
        raise OSError(f"No se pudo leer la imagen: {ruta}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32) / 255.0


def preparar_secuencial(rutas: list[str]) -> np.ndarray:
    """Construye el tensor (N, 64, 64, 3) como lo hacía el laboratorio."""
    return np.array([preparar_imagen(ruta) for ruta in rutas])
