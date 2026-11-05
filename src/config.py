"""Parámetros compartidos. Los valores de datos y modelo son los del Laboratorio 3."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DIR_DATOS = Path(os.environ.get(
    "ASL_DATA_DIR",
    RAIZ.parent.parent / "CC3084 - Data Science" / "CC3084-Laboratorio3" / "asl_alphabet_train",
))
# Fuera de OneDrive: los tensores pesan cerca de 1 GB y no deben sincronizarse.
DIR_CACHE = Path(os.environ.get("ASL_CACHE_DIR", Path.home() / ".cache" / "cc3069-p3"))

DIR_RESULTADOS = RAIZ / "resultados"
DIR_LOGS = DIR_RESULTADOS / "logs"
DIR_MONITOREO = DIR_RESULTADOS / "monitoreo"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"

IMG_SIZE = 64
SAMPLES_PER_CLASS = 650
BATCH_SIZE = 64
EPOCAS = 15
LR = 1e-3
SEMILLA = 42
