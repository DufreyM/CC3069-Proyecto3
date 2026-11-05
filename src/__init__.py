"""Código del proyecto. Limita OpenBLAS antes de que cualquier módulo importe numpy."""
import os

# OpenBLAS reserva memoria por cada hilo posible (32 aquí) en cada proceso. Con 16 o más
# procesos eso agota la memoria virtual de Windows, y estos procesos no hacen álgebra lineal.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
