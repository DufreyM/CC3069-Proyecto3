"""Fase de entrenamiento: el mismo ciclo del Laboratorio 3 en CPU o GPU.

Esta fase no toca archivos de imagen: recibe tensores ya preparados.

Modos:
  cpu_1hilo        línea base secuencial, PyTorch limitado a un hilo
  cpu_nhilos       como corrió el laboratorio, con los hilos por defecto de PyTorch
  cuda             mismo DataLoader, con pin_memory y copia asíncrona de cada lote al GPU
  cuda_optimizado  datos completos en VRAM y precisión mixta (AMP)
"""
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import BATCH_SIZE, LR

MODOS = ("cpu_1hilo", "cpu_nhilos", "cuda", "cuda_optimizado")


@dataclass
class Resultado:
    seg_entrenamiento: float
    seg_validacion: float
    seg_primera_epoca: float
    seg_lote: float
    seg_transferencia: float
    seg_computo: float
    accuracy_test: float
    f1_test: float
    vram_pico_gb: float


def a_tensores(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convierte de (N, H, W, C) a (N, C, H, W), como espera PyTorch."""
    return torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y).long()


def _sincronizar(dispositivo: torch.device) -> None:
    """Espera al GPU; sin esto el reloj se detiene antes de que termine el trabajo encolado."""
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()


def _lotes_vram(X, y, barajar):
    """Genera lotes por índice sobre tensores que ya están en el GPU."""
    orden = torch.randperm(len(y), device=y.device) if barajar else torch.arange(len(y), device=y.device)
    for lote in orden.split(BATCH_SIZE):
        yield X[lote], y[lote]


def entrenar(modelo: nn.Module, datos: dict, modo: str, epocas: int, desglosar: bool = False, al_terminar_epoca=None) -> Resultado:
    """Entrena y evalúa. Con desglosar=True sincroniza en cada paso para separar los tiempos."""
    dispositivo = torch.device("cuda" if modo.startswith("cuda") else "cpu")
    optimizado = modo == "cuda_optimizado"
    usa_cuda = dispositivo.type == "cuda"
    if usa_cuda:
        torch.cuda.reset_peak_memory_stats()
        # Medido: benchmark=True agrega de 5 a 11 s de arranque y no reduce el tiempo por lote.
        torch.backends.cudnn.benchmark = False

    modelo.to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    criterio = nn.CrossEntropyLoss()
    escalador = torch.amp.GradScaler("cuda", enabled=optimizado)

    tensores = {nombre: a_tensores(*datos[nombre]) for nombre in ("train", "val", "test")}
    if optimizado:
        tensores = {n: (X.to(dispositivo), y.to(dispositivo)) for n, (X, y) in tensores.items()}

    def lotes(nombre: str, barajar: bool):
        X, y = tensores[nombre]
        if optimizado:
            return _lotes_vram(X, y, barajar)
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=barajar, pin_memory=usa_cuda)

    seg_entrenamiento = seg_validacion = seg_primera_epoca = seg_lote = seg_transferencia = seg_computo = 0.0
    for epoca in range(epocas):
        _sincronizar(dispositivo)
        inicio_epoca = time.perf_counter()
        modelo.train()
        marca = time.perf_counter()
        for xb, yb in lotes("train", True):
            if desglosar:
                ahora = time.perf_counter()
                seg_lote += ahora - marca
                marca = ahora
            xb = xb.to(dispositivo, non_blocking=True)
            yb = yb.to(dispositivo, non_blocking=True)
            if desglosar:
                _sincronizar(dispositivo)
                ahora = time.perf_counter()
                seg_transferencia += ahora - marca
                marca = ahora
            optimizador.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=optimizado):
                perdida = criterio(modelo(xb), yb)
            escalador.scale(perdida).backward()
            escalador.step(optimizador)
            escalador.update()
            if desglosar:
                _sincronizar(dispositivo)
                ahora = time.perf_counter()
                seg_computo += ahora - marca
                marca = ahora
        _sincronizar(dispositivo)
        fin_entrenamiento = time.perf_counter()
        seg_entrenamiento += fin_entrenamiento - inicio_epoca

        modelo.eval()
        with torch.inference_mode():
            for xb, yb in lotes("val", False):
                criterio(modelo(xb.to(dispositivo, non_blocking=True)), yb.to(dispositivo, non_blocking=True))
        _sincronizar(dispositivo)
        fin_epoca = time.perf_counter()
        seg_validacion += fin_epoca - fin_entrenamiento
        if epoca == 0:
            seg_primera_epoca = fin_epoca - inicio_epoca
        if al_terminar_epoca is not None:
            al_terminar_epoca(epoca)

    predichas, reales = [], []
    modelo.eval()
    with torch.inference_mode():
        for xb, yb in lotes("test", False):
            predichas.append(modelo(xb.to(dispositivo, non_blocking=True)).argmax(1).cpu())
            reales.append(yb.cpu())
    predichas, reales = torch.cat(predichas).numpy(), torch.cat(reales).numpy()

    return Resultado(
        seg_entrenamiento=seg_entrenamiento,
        seg_validacion=seg_validacion,
        seg_primera_epoca=seg_primera_epoca,
        seg_lote=seg_lote,
        seg_transferencia=seg_transferencia,
        seg_computo=seg_computo,
        accuracy_test=accuracy_score(reales, predichas),
        f1_test=f1_score(reales, predichas, average="macro", zero_division=0),
        vram_pico_gb=torch.cuda.max_memory_allocated() / 1024**3 if usa_cuda else 0.0,
    )
