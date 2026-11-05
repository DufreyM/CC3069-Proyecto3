"""Registra el uso de CPU por núcleo y de GPU mientras corre una medición."""
import csv
import subprocess
import threading
import time
from pathlib import Path

import psutil

try:
    import pynvml
except ImportError:
    pynvml = None


class Monitor:
    """Muestrea recursos en un hilo aparte y los guarda en CSV al salir del bloque `with`."""

    def __init__(self, destino: Path, intervalo: float = 0.5):
        self.destino = Path(destino)
        self.intervalo = intervalo
        self._alto = threading.Event()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._gpu = None

    def __enter__(self):
        self.destino.parent.mkdir(parents=True, exist_ok=True)
        if pynvml is not None:
            pynvml.nvmlInit()
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        psutil.cpu_percent(percpu=True)
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._alto.set()
        self._hilo.join()
        if self._gpu is not None:
            pynvml.nvmlShutdown()

    def _muestrear(self) -> None:
        nucleos = psutil.cpu_count()
        columnas = ["t", "ram_gb", "gpu_util", "gpu_mem_gb", "gpu_w"] + [f"cpu{i}" for i in range(nucleos)]
        inicio = time.perf_counter()
        with open(self.destino, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(columnas)
            while not self._alto.wait(self.intervalo):
                gpu = ["", "", ""]
                if self._gpu is not None:
                    gpu = [
                        pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu,
                        round(pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used / 1024**3, 3),
                        round(pynvml.nvmlDeviceGetPowerUsage(self._gpu) / 1000, 1),
                    ]
                escritor.writerow([
                    round(time.perf_counter() - inicio, 2),
                    round(psutil.virtual_memory().used / 1024**3, 2),
                    *gpu,
                    *psutil.cpu_percent(percpu=True),
                ])


def guardar_nvidia_smi(destino: Path) -> None:
    """Guarda la salida de nvidia-smi como evidencia de la carga en ese momento."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    salida = subprocess.run(["nvidia-smi"], capture_output=True, text=True, errors="replace", check=False)
    destino.write_text(salida.stdout + salida.stderr, encoding="utf-8")
