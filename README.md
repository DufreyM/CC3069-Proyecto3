# CC3069 - Proyecto 3: Mejorando algoritmos con CUDA

Universidad del Valle de Guatemala · Computación Paralela y Distribuida

- María José Girón Isidro
- Leonardo Dufrey Mejía Mejía
- Cindy Mishelle Gualim Perez

Acelera el Laboratorio 3 de CC3084 Data Science (reconocimiento del alfabeto ASL con CNN)
separando el trabajo en dos fases:

1. **Preparación de imágenes en CPU con varios procesos**: leer, decodificar, redimensionar a 64×64 y normalizar.
2. **Entrenamiento de la red en GPU con CUDA**: la matemática de la red neuronal (PyTorch + cuDNN), más un
   kernel propio en CUDA Python (Numba) para la primera convolución, con hilos, bloques y grid explícitos.

El informe está en [informe/informe.pdf](informe/informe.pdf) y el análisis de operaciones en
[docs/analisis_y_diseno.md](docs/analisis_y_diseno.md).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/config.py` | Rutas y parámetros (los mismos del Laboratorio 3) |
| `src/datos.py` | Inventario de imágenes y división train/val/test |
| `src/preparacion_secuencial.py` | Preparación original, un solo proceso |
| `src/preparacion_paralela.py` | Preparación con procesos y memoria compartida |
| `src/modelos.py` | CNN básica y CNN profunda del laboratorio |
| `src/entrenamiento.py` | Entrenamiento en CPU (1 hilo / 24 hilos) y GPU (CUDA / optimizado) |
| `src/conv_cuda.py` | Kernel CUDA propio de la primera convolución |
| `src/monitor.py` | Registro de uso por núcleo, GPU, VRAM y potencia |
| `src/medir_*.py` | Mediciones repetidas de cada fase |
| `src/analisis.py` | Promedios, speedup, eficiencia y figuras |
| `scripts/medir_todo.sh` | Corre todas las mediciones en orden |
| `resultados/` | CSV de tiempos, resúmenes, figuras, monitoreo y logs |
| `informe/` | Generador del informe y PDF final |

## Cómo reproducir

Requiere una GPU NVIDIA y el dataset ASL Alphabet de Kaggle (carpeta `asl_alphabet_train`).

```bash
python -m venv .venv   # Python 3.12
.venv/Scripts/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
export ASL_DATA_DIR="ruta/a/asl_alphabet_train"   # por defecto, la carpeta del Laboratorio 3
PY=.venv/Scripts/python bash scripts/medir_todo.sh   # unas 3 horas
.venv/Scripts/python informe/build_informe.py
```

Los tensores preparados se guardan fuera del repositorio en `~/.cache/cc3069-p3` (variable `ASL_CACHE_DIR`).
Para mediciones comparables: equipo conectado a la corriente, sin otros programas pesados abiertos.
