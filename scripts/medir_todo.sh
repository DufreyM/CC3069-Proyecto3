#!/usr/bin/env bash
# Corre todas las mediciones del informe, de la más rápida a la más lenta.
# cpu_1hilo usa 3 épocas porque 15 tardarían más de una hora por corrida.
# Uso: PY=ruta/al/python scripts/medir_todo.sh
set -euo pipefail
PY="${PY:-python}"
export PYTHONIOENCODING=utf8
mkdir -p resultados/logs
"$PY" -m src.medir_preparacion --repeticiones 5 --completo 2>&1 | tee resultados/logs/preparacion.log
"$PY" -m src.medir_entrenamiento --modos cuda cuda_optimizado --repeticiones 5 2>&1 | tee resultados/logs/entrenamiento_gpu.log
"$PY" -m src.medir_entrenamiento --modos cuda --modelos profunda --repeticiones 3 --epocas 5 --desglosar --salida desglose_gpu.csv 2>&1 | tee resultados/logs/desglose_gpu.log
"$PY" -m src.medir_entrenamiento --modos cpu_nhilos --repeticiones 3 2>&1 | tee resultados/logs/entrenamiento_cpu_nhilos.log
"$PY" -m src.medir_entrenamiento --modos cpu_1hilo --repeticiones 3 --epocas 3 2>&1 | tee resultados/logs/entrenamiento_cpu_1hilo.log
