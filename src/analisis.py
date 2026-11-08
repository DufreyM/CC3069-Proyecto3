"""Calcula promedios, speedup y eficiencia, y genera las figuras del informe.

Uso: python -m src.analisis
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DIR_FIGURAS, DIR_MONITOREO, DIR_RESULTADOS, EPOCAS

# Paleta categórica validada (orden fijo); cada modo conserva su color en todas las figuras.
COLORES = {"cpu_1hilo": "#2a78d6", "cpu_nhilos": "#eb6834", "cuda": "#1baf7a", "cuda_optimizado": "#eda100"}
ETIQUETAS = {
    "cpu_1hilo": "CPU 1 hilo",
    "cpu_nhilos": "CPU 24 hilos",
    "cuda": "GPU (CUDA)",
    "cuda_optimizado": "GPU optimizada",
}
MODELOS = {"basica": "CNN básica", "profunda": "CNN profunda"}
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def _estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": TINTA, "figure.dpi": 150,
        "legend.frameon": False,
    })


def resumir_preparacion() -> pd.DataFrame:
    """Promedio, desviación, speedup y eficiencia por cantidad de procesos."""
    df = pd.read_csv(DIR_RESULTADOS / "tiempos_preparacion.csv")
    r = df.groupby(["conjunto", "imagenes", "procesos"]).segundos.agg(["count", "mean", "std", "min", "max"]).reset_index()
    base = r[r.procesos == 1].set_index("conjunto")["mean"]
    r["speedup"] = r.apply(lambda f: base[f.conjunto] / f["mean"], axis=1)
    r["eficiencia"] = r.speedup / r.procesos
    r["imagenes_por_s"] = r.imagenes / r["mean"]
    return r


def resumir_entrenamiento() -> pd.DataFrame:
    """Tiempo total, por época estable, estimación a 15 épocas y speedup frente a ambas bases de CPU."""
    df = pd.read_csv(DIR_RESULTADOS / "tiempos_entrenamiento.csv")
    df = df[~df.desglosado.astype(str).str.lower().eq("true")]
    df["epoca_estable"] = (df.seg_entrenamiento + df.seg_validacion - df.seg_primera_epoca) / (df.epocas - 1)
    # Las corridas con menos épocas se llevan a 15 sumando épocas estables; se marcan como estimadas.
    df["seg_15"] = df.seg_total + (EPOCAS - df.epocas) * df.epoca_estable
    r = df.groupby(["modo", "modelo"]).agg(
        corridas=("seg_total", "size"), epocas=("epocas", "first"), seg_total=("seg_total", "mean"),
        seg_total_std=("seg_total", "std"), epoca_estable=("epoca_estable", "mean"),
        epoca_estable_std=("epoca_estable", "std"), primera_epoca=("seg_primera_epoca", "mean"),
        seg_15=("seg_15", "mean"), accuracy=("accuracy_test", "mean"), accuracy_std=("accuracy_test", "std"),
        f1=("f1_test", "mean"), vram_gb=("vram_pico_gb", "max"),
    ).reset_index()
    r["estimado"] = r.epocas < EPOCAS
    for base in ("cpu_1hilo", "cpu_nhilos"):
        ref = r[r.modo == base].set_index("modelo")
        r[f"speedup_vs_{base}"] = r.apply(lambda f: ref.seg_15.get(f.modelo, np.nan) / f.seg_15, axis=1)
        r[f"speedup_epoca_vs_{base}"] = r.apply(lambda f: ref.epoca_estable.get(f.modelo, np.nan) / f.epoca_estable, axis=1)
    orden = {m: i for i, m in enumerate(COLORES)}
    return r.sort_values(["modelo", "modo"], key=lambda s: s.map(orden) if s.name == "modo" else s).reset_index(drop=True)


def resumir_desglose() -> pd.DataFrame:
    """Reparto del tiempo de entrenamiento en GPU entre obtener lote, transferir y calcular."""
    df = pd.read_csv(DIR_RESULTADOS / "desglose_gpu.csv")
    r = df[["seg_lote", "seg_transferencia", "seg_computo"]].mean()
    return pd.DataFrame({"fase": ["Obtener lote (CPU)", "Copia CPU→GPU", "Cómputo en GPU"],
                         "segundos": r.values, "porcentaje": 100 * r.values / r.sum()})


def resumir_kernel() -> pd.DataFrame:
    """Tiempo de la primera convolución por versión, con speedup frente a la CPU secuencial."""
    df = pd.read_csv(DIR_RESULTADOS / "tiempos_kernel.csv")
    r = df.groupby("version", sort=False).agg(imagenes=("imagenes", "first"), segundos=("segundos", "mean"),
                                              segundos_std=("segundos", "std"),
                                              solo_kernel=("segundos_solo_kernel", "mean")).reset_index()
    base = r.loc[r.version == "cpu_secuencial", "segundos"].iloc[0]
    r["speedup"] = base / r.segundos
    r["speedup_solo_kernel"] = base / r.solo_kernel
    return r


def figura_kernel(ker: pd.DataFrame) -> None:
    etiquetas = {"cpu_secuencial": "CPU 1 hilo (Numba)", "cpu_paralela": "CPU 32 hilos (Numba prange)",
                 "cuda_kernel_propio": "Kernel CUDA propio", "cudnn_pytorch": "cuDNN (PyTorch)"}
    colores = {"cpu_secuencial": COLORES["cpu_1hilo"], "cpu_paralela": COLORES["cpu_nhilos"],
               "cuda_kernel_propio": COLORES["cuda"], "cudnn_pytorch": COLORES["cuda_optimizado"]}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(ker))[::-1]
    ax.barh(y, ker.segundos * 1000, color=[colores[v] for v in ker.version], height=0.55)
    for yi, (_, f) in zip(y, ker.iterrows()):
        texto = f"{f.segundos * 1000:.0f} ms ({f.speedup:.1f}×)"
        if np.isfinite(f.solo_kernel):
            texto += f" · solo kernel {f.solo_kernel * 1000:.0f} ms ({f.speedup_solo_kernel:.0f}×)"
        ax.text(f.segundos * 1000 * 1.03, yi, texto, va="center", color=TINTA, fontsize=9)
    ax.set_yticks(y, [etiquetas[v] for v in ker.version])
    ax.set_xlim(0, ker.segundos.max() * 1000 * 1.9)
    ax.set_xlabel(f"Milisegundos para {int(ker.imagenes.iloc[0])} imágenes, con copias CPU↔GPU")
    ax.set_title("Primera convolución de la CNN profunda (3→32 canales, 3×3)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "kernel_convolucion.png")
    plt.close(fig)


def resumir_monitoreo() -> pd.DataFrame:
    """Uso promedio de CPU, núcleos activos y uso de GPU en cada medición."""
    filas = []
    for archivo in sorted(DIR_MONITOREO.glob("*.csv")):
        m = pd.read_csv(archivo)
        if m.empty:
            continue
        nucleos = m.filter(like="cpu")
        filas.append({
            "medicion": archivo.stem, "muestras": len(m), "cpu_promedio": nucleos.mean(axis=1).mean(),
            "nucleos_sobre_50": (nucleos.mean() > 50).sum(), "nucleos": nucleos.shape[1],
            "gpu_util_promedio": m.gpu_util.mean(), "gpu_util_max": m.gpu_util.max(),
            "gpu_mem_max_gb": m.gpu_mem_gb.max(), "gpu_w_max": m.gpu_w.max(), "ram_max_gb": m.ram_gb.max(),
        })
    return pd.DataFrame(filas)


def figura_preparacion(prep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    procesos = sorted(prep.procesos.unique())
    ax.plot(procesos, procesos, color=TINTA_2, linestyle=(0, (4, 3)), linewidth=1.2, label="Ideal (lineal)")
    for conjunto, color, etiqueta in (("muestra", "#2a78d6", "18,850 imágenes"), ("completo", "#eb6834", "87,000 imágenes")):
        d = prep[prep.conjunto == conjunto].sort_values("procesos")
        if d.empty:
            continue
        ax.plot(d.procesos, d.speedup, color=color, linewidth=2, marker="o", markersize=6, label=etiqueta)
        mejor = d.loc[d.speedup.idxmax()]
        ax.annotate(f"{mejor.speedup:.2f}× con {int(mejor.procesos)}", (mejor.procesos, mejor.speedup),
                    textcoords="offset points", xytext=(0, 9), ha="center", color=TINTA, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(procesos, [str(p) for p in procesos])
    ax.set_ylim(0, max(8.5, prep.speedup.max() + 1.5))
    ax.set_xlabel("Procesos de CPU")
    ax.set_ylabel("Speedup frente a 1 proceso")
    ax.set_title("Preparación de imágenes: speedup por cantidad de procesos")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "speedup_preparacion.png")
    plt.close(fig)


def figura_entrenamiento(ent: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    modos = list(COLORES)
    y = np.arange(len(modos))[::-1]
    for ax, (modelo, nombre) in zip(axes, MODELOS.items()):
        d = ent[ent.modelo == modelo].set_index("modo").reindex(modos)
        ax.barh(y, d.epoca_estable.fillna(0), color=[COLORES[m] for m in modos], height=0.55)
        for yi, valor in zip(y, d.epoca_estable):
            texto = f"{valor:.1f} s" if np.isfinite(valor) else "sin datos"
            ax.text(valor * 1.12 if np.isfinite(valor) else 0.6, yi, texto, va="center", color=TINTA, fontsize=9)
        ax.set_yticks(y, [ETIQUETAS[m] for m in modos])
        ax.set_xscale("log")
        ax.set_xlim(0.5, ent.epoca_estable.max() * 4)
        ax.set_title(nombre)
        ax.set_xlabel("Segundos por época (escala log)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Entrenamiento: tiempo por época estable (entrenamiento + validación)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "tiempo_por_epoca.png")
    plt.close(fig)


def figura_speedup_entrenamiento(ent: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    modos = [m for m in COLORES if m != "cpu_1hilo"]
    ancho = 0.26
    x = np.arange(len(MODELOS))
    for i, modo in enumerate(modos):
        d = ent[ent.modo == modo].set_index("modelo").reindex(list(MODELOS))
        pos = x + (i - 1) * (ancho + 0.02)
        ax.bar(pos, d.speedup_vs_cpu_1hilo, width=ancho, color=COLORES[modo], label=ETIQUETAS[modo])
        for p, v in zip(pos, d.speedup_vs_cpu_1hilo):
            if np.isfinite(v):
                ax.text(p, v * 1.06, f"{v:.0f}×" if v >= 10 else f"{v:.1f}×", ha="center", color=TINTA, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x, list(MODELOS.values()))
    ax.set_ylabel("Speedup frente a CPU 1 hilo (log)")
    ax.set_title("Entrenamiento de 15 épocas: speedup frente a la línea base secuencial")
    ax.legend(loc="upper left", ncols=3)
    maximo = np.nanmax(ent.speedup_vs_cpu_1hilo.to_numpy(dtype=float), initial=1.0)
    ax.set_ylim(1, maximo * 3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "speedup_entrenamiento.png")
    plt.close(fig)


def figura_desglose(des: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    izquierda = 0
    colores = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (_, f), color in zip(des.iterrows(), colores):
        ax.barh(0, f.porcentaje, left=izquierda, color=color, height=0.5, edgecolor="white", linewidth=2, label=f.fase)
        if f.porcentaje > 6:
            ax.text(izquierda + f.porcentaje / 2, 0, f"{f.porcentaje:.0f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        izquierda += f.porcentaje
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("% del tiempo de entrenamiento en GPU (CNN profunda)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "desglose_gpu.png")
    plt.close(fig)


def figura_nucleos(nombre: str, titulo: str) -> None:
    """Mapa de calor del uso de cada núcleo lógico a lo largo de una medición."""
    archivo = DIR_MONITOREO / f"{nombre}.csv"
    if not archivo.exists():
        return
    m = pd.read_csv(archivo)
    nucleos = m.filter(like="cpu").to_numpy().T
    fig, ax = plt.subplots(figsize=(8, 3.2))
    imagen = ax.imshow(nucleos, aspect="auto", cmap="Blues", vmin=0, vmax=100, interpolation="nearest",
                       extent=[m.t.iloc[0], m.t.iloc[-1], nucleos.shape[0] - 0.5, -0.5])
    ax.set_xlabel("Segundos")
    ax.set_ylabel("Núcleo lógico")
    ax.set_title(titulo)
    ax.grid(False)
    fig.colorbar(imagen, ax=ax, label="% de uso", pad=0.01)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"nucleos_{nombre}.png")
    plt.close(fig)


def figura_gpu(nombre: str, titulo: str) -> None:
    """Uso de GPU y de CPU promedio durante un entrenamiento."""
    archivo = DIR_MONITOREO / f"{nombre}.csv"
    if not archivo.exists():
        return
    m = pd.read_csv(archivo)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(m.t, m.gpu_util, color="#1baf7a", linewidth=1.5, label="GPU")
    ax.plot(m.t, m.filter(like="cpu").mean(axis=1), color="#eb6834", linewidth=1.5, label="CPU (promedio de núcleos)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Segundos")
    ax.set_ylabel("% de uso")
    ax.set_title(titulo)
    ax.legend(loc="lower right", ncols=2)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"uso_{nombre}.png")
    plt.close(fig)


def main() -> None:
    _estilo()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    prep, ent, des, mon = resumir_preparacion(), resumir_entrenamiento(), resumir_desglose(), resumir_monitoreo()
    ker = resumir_kernel()
    tablas = (("preparacion", prep), ("entrenamiento", ent), ("desglose_gpu", des), ("kernel", ker), ("monitoreo", mon))
    for nombre, tabla in tablas:
        tabla.to_csv(DIR_RESULTADOS / f"resumen_{nombre}.csv", index=False, float_format="%.4f")
        print(f"\n== {nombre}\n{tabla.round(3).to_string(index=False)}")

    figura_preparacion(prep)
    figura_entrenamiento(ent)
    figura_speedup_entrenamiento(ent)
    figura_desglose(des)
    figura_kernel(ker)
    figura_nucleos("preparacion_muestra_p1", "Preparación secuencial: un solo núcleo trabaja")
    figura_nucleos("preparacion_muestra_p16", "Preparación con 16 procesos: carga repartida entre núcleos")
    figura_nucleos("entrenamiento_cpu_1hilo_profunda", "Entrenamiento en CPU con 1 hilo (CNN profunda)")
    figura_nucleos("entrenamiento_cpu_nhilos_profunda", "Entrenamiento en CPU con 24 hilos (CNN profunda)")
    figura_gpu("entrenamiento_cuda_profunda", "Entrenamiento en GPU (CNN profunda): uso de GPU y CPU")
    figura_gpu("entrenamiento_cuda_optimizado_profunda", "Entrenamiento en GPU optimizada (CNN profunda): uso de GPU y CPU")

    # Pipeline completo de la CNN profunda: preparación de la muestra + 15 épocas.
    seq = prep[(prep.conjunto == "muestra") & (prep.procesos == 1)]["mean"].iloc[0]
    par = prep[prep.conjunto == "muestra"].sort_values("mean").iloc[0]
    tiempos = ent[ent.modelo == "profunda"].set_index("modo").seg_15
    pipeline = {
        "preparacion_secuencial_s": seq, "preparacion_paralela_s": par["mean"], "procesos_optimos": int(par.procesos),
        "entrenamiento_cpu_1hilo_s": tiempos.get("cpu_1hilo"), "entrenamiento_gpu_s": tiempos.get("cuda"),
        "entrenamiento_gpu_optimizado_s": tiempos.get("cuda_optimizado"),
    }
    if pipeline["entrenamiento_cpu_1hilo_s"] is not None and pipeline["entrenamiento_gpu_s"] is not None:
        pipeline["total_secuencial_s"] = seq + pipeline["entrenamiento_cpu_1hilo_s"]
        pipeline["total_paralelo_s"] = par["mean"] + pipeline["entrenamiento_gpu_s"]
        pipeline["speedup_total"] = pipeline["total_secuencial_s"] / pipeline["total_paralelo_s"]
    (DIR_RESULTADOS / "resumen_pipeline.json").write_text(
        json.dumps({k: (round(float(v), 4) if v is not None else None) for k, v in pipeline.items()}, indent=2),
        encoding="utf-8",
    )
    print("\n== pipeline", json.dumps(pipeline, default=float, indent=2))


if __name__ == "__main__":
    main()
