"""Genera informe/informe.html e informe/informe.pdf a partir de los resúmenes de src.analisis.

Las cifras se leen de resultados/; el texto de la discusión está en informe/discusion.html.
Las capturas de pantalla van en informe/capturas/ (PNG o JPG); se incluyen todas en orden alfabético.

Uso: python informe/build_informe.py
"""
import base64
import re
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.analisis import ETIQUETAS, MODELOS  # noqa: E402
from src.config import DIR_FIGURAS, DIR_LOGS, DIR_RESULTADOS, EPOCAS  # noqa: E402

INFORME = RAIZ / "informe"


def imagen(ruta: Path, pie: str) -> str:
    """Incrusta una imagen en base64 para que el HTML no dependa de rutas."""
    if not ruta.exists():
        return f'<p class="falta">Falta la figura {html.escape(ruta.name)}.</p>'
    tipo = "jpeg" if ruta.suffix.lower() in (".jpg", ".jpeg") else "png"
    datos = base64.b64encode(ruta.read_bytes()).decode()
    return f'<figure><img src="data:image/{tipo};base64,{datos}" alt="{html.escape(pie)}"><figcaption>{pie}</figcaption></figure>'


def tabla(df: pd.DataFrame, columnas: dict[str, str], formatos: dict[str, str] | None = None) -> str:
    formatos = formatos or {}
    cabecera = "".join(f"<th>{html.escape(t)}</th>" for t in columnas.values())
    filas = []
    for _, fila in df.iterrows():
        celdas = []
        for col in columnas:
            valor = fila[col]
            if isinstance(valor, float):
                texto = "—" if pd.isna(valor) else formatos.get(col, "{:.2f}").format(valor)
            else:
                texto = str(valor)
            celdas.append(f"<td>{html.escape(texto)}</td>")
        filas.append(f"<tr>{''.join(celdas)}</tr>")
    return f"<table><thead><tr>{cabecera}</tr></thead><tbody>{''.join(filas)}</tbody></table>"


def bloque_log(ruta: Path, titulo: str, lineas: slice | None = None) -> str:
    if not ruta.exists():
        return ""
    texto = ruta.read_text(encoding="utf-8", errors="replace").splitlines()
    if lineas is not None:
        texto = texto[lineas]
    return f"<h4>{html.escape(titulo)}</h4><pre>{html.escape(chr(10).join(texto))}</pre>"


def construir() -> str:
    prep = pd.read_csv(DIR_RESULTADOS / "resumen_preparacion.csv")
    ent = pd.read_csv(DIR_RESULTADOS / "resumen_entrenamiento.csv")
    des = pd.read_csv(DIR_RESULTADOS / "resumen_desglose_gpu.csv")
    ker = pd.read_csv(DIR_RESULTADOS / "resumen_kernel.csv")
    mon = pd.read_csv(DIR_RESULTADOS / "resumen_monitoreo.csv")
    pipe = json.loads((DIR_RESULTADOS / "resumen_pipeline.json").read_text(encoding="utf-8"))

    prep["conjunto"] = prep.conjunto.map({"muestra": "Muestra (18,850)", "completo": "Completo (87,000)"})
    prep_tabla = tabla(prep, {"conjunto": "Conjunto", "procesos": "Procesos", "count": "Corridas", "mean": "Promedio (s)",
                              "std": "Desv. (s)", "speedup": "Speedup", "eficiencia": "Eficiencia"},
                       {"eficiencia": "{:.0%}", "speedup": "{:.2f}×"})

    ent["modo_t"] = ent.modo.map(ETIQUETAS)
    ent["modelo_t"] = ent.modelo.map(MODELOS)
    ent["tiempo_15"] = ent.apply(lambda f: f"{f.seg_15:,.1f}" + (" *" if f.estimado else ""), axis=1)
    ent["corr"] = ent.apply(lambda f: f"{int(f.corridas)} × {int(f.epocas)} ép.", axis=1)
    ent_tabla = tabla(ent, {"modelo_t": "Modelo", "modo_t": "Modo", "corr": "Corridas", "epoca_estable": "s / época",
                            "primera_epoca": "1.ª época (s)", "tiempo_15": f"Total {EPOCAS} ép. (s)",
                            "speedup_vs_cpu_1hilo": "Speedup vs 1 hilo", "speedup_vs_cpu_nhilos": "Speedup vs 24 hilos",
                            "accuracy": "Accuracy test"},
                      {"speedup_vs_cpu_1hilo": "{:.1f}×", "speedup_vs_cpu_nhilos": "{:.1f}×", "accuracy": "{:.4f}",
                       "epoca_estable": "{:.2f}", "primera_epoca": "{:.2f}"})

    ker["version_t"] = ker.version.map({"cpu_secuencial": "CPU 1 hilo (Numba)", "cpu_paralela": "CPU paralela (Numba prange)",
                                        "cuda_kernel_propio": "Kernel CUDA propio (Numba)", "cudnn_pytorch": "cuDNN (PyTorch)"})
    ker["ms"] = ker.segundos * 1000
    ker["ms_kernel"] = ker.solo_kernel * 1000
    ker_tabla = tabla(ker, {"version_t": "Versión", "ms": "Total con copias (ms)", "ms_kernel": "Solo kernel (ms)",
                            "speedup": "Speedup total", "speedup_solo_kernel": "Speedup solo kernel"},
                      {"ms": "{:.1f}", "ms_kernel": "{:.1f}", "speedup": "{:.1f}×", "speedup_solo_kernel": "{:.1f}×"})

    des_tabla = tabla(des, {"fase": "Parte del paso", "segundos": "Segundos", "porcentaje": "%"}, {"porcentaje": "{:.1f}"})

    mon_sel = mon[mon.medicion.str.contains("completo_p1$|completo_p12$|_profunda$", regex=True)].copy()
    mon_sel["cpu_promedio"] = mon_sel.cpu_promedio.round(1)
    mon_tabla = tabla(mon_sel, {"medicion": "Medición", "cpu_promedio": "CPU prom. (%)", "nucleos_sobre_50": "Núcleos > 50 %",
                                "gpu_util_promedio": "GPU prom. (%)", "gpu_mem_max_gb": "VRAM máx. (GB)",
                                "gpu_w_max": "Potencia GPU máx. (W)"},
                      {"cpu_promedio": "{:.1f}", "gpu_util_promedio": "{:.1f}", "gpu_mem_max_gb": "{:.2f}", "gpu_w_max": "{:.1f}"})

    fases = pd.DataFrame([
        {"fase": "Preparación de imágenes (18,850)", "secuencial": pipe["preparacion_secuencial_s"],
         "paralelo": pipe["preparacion_paralela_s"], "como": f"CPU, {int(pipe['procesos_optimos'])} procesos"},
        {"fase": f"Entrenamiento CNN profunda ({EPOCAS} épocas)", "secuencial": pipe["entrenamiento_cpu_1hilo_s"],
         "paralelo": pipe["entrenamiento_gpu_s"], "como": "GPU con CUDA"},
        {"fase": "Pipeline completo", "secuencial": pipe.get("total_secuencial_s"),
         "paralelo": pipe.get("total_paralelo_s"), "como": "CPU paralela + GPU"},
    ])
    fases["speedup"] = fases.secuencial / fases.paralelo
    fases_tabla = tabla(fases, {"fase": "Fase", "secuencial": "Secuencial (s)", "paralelo": "Paralelo (s)",
                                "como": "Paralelo con", "speedup": "Speedup"},
                        {"secuencial": "{:,.1f}", "paralelo": "{:,.1f}", "speedup": "{:.1f}×"})

    capturas = sorted(p for p in (INFORME / "capturas").glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    capturas_html = "".join(imagen(p, p.stem.replace("_", " ")) for p in capturas) or (
        '<p class="falta">Pendiente: agregar capturas del Administrador de Tareas y de nvidia-smi en informe/capturas/.</p>')

    analisis_md = (RAIZ / "docs" / "analisis_y_diseno.md").read_text(encoding="utf-8")
    discusion = (INFORME / "discusion.html").read_text(encoding="utf-8")
    contexto = {
        "TABLA_FASES": fases_tabla, "TABLA_PREP": prep_tabla, "TABLA_ENT": ent_tabla, "TABLA_KER": ker_tabla,
        "TABLA_DES": des_tabla, "TABLA_MON": mon_tabla,
        "FIG_PREP": imagen(DIR_FIGURAS / "speedup_preparacion.png", "Figura 1. Speedup de la preparación según la cantidad de procesos."),
        "FIG_EPOCA": imagen(DIR_FIGURAS / "tiempo_por_epoca.png", "Figura 2. Tiempo por época estable en cada modo (escala logarítmica)."),
        "FIG_SPEEDUP": imagen(DIR_FIGURAS / "speedup_entrenamiento.png", f"Figura 3. Speedup del entrenamiento de {EPOCAS} épocas frente a la CPU con 1 hilo."),
        "FIG_DESGLOSE": imagen(DIR_FIGURAS / "desglose_gpu.png", "Figura 4. Reparto del tiempo de un paso de entrenamiento en GPU."),
        "FIG_KERNEL": imagen(DIR_FIGURAS / "kernel_convolucion.png", "Figura 5. Primera convolución: CPU, kernel CUDA propio y cuDNN."),
        "FIG_NUC_P1": imagen(DIR_FIGURAS / "nucleos_preparacion_completo_p1.png", "Figura 6. Uso por núcleo durante la preparación secuencial (87,000 imágenes)."),
        "FIG_NUC_P16": imagen(DIR_FIGURAS / "nucleos_preparacion_completo_p12.png", "Figura 7. Uso por núcleo durante la preparación con 12 procesos (87,000 imágenes)."),
        "FIG_NUC_1H": imagen(DIR_FIGURAS / "nucleos_entrenamiento_cpu_1hilo_profunda.png", "Figura 8. Uso por núcleo al entrenar en CPU con 1 hilo."),
        "FIG_NUC_NH": imagen(DIR_FIGURAS / "nucleos_entrenamiento_cpu_nhilos_profunda.png", "Figura 9. Uso por núcleo al entrenar en CPU con 24 hilos."),
        "FIG_GPU": imagen(DIR_FIGURAS / "uso_entrenamiento_cuda_profunda.png", "Figura 10. Uso de GPU y CPU al entrenar con CUDA."),
        "FIG_GPU_OPT": imagen(DIR_FIGURAS / "uso_entrenamiento_cuda_optimizado_profunda.png", "Figura 11. Uso de GPU y CPU en el modo GPU optimizado."),
        "CAPTURAS": capturas_html,
        "LOGS": bloque_log(DIR_LOGS / "nvidia_smi_cuda_profunda.txt", "nvidia-smi durante el entrenamiento en GPU (CNN profunda)")
                + bloque_log(DIR_LOGS / "nvidia_smi_cpu_nhilos_profunda.txt", "nvidia-smi durante el entrenamiento en CPU (GPU sin carga)")
                + bloque_log(DIR_LOGS / "entrenamiento_gpu.log", "Log del entrenamiento en GPU", slice(0, 8))
                + bloque_log(DIR_LOGS / "kernel.log", "Log del kernel CUDA", slice(0, 4)),
        # Las subsecciones del documento de análisis quedan dentro de la sección 3 del informe.
        "ANALISIS": re.sub(r"<h3>(\d)\. ", r"<h3>3.\1 ", markdown_simple(analisis_md)),
    }
    for clave, valor in contexto.items():
        discusion = discusion.replace("{{" + clave + "}}", valor)
    for clave, valor in pipe.items():
        discusion = discusion.replace("{{" + clave + "}}", f"{valor:,.2f}" if isinstance(valor, float) else str(valor))
    plantilla = (INFORME / "plantilla.html").read_text(encoding="utf-8")
    logo = base64.b64encode((INFORME / "img" / "logo_uvg.png").read_bytes()).decode()
    return plantilla.replace("{{LOGO}}", f'<img src="data:image/png;base64,{logo}" alt="UVG">').replace("{{CUERPO}}", discusion)


def markdown_simple(texto: str) -> str:
    """Convierte el subconjunto de Markdown de docs/ (títulos, tablas, listas, párrafos, `código`)."""
    import re

    def en_linea(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", t)

    salida, tabla_filas, lista, parrafo = [], [], [], []

    def cerrar():
        if tabla_filas:
            cab = "".join(f"<th>{en_linea(c)}</th>" for c in tabla_filas[0])
            cuerpo = "".join("<tr>" + "".join(f"<td>{en_linea(c)}</td>" for c in f) + "</tr>" for f in tabla_filas[2:])
            salida.append(f"<table><thead><tr>{cab}</tr></thead><tbody>{cuerpo}</tbody></table>")
            tabla_filas.clear()
        if lista:
            salida.append("<ol>" + "".join(f"<li>{en_linea(i)}</li>" for i in lista) + "</ol>")
            lista.clear()
        if parrafo:
            salida.append(f"<p>{en_linea(' '.join(parrafo))}</p>")
            parrafo.clear()

    for linea in texto.splitlines():
        if linea.startswith("|"):
            if parrafo or lista:
                cerrar()
            tabla_filas.append([c.strip() for c in linea.strip().strip("|").split("|")])
        elif re.match(r"^\d+\. ", linea):
            if tabla_filas or parrafo:
                cerrar()
            lista.append(re.sub(r"^\d+\. ", "", linea))
        elif linea.startswith("#"):
            cerrar()
            nivel = len(linea) - len(linea.lstrip("#"))
            if nivel > 1:
                salida.append(f"<h{nivel + 1}>{en_linea(linea.lstrip('# '))}</h{nivel + 1}>")
        elif not linea.strip():
            cerrar()
        else:
            if tabla_filas or lista:
                cerrar()
            parrafo.append(linea.strip())
    cerrar()
    return "\n".join(salida)


def a_pdf(origen: Path, destino: Path) -> None:
    """Imprime el HTML a PDF con Chrome (o Edge) sin interfaz, en un perfil temporal."""
    # Chrome primero: Edge en modo headless termina sin error pero no escribe el PDF en este equipo.
    candidatos = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", shutil.which("chromium") or ""]
    navegador = next((c for c in candidatos if c and os.path.exists(c)), None)
    if navegador is None:
        raise FileNotFoundError("No se encontró Edge ni Chrome para generar el PDF")
    # El navegador falla en silencio con rutas que tienen tildes: se trabaja en una carpeta temporal.
    with tempfile.TemporaryDirectory() as temporal:
        html_temp, pdf_temp = Path(temporal) / "informe.html", Path(temporal) / "informe.pdf"
        shutil.copyfile(origen, html_temp)
        subprocess.run([navegador, "--headless=new", "--disable-gpu", f"--user-data-dir={Path(temporal) / 'perfil'}",
                        "--no-pdf-header-footer", f"--print-to-pdf={pdf_temp}", html_temp.as_uri()],
                       check=True, capture_output=True, timeout=180)
        if not pdf_temp.exists():
            raise RuntimeError("El navegador no generó el PDF")
        shutil.copyfile(pdf_temp, destino)


def main() -> None:
    salida = INFORME / "informe.html"
    salida.write_text(construir(), encoding="utf-8")
    a_pdf(salida, INFORME / "informe.pdf")
    print(f"Informe generado: {INFORME / 'informe.pdf'}")


if __name__ == "__main__":
    main()
