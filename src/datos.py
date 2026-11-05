"""Inventario de imágenes y división train/val/test, igual que en el Laboratorio 3."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DIR_DATOS, SAMPLES_PER_CLASS, SEMILLA


def listar_imagenes(muestra: bool = True) -> pd.DataFrame:
    """Devuelve ruta, clase e id de clase. Con muestra=True toma 650 imágenes por clase."""
    clases = sorted(d.name for d in DIR_DATOS.iterdir() if d.is_dir())
    filas = [
        {"label": clase, "path": str(archivo)}
        for clase in clases
        for archivo in sorted((DIR_DATOS / clase).glob("*.jpg"))
    ]
    df = pd.DataFrame(filas)
    if df.empty:
        raise FileNotFoundError(f"No hay imágenes .jpg en {DIR_DATOS}. Revise ASL_DATA_DIR.")
    if muestra:
        df = df.groupby("label", group_keys=False).sample(n=SAMPLES_PER_CLASS, random_state=SEMILLA)
    df["label_id"] = df["label"].map({clase: i for i, clase in enumerate(clases)})
    return df.reset_index(drop=True)


def dividir(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Divide 70/15/15 de forma estratificada, con la misma semilla del laboratorio."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEMILLA)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEMILLA)
    return {"train": train, "val": val, "test": test}
