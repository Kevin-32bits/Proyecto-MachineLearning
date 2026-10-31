"""Carga, objetivo binario y partición estratificada.

Uso en el notebook de P2 / P3:

    from src.data_prep import get_train_test
    X_train, X_test, y_train, y_test = get_train_test()

Ejecutar como script para generar la evidencia del informe (3.2 y 3.3):

    python -m src.data_prep
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED, DATA_RAW, FIGURES, POSITIVE_LABEL,
    SEED, TARGET_COL, TEST_SIZE, UCI_ID,
)


def load_dataset() -> pd.DataFrame:
    """Devuelve el dataset completo (variables + Target).

    Usa data/raw/data.csv si existe; si no, lo descarga de UCI con ucimlrepo
    y lo guarda ahí para trabajar sin conexión.
    """
    if DATA_RAW.exists():
        df = pd.read_csv(DATA_RAW, sep=";")
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=UCI_ID)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_RAW, sep=";", index=False)
    df.columns = df.columns.str.strip()  # algunas columnas traen espacios/tabs
    return df


def make_binary_target(df: pd.DataFrame):
    """Separa X e y. y = 1 si Target == 'Dropout'; 0 si Enrolled o Graduate."""
    y = (df[TARGET_COL] == POSITIVE_LABEL).astype(int).rename("dropout")
    X = df.drop(columns=[TARGET_COL])
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Partición estratificada 80/20 con semilla fija."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )


def get_train_test():
    """Atajo para P2 y P3: carga + binariza + divide (determinista)."""
    X, y = make_binary_target(load_dataset())
    return split_data(X, y)


def main() -> None:
    df = load_dataset()
    X, y = make_binary_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    lines = []

    def out(txt=""):
        print(txt)
        lines.append(str(txt))

    out("=== EVIDENCIA 1: forma y objetivo ===")
    out(f"df.shape (con Target): {df.shape}")
    out(f"X.shape (predictoras): {X.shape}")
    out("\nTarget original:")
    out(df[TARGET_COL].value_counts().to_string())
    out("\nTarget original (%):")
    out((df[TARGET_COL].value_counts(normalize=True) * 100).round(1).to_string())
    out("\nTarget binario (1 = Dropout):")
    out(y.value_counts().to_string())
    out((y.value_counts(normalize=True) * 100).round(1).to_string())

    out("\n=== EVIDENCIA 3: calidad de datos ===")
    out(f"Valores nulos totales: {int(df.isna().sum().sum())}")
    out(f"Filas duplicadas: {int(df.duplicated().sum())}")
    out("\nTipos de datos:")
    out(df.dtypes.value_counts().to_string())

    out("\n=== EVIDENCIA 4: partición estratificada ===")
    out(f"SEED = {SEED} | TEST_SIZE = {TEST_SIZE}")
    out(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    out(f"Dropout en train: {int(y_train.sum())} ({y_train.mean()*100:.1f} %)")
    out(f"Dropout en test : {int(y_test.sum())} ({y_test.mean()*100:.1f} %)")

    # Comprobaciones: si fallan, el split no es el acordado
    assert abs(y_train.mean() - y_test.mean()) < 0.005, "Estratificación incorrecta"
    assert len(X_train) + len(X_test) == len(X)
    assert not set(X_train.index) & set(X_test.index), "Fuga: índices repetidos"
    out("\nChequeos OK: estratificado, sin solapamiento train/test.")

    # Guardar partición y evidencia
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_train.assign(dropout=y_train).to_csv(DATA_PROCESSED / "train.csv", index=False)
    X_test.assign(dropout=y_test).to_csv(DATA_PROCESSED / "test.csv", index=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES.parent / "evidencia_01_datos.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Gráfico de distribución de clases (para el informe)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    orig = df[TARGET_COL].value_counts()
    ax[0].bar(orig.index, orig.values,
              color=["#4c78a8", "#f58518", "#e45756"][: len(orig)])
    ax[0].set_title("Target original (3 clases)")
    for i, v in enumerate(orig.values):
        ax[0].text(i, v, str(v), ha="center", va="bottom")
    binv = y.value_counts().sort_index()
    ax[1].bar(["Resto (0)", "Dropout (1)"], binv.values, color=["#4c78a8", "#e45756"])
    ax[1].set_title("Target binario (Dropout vs resto)")
    for i, v in enumerate(binv.values):
        ax[1].text(i, v, f"{v} ({v/len(y)*100:.1f} %)", ha="center", va="bottom")
    for a in ax:
        a.set_ylabel("Estudiantes")
    plt.tight_layout()
    plt.savefig(FIGURES / "distribucion_clases.png", dpi=200)
    print(f"\nGuardado: {FIGURES / 'distribucion_clases.png'}")


if __name__ == "__main__":
    main()
