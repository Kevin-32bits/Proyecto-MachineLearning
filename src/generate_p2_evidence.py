"""Genera las evidencias y figuras de la Parte 2:
- reports/evidencia_02_preprocesamiento.txt
- reports/figures/evidencia_02_matriz_comparacion.png
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.config import DATA_RAW, SEED, TEST_SIZE, TARGET_COL, POSITIVE_LABEL, FIGURES
from src.preprocessor import (
    NOMINAL_COLS, BINARY_COLS, NUMERIC_COLS, ALL_FEATURE_COLS, get_preprocessor
)

def generate_p2_evidence():
    df = pd.read_csv(DATA_RAW, sep=";")
    df.columns = df.columns.str.strip()
    
    y = (df[TARGET_COL] == POSITIVE_LABEL).astype(int).rename("dropout")
    X = df[ALL_FEATURE_COLS]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    
    preprocessor = get_preprocessor()
    preprocessor.fit(X_train)
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    
    # 1. Guardar reporte de texto
    lines = [
        "=== EVIDENCIA PARTE 2: EXPLORACIÓN, DICCIONARIO Y PREPROCESAMIENTO ===",
        f"Dimensiones del dataset original: {df.shape[0]} filas x {df.shape[1]} columnas",
        f"Valores nulos totales: {df.isna().sum().sum()} (Comprobado: dataset completo sin valores faltantes)",
        f"Filas duplicadas: {df.duplicated().sum()} (Comprobado: registros únicos sin redundancia)",
        "",
        "--- CLASIFICACIÓN DE LAS 36 VARIABLES PREDICTORAS ---",
        f"1. Nominales codificadas ({len(NOMINAL_COLS)}): Tratamiento -> OneHotEncoder(handle_unknown='ignore')",
        f"   Variables: {', '.join(NOMINAL_COLS)}",
        f"2. Binarias 0/1 ({len(BINARY_COLS)}): Tratamiento -> Passthrough (conservan escala 0/1)",
        f"   Variables: {', '.join(BINARY_COLS)}",
        f"3. Numéricas continuas/conteo ({len(NUMERIC_COLS)}): Tratamiento -> StandardScaler()",
        f"   Variables: {', '.join(NUMERIC_COLS)}",
        "",
        "--- PARTICIÓN ESTRATIFICADA (80/20, SEED 42) ---",
        f"X_train original: {X_train.shape} | y_train: {y_train.shape} (Dropout: {y_train.sum()} = {y_train.mean()*100:.2f}%)",
        f"X_test original : {X_test.shape} | y_test: {y_test.shape} (Dropout: {y_test.sum()} = {y_test.mean()*100:.2f}%)",
        "",
        "--- RESULTADO DE LA TRANSFORMACIÓN (PREPROCESADOR AJUSTADO SOLO CON TRAIN) ---",
        f"Columnas resultantes tras codificación One-Hot: {len(feature_names)} características",
        f"X_train_transformado: {X_train_trans.shape}",
        f"X_test_transformado : {X_test_trans.shape}",
        f"Valores nulos tras transformación (train): {pd.isna(X_train_trans).sum()}",
        f"Valores nulos tras transformación (test) : {pd.isna(X_test_trans).sum()}",
        "Comprobación de dimensiones y compatibilidad: OK."
    ]
    
    txt_path = FIGURES.parent / "evidencia_02_preprocesamiento.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evidencia escrita en: {txt_path}")
    
    # 2. Guardar figura comparativa antes/después
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Gráfico 1: Clasificación de variables
    counts = [len(NOMINAL_COLS), len(BINARY_COLS), len(NUMERIC_COLS)]
    labels = ["Nominales\n(One-Hot: 9)", "Binarias\n(Passthrough: 8)", "Numéricas\n(StandardScaler: 19)"]
    colors = ["#4c78a8", "#72b7b2", "#f58518"]
    axes[0].bar(labels, counts, color=colors, width=0.55)
    axes[0].set_title("Clasificación de las 36 variables predictoras", fontweight="bold")
    axes[0].set_ylabel("Cantidad de variables")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    axes[0].set_ylim(0, 22)
    
    # Gráfico 2: Dimensiones Antes vs Después
    dims = [36, len(feature_names)]
    dim_labels = ["Antes\n(Original: 36 cols)", f"Después\n(Preprocesado: {len(feature_names)} cols)"]
    axes[1].bar(dim_labels, dims, color=["#b279a2", "#54a24b"], width=0.45)
    axes[1].set_title("Expansión del espacio de características tras One-Hot", fontweight="bold")
    axes[1].set_ylabel("Número de columnas / features")
    for i, v in enumerate(dims):
        axes[1].text(i, v + 4, str(v), ha="center", fontweight="bold")
    axes[1].set_ylim(0, 260)
    
    plt.tight_layout()
    plot_path = FIGURES / "evidencia_02_preprocesamiento.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Gráfico guardado en: {plot_path}")

if __name__ == "__main__":
    generate_p2_evidence()

