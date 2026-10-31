"""Puntuación de estudiantes para la app: semáforo, simulador, lotes y métricas.

Sin dependencias de Streamlit, para poder probarse por separado.
"""
import io

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src.config import POSITIVE_LABEL, TARGET_COL
from src.preprocessor import ALL_FEATURE_COLS, NOMINAL_COLS

CORTE_ROJO = 0.60  # desde aquí la alerta pasa de amarilla a roja

# Criterios de éxito de la sección 1.4 (clase Dropout)
CRITERIOS = {"accuracy": 0.75, "precision": 0.75, "recall": 0.80, "f1": 0.78}

# El CSV del zip de UCI trae "Marital status"; ucimlrepo entrega "Marital Status"
ALIAS_COLUMNAS = {"Marital status": "Marital Status"}

SEMESTRES = ("1st", "2nd")


def semaforo(probabilidades, umbral: float) -> np.ndarray:
    """Verde bajo el umbral del modelo; sobre él, Amarillo y luego Rojo."""
    prob = np.asarray(probabilidades)
    return np.select(
        [prob >= max(CORTE_ROJO, umbral), prob >= umbral],
        ["Rojo", "Amarillo"],
        default="Verde",
    )


def validar_perfil(entradas: dict, rangos: dict, etiquetas: dict) -> list:
    """Devuelve los errores de un perfil del simulador (vacío si es válido)."""
    errores = [
        f"{etiquetas[col]}: debe estar entre {minimo} y {maximo}."
        for col, (minimo, maximo) in rangos.items()
        if not minimo <= entradas[col] <= maximo
    ]
    for sem, nombre in zip(SEMESTRES, ("1er", "2do")):
        matriculadas = entradas[f"Curricular units {sem} sem (enrolled)"]
        if entradas[f"Curricular units {sem} sem (approved)"] > matriculadas:
            errores.append(f"{nombre} semestre: aprobadas no puede superar a matriculadas.")
        if entradas[f"Curricular units {sem} sem (credited)"] > matriculadas:
            errores.append(f"{nombre} semestre: convalidadas no puede superar a matriculadas.")
    return errores


def construir_perfil(medianas: dict, entradas: dict) -> pd.DataFrame:
    """Completa con las medianas de train las variables que el simulador no pide.

    En el dataset, 0 materias aprobadas siempre implica promedio 0.
    """
    perfil = {**medianas, **entradas}
    for sem in SEMESTRES:
        if perfil[f"Curricular units {sem} sem (approved)"] == 0:
            perfil[f"Curricular units {sem} sem (grade)"] = 0.0
    return pd.DataFrame([perfil])[ALL_FEATURE_COLS]


def factores_riesgo(pipeline, medianas: dict, entradas: dict):
    """Mide cada dato por separado frente a un estudiante típico.

    Parte del perfil con las medianas de train y cambia un solo dato cada vez:
    positivo sube el riesgo, negativo lo baja. Devuelve (riesgo_tipico, aportes
    de mayor a menor). Los aportes no se suman y no son una relación causal.
    """
    def probabilidad(cambios):
        return float(pipeline.predict_proba(construir_perfil(medianas, cambios))[0, 1])

    tipico = probabilidad({})
    aportes = {
        col: probabilidad({col: valor}) - tipico
        for col, valor in entradas.items()
        if valor != medianas[col]
    }
    return tipico, sorted(aportes.items(), key=lambda item: item[1], reverse=True)


def leer_lote(contenido: bytes) -> pd.DataFrame:
    """Lee un CSV separado por ',' o ';'. Lanza ValueError si no se puede leer."""
    cabecera = contenido.split(b"\n", 1)[0]
    sep = ";" if cabecera.count(b";") > cabecera.count(b",") else ","
    try:
        df = pd.read_csv(io.BytesIO(contenido), sep=sep, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"No se pudo leer el CSV: {e}") from e
    df.columns = df.columns.astype(str).str.strip()
    return df.rename(columns=ALIAS_COLUMNAS)


def preparar_lote(df: pd.DataFrame):
    """Devuelve (X, validas): filas puntuables y máscara sobre el df original."""
    faltan = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if faltan:
        raise ValueError(
            f"Faltan {len(faltan)} de {len(ALL_FEATURE_COLS)} columnas: {', '.join(faltan[:5])}"
            + (" ..." if len(faltan) > 5 else "")
        )
    X = df[ALL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    validas = X.notna().all(axis=1)
    return X[validas], validas


def categorias_desconocidas(pipeline, X: pd.DataFrame) -> dict:
    """Cuenta por columna los códigos nominales que el modelo no vio en train."""
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["nominal"]
    conteo = {}
    for col, conocidas in zip(NOMINAL_COLS, encoder.categories_):
        n = int((~X[col].isin(conocidas)).sum())
        if n:
            conteo[col] = n
    return conteo


def etiquetas_reales(df: pd.DataFrame):
    """Devuelve y (0/1) si el lote trae resultado real utilizable; si no, None."""
    if "dropout" in df.columns:
        y = pd.to_numeric(df["dropout"], errors="coerce")
    elif TARGET_COL in df.columns:
        y = (df[TARGET_COL].astype(str).str.strip() == POSITIVE_LABEL).astype(int)
    else:
        return None
    return y if y.isin([0, 1]).all() and y.nunique() == 2 else None


def metricas(y_true, y_proba, umbral: float) -> dict:
    """Métricas de la clase Dropout con el umbral indicado."""
    y_pred = (np.asarray(y_proba) >= umbral).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "cm": confusion_matrix(y_true, y_pred),
    }
