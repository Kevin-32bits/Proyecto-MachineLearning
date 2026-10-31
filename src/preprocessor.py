"""Definición de tipos de variables y preprocesador para el Pipeline de Scikit-Learn.

Agrupa las 36 variables predictoras en:
- Nominales (One-Hot Encoding)
- Binarias (Passthrough)
- Numéricas continuas/discretas (StandardScaler)
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Variables Nominales (Categorías codificadas como enteros -> One-Hot)
NOMINAL_COLS = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# 2. Variables Binarias (Valores 0 y 1 ya balanceados de origen -> Passthrough)
BINARY_COLS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# 3. Variables Numéricas Continuas y Discretas de Conteo -> StandardScaler
NUMERIC_COLS = [
    "Application order",
    "Age at enrollment",
    "Previous qualification (grade)",
    "Admission grade",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

ALL_FEATURE_COLS = NOMINAL_COLS + BINARY_COLS + NUMERIC_COLS


def get_preprocessor() -> ColumnTransformer:
    """Retorna un ColumnTransformer listo para encadenar en un Pipeline.

    - Nominales: OneHotEncoder (handle_unknown='ignore' para evitar fallos en test)
    - Binarias: passthrough (se conservan como 0 y 1)
    - Numéricas: StandardScaler (media 0, desviación 1)
    """
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_COLS,
            ),
            (
                "binary",
                "passthrough",
                BINARY_COLS,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

