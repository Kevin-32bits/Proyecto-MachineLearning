"""Configuración compartida del proyecto.

ACUERDO DE EQUIPO (P1 -> P2 y P3): estos valores NO se cambian sin avisar.
Todos los modelos deben usar exactamente la misma partición.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "data.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

UCI_ID = 697                 # Predict Students' Dropout and Academic Success
TARGET_COL = "Target"        # Dropout / Enrolled / Graduate
POSITIVE_LABEL = "Dropout"   # clase positiva (1); el resto (Enrolled + Graduate) = 0

SEED = 42                    # random_state único para todo el proyecto
TEST_SIZE = 0.20             # partición 80 % entrenamiento / 20 % prueba
PRIMARY_METRIC = "f1"        # Métrica principal para elegir al modelo ganador (F1-score clase positiva)

