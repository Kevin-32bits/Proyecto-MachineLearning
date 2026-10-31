"""Genera la Figura 1 del informe: diagrama del pipeline.

    python -m src.pipeline_diagram
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import FIGURES

STEPS = [
    ("1. Carga", "Dataset UCI id 697\n4 424 estudiantes"),
    ("2. Objetivo binario", "Dropout = 1\nEnrolled + Graduate = 0"),
    ("3. Partición", "80 % train / 20 % test\nestratificada, seed = 42"),
    ("4. Preprocesamiento", "One-hot + estandarizar\najustado solo con train"),
    ("5. Entrenamiento", "Reg. Logística, Árbol,\nRandom Forest, Grad. Boosting"),
    ("6. Predicción", "Sobre el conjunto\nde prueba (test)"),
    ("7. Evaluación", "Accuracy, Precision,\nRecall, F1 + matriz conf."),
    ("8. Importancia", "Permutación en el\nmejor modelo (OE5)"),
]
COLORS = ["#dbe9f6", "#dbe9f6", "#dbe9f6", "#e2f0d9",
          "#fff2cc", "#fff2cc", "#fbe0d6", "#fbe0d6"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    w, h = 2.4, 1.5
    xs = [0.3 + i * 2.9 for i in range(4)]
    rows = [2.6, 0.5]  # fila superior e inferior

    pos = []
    for i in range(8):
        r, c = divmod(i, 4)
        x = xs[c] if r == 0 else xs[3 - c]  # fila inferior va de derecha a izquierda
        pos.append((x, rows[r]))

    for (x, y), (title, body), col in zip(pos, STEPS, COLORS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=col, ec="#444444", lw=1.2))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + 0.55, body, ha="center", va="center", fontsize=8.5)

    def arrow(p, q):
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))

    for i in range(7):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if i == 3:  # baja de la fila superior a la inferior
            arrow((x1 + w / 2, y1 - 0.05), (x2 + w / 2, y2 + h + 0.05))
        elif i < 3:
            arrow((x1 + w + 0.05, y1 + h / 2), (x2 - 0.05, y2 + h / 2))
        else:  # fila inferior, de derecha a izquierda
            arrow((x1 - 0.05, y1 + h / 2), (x2 + w + 0.05, y2 + h / 2))

    ax.text(6, 4.7, "Figura 1. Pipeline del proyecto (CRISP-DM: fases 2 a 5)",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(6, 0.1, "El conjunto de prueba solo se usa en los pasos 6 y 7 (sin fuga de datos).",
            ha="center", fontsize=9, style="italic")

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "figura1_pipeline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
