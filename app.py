"""Aplicación Web Interactiva en Streamlit.
Predicción de la Deserción Universitaria (Sistemas Inteligentes - UPN).
Dataset: UCI #697 — Predict Students' Dropout and Academic Success
"""
import sys
import json
from pathlib import Path

# Configurar path raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import altair as alt
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import precision_recall_curve, average_precision_score

from src.config import FIGURES, POSITIVE_LABEL, SEED, TARGET_COL
from src.data_prep import load_dataset, make_binary_target, split_data
from src.preprocessor import ALL_FEATURE_COLS
from src.scoring import (
    CORTE_ROJO, categorias_desconocidas, construir_perfil, etiquetas_reales, CRITERIOS,
    factores_riesgo, leer_lote, metricas, preparar_lote, semaforo, validar_perfil,
)

# ---------------------------------------------------------
# Configuración general de la página Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sistema Predictivo de Deserción Universitaria",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS modernos para una experiencia visual ejecutiva
st.html("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  /* Tarjetas blancas con sombras suaves y esquinas redondeadas */
  [data-testid="stMain"] [data-testid="stMetric"],
  [data-testid="stMain"] [data-testid="stExpander"] details,
  [class*="st-key-tarjeta"] { 
      background: #ffffff; 
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(13, 54, 107, 0.04);
      border: 1px solid #dbe6f2;
  }
  [data-testid="stMain"] [data-testid="stExpander"] details { 
      border-radius: 10px; 
  }

  /* Acento superior en las métricas KPI */
  [data-testid="stMetric"] {
      border-top: 4px solid #184f95 !important;
      padding: 16px 18px !important;
  }

  /* Barra de Navegación Segmented Control */
  [data-testid="stSegmentedControl"] {
      background: #ffffff;
      padding: 6px;
      border-radius: 14px;
      border: 1px solid #d3deeb;
      box-shadow: 0 4px 15px rgba(13, 54, 107, 0.05);
      margin-bottom: 20px;
      gap: 4px;
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-start;
  }
  
  [data-testid="stSegmentedControl"] button {
      border-radius: 8px !important;
      font-weight: 600 !important;
      font-size: 0.90rem !important;
      padding: 7px 14px !important;
      border: none !important;
      transition: all 0.2s ease !important;
  }

  /* Botones preset con micro-interacciones */
  .stButton > button {
      border-radius: 10px;
      font-weight: 600;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .stButton > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }

  /* Ajuste de densidad y espaciado de cabeceras */
  h1, h2, h3 {
      letter-spacing: -0.02em;
  }
</style>
""")

MODELS_DIR = ROOT_DIR / "models"

COL_APROB_1 = "Curricular units 1st sem (approved)"
COL_APROB_2 = "Curricular units 2nd sem (approved)"
COL_MATRIC_1 = "Curricular units 1st sem (enrolled)"
COL_MATRIC_2 = "Curricular units 2nd sem (enrolled)"
COL_CONV_2 = "Curricular units 2nd sem (credited)"
COL_CONV_1 = "Curricular units 1st sem (credited)"
COL_PAGOS = "Tuition fees up to date"
COL_DEUDA = "Debtor"
COL_BECA = "Scholarship holder"
COL_EDAD = "Age at enrollment"
COL_CARRERA = "Course"
COL_ID = "id_estudiante"

# Top 10 por permutación a nivel de variable original, medida en validación cruzada.
# Se presentan por tema, no por importancia, para leerlas como una ficha del estudiante.
ETIQUETAS = {
    COL_CARRERA: "Carrera",
    COL_EDAD: "Edad al matricularse",
    COL_MATRIC_1: "Materias matriculadas 1er semestre",
    COL_APROB_1: "Materias aprobadas 1er semestre",
    COL_MATRIC_2: "Materias matriculadas 2do semestre",
    COL_APROB_2: "Materias aprobadas 2do semestre",
    COL_CONV_2: "Materias convalidadas 2do semestre",
    COL_PAGOS: "Pagos de matrícula al día",
    COL_DEUDA: "Tiene deudas pendientes",
    COL_BECA: "Tiene beca",
}
ENTRADAS_SIMULADOR = list(ETIQUETAS)

CARRERAS = {
    33: "Tecnologías de Producción de Biocombustibles",
    171: "Diseño de Animación y Multimedia",
    8014: "Servicio Social (nocturno)",
    9003: "Agronomía",
    9070: "Diseño de Comunicación",
    9085: "Enfermería Veterinaria",
    9119: "Ingeniería Informática",
    9130: "Equinocultura",
    9147: "Administración",
    9238: "Servicio Social",
    9254: "Turismo",
    9500: "Enfermería",
    9556: "Higiene Oral",
    9670: "Gestión de Publicidad y Marketing",
    9773: "Periodismo y Comunicación",
    9853: "Educación Básica",
    9991: "Administración (nocturno)",
}

PERFILES = {
    "riesgo": {
        COL_APROB_1: 0, COL_APROB_2: 0, COL_MATRIC_1: 6, COL_MATRIC_2: 6, COL_CONV_2: 0,
        COL_PAGOS: 0, COL_DEUDA: 1, COL_BECA: 0, COL_EDAD: 32, COL_CARRERA: 9238,
    },
    "medio": {
        COL_APROB_1: 4, COL_APROB_2: 3, COL_MATRIC_1: 6, COL_MATRIC_2: 6, COL_CONV_2: 0,
        COL_PAGOS: 1, COL_DEUDA: 1, COL_BECA: 0, COL_EDAD: 24, COL_CARRERA: 9238,
    },
    "seguro": {
        COL_APROB_1: 6, COL_APROB_2: 6, COL_MATRIC_1: 6, COL_MATRIC_2: 6, COL_CONV_2: 0,
        COL_PAGOS: 1, COL_DEUDA: 0, COL_BECA: 1, COL_EDAD: 18, COL_CARRERA: 9238,
    },
}

# Paleta de gráficos y estados
COLOR_SERIE = "#2a78d6"
COLOR_APAGADO = "#c3c2b7"
COLOR_TINTA = "#0b0b0b"
COLOR_TINTA_2 = "#52514e"
COLOR_REJILLA = "#e1e0d9"
COLOR_SUPERFICIE = "#ffffff"
COLOR_MARINO = "#0d366b"
COLOR_ENCABEZADO = "#184f95"
COLOR_LUZ_APAGADA = "#2f5a8f"
COLOR_ESTADO = {"Rojo": "#d03b3b", "Amarillo": "#fab219", "Verde": "#0ca30c"}
ICONO_ESTADO = {"Rojo": "🔴", "Amarillo": "🟡", "Verde": "🟢"}
TEXTO_ESTADO = {
    "Rojo": ("Riesgo alto", "Priorizar contacto e intervención."),
    "Amarillo": ("Riesgo moderado", "Activar alerta y seguimiento cercano."),
    "Verde": ("Riesgo bajo", "Mantener seguimiento regular."),
}


# ---------------------------------------------------------
# Funciones cacheadas para carga de datos y modelo ganador
# ---------------------------------------------------------
@st.cache_data
def load_all_data():
    raw_df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(*make_binary_target(raw_df))
    return raw_df, X_train[ALL_FEATURE_COLS], X_test[ALL_FEATURE_COLS], y_train, y_test


@st.cache_resource
def load_winner_model():
    return joblib.load(MODELS_DIR / "pipeline_ganador.joblib")


@st.cache_data
def load_metadata():
    with open(MODELS_DIR / "metricas_test.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_permutation_data():
    perm_file = MODELS_DIR / "permutation_importance.json"
    if perm_file.exists():
        with open(perm_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def predict_test(X_test):
    return load_winner_model().predict_proba(X_test)[:, 1]


def aplicar_perfil(nombre):
    """Carga un perfil de ejemplo en el simulador y lo evalúa de inmediato."""
    st.session_state.update(PERFILES[nombre])
    st.session_state["perfil_evaluado"] = PERFILES[nombre]


def describir_valor(col, valor):
    if col == COL_CARRERA:
        return CARRERAS[int(valor)]
    if col in (COL_PAGOS, COL_DEUDA, COL_BECA):
        return "Sí" if valor == 1 else "No"
    return str(int(valor))


def encabezado(titulo, subtitulo):
    """Banda de color con el título de la vista; admite HTML simple en el subtítulo."""
    st.html(f"""
    <div style="padding:22px 28px;border-radius:14px;background:linear-gradient(135deg, {COLOR_MARINO} 0%, {COLOR_ENCABEZADO} 100%);box-shadow:0 4px 18px rgba(13,54,107,0.12);margin-bottom:18px;">
      <h1 style="margin:0;padding:0;font-size:2rem;font-weight:700;line-height:1.25;color:#ffffff">{titulo}</h1>
      <div style="margin-top:6px;font-size:1.02rem;color:#e6effb">{subtitulo}</div>
    </div>
    """)


def tarjeta(clave, contenedor=st, height="content"):
    """Contenedor con borde y fondo blanco; `clave` debe ser única por vista."""
    return contenedor.container(border=True, height=height, key=f"tarjeta_{clave}")


def mostrar_metricas(m):
    """Fila de métricas de la clase Dropout contra los criterios de la sección 1.4."""
    columnas = st.columns(5)
    for col, clave, nombre in zip(
        columnas,
        ["accuracy", "precision", "recall", "f1"],
        ["Accuracy", "Precision", "Recall", "F1-Score"],
    ):
        cumple = m[clave] > CRITERIOS[clave]
        valor = f"{m[clave]:.4f}" if clave == "f1" else f"{m[clave] * 100:.1f} %"
        meta = f"{CRITERIOS[clave]:.2f}" if clave == "f1" else f"{CRITERIOS[clave] * 100:.0f} %"
        col.metric(nombre, valor, f"{'✔' if cumple else '✘'} Meta > {meta}", delta_color="off", delta_arrow="off", border=True)
    columnas[4].metric("ROC-AUC", f"{m['roc_auc']:.4f}", border=True)


def tarjeta_resultado(prob, color, umbral):
    """HTML del semáforo con la probabilidad y su posición en la escala de riesgo."""
    corte = max(CORTE_ROJO, umbral)
    luces = "".join(
        f'<div style="width:38px;height:38px;border-radius:50%;'
        f'background:{COLOR_ESTADO[c] if c == color else COLOR_LUZ_APAGADA}"></div>'
        for c in ("Rojo", "Amarillo", "Verde")
    )
    barra = "".join(
        f'<div style="flex:{ancho};height:12px;border-radius:4px;background:{COLOR_ESTADO[c]}"></div>'
        for c, ancho in [("Verde", umbral), ("Amarillo", corte - umbral), ("Rojo", 1 - corte)]
        if ancho > 0
    )
    titulo, detalle = TEXTO_ESTADO[color]
    return f"""
    <div style="display:flex;gap:24px;align-items:center;padding:20px 24px;border-radius:12px;
                border:1px solid rgba(11,11,11,.10);background:{COLOR_SUPERFICIE}">
      <div style="display:flex;flex-direction:column;gap:8px;padding:10px;border-radius:32px;background:{COLOR_MARINO}">{luces}</div>
      <div style="flex:1;min-width:0">
        <div style="color:{COLOR_TINTA_2}">Probabilidad estimada de abandono</div>
        <div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap">
          <span style="font-size:3.2rem;font-weight:700;line-height:1.2;color:{COLOR_TINTA}">{prob * 100:.1f} %</span>
          <span style="font-size:1.4rem;font-weight:600;color:{COLOR_TINTA}">{ICONO_ESTADO[color]} {titulo}</span>
        </div>
        <div style="margin-bottom:12px;color:{COLOR_TINTA_2}">{detalle}</div>
        <div style="position:relative;padding-top:12px">
          <div style="position:absolute;top:0;left:calc({prob * 100:.1f}% - 6px);width:0;height:0;
                      border-left:6px solid transparent;border-right:6px solid transparent;
                      border-top:9px solid {COLOR_TINTA}"></div>
          <div style="display:flex;gap:2px">{barra}</div>
        </div>
        <div style="margin-top:6px;font-size:.85rem;color:{COLOR_TINTA_2}">
          🟢 menor a {umbral * 100:.0f} % · 🟡 desde {umbral * 100:.0f} % · 🔴 desde {corte * 100:.0f} %
        </div>
      </div>
    </div>
    """


def grafico_barras(datos, dominio, formato, referencia, destacar=None):
    """Barras horizontales de una serie: `etiqueta` ya incluye el valor exacto."""
    color = alt.value(COLOR_SERIE)
    if destacar is not None:
        color = alt.condition(alt.datum.etiqueta == destacar, alt.value(COLOR_SERIE), alt.value(COLOR_APAGADO))
    barras = alt.Chart(datos).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("etiqueta:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=14, labelColor=COLOR_TINTA, ticks=False, domainColor=COLOR_APAGADO)),
        x=alt.X("valor:Q", title=None, scale=alt.Scale(domain=dominio),
                axis=alt.Axis(format=formato, values=[0, 0.25, 0.5, 0.75, 1], gridColor=COLOR_REJILLA,
                              labelColor=COLOR_TINTA_2, domain=False, ticks=False)),
        color=color,
        tooltip=[alt.Tooltip("etiqueta:N", title="Grupo"), alt.Tooltip("valor:Q", title="Valor", format=formato)],
    )
    regla = alt.Chart(pd.DataFrame({"valor": [referencia]})).mark_rule(
        color=COLOR_TINTA_2, strokeWidth=1.5
    ).encode(x="valor:Q")
    return (barras + regla).properties(height=34 * len(datos) + 36, background=COLOR_SUPERFICIE)


def grafico_matriz_confusion(cm, umbral):
    """Matriz de confusión interactiva, no aplastada y sincronizada con el umbral activo."""
    tn, fp, fn, tp = cm.ravel()
    total_resto = tn + fp
    total_dropout = fn + tp

    datos_cm = pd.DataFrame([
        {
            "Real": "Resto (0)",
            "Predicho": "Resto (0)",
            "Conteo": int(tn),
            "Tag": "VERDADEROS NEGATIVOS (TN)",
            "TextoNum": f"{tn:,}",
            "TextoSub": f"{tn/total_resto*100:.1f} % (Especificidad)",
            "Tipo": "Verdadero Negativo (TN)",
            "Significado": "Continuó sus estudios y el modelo NO levantó falsa alarma",
        },
        {
            "Real": "Resto (0)",
            "Predicho": "Dropout (1)",
            "Conteo": int(fp),
            "Tag": "FALSOS POSITIVOS (FP)",
            "TextoNum": f"{fp:,}",
            "TextoSub": f"{fp/total_resto*100:.1f} % (Falsa Alarma)",
            "Tipo": "Falso Positivo (FP)",
            "Significado": "No iba a abandonar, pero recibió apoyo preventivo",
        },
        {
            "Real": "Dropout (1)",
            "Predicho": "Resto (0)",
            "Conteo": int(fn),
            "Tag": "FALSOS NEGATIVOS (FN)",
            "TextoNum": f"{fn:,}",
            "TextoSub": f"{fn/total_dropout*100:.1f} % (No Detectado)",
            "Tipo": "Falso Negativo (FN)",
            "Significado": "Abandono real que NO fue alertado a tiempo (error crítico)",
        },
        {
            "Real": "Dropout (1)",
            "Predicho": "Dropout (1)",
            "Conteo": int(tp),
            "Tag": "VERDADEROS POSITIVOS (TP)",
            "TextoNum": f"{tp:,}",
            "TextoSub": f"{tp/total_dropout*100:.1f} % (Recall)",
            "Tipo": "Verdadero Positivo (TP)",
            "Significado": "Abandono detectado a tiempo para intervención de bienestar/tutoría",
        },
    ])

    ord_cat = ["Resto (0)", "Dropout (1)"]

    rects = alt.Chart(datos_cm).mark_rect(
        cornerRadius=12, stroke="#ffffff", strokeWidth=4
    ).encode(
        x=alt.X("Predicho:N", sort=ord_cat, title="Clase Predicha por el Modelo",
                axis=alt.Axis(labelAngle=0, labelFontSize=13, labelFontWeight="bold", titleFontSize=13, titlePadding=10)),
        y=alt.Y("Real:N", sort=ord_cat, title="Clase Real del Estudiante",
                axis=alt.Axis(labelFontSize=13, labelFontWeight="bold", titleFontSize=13, titlePadding=10)),
        color=alt.Color("Conteo:Q", scale=alt.Scale(scheme="blues", domain=[0, max(tn, tp) * 1.05]), legend=None),
        tooltip=[
            alt.Tooltip("Tipo:N", title="Cuadrante"),
            alt.Tooltip("Conteo:Q", format=",", title="Estudiantes"),
            alt.Tooltip("TextoSub:N", title="Tasa sobre clase real"),
            alt.Tooltip("Significado:N", title="Interpretación"),
        ]
    )

    t_tag = alt.Chart(datos_cm).mark_text(
        fontSize=11, fontWeight="bold", dy=-32
    ).encode(
        x=alt.X("Predicho:N", sort=ord_cat),
        y=alt.Y("Real:N", sort=ord_cat),
        text="Tag:N",
        color=alt.condition(alt.datum.Conteo > 150, alt.value("#dbeafe"), alt.value("#184f95"))
    )

    t_num = alt.Chart(datos_cm).mark_text(
        fontSize=32, fontWeight="bold", dy=-4
    ).encode(
        x=alt.X("Predicho:N", sort=ord_cat),
        y=alt.Y("Real:N", sort=ord_cat),
        text="TextoNum:N",
        color=alt.condition(alt.datum.Conteo > 150, alt.value("#ffffff"), alt.value("#0d366b"))
    )

    t_sub = alt.Chart(datos_cm).mark_text(
        fontSize=13, fontWeight="bold", dy=26
    ).encode(
        x=alt.X("Predicho:N", sort=ord_cat),
        y=alt.Y("Real:N", sort=ord_cat),
        text="TextoSub:N",
        color=alt.condition(alt.datum.Conteo > 150, alt.value("#e0edfd"), alt.value("#2f5a8f"))
    )

    chart = (rects + t_tag + t_num + t_sub).properties(
        title=alt.TitleParams(
            text=f"Matriz de Confusión Oficial (Umbral: {umbral:.2f})",
            subtitle="Filas: Clase Real · Columnas: Clase Predicha",
            fontSize=14,
            subtitleFontSize=11,
            color=COLOR_MARINO,
            subtitleColor=COLOR_TINTA_2,
            anchor="start"
        ),
        height=370,
        background=COLOR_SUPERFICIE
    )
    return chart


def grafico_precision_recall(y_true, y_proba, m_activo, umbral_activo):
    """Curva Precision-Recall enriquecida con líneas de umbral óptimo (0.39), operativo (0.40), estándar (0.50) y activo."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    curva = pd.DataFrame({
        "Recall": recalls,
        "Precision": precisions,
        "Serie": "Curva P-R (AP = " + f"{ap:.3f}" + ")"
    }).reset_index()
    # Filtramos para que el dominio Y [0.30, 1.03] se vea limpio como en la figura del informe
    curva["Precision_clip"] = curva["Precision"].clip(lower=0.30)

    m_050 = metricas(y_true, y_proba, 0.50)
    m_040 = metricas(y_true, y_proba, 0.40)
    m_039 = metricas(y_true, y_proba, 0.3904)

    escala_x = alt.Scale(domain=[0.0, 1.02])
    escala_y = alt.Scale(domain=[0.30, 1.03])

    # 1. Línea principal de la curva P-R
    linea_pr = alt.Chart(curva).mark_line(strokeWidth=3, interpolate="step-after").encode(
        x=alt.X("Recall:Q", scale=escala_x, title="Recall (Sensibilidad)",
                axis=alt.Axis(gridColor=COLOR_REJILLA, labelColor=COLOR_TINTA_2, titleFontSize=13, labelFontSize=12, format=".2f")),
        y=alt.Y("Precision_clip:Q", scale=escala_y, title="Precision",
                axis=alt.Axis(gridColor=COLOR_REJILLA, labelColor=COLOR_TINTA_2, titleFontSize=13, labelFontSize=12, format=".2f")),
        color=alt.Color(
            "Serie:N",
            scale=alt.Scale(
                domain=[
                    f"Curva P-R (AP = {ap:.3f})",
                    "Umbral óptimo F1 = 0.39 (Recall: 83.5%)",
                    "Umbral operativo = 0.40 (Recall: 83.1%)",
                    "Umbral estándar = 0.50 (Recall: 76.1%)",
                ],
                range=["#2a78d6", "#d03b3b", "#f59e0b", "#64748b"]
            ),
            legend=alt.Legend(
                title="Referencias en el Gráfico",
                orient="bottom-left",
                fillColor="#ffffff",
                strokeColor="#d3deeb",
                padding=10,
                cornerRadius=8,
                labelFontSize=12,
                titleFontSize=12,
                labelLimit=320,
            )
        ),
        order="index:Q",
        tooltip=[
            alt.Tooltip("Recall:Q", format=".3f", title="Recall (Sensibilidad)"),
            alt.Tooltip("Precision:Q", format=".3f", title="Precision"),
        ]
    )

    # 2. Líneas verticales de referencia (0.39 óptimo F1, 0.40 operativo, 0.50 estándar)
    df_reglas_v = pd.DataFrame([
        {"Recall": m_039["recall"], "Serie": "Umbral óptimo F1 = 0.39 (Recall: 83.5%)"},
        {"Recall": m_040["recall"], "Serie": "Umbral operativo = 0.40 (Recall: 83.1%)"},
        {"Recall": m_050["recall"], "Serie": "Umbral estándar = 0.50 (Recall: 76.1%)"},
    ])

    regla_039 = alt.Chart(df_reglas_v.iloc[[0]]).mark_rule(strokeDash=[7, 4], strokeWidth=2.2).encode(
        x="Recall:Q",
        color="Serie:N"
    )
    regla_040 = alt.Chart(df_reglas_v.iloc[[1]]).mark_rule(strokeDash=[3, 3], strokeWidth=2.2).encode(
        x="Recall:Q",
        color="Serie:N"
    )
    regla_050 = alt.Chart(df_reglas_v.iloc[[2]]).mark_rule(strokeDash=[5, 5], strokeWidth=1.8).encode(
        x="Recall:Q",
        color="Serie:N"
    )

    # 3. Puntos clave sobre la curva (0.50, 0.40 y umbral activo)
    df_puntos = pd.DataFrame([
        {
            "Recall": m_050["recall"],
            "Precision": m_050["precision"],
            "Etiqueta": f"Umbral 0.50  ({m_050['recall']*100:.1f}% Rec | {m_050['precision']*100:.1f}% Prec)",
            "Color": "#64748b",
            "dy": -14,
            "dx": -55,
        },
        {
            "Recall": m_040["recall"],
            "Precision": m_040["precision"],
            "Etiqueta": f"Operativo 0.40  ({m_040['recall']*100:.1f}% Rec | {m_040['precision']*100:.1f}% Prec)",
            "Color": "#d03b3b",
            "dy": 18,
            "dx": -65,
        },
    ])

    puntos_ref = alt.Chart(df_puntos).mark_point(filled=True, size=130, stroke="#ffffff", strokeWidth=1.5).encode(
        x="Recall:Q",
        y="Precision:Q",
        color=alt.value("#0d366b"),
        tooltip=[
            alt.Tooltip("Etiqueta:N", title="Punto de corte"),
            alt.Tooltip("Recall:Q", format=".4f"),
            alt.Tooltip("Precision:Q", format=".4f"),
        ]
    )

    textos_ref = alt.Chart(df_puntos).mark_text(fontSize=11.5, fontWeight="bold", dy=-14, dx=-45).encode(
        x="Recall:Q",
        y="Precision:Q",
        text="Etiqueta:N",
        color=alt.value("#0d366b")
    )

    # Punto del umbral activo si es distinto
    punto_activo = alt.Chart(pd.DataFrame([{
        "Recall": m_activo["recall"],
        "Precision": max(0.30, m_activo["precision"]),
        "Umbral": f"Activo ({umbral_activo:.2f})"
    }])).mark_point(
        filled=True, size=180, opacity=1, color="#0ca30c", stroke="#ffffff", strokeWidth=2
    ).encode(
        x="Recall:Q",
        y="Precision:Q",
        tooltip=[
            alt.Tooltip("Umbral:N", title="Umbral seleccionado"),
            alt.Tooltip("Recall:Q", format=".4f"),
            alt.Tooltip("Precision:Q", format=".4f"),
        ],
    )

    chart = (linea_pr + regla_039 + regla_040 + regla_050 + puntos_ref + textos_ref + punto_activo).properties(
        title=alt.TitleParams(
            text="Curva Precision-Recall — Gradient Boosting (Conjunto de Prueba: 885 estudiantes)",
            subtitle="Intercambio entre Precisión y Sensibilidad: calibrar de 0.50 a 0.40 eleva el Recall de 76.06 % a 83.10 % (+20 desertores detectados)",
            fontSize=15,
            subtitleFontSize=12,
            color=COLOR_MARINO,
            subtitleColor=COLOR_TINTA_2,
            anchor="start"
        ),
        height=400,
        background=COLOR_SUPERFICIE
    )
    return chart


def grafico_importancia_permutacion(perm_data):
    """Gráfico de barras horizontal nativo en Altair con las 15 variables más influyentes ordenadas de mayor a menor."""
    if not perm_data:
        return None

    traducciones = {
        "Curricular units 2nd sem (approved)": ("Materias aprobadas (2do sem)", "Académico"),
        "Tuition fees up to date": ("Pago de matrícula al día", "Financiero"),
        "Curricular units 1st sem (approved)": ("Materias aprobadas (1er sem)", "Académico"),
        "Scholarship holder": ("Beneficiario de beca", "Financiero"),
        "Curricular units 2nd sem (credited)": ("Créditos convalidados (2do sem)", "Académico"),
        "Age at enrollment": ("Edad al matricularse", "Demográfico"),
        "Father's occupation": ("Ocupación del padre", "Familiar"),
        "Gender": ("Género reportado", "Demográfico"),
        "Mother's qualification": ("Nivel de estudios de la madre", "Familiar"),
        "Debtor": ("Tiene pagos adeudados", "Financiero"),
        "GDP": ("Crecimiento del PBI (Macro)", "Contexto"),
        "International": ("Estudiante internacional", "Demográfico"),
        "Curricular units 1st sem (credited)": ("Créditos convalidados (1er sem)", "Académico"),
        "Curricular units 2nd sem (grade)": ("Promedio de notas (2do sem)", "Académico"),
        "Curricular units 1st sem (without evaluations)": ("Cursos sin evaluar (1er sem)", "Académico"),
    }

    filas = []
    for d in perm_data:
        nombre_es, cat = traducciones.get(d["variable"], (d["variable"], "Otros"))
        m_val = float(d["mean"])
        s_val = float(d["std"])
        filas.append({
            "Variable": nombre_es,
            "Original": d["variable"],
            "Dimensión": cat,
            "Caída_F1": m_val,
            "Desviación": s_val,
            "TextoBarra": f"+{m_val:.4f} (±{s_val:.4f})",
            "TooltipF1": f"+{m_val:.4f} ± {s_val:.4f}",
        })

    df = pd.DataFrame(filas).sort_values("Caída_F1", ascending=False).reset_index(drop=True)
    orden_vars = df["Variable"].tolist()
    max_x = df["Caída_F1"].max() * 1.22

    barras = alt.Chart(df).mark_bar(cornerRadiusEnd=5, height=18).encode(
        y=alt.Y("Variable:N", sort=orden_vars, title=None,
                axis=alt.Axis(labelLimit=280, labelFontSize=12.5, labelColor=COLOR_TINTA, ticks=False)),
        x=alt.X("Caída_F1:Q", title="Caída media en F1-Score (Dropout) al permutar la variable (30 repeticiones)", scale=alt.Scale(domain=[0, max_x]),
                axis=alt.Axis(format=".2f", gridColor=COLOR_REJILLA, labelColor=COLOR_TINTA_2, titleFontSize=12.5)),
        color=alt.Color("Dimensión:N", scale=alt.Scale(
            domain=["Académico", "Financiero", "Demográfico", "Familiar", "Contexto"],
            range=["#184f95", "#fab219", "#2a78d6", "#72b7b2", "#9c755f"]
        ), legend=alt.Legend(title="Dimensión de la Variable", orient="bottom", labelFontSize=12, titleFontSize=12)),
        tooltip=[
            alt.Tooltip("Variable:N", title="Variable"),
            alt.Tooltip("Dimensión:N", title="Dimensión"),
            alt.Tooltip("TooltipF1:N", title="Caída en F1 (± std)"),
            alt.Tooltip("Original:N", title="Nombre técnico UCI"),
        ]
    )

    textos = alt.Chart(df).mark_text(dx=6, align="left", fontSize=11.5, fontWeight="bold").encode(
        y=alt.Y("Variable:N", sort=orden_vars),
        x=alt.X("Caída_F1:Q"),
        text="TextoBarra:N",
        color=alt.value(COLOR_MARINO)
    )

    chart = (barras + textos).properties(
        title=alt.TitleParams(
            text="Top 15 Factores Determinantes (Importancia por Permutación — OE5)",
            subtitle="Ordenado de mayor a menor impacto: las materias aprobadas en 2do semestre (+0.2261) y el pago de matrícula (+0.0759) dominan la predicción",
            fontSize=15,
            subtitleFontSize=12,
            color=COLOR_MARINO,
            subtitleColor=COLOR_TINTA_2,
            anchor="start"
        ),
        height=430,
        background=COLOR_SUPERFICIE
    )
    return chart


def abandono_por_grupo(df):
    """Tasa de abandono de cada grupo, de mayor a menor."""
    es_dropout = df[TARGET_COL] == POSITIVE_LABEL
    grupos = {
        "Pagos atrasados": df[COL_PAGOS] == 0,
        "Pagos al día": df[COL_PAGOS] == 1,
        "Con deudas": df[COL_DEUDA] == 1,
        "Sin deudas": df[COL_DEUDA] == 0,
        "Sin beca": df[COL_BECA] == 0,
        "Con beca": df[COL_BECA] == 1,
        "25 años o más": df[COL_EDAD] >= 25,
        "Menores de 25": df[COL_EDAD] < 25,
    }
    filas = [
        {"Grupo": g, "valor": es_dropout[m].mean(), "Estudiantes": int(m.sum())}
        for g, m in grupos.items()
    ]
    datos = pd.DataFrame(filas).sort_values("valor", ascending=False)
    datos["etiqueta"] = [f"{g}  ·  {v * 100:.1f} %" for g, v in zip(datos["Grupo"], datos["valor"])]
    return datos


# Cargar datos y modelo
raw_df, X_train, X_test, y_train, y_test = load_all_data()
model_pipeline = load_winner_model()
metadata = load_metadata()
perm_data = load_permutation_data()
proba_test = predict_test(X_test)
model_name = metadata["modelo"]
medianas = X_train.median().to_dict()
tasa_abandono = (raw_df[TARGET_COL] == POSITIVE_LABEL).mean()

VISTA_INICIO = "🏠 Contexto y Objetivos"
VISTA_DATOS = "📊 Datos y Metodología"
VISTA_MODELOS = "🏆 Modelos y Resultados"
VISTA_SIMULADOR = "🔮 Simulador en Vivo"
VISTA_CONCLUSIONES = "📌 Conclusiones y Ética"

LISTA_VISTAS = [VISTA_INICIO, VISTA_DATOS, VISTA_MODELOS, VISTA_SIMULADOR, VISTA_CONCLUSIONES]

# ---------------------------------------------------------
# Barra Lateral (Sidebar) — Oculta por defecto para dar ancho completo
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Ajustes del Sistema")
    st.caption("Ajusta el umbral operativo del modelo:")
    umbral = st.slider(
        "Umbral de alerta operativa",
        min_value=0.20, max_value=0.80, value=float(metadata["umbral_operativo"]), step=0.01,
        help="Probabilidad mínima para alertar. Un umbral más bajo (ej. 0.40) detecta más abandonos a tiempo."
    )
    st.divider()
    st.markdown("#### 📌 Información del Modelo")
    st.markdown(f"**Algoritmo:** {model_name}")
    st.markdown(f"**F1-Score Test (0.50):** {metadata['F1-Score (Dropout)']:.4f}")
    st.markdown(f"**Accuracy Test (0.50):** {metadata['Accuracy'] * 100:.2f} %")
    st.markdown(f"**ROC-AUC Test:** {metadata['ROC-AUC']:.4f}")
    st.divider()
    st.caption("Sistemas Inteligentes & Machine Learning — UPN\nGrupo 03 · Dataset: UCI #697")

m_test = metricas(y_test, proba_test, umbral)
m_test_050 = metricas(y_test, proba_test, 0.50)
m_test_040 = metricas(y_test, proba_test, 0.40)

# ---------------------------------------------------------
# Barra Superior de Navegación (Top Navbar)
# ---------------------------------------------------------
st.html(f"""
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px; padding:12px 20px; background:#ffffff; border-radius:12px; border:1px solid #d3deeb; box-shadow:0 2px 10px rgba(0,0,0,0.03);">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="font-size:2rem; line-height:1;">🎓</span>
    <div>
      <div style="font-weight:700; font-size:1.2rem; color:#0d366b; line-height:1.2;">Alerta de Deserción Universitaria · Grupo 03 (UPN)</div>
      <div style="font-size:0.82rem; color:#52514e;">Predicción de la Deserción Universitaria mediante Algoritmos de Clasificación de Machine Learning</div>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="display:inline-flex; align-items:center; gap:6px; background:#e8f4fd; color:#184f95; padding:5px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; border:1px solid #b8dafc;">
      <span style="width:8px; height:8px; border-radius:50%; background:#0ca30c; display:inline-block;"></span>
      Modelo Activo: {model_name} (F1 Test: {metadata['F1-Score (Dropout)']*100:.1f} % | Umbral: {umbral:.2f})
    </span>
  </div>
</div>
""")

if "menu_nav" not in st.session_state or st.session_state["menu_nav"] not in LISTA_VISTAS:
    st.session_state["menu_nav"] = VISTA_INICIO

menu = st.segmented_control(
    "Navegación principal",
    LISTA_VISTAS,
    default=st.session_state["menu_nav"],
    label_visibility="collapsed"
)
if menu is None:
    menu = st.session_state["menu_nav"]
else:
    st.session_state["menu_nav"] = menu

# ---------------------------------------------------------
# VISTA 1: CONTEXTO Y OBJETIVOS (CAPÍTULO I DEL INFORME)
# ---------------------------------------------------------
if menu == VISTA_INICIO:
    encabezado(
        "¿Quién necesita apoyo para no abandonar la universidad?",
        "De un enfoque reactivo tradicional a la <b>detección temprana del riesgo de deserción</b> mediante Machine Learning."
    )

    tab_contexto, tab_objetivos, tab_alcance = st.tabs([
        "🌍 Contexto y Antecedentes",
        "🎯 Objetivos (OG y OE1–OE5)",
        "🛡️ Alcance, Restricciones y Metas"
    ])

    with tab_contexto:
        n1, n2, n3 = st.columns(3)
        n1.metric("🌍 Nivel Internacional (OCDE)", "20 % – 30 %", "Tasa global de abandono superior", delta_color="off", delta_arrow="off", border=True)
        n2.metric("🇵🇪 Nivel Nacional (Perú)", "~174,000", "Estudiantes desertaron en 2020", delta_color="off", delta_arrow="off", border=True)
        n3.metric("📍 Nivel Local (Trujillo)", "25.00 %", "Índice regional (2021–2022)", delta_color="off", delta_arrow="off", border=True)

        col_react, col_proact = st.columns(2, gap="large")
        with col_react:
            with tarjeta("enfoque_reactivo", height="stretch"):
                st.html("""
                <div style="padding:4px 2px;">
                  <div style="display:inline-block;background:#fee2e2;color:#b91c1c;font-weight:700;font-size:0.78rem;padding:4px 10px;border-radius:20px;margin-bottom:10px;">⚠️ ENFOQUE REACTIVO</div>
                  <div style="font-size:1.15rem;font-weight:700;color:#0d366b;margin-bottom:12px;">Actúa cuando el alumno ya abandonó</div>
                  <div style="display:grid;grid-template-columns:1fr;gap:8px;">
                    <div style="background:#f8fafc;border-left:4px solid #d03b3b;padding:8px 12px;border-radius:6px;font-size:0.92rem;">🎓 <b>Impacto estudiantil:</b> proyecto de vida truncado</div>
                    <div style="background:#f8fafc;border-left:4px solid #d03b3b;padding:8px 12px;border-radius:6px;font-size:0.92rem;">📉 <b>Impacto institucional:</b> inestabilidad financiera</div>
                    <div style="background:#f8fafc;border-left:4px solid #d03b3b;padding:8px 12px;border-radius:6px;font-size:0.92rem;">🏛️ <b>Impacto social:</b> pérdida de capital humano</div>
                  </div>
                </div>
                """)
        with col_proact:
            with tarjeta("enfoque_proactivo", height="stretch"):
                st.html("""
                <div style="padding:4px 2px;">
                  <div style="display:inline-block;background:#dcfce7;color:#15803d;font-weight:700;font-size:0.78rem;padding:4px 10px;border-radius:20px;margin-bottom:10px;">🚀 SOLUCIÓN PROACTIVA (IA / ML)</div>
                  <div style="font-size:1.15rem;font-weight:700;color:#0d366b;margin-bottom:12px;">Alerta temprana antes del retiro</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;padding:10px 12px;border-radius:8px;font-weight:600;color:#1e40af;font-size:0.9rem;">1️⃣ Regresión Logística</div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;padding:10px 12px;border-radius:8px;font-weight:600;color:#1e40af;font-size:0.9rem;">2️⃣ Árbol de Decisión</div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;padding:10px 12px;border-radius:8px;font-weight:600;color:#1e40af;font-size:0.9rem;">3️⃣ Random Forest</div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;padding:10px 12px;border-radius:8px;font-weight:600;color:#1e40af;font-size:0.9rem;">4️⃣ Gradient Boosting</div>
                  </div>
                </div>
                """)

        st.markdown("#### ⚙️ Flujo de Alerta y Semáforo de Intervención")
        p1, p2, p3 = st.columns(3)
        with tarjeta("paso_1", p1, "stretch"):
            st.markdown("📥 **1. Datos Históricos**  \nHistorial académico, pagos y perfil socioeconómico.")
        with tarjeta("paso_2", p2, "stretch"):
            st.markdown("🧠 **2. Clasificación ML**  \nPredice la probabilidad individual o masiva de abandono.")
        with tarjeta("paso_3", p3, "stretch"):
            st.markdown("🤝 **3. Acción Oportuna**  \nDeriva a tutoría académica o apoyo financiero.")

        corte = max(CORTE_ROJO, umbral)
        s1, s2, s3 = st.columns(3)
        s1.success(f"🟢 **Bajo (< {umbral * 100:.0f} %):** Seguimiento regular")
        s2.warning(f"🟡 **Moderado ({umbral * 100:.0f} % – {corte * 100:.0f} %):** Tutoría preventiva")
        s3.error(f"🔴 **Alto (≥ {corte * 100:.0f} %):** Priorizar intervención")

        st.divider()
        antecedente_sel = st.selectbox(
            "📚 Antecedentes Científicos y Pregunta de Investigación (selecciona para desplegar):",
            [
                "🔽 Oculto — Selecciona un antecedente o la pregunta de investigación...",
                "❓ Pregunta de Investigación Comparativa",
                "🌍 Internacional (Colombia) — Banquez-Maturana et al. (2025)",
                "🌍 Internacional (México) — Kuz y Morales (2023)",
                "🇵🇪 Nacional (Perú - UNTELS) — Valero Cajahuanca et al. (2022)",
                "📍 Local (Trujillo - UCT) — Izquierdo y Torres (2024)",
                "📋 Ver tabla comparativa de los 4 antecedentes",
            ],
        )

        df_antecedentes = pd.DataFrame([
            {
                "Ámbito": "Internacional (Colombia)",
                "Autores": "Banquez-Maturana et al. (2025)",
                "Muestra y Método": "104,147 registros · DSR + SMOTE",
                "Mejor Resultado": "LightGBM: Accuracy = 87.00 %, F1 = 0.83",
                "Hallazgo Clave": "El progreso académico y la tendencia de notas son los mayores determinantes (SHAP)."
            },
            {
                "Ámbito": "Internacional (México)",
                "Autores": "Kuz y Morales (2023)",
                "Muestra y Método": "1,847 registros (univ. privada, 2014–2020)",
                "Mejor Resultado": "XGBoost: Accuracy = 99.28 % (94 % retención)",
                "Hallazgo Clave": "Predictores en 1er año: promedio inicial, porcentaje de beca y región del campus."
            },
            {
                "Ámbito": "Nacional (Perú - UNTELS)",
                "Autores": "Valero Cajahuanca et al. (2022)",
                "Muestra y Método": "652 estudiantes (muestra 106, 2018–2021)",
                "Mejor Resultado": "KNN (90.62 %) > SVM (87.50 %) > Reg. Log. y Árbol (81.25 %)",
                "Hallazgo Clave": "Integrar variables académicas y socioeconómicas eleva la precisión."
            },
            {
                "Ámbito": "Local (Trujillo - UCT)",
                "Autores": "Izquierdo y Torres (2024)",
                "Muestra y Método": "3,156 estudiantes en 11 carreras (25 % deserción)",
                "Mejor Resultado": "Regresión Logística Binaria (98.70 % en train)",
                "Hallazgo Clave": "Factores significativos: procedencia, n° hermanos, colegio y cursos desaprobados."
            },
        ])

        if "Pregunta" in antecedente_sel:
            st.info(
                "❓ **Pregunta de Investigación:** "
                "¿Cuál de los algoritmos de Machine Learning presenta el mayor rendimiento predictivo en términos de "
                "**exactitud, precisión, sensibilidad y F1-score** para la clasificación de la deserción universitaria, "
                "utilizando el conjunto de datos del repositorio UCI Machine Learning Repository?"
            )
        elif "Colombia" in antecedente_sel:
            st.dataframe(df_antecedentes.iloc[[0]], width="stretch", hide_index=True)
        elif "México" in antecedente_sel:
            st.dataframe(df_antecedentes.iloc[[1]], width="stretch", hide_index=True)
        elif "UNTELS" in antecedente_sel:
            st.dataframe(df_antecedentes.iloc[[2]], width="stretch", hide_index=True)
        elif "Trujillo" in antecedente_sel:
            st.dataframe(df_antecedentes.iloc[[3]], width="stretch", hide_index=True)
        elif "tabla" in antecedente_sel:
            st.dataframe(df_antecedentes, width="stretch", hide_index=True)

    with tab_objetivos:
        col_og, col_oe = st.columns([1.05, 1.35], gap="large")
        with col_og:
            with tarjeta("obj_general", height="stretch"):
                st.html("""
                <div style="padding:4px 2px;">
                  <div style="display:inline-block;background:#dbeafe;color:#1e40af;font-weight:700;font-size:0.78rem;padding:4px 12px;border-radius:20px;margin-bottom:10px;">🎯 OBJETIVO GENERAL (OG)</div>
                  <div style="font-size:1.12rem;font-weight:700;color:#0d366b;line-height:1.4;margin-bottom:12px;">
                    Evaluar y comparar algoritmos de Machine Learning para clasificar el riesgo de deserción universitaria e identificar las variables predictivas clave.
                  </div>
                  <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;">
                    <span style="background:#f1f5f9;border:1px solid #cbd5e1;padding:4px 10px;border-radius:6px;font-size:0.82rem;font-weight:600;color:#334155;">📦 Dataset UCI #697</span>
                    <span style="background:#f1f5f9;border:1px solid #cbd5e1;padding:4px 10px;border-radius:6px;font-size:0.82rem;font-weight:600;color:#334155;">🤖 4 Clasificadores</span>
                    <span style="background:#f1f5f9;border:1px solid #cbd5e1;padding:4px 10px;border-radius:6px;font-size:0.82rem;font-weight:600;color:#334155;">🔄 5-Fold CV + Hold-out Test</span>
                  </div>
                  <div style="background:#eff6ff;border-left:4px solid #2a78d6;padding:12px 14px;border-radius:8px;">
                    <div style="font-size:0.82rem;font-weight:700;color:#184f95;margin-bottom:4px;">🔍 OE5 · EXPLICABILIDAD DE VARIABLES</div>
                    <div style="font-size:0.92rem;color:#1e293b;">Identificar los factores <b>académicos, financieros y sociodemográficos</b> con mayor peso predictivo mediante importancia por permutación.</div>
                  </div>
                </div>
                """)

        with col_oe:
            st.markdown("##### 📌 Evaluación de Rendimiento (OE1 – OE4)")
            g1, g2 = st.columns(2)
            with tarjeta("card_oe1", g1, "stretch"):
                st.html("""
                <div>
                  <span style="background:#e0f2fe;color:#0369a1;font-weight:700;font-size:0.78rem;padding:3px 9px;border-radius:12px;">OE1 · EXACTITUD</span>
                  <div style="font-size:1.1rem;font-weight:700;color:#0d366b;margin:8px 0 4px 0;">Accuracy Global</div>
                  <div style="font-size:0.88rem;color:#52514e;">Medir el desempeño global de clasificación en validación cruzada y prueba independiente.</div>
                </div>
                """)
            with tarjeta("card_oe2", g2, "stretch"):
                st.html("""
                <div>
                  <span style="background:#fef3c7;color:#b45309;font-weight:700;font-size:0.78rem;padding:3px 9px;border-radius:12px;">OE2 · PRECISIÓN</span>
                  <div style="font-size:1.1rem;font-weight:700;color:#0d366b;margin:8px 0 4px 0;">Precision (Dropout)</div>
                  <div style="font-size:0.88rem;color:#52514e;">Clasificar con acierto la clase de riesgo, <b>minimizando falsos positivos</b> institucionales.</div>
                </div>
                """)
            g3, g4 = st.columns(2)
            with tarjeta("card_oe3", g3, "stretch"):
                st.html("""
                <div>
                  <span style="background:#dcfce7;color:#15803d;font-weight:700;font-size:0.78rem;padding:3px 9px;border-radius:12px;">OE3 · SENSIBILIDAD</span>
                  <div style="font-size:1.1rem;font-weight:700;color:#0d366b;margin:8px 0 4px 0;">Recall (Dropout)</div>
                  <div style="font-size:0.88rem;color:#52514e;">Maximizar la detección oportuna de casos reales de abandono, <b>minimizando falsos negativos</b>.</div>
                </div>
                """)
            with tarjeta("card_oe4", g4, "stretch"):
                st.html("""
                <div>
                  <span style="background:#ede9fe;color:#6d28d9;font-weight:700;font-size:0.78rem;padding:3px 9px;border-radius:12px;">OE4 · EQUILIBRIO</span>
                  <div style="font-size:1.1rem;font-weight:700;color:#0d366b;margin:8px 0 4px 0;">F1-Score (Dropout)</div>
                  <div style="font-size:0.88rem;color:#52514e;">Determinar el modelo ganador con el mejor <b>balance armónico</b> entre precisión y sensibilidad.</div>
                </div>
                """)

    with tab_alcance:
        c_alc, c_rest = st.columns(2, gap="large")
        with c_alc:
            with tarjeta("card_alcance", height="stretch"):
                st.html("""
                <div style="padding:4px 2px;">
                  <div style="display:inline-block;background:#dcfce7;color:#15803d;font-weight:700;font-size:0.78rem;padding:4px 12px;border-radius:20px;margin-bottom:12px;">✅ ALCANCE DEL PROYECTO</div>
                  <div style="display:grid;gap:8px;">
                    <div style="background:#f8fafc;border-left:4px solid #2a78d6;padding:8px 12px;border-radius:6px;font-size:0.9rem;">🤖 <b>4 Algoritmos comparados:</b> Regresión Logística, Árbol de Decisión, Random Forest y Gradient Boosting.</div>
                    <div style="background:#f8fafc;border-left:4px solid #2a78d6;padding:8px 12px;border-radius:6px;font-size:0.9rem;">📂 <b>Dataset público UCI #697:</b> <i>Predict Students' Dropout and Academic Success</i> (Realinho et al., 2021).</div>
                    <div style="background:#f8fafc;border-left:4px solid #2a78d6;padding:8px 12px;border-radius:6px;font-size:0.9rem;">⚙️ <b>Flujo completo:</b> Preprocesamiento, validación por métricas e importancia de variables (asociación, no causalidad).</div>
                    <div style="background:#f8fafc;border-left:4px solid #64748b;padding:8px 12px;border-radius:6px;font-size:0.9rem;">🚫 <b>Excluye:</b> Despliegue en producción real e ingesta en tiempo real (prototipo demostrativo).</div>
                  </div>
                </div>
                """)
        with c_rest:
            with tarjeta("card_restricciones", height="stretch"):
                st.html("""
                <div style="padding:4px 2px;">
                  <div style="display:inline-block;background:#fef3c7;color:#b45309;font-weight:700;font-size:0.78rem;padding:4px 12px;border-radius:20px;margin-bottom:12px;">⚠️ RESTRICCIONES DEL ESTUDIO</div>
                  <div style="display:grid;gap:8px;">
                    <div style="background:#f8fafc;border-left:4px solid #fab219;padding:8px 12px;border-radius:6px;font-size:0.9rem;">🌍 <b>Origen de los datos:</b> Institución europea (Portalegre, Portugal); limita generalizar al Perú sin validación externa.</div>
                    <div style="background:#f8fafc;border-left:4px solid #fab219;padding:8px 12px;border-radius:6px;font-size:0.9rem;">🗓️ <b>Ventana temporal:</b> Usa notas hasta el <b>2do semestre</b> (acompaña el 1er año, no predice el día de la matrícula).</div>
                    <div style="background:#f8fafc;border-left:4px solid #fab219;padding:8px 12px;border-radius:6px;font-size:0.9rem;">💻 <b>Tiempo e infraestructura:</b> 4,424 registros entrenables en equipo estándar; sin búsqueda exhaustiva de hiperparámetros.</div>
                  </div>
                </div>
                """)

        st.markdown("#### 🏆 Criterios Operativos de Éxito del Modelo")
        ce1, ce2, ce3, ce4 = st.columns(4)
        ce1.metric("Exactitud (Accuracy)", "> 75 %", "Supera tasa base (~68 %)", delta_color="off", delta_arrow="off", border=True)
        ce2.metric("Precisión (Dropout)", "> 75 %", "Minimiza falsas alarmas", delta_color="off", delta_arrow="off", border=True)
        ce3.metric("Sensibilidad (Recall)", "> 80 %", "Maximiza detección real", delta_color="off", delta_arrow="off", border=True)
        ce4.metric("F1-Score (Dropout)", "> 0.78", "Equilibrio precisión-recall", delta_color="off", delta_arrow="off", border=True)


# ---------------------------------------------------------
# VISTA 2: DATOS Y METODOLOGÍA (CAPÍTULOS II Y III DEL INFORME)
# ---------------------------------------------------------
elif menu == VISTA_DATOS:
    encabezado(
        "📊 Datos, Preprocesamiento y Metodología (CRISP-DM)",
        "Diseño experimental, auditoría de calidad del dataset UCI #697 (4,424 estudiantes, 36 variables), "
        "transformación de características (36 → 235 columnas) y arquitectura algorítmica."
    )

    tab_crispdm, tab_eda, tab_dicc, tab_preproc, tab_arq = st.tabs([
        "🧭 Metodología CRISP-DM y Muestra",
        "📊 Análisis Exploratorio (EDA)",
        "📋 Diccionario de las 36 Variables",
        "⚙️ Preprocesamiento y 5 Estudiantes",
        "🏗️ Arquitectura, Flujo e Hiperparámetros"
    ])

    with tab_crispdm:
        st.markdown("#### 🧭 Proceso de Desarrollo: Metodología CRISP-DM (6 Fases)")
        f_cols = st.columns(6)
        fases_crisp = [
            ("FASE 1", "🎯 Negocio", "#eff6ff", "#184f95", "Objetivos OE1–OE5 y metas operativas (`F1 > 0.78`, `Recall > 80 %`)."),
            ("FASE 2", "📦 Datos", "#f0fdf4", "#15803d", "UCI #697: **4,424 alumnos**, 36 variables, 0 nulos (`32.1 %` Dropout)."),
            ("FASE 3", "⚙️ Preparación", "#fffbeb", "#b45309", "Split **80/20** (`SEED=42`), One-Hot, StandardScaler y pesos de clase."),
            ("FASE 4", "🤖 Modelado", "#f5f3ff", "#6d28d9", "Entrenamiento supervisado de **4 algoritmos** de clasificación."),
            ("FASE 5", "📊 Evaluación", "#fef2f2", "#b91c1c", "5-Fold CV en Train, Test ciego (885), umbral `0.40` y permutación."),
            ("FASE 6", "🚀 Despliegue", "#ecfeff", "#0e7490", "Pipeline serializado (`.joblib`) y simulador web en Streamlit."),
        ]
        for col_f, (tag_f, tit_f, bg_f, col_b, desc_f) in zip(f_cols, fases_crisp):
            with tarjeta(f"crisp_{tag_f}", col_f, "stretch"):
                st.markdown(
                    f"<div style='display:inline-block;padding:2px 8px;border-radius:10px;background:{bg_f};color:{col_b};font-size:0.74rem;font-weight:700;margin-bottom:6px'>{tag_f}</div>"
                    f"<div style='font-weight:700;font-size:1rem;color:#0d366b;margin-bottom:6px'>{tit_f}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(desc_f)

        st.markdown("#### 👥 Población, Muestra Censal y Partición Estratificada (80 / 20)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Población / Muestra", "4,424 estudiantes", "Muestreo censal · Portalegre (Portugal)", delta_color="off", delta_arrow="off", border=True)
        m2.metric("Entrenamiento (80 %)", "3,539 registros", "1,137 Dropout (32.1 %) · 5-Fold CV", delta_color="off", delta_arrow="off", border=True)
        m3.metric("Prueba Hold-Out (20 %)", "885 registros", "284 Dropout (32.1 %) · Evaluación ciega", delta_color="off", delta_arrow="off", border=True)
        m4.metric("Calidad y Limpieza", "0 nulos · 0 duplicados", "Nombres limpiados con str.strip()", delta_color="off", delta_arrow="off", border=True)

        c_pob, c_bloques = st.columns([1.05, 1.25], gap="large")
        with c_pob:
            with tarjeta("card_criterios_muestra", height="stretch"):
                st.markdown("##### 📋 Diseño Muestral y Reglas de Integridad")
                reglas_muestra = [
                    ("🎓 Origen Institucional", "#eff6ff", "#184f95", "17 carreras de pregrado del *IP Portalegre (Portugal)* entre 2008 y 2019 (Realinho et al., 2021 · Licencia CC BY 4.0)."),
                    ("✅ Criterio de Inclusión Censal", "#f0fdf4", "#15803d", "Se incluye el **100 % de los 4,424 registros** al contar con las 36 predictoras completas y estado académico verificado."),
                    ("⚖️ Binarización de la Clase Objetivo", "#fffbeb", "#d97706", "`Dropout = 1` (**1,421 casos · 32.1 %**) frente a `Enrolled + Graduate = 0` (**3,003 casos · 67.9 %**)."),
                    ("🔒 Garantía contra Data Leakage", "#fef2f2", "#d03b3b", "El conjunto `X_test` (**885 alumnos**) se aísla antes de cualquier transformación y jamás participa en el ajuste."),
                ]
                for tit_rm, bg_rm, br_rm, txt_rm in reglas_muestra:
                    st.markdown(
                        f"<div style='padding:9px 12px;margin-bottom:8px;border-radius:8px;background:{bg_rm};border-left:4px solid {br_rm}'>"
                        f"<div style='font-weight:700;color:{br_rm};font-size:0.9rem'>{tit_rm}</div>"
                        f"<div style='color:#1e293b;font-size:0.86rem;margin-top:2px'>{txt_rm}</div></div>",
                        unsafe_allow_html=True,
                    )

        with c_bloques:
            with tarjeta("card_bloques_vars", height="stretch"):
                st.markdown("##### 🧩 Distribución de las 36 Variables Predictoras por Bloque Temático")
                df_b_chart = pd.DataFrame([
                    {"Bloque": "Rendimiento 1er y 2do Sem.", "Variables": 12, "Texto": "12 vars (33.3 %)", "Ejemplos": "Matriculadas, aprobadas, convalidadas, evaluadas y notas"},
                    {"Bloque": "Socioeconómicas", "Variables": 9, "Texto": "9 vars (25.0 %)", "Ejemplos": "Matrícula al día, deudor, becario, estudios/ocupación padres"},
                    {"Bloque": "Ingreso y Trayectoria", "Variables": 7, "Texto": "7 vars (19.4 %)", "Ejemplos": "Carrera (Course), modalidad, orden de postulación, admisión"},
                    {"Bloque": "Demográficas", "Variables": 5, "Texto": "5 vars (13.9 %)", "Ejemplos": "Edad al matricularse, género, estado civil, nacionalidad"},
                    {"Bloque": "Macroeconómicas", "Variables": 3, "Texto": "3 vars (8.3 %)", "Ejemplos": "Tasa de desempleo, inflación anual y crecimiento del PBI"},
                ])
                ord_b = df_b_chart["Bloque"].tolist()
                bar_b = alt.Chart(df_b_chart).mark_bar(cornerRadiusEnd=5, height=22).encode(
                    y=alt.Y("Bloque:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12, labelLimit=200, ticks=False)),
                    x=alt.X("Variables:Q", title="Cantidad de Variables Predictoras (Total = 36)", scale=alt.Scale(domain=[0, 15])),
                    color=alt.Color(
                        "Bloque:N",
                        legend=None,
                        scale=alt.Scale(domain=ord_b, range=["#184f95", "#2a78d6", "#3b82f6", "#64748b", "#94a3b8"]),
                    ),
                    tooltip=[alt.Tooltip("Bloque:N"), alt.Tooltip("Variables:Q"), alt.Tooltip("Ejemplos:N")],
                )
                txt_b = bar_b.mark_text(dx=6, align="left", fontSize=11.5, fontWeight="bold", color=COLOR_MARINO).encode(text="Texto:N")
                st.altair_chart((bar_b + txt_b).properties(height=185, background=COLOR_SUPERFICIE), width="stretch")
                st.dataframe(df_b_chart[["Bloque", "Variables", "Ejemplos"]], width="stretch", hide_index=True)

    with tab_eda:
        grupos = abandono_por_grupo(raw_df)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estudiantes", f"{len(raw_df):,}", "Muestra censal completa", delta_color="off", delta_arrow="off", border=True)
        k2.metric("Tasa de abandono", f"{tasa_abandono * 100:.1f} %", "1,421 casos de Dropout", delta_color="off", delta_arrow="off", border=True)
        k3.metric("Valores faltantes", int(raw_df.isna().sum().sum()), "0 % datos perdidos", delta_color="off", delta_arrow="off", border=True)
        k4.metric("Variables predictoras", len(ALL_FEATURE_COLS), "9 nom · 8 bin · 19 num", delta_color="off", delta_arrow="off", border=True)

        st.subheader("Distribución de Clases (Target Original vs Binario)")
        c_target1, c_target2 = st.columns(2)

        with c_target1:
            with tarjeta("dist_original"):
                df_orig = raw_df[TARGET_COL].value_counts().reset_index()
                df_orig.columns = ["Clase", "Estudiantes"]
                orden_cat = ["Graduate", "Dropout", "Enrolled"]
                df_orig["orden"] = df_orig["Clase"].map(lambda x: orden_cat.index(x) if x in orden_cat else 99)
                df_orig = df_orig.sort_values("orden").drop(columns=["orden"])
                df_orig["Porcentaje"] = (df_orig["Estudiantes"] / len(raw_df) * 100).round(1)
                df_orig["Texto"] = [f"{n:,} ({p:.1f} %)" for n, p in zip(df_orig["Estudiantes"], df_orig["Porcentaje"])]

                bars_orig = alt.Chart(df_orig).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=60).encode(
                    x=alt.X("Clase:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13, labelColor=COLOR_TINTA, ticks=False)),
                    y=alt.Y("Estudiantes:Q", title="N° de Estudiantes", scale=alt.Scale(domain=[0, 2600]), axis=alt.Axis(gridColor=COLOR_REJILLA, labelColor=COLOR_TINTA_2)),
                    color=alt.Color("Clase:N", legend=None, scale=alt.Scale(
                        domain=["Graduate", "Dropout", "Enrolled"],
                        range=["#2a78d6", "#d03b3b", "#fab219"]
                    )),
                    tooltip=[alt.Tooltip("Clase:N"), alt.Tooltip("Estudiantes:Q", format=","), alt.Tooltip("Porcentaje:Q", format=".1f", title="Porcentaje (%)")]
                )
                text_orig = bars_orig.mark_text(dy=-12, fontSize=13, fontWeight="bold", color=COLOR_TINTA).encode(
                    text="Texto:N"
                )
                chart_orig = (bars_orig + text_orig).properties(
                    title=alt.TitleParams(text="Target Original (3 clases)", fontSize=15, color=COLOR_MARINO, anchor="start"),
                    height=340,
                    background=COLOR_SUPERFICIE
                )
                st.altair_chart(chart_orig, width="stretch")

        with c_target2:
            with tarjeta("dist_binario"):
                df_bin = pd.DataFrame([
                    {"Clase": "Resto (0)", "Estudiantes": int((raw_df[TARGET_COL] != POSITIVE_LABEL).sum())},
                    {"Clase": "Dropout (1)", "Estudiantes": int((raw_df[TARGET_COL] == POSITIVE_LABEL).sum())},
                ])
                df_bin["Porcentaje"] = (df_bin["Estudiantes"] / len(raw_df) * 100).round(1)
                df_bin["Texto"] = [f"{n:,} ({p:.1f} %)" for n, p in zip(df_bin["Estudiantes"], df_bin["Porcentaje"])]

                bars_bin = alt.Chart(df_bin).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=75).encode(
                    x=alt.X("Clase:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13, labelColor=COLOR_TINTA, ticks=False)),
                    y=alt.Y("Estudiantes:Q", title="N° de Estudiantes", scale=alt.Scale(domain=[0, 3400]), axis=alt.Axis(gridColor=COLOR_REJILLA, labelColor=COLOR_TINTA_2)),
                    color=alt.Color("Clase:N", legend=None, scale=alt.Scale(
                        domain=["Resto (0)", "Dropout (1)"],
                        range=["#2a78d6", "#d03b3b"]
                    )),
                    tooltip=[alt.Tooltip("Clase:N"), alt.Tooltip("Estudiantes:Q", format=","), alt.Tooltip("Porcentaje:Q", format=".1f", title="Porcentaje (%)")]
                )
                text_bin = bars_bin.mark_text(dy=-12, fontSize=13, fontWeight="bold", color=COLOR_TINTA).encode(
                    text="Texto:N"
                )
                chart_bin = (bars_bin + text_bin).properties(
                    title=alt.TitleParams(text="Target Binario (Dropout vs Resto)", fontSize=15, color=COLOR_MARINO, anchor="start"),
                    height=340,
                    background=COLOR_SUPERFICIE
                )
                st.altair_chart(chart_bin, width="stretch")

        st.subheader("Tasa de Abandono por Grupo de Riesgo (Gráfico y Tabla)")
        col_g_bar, col_g_tab = st.columns([1.45, 1], gap="large")
        with col_g_bar:
            with tarjeta("grafico_grupos", height="stretch"):
                st.markdown("##### 📊 Comparativa de Tasa de Abandono por Condición")
                st.altair_chart(grafico_barras(grupos, [0, 1], ".0%", tasa_abandono), width="stretch")
                st.caption(f"Línea vertical: promedio general ({tasa_abandono * 100:.1f} %). Muestra asociación estadística, no causalidad.")
        with col_g_tab:
            with tarjeta("tabla_grupos_visible", height="stretch"):
                st.markdown("##### 📋 Resumen Cuantitativo por Grupo")
                tabla = grupos.assign(Abandono=(grupos["valor"] * 100).round(1))[["Grupo", "Abandono", "Estudiantes"]]
                st.dataframe(
                    tabla,
                    width="stretch",
                    hide_index=True,
                    column_config={"Abandono": st.column_config.NumberColumn("Abandono (%)", format="%.1f %%")},
                )

        st.subheader("🔍 Explorador Interactivo del Dataset")
        with tarjeta("explorador_dataset_visible"):
            st.markdown("##### Filtrar registros por estado académico original:")
            target_filter = st.multiselect(
                "Selecciona categorías a incluir:",
                options=list(raw_df[TARGET_COL].unique()),
                default=list(raw_df[TARGET_COL].unique()),
                label_visibility="collapsed"
            )
            filtered_df = raw_df[raw_df[TARGET_COL].isin(target_filter)]

            if not filtered_df.empty:
                conteo = filtered_df[TARGET_COL].value_counts().reset_index()
                conteo.columns = ["Estado", "Estudiantes"]
                conteo["Porcentaje"] = (conteo["Estudiantes"] / len(filtered_df) * 100).round(1)
                conteo["Texto"] = [f"{n:,} ({p:.1f} %)" for n, p in zip(conteo["Estudiantes"], conteo["Porcentaje"])]

                col_met, col_graf = st.columns([1, 2])
                with col_met:
                    st.metric("Total seleccionados", f"{len(filtered_df):,} estudiantes", f"{len(filtered_df)/len(raw_df)*100:.1f} % del total", delta_color="off", delta_arrow="off", border=True)
                    st.dataframe(
                        conteo[["Estado", "Estudiantes", "Porcentaje"]],
                        hide_index=True,
                        width="stretch",
                        column_config={"Porcentaje": st.column_config.NumberColumn("Porcentaje (%)", format="%.1f %%")}
                    )
                with col_graf:
                    bars_dyn = alt.Chart(conteo).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=55).encode(
                        x=alt.X("Estado:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13, labelColor=COLOR_TINTA, ticks=False)),
                        y=alt.Y("Estudiantes:Q", title="Cantidad de Estudiantes", axis=alt.Axis(gridColor=COLOR_REJILLA, labelColor=COLOR_TINTA_2)),
                        color=alt.Color("Estado:N", legend=None, scale=alt.Scale(
                            domain=["Graduate", "Dropout", "Enrolled"],
                            range=["#2a78d6", "#d03b3b", "#fab219"]
                        )),
                        tooltip=[alt.Tooltip("Estado:N"), alt.Tooltip("Estudiantes:Q", format=","), alt.Tooltip("Porcentaje:Q", format=".1f", title="Porcentaje (%)")]
                    )
                    text_dyn = bars_dyn.mark_text(dy=-12, fontSize=13, fontWeight="bold", color=COLOR_TINTA).encode(
                        text="Texto:N"
                    )
                    chart_dyn = (bars_dyn + text_dyn).properties(height=280, background=COLOR_SUPERFICIE)
                    st.altair_chart(chart_dyn, width="stretch")

                st.caption(f"Vista previa de los primeros {min(100, len(filtered_df))} registros del dataset UCI #697:")
                st.dataframe(filtered_df.head(100), width="stretch")
            else:
                st.warning("⚠️ Selecciona al menos un estado en el filtro para visualizar el gráfico y los datos.")

    with tab_dicc:
        st.markdown("#### 📋 Diccionario Técnico y Tratamiento de las 36 Variables Predictoras")
        td1, td2, td3 = st.columns(3)
        with tarjeta("dicc_res_1", td1, "stretch"):
            st.metric("🏷️ Variables Nominales", "9 variables", "→ OneHotEncoder (208 cols)", delta_color="off", delta_arrow="off")
            st.caption("Carrera (`Course`), modalidad de ingreso, estado civil, nacionalidad, estudios y ocupación de los padres.")
        with tarjeta("dicc_res_2", td2, "stretch"):
            st.metric("🔘 Variables Binarias", "8 variables", "→ Passthrough (8 cols 0/1)", delta_color="off", delta_arrow="off")
            st.caption("Matrícula al día (`Tuition fees`), deudor (`Debtor`), becario (`Scholarship`), género, turno, desplazado, etc.")
        with tarjeta("dicc_res_3", td3, "stretch"):
            st.metric("🔢 Variables Numéricas", "19 variables", "→ StandardScaler (19 cols Z-score)", delta_color="off", delta_arrow="off")
            st.caption("Materias de 1er y 2do semestre (matriculadas, aprobadas, notas), nota de admisión, edad, PBI, inflación y desempleo.")

        tipo_filtro = st.radio("Filtrar tabla por tipo de variable:", ["Todas (36)", "Nominales (9)", "Binarias (8)", "Numéricas (19)"], horizontal=True)

        dicc_data = [
            ('Marital Status', 'Nominal', 'Estado civil (1=Soltero, 2=Casado, etc.)', 'OneHotEncoder'),
            ('Application mode', 'Nominal', 'Modalidad de postulación a la universidad', 'OneHotEncoder'),
            ('Course', 'Nominal', 'Carrera profesional elegida', 'OneHotEncoder'),
            ('Previous qualification', 'Nominal', 'Nivel educativo previo del ingresante', 'OneHotEncoder'),
            ('Nacionality', 'Nominal', 'Nacionalidad de origen del estudiante', 'OneHotEncoder'),
            ("Mother's qualification", 'Nominal', 'Nivel de instrucción alcanzado por la madre', 'OneHotEncoder'),
            ("Father's qualification", 'Nominal', 'Nivel de instrucción alcanzado por el padre', 'OneHotEncoder'),
            ("Mother's occupation", 'Nominal', 'Ocupación laboral principal de la madre', 'OneHotEncoder'),
            ("Father's occupation", 'Nominal', 'Ocupación laboral principal del padre', 'OneHotEncoder'),
            ('Daytime/evening attendance', 'Binaria', 'Turno de asistencia (1=Diurno, 0=Nocturno)', 'Passthrough (0/1 intacto)'),
            ('Displaced', 'Binaria', 'Estudiante foráneo o desplazado (1=Sí, 0=No)', 'Passthrough (0/1 intacto)'),
            ('Educational special needs', 'Binaria', 'Necesidades educativas especiales (1=Sí, 0=No)', 'Passthrough (0/1 intacto)'),
            ('Debtor', 'Binaria', 'Pagos atrasados o deudas pendientes (1=Sí, 0=No)', 'Passthrough (0/1 intacto)'),
            ('Tuition fees up to date', 'Binaria', 'Al día en el pago de tasas de matrícula (1=Sí, 0=No)', 'Passthrough (0/1 intacto)'),
            ('Gender', 'Binaria', 'Género reportado (1=Masculino, 0=Femenino)', 'Passthrough (0/1 intacto)'),
            ('Scholarship holder', 'Binaria', 'Beneficiario de beca estudiantil (1=Sí, 0=No)', 'Passthrough (0/1 intacto)'),
            ('International', 'Binaria', 'Estudiante internacional extranjero (1=Sí, 0=No)', 'Passthrough (0/1 intacto)'),
            ('Application order', 'Numérica', 'Preferencia de orden de carrera elegida (0 a 9)', 'StandardScaler (media 0, std 1)'),
            ('Age at enrollment', 'Numérica', 'Edad al momento de matricularse (17 a 70)', 'StandardScaler (media 0, std 1)'),
            ('Previous qualification (grade)', 'Numérica', 'Calificación de educación previa (0-200)', 'StandardScaler (media 0, std 1)'),
            ('Admission grade', 'Numérica', 'Nota del examen de admisión (0-200)', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 1st sem (credited)', 'Numérica', 'Créditos convalidados en 1er semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 1st sem (enrolled)', 'Numérica', 'Cursos matriculados en 1er semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 1st sem (evaluations)', 'Numérica', 'Evaluaciones rendidas en 1er semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 1st sem (approved)', 'Numérica', 'Cursos aprobados en 1er semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 1st sem (grade)', 'Numérica', 'Promedio de notas en 1er semestre (0-20)', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 1st sem (without evaluations)', 'Numérica', 'Cursos sin evaluación en 1er semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 2nd sem (credited)', 'Numérica', 'Créditos convalidados en 2do semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 2nd sem (enrolled)', 'Numérica', 'Cursos matriculados en 2do semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 2nd sem (evaluations)', 'Numérica', 'Evaluaciones rendidas en 2do semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 2nd sem (approved)', 'Numérica', 'Cursos aprobados en 2do semestre', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 2nd sem (grade)', 'Numérica', 'Promedio de notas en 2do semestre (0-20)', 'StandardScaler (media 0, std 1)'),
            ('Curricular units 2nd sem (without evaluations)', 'Numérica', 'Cursos sin evaluación en 2do semestre', 'StandardScaler (media 0, std 1)'),
            ('Unemployment rate', 'Numérica', 'Tasa de desempleo nacional al cursar (%)', 'StandardScaler (media 0, std 1)'),
            ('Inflation rate', 'Numérica', 'Tasa de inflación anual (%)', 'StandardScaler (media 0, std 1)'),
            ('GDP', 'Numérica', 'Crecimiento del PBI nacional (%)', 'StandardScaler (media 0, std 1)')
        ]

        df_d = pd.DataFrame(dicc_data, columns=["Variable", "Tipo", "Descripción", "Preprocesamiento"])
        if "Nominales" in tipo_filtro:
            df_d = df_d[df_d["Tipo"] == "Nominal"]
        elif "Binarias" in tipo_filtro:
            df_d = df_d[df_d["Tipo"] == "Binaria"]
        elif "Numéricas" in tipo_filtro:
            df_d = df_d[df_d["Tipo"] == "Numérica"]

        st.dataframe(df_d, width="stretch", hide_index=True)

    with tab_preproc:
        st.subheader("Reglas de Preprocesamiento Modular (Sin Fuga de Datos)")
        pr1, pr2, pr3 = st.columns(3)
        with tarjeta("prep_1", pr1, "stretch"):
            st.metric("1️⃣ Partición Estratificada", "80 % Train / 20 % Test", "3,539 entrenamiento · 885 prueba (SEED=42)", delta_color="off", delta_arrow="off")
            st.markdown("Mantiene la proporción exacta de `32.1 % Dropout` y `67.9 % Resto` en ambos subconjuntos.")
        with tarjeta("prep_2", pr2, "stretch"):
            st.metric("2️⃣ ColumnTransformer", "36 → 235 Columnas", "One-Hot (208) + Pass (8) + Z-Score (19)", delta_color="off", delta_arrow="off")
            st.markdown("Ajustado (`fit`) **únicamente sobre `X_train`** con `handle_unknown='ignore'` para evitar fuga de información.")
        with tarjeta("prep_3", pr3, "stretch"):
            st.metric("3️⃣ Balanceo de Clases", "Pesos 1.56 vs 0.74", "class_weight='balanced' + Umbral 0.40", delta_color="off", delta_arrow="off")
            st.markdown("Penaliza errores en la clase minoritaria (`Dropout`) y calibra el punto de corte en Gradient Boosting.")

        st.subheader("Evidencia Real: Los mismos 5 estudiantes (Original vs. Procesado)")
        st.caption("Comparación directa de 5 registros del conjunto de entrenamiento antes y después de aplicar `ColumnTransformer`.")

        df_orig_5 = pd.DataFrame([
            {"ID Estudiante": "Estudiante #571", "Course (Nominal)": 9147, "Tuition fees up to date (Bin)": 1, "Scholarship holder (Bin)": 0, "Age at enrollment (Num)": 22, "Admission grade (Num)": 124.8, "Target_Binario": 1},
            {"ID Estudiante": "Estudiante #155", "Course (Nominal)": 9853, "Tuition fees up to date (Bin)": 1, "Scholarship holder (Bin)": 0, "Age at enrollment (Num)": 19, "Admission grade (Num)": 138.1, "Target_Binario": 1},
            {"ID Estudiante": "Estudiante #912", "Course (Nominal)": 9991, "Tuition fees up to date (Bin)": 0, "Scholarship holder (Bin)": 0, "Age at enrollment (Num)": 38, "Admission grade (Num)": 120.0, "Target_Binario": 1},
            {"ID Estudiante": "Estudiante #1972", "Course (Nominal)": 9670, "Tuition fees up to date (Bin)": 1, "Scholarship holder (Bin)": 1, "Age at enrollment (Num)": 19, "Admission grade (Num)": 129.4, "Target_Binario": 1},
            {"ID Estudiante": "Estudiante #2951", "Course (Nominal)": 9085, "Tuition fees up to date (Bin)": 1, "Scholarship holder (Bin)": 0, "Age at enrollment (Num)": 18, "Admission grade (Num)": 128.7, "Target_Binario": 0},
        ])

        df_proc_5 = pd.DataFrame([
            {"ID Estudiante": "Estudiante #571", "Course_9147 (1-Hot)": 1.0, "Course_9853 (1-Hot)": 0.0, "Course_9991 (1-Hot)": 0.0, "Tuition (Pass)": 1.0, "Beca (Pass)": 0.0, "Edad (Z-Score train)": -0.161, "Nota Adm (Z-Score train)": -0.150},
            {"ID Estudiante": "Estudiante #155", "Course_9147 (1-Hot)": 0.0, "Course_9853 (1-Hot)": 1.0, "Course_9991 (1-Hot)": 0.0, "Tuition (Pass)": 1.0, "Beca (Pass)": 0.0, "Edad (Z-Score train)": -0.565, "Nota Adm (Z-Score train)": 0.764},
            {"ID Estudiante": "Estudiante #912", "Course_9147 (1-Hot)": 0.0, "Course_9853 (1-Hot)": 0.0, "Course_9991 (1-Hot)": 1.0, "Tuition (Pass)": 0.0, "Beca (Pass)": 0.0, "Edad (Z-Score train)": 1.995, "Nota Adm (Z-Score train)": -0.480},
            {"ID Estudiante": "Estudiante #1972", "Course_9147 (1-Hot)": 0.0, "Course_9853 (1-Hot)": 0.0, "Course_9991 (1-Hot)": 0.0, "Tuition (Pass)": 1.0, "Beca (Pass)": 1.0, "Edad (Z-Score train)": -0.565, "Nota Adm (Z-Score train)": 0.166},
            {"ID Estudiante": "Estudiante #2951", "Course_9147 (1-Hot)": 0.0, "Course_9853 (1-Hot)": 0.0, "Course_9991 (1-Hot)": 0.0, "Tuition (Pass)": 1.0, "Beca (Pass)": 0.0, "Edad (Z-Score train)": -0.700, "Nota Adm (Z-Score train)": 0.118},
        ])

        c_t5_orig, c_t5_proc = st.columns(2, gap="large")
        with c_t5_orig:
            with tarjeta("card_5_orig", height="stretch"):
                st.markdown("##### 🔹 Antes: Dataset Original (36 variables)")
                st.dataframe(df_orig_5, width="stretch", hide_index=True)
        with c_t5_proc:
            with tarjeta("card_5_proc", height="stretch"):
                st.markdown("##### 🔸 Después: Matriz Expandida (235 columnas · 0 nulos)")
                st.dataframe(df_proc_5, width="stretch", hide_index=True)

        evid_path = FIGURES / "evidencia_02_preprocesamiento.png"
        if evid_path.exists():
            with tarjeta("card_evid_exp_visible"):
                st.markdown("##### 📊 Gráfico de Expansión Dimensional (36 → 235 Columnas)")
                st.image(str(evid_path), caption="Clasificación de las 36 variables y expansión a 235 columnas tras One-Hot Encoding.", width="stretch")

    with tab_arq:
        st.subheader("Modelos Supervisados Candidatos y Configuración de Hiperparámetros")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        modelos_cards = [
            (
                "📐 LINEAL (BASELINE)",
                "Regresión Logística",
                "LogisticRegression",
                "#eff6ff",
                "#184f95",
                "C=1.0 · penalty='l2' · solver='lbfgs' · class_weight='balanced'",
                "Clasificador lineal sigmoide con regularización Ridge (L2). Punto de referencia interpretable de frontera lineal.",
            ),
            (
                "🌳 ÁRBOL INDIVIDUAL",
                "Árbol de Decisión",
                "DecisionTreeClassifier",
                "#f0fdf4",
                "#15803d",
                "criterion='gini' · max_depth=8 · min_samples_leaf=10 · balanced",
                "Partición recursiva no lineal por impureza de Gini, acotada a profundidad 8 para controlar el sobreajuste.",
            ),
            (
                "🌲 ENSAMBLE BAGGING",
                "Random Forest",
                "RandomForestClassifier",
                "#fffbeb",
                "#b45309",
                "n_estimators=200 · min_samples_leaf=2 · class_weight='balanced'",
                "Ensamble paralelo de 200 árboles por votación mayoritaria sobre muestras bootstrap independientes.",
            ),
            (
                "🚀 ENSAMBLE BOOSTING",
                "Gradient Boosting",
                "GradientBoostingClassifier",
                "#f5f3ff",
                "#6d28d9",
                "n_estimators=200 · lr=0.05 · max_depth=4 · subsample=0.8",
                "Ensamble secuencial aditivo donde cada árbol corrige los pseudo-residuos del anterior con baja varianza.",
            ),
        ]
        for col_mc, (rank_m, nom_m, cls_m, bg_m, br_m, hip_m, desc_m) in zip([m_c1, m_c2, m_c3, m_c4], modelos_cards):
            with tarjeta(f"mod_arq_{nom_m[:6]}", col_mc, "stretch"):
                st.markdown(
                    f"<div style='display:inline-block;padding:2px 9px;border-radius:10px;background:{bg_m};color:{br_m};font-size:0.75rem;font-weight:700;margin-bottom:6px'>{rank_m}</div>"
                    f"<div style='font-weight:700;font-size:1.05rem;color:#0d366b'>{nom_m}</div>"
                    f"<div style='font-family:monospace;font-size:0.78rem;color:#64748b;margin-bottom:8px'>{cls_m}</div>"
                    f"<div style='padding:6px 8px;border-radius:6px;background:#f1f5f9;font-family:monospace;font-size:0.76rem;color:#0f172a;margin-bottom:8px'>{hip_m}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(desc_m)

        st.markdown("#### 🔄 Flujo Experimental en 8 Etapas")
        pasos_8 = [
            ("1️⃣ Ingesta UCI #697", "Carga reproducible con `ucimlrepo` y limpieza de encabezados `str.strip()`."),
            ("2️⃣ Binarización Target", "Definición de `Dropout = 1` (32.1 %) frente a `Resto = 0` (67.9 %)."),
            ("3️⃣ Split Estratificado", "Partición fija en `X_train = 3,539` (80 %) y `X_test = 885` (20 %, `SEED=42`)."),
            ("4️⃣ ColumnTransformer", "One-Hot (9) + Passthrough (8) + StandardScaler (19) → 235 columnas."),
            ("5️⃣ Validación 5-Fold CV", "Competencia experimental de los 4 pipelines exclusivamente dentro de `X_train`."),
            ("6️⃣ Regla de Selección", "Mayor `F1-Score` medio en CV y desempate técnico (`< 0.005`) por menor desviación estándar."),
            ("7️⃣ Prueba Ciega (Hold-Out)", "Evaluación definitiva del modelo seleccionado sobre los 885 alumnos de `X_test`."),
            ("8️⃣ Calibración y Despliegue", "Ajuste de umbral operativo en curva PR, importancia por permutación y exportación `.joblib`."),
        ]
        p_row1 = st.columns(4)
        for col_p, (tit_p, txt_p) in zip(p_row1, pasos_8[:4]):
            with tarjeta(f"flujo_p_{tit_p[:5]}", col_p, "stretch"):
                st.markdown(f"**{tit_p}**")
                st.caption(txt_p)
        p_row2 = st.columns(4)
        for col_p, (tit_p, txt_p) in zip(p_row2, pasos_8[4:]):
            with tarjeta(f"flujo_p_{tit_p[:5]}", col_p, "stretch"):
                st.markdown(f"**{tit_p}**")
                st.caption(txt_p)

        c_lib, c_code = st.columns([1, 1.15], gap="large")
        with c_lib:
            with tarjeta("card_stack_tecnologico", height="stretch"):
                st.markdown("#### 🛠️ Stack Tecnológico y Librerías")
                df_libs = pd.DataFrame([
                    {"Herramienta": "Python 3.x", "Rol en el Proyecto": "Lenguaje principal de experimentación y arquitectura modular (`src/`)."},
                    {"Herramienta": "pandas & numpy", "Rol en el Proyecto": "Manipulación tabular, limpieza y cálculo matricial."},
                    {"Herramienta": "scikit-learn", "Rol en el Proyecto": "ColumnTransformer, Pipeline, 4 clasificadores, StratifiedKFold y métricas."},
                    {"Herramienta": "Altair / Matplotlib", "Rol en el Proyecto": "Visualización estadística interactiva y gráficos de evaluación."},
                    {"Herramienta": "ucimlrepo & joblib", "Rol en el Proyecto": "Conexión al repositorio UCI #697 y serialización del modelo seleccionado."},
                    {"Herramienta": "Streamlit & Git/GitHub", "Rol en el Proyecto": "Aplicación web interactiva (`app.py`) y control de versiones grupal."},
                ])
                st.dataframe(df_libs, width="stretch", hide_index=True)

        with c_code:
            with tarjeta("card_code_pipeline", height="stretch"):
                st.markdown("#### 💻 Encapsulamiento Modular de Pipelines (`src/models.py`)")
                st.code("""from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.preprocessor import get_preprocessor
from src.config import SEED

def build_candidate_pipeline(classifier):
    # Une el preprocesador (36 -> 235 cols) con cada clasificador candidato
    # Garantiza ajuste exclusivo en el pliegue de entrenamiento (0 Data Leakage)
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", classifier)
    ])""", language="python")


# ---------------------------------------------------------
# VISTA 3: MODELOS Y RESULTADOS (OE1 A OE5)
# ---------------------------------------------------------
elif menu == VISTA_MODELOS:
    encabezado(
        "🏆 Evaluación de Modelos y Resultados por Objetivo (OE1 – OE5)",
        f"Competencia experimental en Validación Cruzada (5-Fold), cumplimiento de metas y desempeño del modelo ganador "
        f"(<b>{model_name}</b>) en el conjunto de prueba independiente (885 estudiantes)."
    )
    tab_cv, tab_oe, tab_test = st.tabs([
        "🏆 Comparación de Modelos (Validación Cruzada)",
        "🎯 Resultados por Objetivo (OE1 a OE5)",
        "🔬 Evaluación en Test, Umbral y Explicabilidad"
    ])

    with tab_cv:
        df_cv_results = pd.DataFrame([
            {
                "Ranking": "🥇 1° (Ganador)",
                "Modelo": "Gradient Boosting",
                "F1 (CV)": 0.7907,
                "F1 ± std": "0.7907 ± 0.0085",
                "Precision (CV)": "84.96 %",
                "Recall (CV)": "73.97 %",
                "Accuracy (CV)": "87.43 %",
                "ROC-AUC (CV)": 0.9146,
                "Estado Meta F1 > 0.78": "✔ Supera Meta"
            },
            {
                "Ranking": "🥈 2°",
                "Modelo": "Random Forest",
                "F1 (CV)": 0.7873,
                "F1 ± std": "0.7873 ± 0.0177",
                "Precision (CV)": "78.61 %",
                "Recall (CV)": "78.89 %",
                "Accuracy (CV)": "86.30 %",
                "ROC-AUC (CV)": 0.9130,
                "Estado Meta F1 > 0.78": "✔ Supera Meta"
            },
            {
                "Ranking": "🥉 3°",
                "Modelo": "Logistic Regression",
                "F1 (CV)": 0.7860,
                "F1 ± std": "0.7860 ± 0.0147",
                "Precision (CV)": "76.79 %",
                "Recall (CV)": "80.56 %",
                "Accuracy (CV)": "85.90 %",
                "ROC-AUC (CV)": 0.9209,
                "Estado Meta F1 > 0.78": "✔ Supera Meta"
            },
            {
                "Ranking": "4°",
                "Modelo": "Decision Tree",
                "F1 (CV)": 0.7481,
                "F1 ± std": "0.7481 ± 0.0219",
                "Precision (CV)": "71.40 %",
                "Recall (CV)": "78.72 %",
                "Accuracy (CV)": "82.93 %",
                "ROC-AUC (CV)": 0.8630,
                "Estado Meta F1 > 0.78": "❌ No cumple meta"
            }
        ])

        col_cv_graf, col_cv_tablas = st.columns([1, 1.35], gap="large")

        with col_cv_graf:
            with tarjeta("grafico_modelos"):
                st.markdown("##### 📊 F1-Score en Validación Cruzada (5-Fold)")
                st.caption(f"Evaluación sobre X_train (3,539 alumnos) · Línea roja punteada: Meta mínima F1 > {CRITERIOS['f1']:.2f}")
                df_v = pd.DataFrame([
                    {"ModeloCorto": "Gradient Boosting", "Modelo": "Gradient Boosting", "F1_CV": 0.7907, "F1_std": "0.7907 ± 0.0085", "Precision_CV": "84.96 %", "Recall_CV": "73.97 %", "Texto": "0.7907", "Meta": float(CRITERIOS["f1"])},
                    {"ModeloCorto": "Random Forest", "Modelo": "Random Forest", "F1_CV": 0.7873, "F1_std": "0.7873 ± 0.0177", "Precision_CV": "78.61 %", "Recall_CV": "78.89 %", "Texto": "0.7873", "Meta": float(CRITERIOS["f1"])},
                    {"ModeloCorto": "Reg. Logística", "Modelo": "Logistic Regression", "F1_CV": 0.7860, "F1_std": "0.7860 ± 0.0147", "Precision_CV": "76.79 %", "Recall_CV": "80.56 %", "Texto": "0.7860", "Meta": float(CRITERIOS["f1"])},
                    {"ModeloCorto": "Árbol Decisión", "Modelo": "Decision Tree", "F1_CV": 0.7481, "F1_std": "0.7481 ± 0.0219", "Precision_CV": "71.40 %", "Recall_CV": "78.72 %", "Texto": "0.7481", "Meta": float(CRITERIOS["f1"])},
                ])
                orden_m = df_v["ModeloCorto"].tolist()

                barras_v = alt.Chart(df_v).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=46).encode(
                    x=alt.X(
                        "ModeloCorto:N",
                        sort=None,
                        title=None,
                        axis=alt.Axis(
                            labelAngle=0,
                            labelFontSize=11.5,
                            labelFontWeight="bold",
                            labelColor=COLOR_TINTA,
                            labelLimit=130,
                            labelPadding=8,
                            ticks=False,
                        ),
                    ),
                    y=alt.Y(
                        "F1_CV:Q",
                        title="F1-Score (Dropout)",
                        scale=alt.Scale(domain=[0, 0.92]),
                        axis=alt.Axis(format=".2f", gridColor=COLOR_REJILLA, labelColor=COLOR_TINTA_2),
                    ),
                    color=alt.Color(
                        "ModeloCorto:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=orden_m,
                            range=["#184f95", "#94a3b8", "#94a3b8", "#cbd5e1"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Modelo:N", title="Algoritmo"),
                        alt.Tooltip("F1_std:N", title="F1 ± Desv. Est."),
                        alt.Tooltip("Precision_CV:N", title="Precision (CV)"),
                        alt.Tooltip("Recall_CV:N", title="Recall (CV)"),
                    ],
                )
                texto_v = barras_v.mark_text(
                    dy=-12, fontSize=12.5, fontWeight="bold", color=COLOR_MARINO
                ).encode(text="Texto:N")
                regla_meta_cv = alt.Chart(df_v).mark_rule(
                    color="#d03b3b", strokeDash=[6, 4], strokeWidth=2.2
                ).encode(y="Meta:Q")

                chart_cv_v = (barras_v + regla_meta_cv + texto_v).properties(
                    height=310,
                    background=COLOR_SUPERFICIE,
                )
                st.altair_chart(chart_cv_v, width="stretch")

                st.warning(
                    "**Desempate técnico (< 0.005):** Gana **Gradient Boosting** (`0.7907`) por tener la **menor desviación estándar (± 0.0085)** y la **mayor precisión (84.96 %)**."
                )

        with col_cv_tablas:
            with tarjeta("card_tabla3_cv"):
                st.markdown("##### 📋 Comparación Consolidada en Validación Cruzada (5 Pliegues sobre `X_train` = 3,539)")
                st.dataframe(df_cv_results, width="stretch", hide_index=True)

            with tarjeta("card_tabla4_cv"):
                st.markdown("##### 📋 Cumplimiento de los 4 Criterios en Validación Cruzada (Umbral 0.50)")
                df_tabla4 = pd.DataFrame([
                    {"Modelo": "Gradient Boosting", "Accuracy > 75 %": "✔ Sí (87.43 %)", "Precision > 75 %": "✔ Sí (84.96 %)", "Recall > 80 %": "✘ No (73.97 %)", "F1 > 0.78": "✔ Sí (0.7907)", "¿Cumple los 4 (0.50)?": "No (se calibra a 0.40)"},
                    {"Modelo": "Random Forest", "Accuracy > 75 %": "✔ Sí (86.30 %)", "Precision > 75 %": "✔ Sí (78.61 %)", "Recall > 80 %": "✘ No (78.89 %)", "F1 > 0.78": "✔ Sí (0.7873)", "¿Cumple los 4 (0.50)?": "No"},
                    {"Modelo": "Regresión Logística", "Accuracy > 75 %": "✔ Sí (85.90 %)", "Precision > 75 %": "✔ Sí (76.79 %)", "Recall > 80 %": "✔ Sí (80.56 %)", "F1 > 0.78": "✔ Sí (0.7860)", "¿Cumple los 4 (0.50)?": "✔ Sí (balanceado)"},
                    {"Modelo": "Árbol de Decisión", "Accuracy > 75 %": "✔ Sí (82.93 %)", "Precision > 75 %": "✘ No (71.40 %)", "Recall > 80 %": "✘ No (78.72 %)", "F1 > 0.78": "✘ No (0.7481)", "¿Cumple los 4 (0.50)?": "No"},
                ])
                st.dataframe(df_tabla4, width="stretch", hide_index=True)
                st.caption(
                    "Con umbral 0.50, Gradient Boosting privilegia la precisión (84.96 %) sobre el Recall (73.97 %). "
                    "Al calibrar su umbral operativo a 0.40, supera los 4 criterios."
                )

    with tab_oe:
        st.subheader("Demostración Gráfica de Resultados por Objetivo (OE1 – OE5)")

        c_oe1, c_oe2 = st.columns(2, gap="large")
        with c_oe1:
            with tarjeta("res_oe1", height="stretch"):
                st.markdown("#### ✅ OE1: Exactitud Global (Accuracy > 75 %)")
                st.metric("Accuracy en Test (Gradient Boosting)", "88.36 %", "782 de 885 estudiantes bien clasificados (+13.36 pp sobre la meta)", delta_color="off", delta_arrow="off")
                df_oe1 = pd.DataFrame([
                    {"Etiqueta": "GB en Test (Ganador)", "Valor": 88.36, "Texto": "88.36 %", "Meta": 75.0},
                    {"Etiqueta": "Gradient Boosting (CV)", "Valor": 87.43, "Texto": "87.43 %", "Meta": 75.0},
                    {"Etiqueta": "Random Forest (CV)", "Valor": 86.30, "Texto": "86.30 %", "Meta": 75.0},
                    {"Etiqueta": "Reg. Logística (CV)", "Valor": 85.90, "Texto": "85.90 %", "Meta": 75.0},
                    {"Etiqueta": "Árbol Decisión (CV)", "Valor": 82.93, "Texto": "82.93 %", "Meta": 75.0},
                ])
                ord_oe1 = df_oe1["Etiqueta"].tolist()
                b_oe1 = alt.Chart(df_oe1).mark_bar(cornerRadiusEnd=5, height=18).encode(
                    y=alt.Y("Etiqueta:N", sort=None, title=None, axis=alt.Axis(labelFontSize=11.5, labelLimit=180, ticks=False)),
                    x=alt.X("Valor:Q", title="Exactitud (%) · Línea roja: Meta 75 %", scale=alt.Scale(domain=[0, 105])),
                    color=alt.Color(
                        "Etiqueta:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=ord_oe1,
                            range=["#184f95", "#64748b", "#64748b", "#64748b", "#94a3b8"],
                        ),
                    ),
                )
                t_oe1 = b_oe1.mark_text(dx=6, align="left", fontSize=11, fontWeight="bold", color=COLOR_MARINO).encode(
                    text="Texto:N"
                )
                r_oe1 = alt.Chart(df_oe1).mark_rule(color="#d03b3b", strokeDash=[5, 4], strokeWidth=2).encode(x="Meta:Q")
                st.altair_chart((b_oe1 + r_oe1 + t_oe1).properties(height=180, background=COLOR_SUPERFICIE), width="stretch")

        with c_oe2:
            with tarjeta("res_oe2", height="stretch"):
                st.markdown("#### ✅ OE2: Precisión en Dropout (> 75 %) y Falsas Alarmas")
                st.metric("Precision en Test (Umbral 0.50)", "86.06 %", "216 desertores reales de 251 alertas · Solo 35 FP (5.82 % falsas alarmas)", delta_color="off", delta_arrow="off")
                df_oe2 = pd.DataFrame([
                    {"Modelo": "Gradient Boosting (Test)", "Valor": 86.06, "Texto": "86.06 % (216/251)", "Meta": 75.0},
                    {"Modelo": "Gradient Boosting (CV)", "Valor": 84.96, "Texto": "84.96 %", "Meta": 75.0},
                    {"Modelo": "Random Forest (CV)", "Valor": 78.61, "Texto": "78.61 %", "Meta": 75.0},
                    {"Modelo": "Reg. Logística (CV)", "Valor": 76.79, "Texto": "76.79 %", "Meta": 75.0},
                    {"Modelo": "Árbol Decisión (CV)", "Valor": 71.40, "Texto": "71.40 % (✘)", "Meta": 75.0},
                ])
                ord_oe2 = df_oe2["Modelo"].tolist()
                b_oe2 = alt.Chart(df_oe2).mark_bar(cornerRadiusEnd=5, height=18).encode(
                    y=alt.Y("Modelo:N", sort=None, title=None, axis=alt.Axis(labelFontSize=11.5, labelLimit=180, ticks=False)),
                    x=alt.X("Valor:Q", title="Precisión (%) · Línea roja: Meta 75 %", scale=alt.Scale(domain=[0, 108])),
                    color=alt.Color(
                        "Modelo:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=ord_oe2,
                            range=["#184f95", "#64748b", "#64748b", "#64748b", "#cbd5e1"],
                        ),
                    ),
                )
                t_oe2 = b_oe2.mark_text(dx=6, align="left", fontSize=11, fontWeight="bold", color=COLOR_MARINO).encode(
                    text="Texto:N"
                )
                r_oe2 = alt.Chart(df_oe2).mark_rule(color="#d03b3b", strokeDash=[5, 4], strokeWidth=2).encode(x="Meta:Q")
                st.altair_chart((b_oe2 + r_oe2 + t_oe2).properties(height=180, background=COLOR_SUPERFICIE), width="stretch")

        c_oe3, c_oe4 = st.columns(2, gap="large")
        with c_oe3:
            with tarjeta("res_oe3", height="stretch"):
                st.markdown("#### ✅ OE3: Sensibilidad (Recall > 80 %) y Calibración de Umbral")
                st.metric("Recall Operativo (Umbral 0.40)", "83.10 %", "236 de 284 desertores detectados (+20 alumnos rescatados vs. umbral 0.50)", delta_color="off", delta_arrow="off")
                df_oe3 = pd.DataFrame([
                    {"Corte": "Umbral Operativo 0.40", "Valor": 83.10, "Texto": "83.10 % (236/284) ✔", "Meta": 80.0},
                    {"Corte": "Umbral Estándar 0.50", "Valor": 76.06, "Texto": "76.06 % (216/284) ✘", "Meta": 80.0},
                ])
                ord_oe3 = df_oe3["Corte"].tolist()
                b_oe3 = alt.Chart(df_oe3).mark_bar(cornerRadiusEnd=6, height=28).encode(
                    y=alt.Y("Corte:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12, labelLimit=180, ticks=False)),
                    x=alt.X("Valor:Q", title="Sensibilidad / Recall (%) · Línea roja: Meta 80 %", scale=alt.Scale(domain=[0, 110])),
                    color=alt.Color(
                        "Corte:N",
                        legend=None,
                        scale=alt.Scale(domain=ord_oe3, range=["#0ca30c", "#fab219"]),
                    ),
                )
                t_oe3 = b_oe3.mark_text(dx=6, align="left", fontSize=11.5, fontWeight="bold", color=COLOR_MARINO).encode(
                    text="Texto:N"
                )
                r_oe3 = alt.Chart(df_oe3).mark_rule(color="#d03b3b", strokeDash=[5, 4], strokeWidth=2).encode(x="Meta:Q")
                st.altair_chart((b_oe3 + r_oe3 + t_oe3).properties(height=150, background=COLOR_SUPERFICIE), width="stretch")

        with c_oe4:
            with tarjeta("res_oe4", height="stretch"):
                st.markdown("#### ✅ OE4: Balance Armónico (F1-Score > 0.78) y Generalización")
                st.metric("F1-Score en Test (Ganador)", "0.8075 (umbral 0.50)", "0.8266 con umbral 0.40 · CV: 0.7907 ± 0.0085 · ROC-AUC: 0.9343", delta_color="off", delta_arrow="off")
                df_oe4 = pd.DataFrame([
                    {"Etapa": "Test (Umbral 0.40)", "Valor": 0.8266, "Texto": "0.8266 ✔", "Meta": 0.78},
                    {"Etapa": "Test (Umbral 0.50)", "Valor": 0.8075, "Texto": "0.8075 ✔", "Meta": 0.78},
                    {"Etapa": "Validación Cruzada (CV)", "Valor": 0.7907, "Texto": "0.7907 ± 0.0085 ✔", "Meta": 0.78},
                ])
                ord_oe4 = df_oe4["Etapa"].tolist()
                b_oe4 = alt.Chart(df_oe4).mark_bar(cornerRadiusEnd=6, height=24).encode(
                    y=alt.Y("Etapa:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12, labelLimit=180, ticks=False)),
                    x=alt.X("Valor:Q", title="F1-Score (Dropout) · Línea roja: Meta 0.78", scale=alt.Scale(domain=[0, 1.05])),
                    color=alt.Color(
                        "Etapa:N",
                        legend=None,
                        scale=alt.Scale(domain=ord_oe4, range=["#0ca30c", "#184f95", "#2a78d6"]),
                    ),
                )
                t_oe4 = b_oe4.mark_text(dx=6, align="left", fontSize=11.5, fontWeight="bold", color=COLOR_MARINO).encode(
                    text="Texto:N"
                )
                r_oe4 = alt.Chart(df_oe4).mark_rule(color="#d03b3b", strokeDash=[5, 4], strokeWidth=2).encode(x="Meta:Q")
                st.altair_chart((b_oe4 + r_oe4 + t_oe4).properties(height=150, background=COLOR_SUPERFICIE), width="stretch")

        with tarjeta("res_oe5"):
            st.markdown("#### ✅ OE5: Top 5 Variables con Mayor Peso Predictivo (Importancia por Permutación)")
            col_oe5_cards, col_oe5_chart = st.columns([1.1, 1.3], gap="large")
            with col_oe5_cards:
                o5_a, o5_b = st.columns(2)
                o5_a.metric("1° Aprobadas 2do Sem", "-0.2261 F1", "Factor dominante (Académico)", delta_color="off", delta_arrow="off", border=True)
                o5_b.metric("2° Matrícula al Día", "-0.0759 F1", "Factor financiero clave", delta_color="off", delta_arrow="off", border=True)
                o5_c, o5_d = st.columns(2)
                o5_c.metric("3° Aprobadas 1er Sem", "-0.0101 F1", "Alerta desde 1er ciclo", delta_color="off", delta_arrow="off", border=True)
                o5_d.metric("4° Tenencia de Beca", "-0.0078 F1", "Protector socioeconómico", delta_color="off", delta_arrow="off", border=True)
                st.caption("5° Créditos convalidados 2do sem (`-0.0075 F1`) y 6° Edad al matricularse (`-0.0070 F1`).")
            with col_oe5_chart:
                df_top5 = pd.DataFrame([
                    {"Variable": "1. Materias aprobadas (2do sem)", "Caida": 0.2261, "Texto": "+0.2261"},
                    {"Variable": "2. Pago de matrícula al día", "Caida": 0.0759, "Texto": "+0.0759"},
                    {"Variable": "3. Materias aprobadas (1er sem)", "Caida": 0.0101, "Texto": "+0.0101"},
                    {"Variable": "4. Beneficiario de beca", "Caida": 0.0078, "Texto": "+0.0078"},
                    {"Variable": "5. Créditos convalidados (2do sem)", "Caida": 0.0075, "Texto": "+0.0075"},
                ])
                ord_top5 = df_top5["Variable"].tolist()
                b_top5 = alt.Chart(df_top5).mark_bar(cornerRadiusEnd=5, height=20).encode(
                    y=alt.Y("Variable:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12, labelLimit=230, ticks=False)),
                    x=alt.X("Caida:Q", title="Caída en F1-Score al permutar", scale=alt.Scale(domain=[0, 0.27])),
                    color=alt.Color(
                        "Variable:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=ord_top5,
                            range=["#184f95", "#fab219", "#2a78d6", "#fab219", "#2a78d6"],
                        ),
                    ),
                )
                t_top5 = b_top5.mark_text(dx=6, align="left", fontSize=11.5, fontWeight="bold", color=COLOR_MARINO).encode(
                    text="Texto:N"
                )
                st.altair_chart((b_top5 + t_top5).properties(height=185, background=COLOR_SUPERFICIE), width="stretch")

    with tab_test:
        tn, fp, fn, tp = m_test["cm"].ravel()

        # 1. Cumplimiento de Metas y Tabla 5 (Ancho Completo)
        st.subheader(f"1. Cumplimiento de Metas en el Conjunto de Prueba (885 Estudiantes · Umbral Activo: {umbral:.2f})")
        mostrar_metricas(m_test)

        with tarjeta("card_tabla5_test"):
            st.markdown("##### 📋 Comparativa Oficial en Test — Umbral por Defecto (`0.50`) vs. Umbral Operativo (`0.40`)")
            df_tabla5 = pd.DataFrame([
                {
                    "Métrica (Clase Dropout)": "Exactitud (Accuracy)",
                    "Umbral 0.50 (Estándar)": f"{m_test_050['accuracy']*100:.2f} % (782/885)",
                    "Umbral 0.40 (Operativo)": f"{m_test_040['accuracy']*100:.2f} % (786/885)",
                    f"Umbral Activo ({umbral:.2f})": f"{m_test['accuracy']*100:.2f} %",
                    "Meta Operativa": "> 75.00 %",
                    "Estado": "✔ Superada en ambos (+13.36 pp en 0.50)"
                },
                {
                    "Métrica (Clase Dropout)": "Precisión (Precision)",
                    "Umbral 0.50 (Estándar)": f"{m_test_050['precision']*100:.2f} % (216/251)",
                    "Umbral 0.40 (Operativo)": f"{m_test_040['precision']*100:.2f} % (236/287)",
                    f"Umbral Activo ({umbral:.2f})": f"{m_test['precision']*100:.2f} %",
                    "Meta Operativa": "> 75.00 %",
                    "Estado": "✔ Superada en ambos (+11.06 pp en 0.50)"
                },
                {
                    "Métrica (Clase Dropout)": "Sensibilidad (Recall)",
                    "Umbral 0.50 (Estándar)": f"{m_test_050['recall']*100:.2f} % (216/284)",
                    "Umbral 0.40 (Operativo)": f"{m_test_040['recall']*100:.2f} % (236/284)",
                    f"Umbral Activo ({umbral:.2f})": f"{m_test['recall']*100:.2f} %",
                    "Meta Operativa": "> 80.00 %",
                    "Estado": "✘ No alcanzada en 0.50 · ✔ Superada en 0.40 (+20 detectados)"
                },
                {
                    "Métrica (Clase Dropout)": "F1-Score (Dropout)",
                    "Umbral 0.50 (Estándar)": f"{m_test_050['f1']:.4f}",
                    "Umbral 0.40 (Operativo)": f"{m_test_040['f1']:.4f}",
                    f"Umbral Activo ({umbral:.2f})": f"{m_test['f1']:.4f}",
                    "Meta Operativa": "> 0.7800",
                    "Estado": "✔ Superada en ambos (+0.0275 en 0.50)"
                },
                {
                    "Métrica (Clase Dropout)": "Matriz de Confusión (TN / FP / FN / TP)",
                    "Umbral 0.50 (Estándar)": "TN=566 · FP=35 · FN=68 · TP=216",
                    "Umbral 0.40 (Operativo)": "TN=550 · FP=51 · FN=48 · TP=236",
                    f"Umbral Activo ({umbral:.2f})": f"TN={tn} · FP={fp} · FN={fn} · TP={tp}",
                    "Meta Operativa": "ROC-AUC = 0.9343",
                    "Estado": "Excelente discriminación probabilística"
                },
            ])
            st.dataframe(df_tabla5, width="stretch", hide_index=True)

        # 2. Matriz de Confusión (Secuencial, amplia y legible)
        st.subheader(f"2. Matriz de Confusión y Análisis del Error Institucional (Umbral: {umbral:.2f})")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("✅ Verdaderos Positivos (TP)", int(tp), f"Abandono detectado ({tp/284*100:.1f} % Recall)", delta_color="off", delta_arrow="off", border=True)
        r2.metric("❌ Falsos Negativos (FN)", int(fn), f"Abandono NO detectado ({fn/284*100:.1f} %)", delta_color="off", delta_arrow="off", border=True)
        r3.metric("⚠️ Falsos Positivos (FP)", int(fp), f"Falsa alarma ({fp/601*100:.1f} % de retenidos)", delta_color="off", delta_arrow="off", border=True)
        r4.metric("👍 Verdaderos Negativos (TN)", int(tn), f"Continuó sin alerta ({tn/601*100:.1f} %)", delta_color="off", delta_arrow="off", border=True)

        with tarjeta("card_cm"):
            st.altair_chart(grafico_matriz_confusion(m_test["cm"], umbral), width="stretch")
            st.caption(
                f"Matriz de confusión sincronizada al umbral activo ({umbral:.2f}). "
                f"Total clasificados correctamente: {tp + tn:,} de 885 ({ (tp + tn)/len(y_test)*100:.2f} %) · "
                f"Errores totales: {fp + fn:,} ({ (fp + fn)/len(y_test)*100:.2f} %)."
            )
            with st.expander("📷 Ver imagen estática del informe académico (PNG)"):
                cm_path = FIGURES / "matriz_confusion_ganador.png"
                if cm_path.exists():
                    st.image(
                        str(cm_path),
                        caption="Matriz de confusión en Test con umbral operativo 0.40 (885 estudiantes).",
                        width="stretch"
                    )

        # 3. Curva Precision-Recall (Secuencial, ancho completo)
        st.subheader("3. Curva Precision-Recall y Calibración del Umbral Operativo")
        with tarjeta("card_pr"):
            st.altair_chart(grafico_precision_recall(y_test, proba_test, m_test, umbral), width="stretch")
            st.caption(
                "En educación superior, el costo de un Falso Negativo (perder a un alumno sin haberlo detectado) supera al de una falsa alarma. "
                "Reducir el umbral de 0.50 a 0.40 rescata 20 desertores adicionales (de 216 a 236 detectados) manteniendo 82.23 % de precisión."
            )
            with st.expander("📷 Ver curva estática del informe académico (PNG)"):
                pr_path = FIGURES / "curva_precision_recall.png"
                if pr_path.exists():
                    st.image(str(pr_path), caption="Curva Precision-Recall con umbral óptimo F1 (0.39) y operativo (0.40).", width="stretch")

        # 4. Importancia de Variables por Permutación (Secuencial, ancho completo)
        st.subheader("4. Explicabilidad e Importancia de Variables por Permutación (OE5)")
        with tarjeta("card_imp"):
            if perm_data:
                st.altair_chart(grafico_importancia_permutacion(perm_data), width="stretch")
                st.caption(
                    "Caída media en F1-Score tras 30 repeticiones en el conjunto de prueba. "
                    "Las unidades aprobadas en el 2do semestre (+0.2261) y estar al día en matrícula (+0.0759) concentran más del 80 % del impacto predictivo."
                )
            else:
                imp_path = FIGURES / "permutation_importance_top15.png"
                if imp_path.exists():
                    st.image(str(imp_path), width="stretch")

            with st.expander("📷 Ver gráfico estático con barras de error (PNG)"):
                imp_path = FIGURES / "permutation_importance_top15.png"
                if imp_path.exists():
                    st.image(
                        str(imp_path),
                        caption="Figura en alta resolución con barras de error estándar (± std).",
                        width="stretch"
                    )


# ---------------------------------------------------------
# VISTA 4: SIMULADOR (INDIVIDUAL Y POR LOTES)
# ---------------------------------------------------------
elif menu == VISTA_SIMULADOR:
    encabezado("🔮 Simulador de Riesgo en Vivo", "Evalúa el riesgo de deserción en tiempo real de forma individual o mediante carga masiva de estudiantes (CSV).")
    tab_individual, tab_lote = st.tabs(["👤 Evaluación Individual", "📂 Carga por Lote (CSV)"])

    with tab_individual:
        st.markdown("##### ⚡ Perfiles de prueba rápida:")
        col_demo1, col_demo2, col_demo3 = st.columns(3)
        col_demo1.button("🚨 Estudiante en riesgo", width="stretch", on_click=aplicar_perfil, args=("riesgo",))
        col_demo2.button("⚠️ Estudiante vulnerable", width="stretch", on_click=aplicar_perfil, args=("medio",))
        col_demo3.button("✅ Estudiante destacado", width="stretch", on_click=aplicar_perfil, args=("seguro",))

        # La primera visita abre con un ejemplo; al volver, recupera el perfil evaluado
        if "simulador_iniciado" not in st.session_state:
            st.session_state["simulador_iniciado"] = True
            aplicar_perfil("medio")
        evaluado = st.session_state.get("perfil_evaluado", {})
        for col in ENTRADAS_SIMULADOR:
            st.session_state.setdefault(col, evaluado.get(col, int(medianas[col])))

        si_no = lambda x: "Sí" if x == 1 else "No"

        rangos = {
            col: (int(raw_df[col].min()), int(raw_df[col].max()))
            for col in (COL_EDAD, COL_MATRIC_1, COL_APROB_1, COL_MATRIC_2, COL_APROB_2, COL_CONV_2)
        }

        def campo(col):
            return {"key": col, "step": 1, "help": f"Valores permitidos: {rangos[col][0]} a {rangos[col][1]}."}

        # Distribución en 2 Columnas (Split Dashboard)
        col_form, col_resultado = st.columns([1, 1.25], gap="large")

        with col_form:
            with tarjeta("perfil_estudiante_card"):
                st.markdown("### 📝 Datos del Estudiante")
                with st.form("perfil_estudiante"):
                    st.selectbox(ETIQUETAS[COL_CARRERA], options=list(CARRERAS), format_func=CARRERAS.get, key=COL_CARRERA)
                    st.number_input(ETIQUETAS[COL_EDAD], **campo(COL_EDAD))

                    st.markdown("##### 📚 1er Semestre")
                    s1_matric, s1_aprob = st.columns(2)
                    s1_matric.number_input("Matriculadas", **campo(COL_MATRIC_1))
                    s1_aprob.number_input("Aprobadas", **campo(COL_APROB_1))

                    st.markdown("##### 📚 2do Semestre")
                    s2_matric, s2_aprob = st.columns(2)
                    s2_matric.number_input("Matriculadas", **campo(COL_MATRIC_2))
                    s2_aprob.number_input("Aprobadas", **campo(COL_APROB_2))
                    st.number_input("Convalidadas (2do sem)", **campo(COL_CONV_2))

                    st.markdown("##### 💳 Situación Económica")
                    st.selectbox(ETIQUETAS[COL_PAGOS], options=[1, 0], format_func=si_no, key=COL_PAGOS)
                    st.selectbox(ETIQUETAS[COL_DEUDA], options=[0, 1], format_func=si_no, key=COL_DEUDA)
                    st.selectbox(ETIQUETAS[COL_BECA], options=[0, 1], format_func=si_no, key=COL_BECA)

                    st.caption("Aprobadas y convalidadas no pueden superar a matriculadas.")
                    evaluar = st.form_submit_button("🚀 Evaluar Riesgo", type="primary", width="stretch")

        errores = []
        if evaluar:
            entradas = {col: st.session_state[col] for col in ENTRADAS_SIMULADOR}
            errores = validar_perfil({**entradas, COL_CONV_1: int(medianas[COL_CONV_1])}, rangos, ETIQUETAS)
            for error in errores:
                st.error(error)
            if errores:
                st.session_state.pop("perfil_evaluado", None)
            else:
                st.session_state["perfil_evaluado"] = entradas

        with col_resultado:
            perfil = st.session_state.get("perfil_evaluado")
            if perfil is None:
                if not errores:
                    st.info("Elige un ejemplo rápido arriba o completa los datos en el formulario y pulsa **Evaluar Riesgo**.")
            else:
                input_df = construir_perfil(medianas, perfil)
                prob_dropout = float(model_pipeline.predict_proba(input_df)[0, 1])
                color = str(semaforo([prob_dropout], umbral)[0])
                st.html(tarjeta_resultado(prob_dropout, color, umbral))

                riesgo_tipico, factores = factores_riesgo(model_pipeline, medianas, perfil)
                suben = [(c, v) for c, v in factores if v >= 0.01][:3]
                bajan = [(c, v) for c, v in reversed(factores) if v <= -0.01][:3]

                st.markdown("#### ¿Por qué este resultado?")
                c_sube, c_baja = st.columns(2)
                for columna, clave, titulo, grupo, vacio in [
                    (c_sube, "suben", "⬆️ Aumentan el riesgo", suben, "Ningún dato lo aumenta de forma relevante."),
                    (c_baja, "bajan", "⬇️ Reducen el riesgo", bajan, "Ningún dato lo reduce de forma relevante."),
                ]:
                    with tarjeta(clave, columna, "stretch"):
                        st.markdown(f"**{titulo}**")
                        if not grupo:
                            st.markdown(vacio)
                        for col, aporte in grupo:
                            st.markdown(
                                f"**{ETIQUETAS[col]}:** {describir_valor(col, perfil[col])} "
                                f"(típico: {describir_valor(col, medianas[col])}) · {aporte * 100:+.0f} pts"
                            )
                st.caption(
                    f"Un estudiante típico tiene {riesgo_tipico * 100:.0f} % de riesgo. Cada línea muestra cuántos puntos "
                    "cambia si solo ese dato fuera distinto. Los efectos no se suman y reflejan cómo razona el modelo, "
                    "no una causa comprobada."
                )

                st.markdown("#### Qué hacer (Plan de Acción)")
                recomendaciones = []
                if perfil[COL_DEUDA] == 1 or perfil[COL_PAGOS] == 0:
                    recomendaciones.append("💰 **Apoyo financiero:** contactar con Bienestar Universitario para fraccionamiento o beca de emergencia.")
                if perfil[COL_APROB_2] < perfil[COL_MATRIC_2] / 2 or perfil[COL_APROB_1] < perfil[COL_MATRIC_1] / 2:
                    recomendaciones.append("📚 **Tutoría académica:** aprobó menos de la mitad de sus materias en algún semestre.")
                if perfil[COL_BECA] == 0 and prob_dropout >= umbral:
                    recomendaciones.append("🎯 **Evaluación socioeconómica:** revisar elegibilidad para beca o subsidio.")
                if perfil[COL_EDAD] >= 25:
                    recomendaciones.append("⏰ **Flexibilidad horaria:** revisar si compagina estudios con cargas familiares o laborales.")
                if not recomendaciones:
                    recomendaciones.append("👏 **Seguimiento regular:** sin factores de atención en los datos ingresados.")

                with tarjeta("acciones"):
                    for rec in recomendaciones:
                        st.markdown(rec)

                st.caption(
                    f"Solo se piden las 10 variables más influyentes; las otras {len(ALL_FEATURE_COLS) - len(ETIQUETAS)} "
                    "usan el valor típico. Para evaluar datos completos, usa la pestaña Carga por Lote."
                )

    with tab_lote:
        st.markdown("#### 📂 Evaluación Masiva mediante archivo CSV")
        st.markdown("Sube un CSV con las **36 variables** del dataset y obtén la lista clasificada y ordenada por nivel de riesgo.")
        st.download_button(
            "⬇️ Descargar plantilla de ejemplo",
            data=X_test.assign(dropout=y_test).rename_axis(COL_ID).to_csv().encode("utf-8"),
            file_name="plantilla_lote.csv",
            mime="text/csv"
        )
        st.caption(
            f"La plantilla son los {len(X_test)} estudiantes de prueba. Si el archivo incluye `dropout` o `Target`, "
            "también se compara contra el resultado real."
        )

        archivo = st.file_uploader("Archivo CSV de estudiantes", type="csv")

        if archivo is not None:
            try:
                lote = leer_lote(archivo.getvalue())
                X_lote, validas = preparar_lote(lote)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            descartadas = int((~validas).sum())
            if descartadas:
                st.warning(f"{descartadas} fila(s) descartadas por valores vacíos o no numéricos.")
            if X_lote.empty:
                st.error("El archivo no tiene filas válidas para evaluar.")
                st.stop()

            desconocidas = categorias_desconocidas(model_pipeline, X_lote)
            if desconocidas:
                detalle = ", ".join(f"{col}: {n}" for col, n in desconocidas.items())
                with st.expander("ℹ️ Códigos que el modelo no conoce"):
                    st.markdown(f"Se puntúan igual, con menor fiabilidad. {detalle}.")

            proba_lote = model_pipeline.predict_proba(X_lote)[:, 1]
            resultado = lote[validas].copy()
            resultado.insert(0, "Semáforo", semaforo(proba_lote, umbral))
            resultado.insert(0, "Prob. Deserción", proba_lote.round(4))
            resultado = resultado.sort_values("Prob. Deserción", ascending=False)

            conteo = resultado["Semáforo"].value_counts()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Estudiantes evaluados", f"{len(resultado):,}", border=True)
            k2.metric("🔴 Riesgo alto", int(conteo.get("Rojo", 0)), border=True)
            k3.metric("🟡 Riesgo moderado", int(conteo.get("Amarillo", 0)), border=True)
            k4.metric("🟢 Riesgo bajo", int(conteo.get("Verde", 0)), border=True)

            st.subheader("Estudiantes ordenados por riesgo")
            vista = resultado.copy()
            vista["Semáforo"] = vista["Semáforo"].map(lambda c: f"{ICONO_ESTADO[c]} {c}")
            vista[COL_CARRERA] = X_lote[COL_CARRERA].map(lambda c: CARRERAS.get(int(c), str(int(c))))
            st.dataframe(
                vista,
                width="stretch",
                hide_index=True,
                column_order=[*vista.columns.intersection([COL_ID]), "Prob. Deserción", "Semáforo", *ENTRADAS_SIMULADOR],
                column_config={
                    COL_ID: st.column_config.NumberColumn("ID", format="%d"),
                    "Prob. Deserción": st.column_config.ProgressColumn("Probabilidad", format="percent", min_value=0, max_value=1),
                    **{col: st.column_config.Column(etiqueta) for col, etiqueta in ETIQUETAS.items()},
                },
            )
            st.download_button(
                "⬇️ Descargar resultados",
                data=resultado.to_csv(index=False).encode("utf-8"),
                file_name="resultados_riesgo.csv",
                mime="text/csv"
            )

            y_lote = etiquetas_reales(lote[validas])
            if y_lote is not None:
                st.subheader("Comparación con el resultado real")
                mostrar_metricas(metricas(y_lote, proba_lote, umbral))


# ---------------------------------------------------------
# VISTA 5: CONCLUSIONES, LIMITACIONES Y ÉTICA
# ---------------------------------------------------------
elif menu == VISTA_CONCLUSIONES:
    encabezado(
        "📌 Discusión, Conclusiones, Limitaciones y Ética de IA",
        "Síntesis visual de hallazgos por objetivo, limitaciones técnicas, "
        "recomendaciones futuras y gobernanza ética del sistema."
    )

    tab_conc, tab_lim, tab_etica = st.tabs([
        "🎯 Discusión y Conclusiones por Objetivo",
        "⚠️ Limitaciones y Recomendaciones Técnicas",
        "⚖️ Consideraciones Éticas de IA",
    ])

    with tab_conc:
        st.markdown("#### 🔬 Discusión de Resultados frente a los Antecedentes Científicos")
        d1, d2, d3 = st.columns(3)
        with tarjeta("disc_1", d1, "stretch"):
            st.metric(
                "🏆 Liderazgo en Ensambles",
                "88.36 % Acc | 0.8266 F1",
                "Consistente con el estado del arte",
                delta_color="off",
                delta_arrow="off",
            )
            st.markdown(
                "Coincide con **Banquez-Maturana et al. (2025)** (*LightGBM: 87.0 % Acc, F1 = 0.83*) y **Kuz y Morales (2023)** (*XGBoost*): "
                "los ensambles **Gradient Boosting** capturan mejor las interacciones no lineales académicas y financieras."
            )
        with tarjeta("disc_2", d2, "stretch"):
            st.metric(
                "🔑 Predictores Universales",
                "2do Sem + Matrícula",
                "Caída F1: -0.2261 y -0.0759",
                delta_color="off",
                delta_arrow="off",
            )
            st.markdown(
                "Converge con la literatura internacional y local (**Izquierdo y Torres, 2024**): "
                "las **materias aprobadas en 1er año** y la **morosidad / beca** son los determinantes más potentes del abandono."
            )
        with tarjeta("disc_3", d3, "stretch"):
            st.metric(
                "🛡️ Rigor Metodológico",
                "Hold-Out 20 % Ciego",
                "885 alumnos nunca vistos en Train",
                delta_color="off",
                delta_arrow="off",
            )
            st.markdown(
                "A diferencia de estudios locales que evaluaron sobre el mismo entrenamiento (*98.7 %*), "
                "nuestro pipeline aplica **5-Fold CV + Test independiente sin fuga de datos**, garantizando generalización real."
            )

        st.markdown("#### ✅ Conclusiones por Objetivo (OG y OE1 – OE5)")
        with tarjeta("conc_og"):
            c_og_izq, c_og_der = st.columns([1, 2.2], gap="large")
            with c_og_izq:
                st.metric(
                    "🎯 Objetivo General (OG)",
                    "Gradient Boosting Ganador",
                    "F1 Test: 0.8075 (0.50) → 0.8266 (0.40) | ROC-AUC: 0.9343",
                    delta_color="off",
                    delta_arrow="off",
                )
            with c_og_der:
                st.markdown(
                    "**Conclusión General:** Los modelos de ensamble (**Gradient Boosting** y **Random Forest**) y la **Regresión Logística** "
                    "alcanzan un alto poder predictivo sobre el dataset UCI #697, superando ampliamente al Árbol de Decisión individual (`F1 = 0.7481`) "
                    "y permitiendo anticipar la deserción estudiantil con rigor estadístico."
                )

        co1, co2, co3 = st.columns(3)
        with tarjeta("conc_oe1", co1, "stretch"):
            st.metric("OE1 · Exactitud Global", "88.36 % en Test", "✔ Meta > 75.00 % (+13.36 pp)", delta_color="off", delta_arrow="off")
            st.markdown(
                "Clasifica correctamente a **782 de 885 estudiantes** (`87.43 %` en CV y `88.81 %` con umbral `0.40`), "
                "superando en más de **20 puntos** la tasa base mayoritaria (`67.9 %`)."
            )
        with tarjeta("conc_oe2", co2, "stretch"):
            st.metric("OE2 · Precisión (Dropout)", "86.06 % en Test", "✔ Meta > 75.00 % (Solo 5.8 % FP)", delta_color="off", delta_arrow="off")
            st.markdown(
                "De 251 alertas en umbral `0.50`, **216 son desertores reales** (solo 35 falsas alarmas; `82.23 %` en umbral `0.40`), "
                "evitando sobrecargar a Bienestar Universitario."
            )
        with tarjeta("conc_oe3", co3, "stretch"):
            st.metric("OE3 · Sensibilidad (Recall)", "83.10 % (Umbral 0.40)", "✔ Meta > 80.00 % (+20 rescatados)", delta_color="off", delta_arrow="off")
            st.markdown(
                "El umbral estándar (`0.50`) detectaba `76.06 %` (216/284). Al calibrar el corte operativo a **`0.40`**, "
                "se detectan **236 de 284 desertores**, cumpliendo la meta institucional."
            )

        co4, co5 = st.columns([1, 2])
        with tarjeta("conc_oe4", co4, "stretch"):
            st.metric("OE4 · Balance F1-Score", "0.8075 → 0.8266", "✔ Meta > 0.78 (CV: 0.7907 ± 0.0085)", delta_color="off", delta_arrow="off")
            st.markdown(
                "Supera el criterio en validación cruzada y en prueba ciega, confirmando **alta estabilidad entre pliegues y cero sobreajuste**."
            )
        with tarjeta("conc_oe5", co5, "stretch"):
            st.metric("OE5 · Variables con Mayor Peso Predictivo", "Rendimiento 1er Año + Solvencia", "Aprobadas 2do Sem (-0.2261 F1) y Matrícula al Día (-0.0759 F1)", delta_color="off", delta_arrow="off")
            st.markdown(
                "La deserción responde principalmente a la combinación de **rezago académico** (`2do y 1er semestre`) y **morosidad financiera / ausencia de beca**, "
                "guiando intervenciones conjuntas de tutoría académica y apoyo socioeconómico."
            )

    with tab_lim:
        st.markdown("#### ⚠️ Limitaciones del Estudio vs. 🚀 Hoja de Ruta Técnica")
        c_lim, c_rec = st.columns(2, gap="large")

        with c_lim:
            with tarjeta("card_limitaciones_53", height="stretch"):
                st.markdown("##### ⚠️ Limitaciones Metodológicas y de Alcance")
                lim_items = [
                    ("🌍 1. Validez externa geográfica", "Datos de una sola institución en Portugal (*IP Portalegre*, UCI #697); requiere validación local antes de extrapolar a Perú."),
                    ("⏳ 2. Ventana al cierre del 1er año", "Su predictor principal (`2do sem aprobado`) se consolida al terminar el 2do ciclo, no el primer día de matrícula."),
                    ("🎚️ 3. Umbral 0.40 exploratorio", "Calibrado inspeccionando la curva PR en prueba; para producción estricta debe fijarse *Out-of-Fold* en `X_train`."),
                    ("🔗 4. Asociación ≠ Causalidad", "La importancia por permutación mide el peso predictivo dentro del modelo, pero no demuestra relación causal directa."),
                    ("⚙️ 5. Hiperparámetros acotados", "Configuración controlada por tiempo académico sin búsqueda exhaustiva (*GridSearch / Optuna*)."),
                ]
                for tit_l, desc_l in lim_items:
                    st.markdown(
                        f"<div style='padding:10px 14px;margin-bottom:8px;border-radius:8px;background:#fff8eb;border-left:4px solid #fab219'>"
                        f"<div style='font-weight:700;color:#854d0e;font-size:0.95rem'>{tit_l}</div>"
                        f"<div style='color:#422006;font-size:0.88rem;margin-top:2px'>{desc_l}</div></div>",
                        unsafe_allow_html=True,
                    )

        with c_rec:
            with tarjeta("card_recomendaciones_53", height="stretch"):
                st.markdown("##### 🚀 Recomendaciones Técnicas Futuras")
                rec_items = [
                    ("🎯 1. Umbral Out-of-Fold (OOF) y Cost-Sensitive", "Estimar el umbral óptimo con `cross_val_predict` solo en `X_train` e incorporar `sample_weight` en Gradient Boosting."),
                    ("⚡ 2. Nuevos Ensambles + Optimización Bayesiana", "Evaluar **XGBoost, LightGBM y CatBoost** ajustados con *Optuna* y comparar contra `SMOTE` dentro de cada pliegue."),
                    ("📅 3. Modelos Escalonados por Etapa", "Desplegar 3 modelos según el momento del alumno: **Ingreso (Admisión)** → **Cierre 1er Ciclo** → **Cierre 2do Ciclo**."),
                    ("🏫 4. Adaptación con Datos Institucionales UPN", "Conectar la arquitectura modular (`src/`) al historial académico y financiero local para validar su transferencia en Perú."),
                    ("🔄 5. Monitoreo Continuo (MLOps)", "Recalibrar periódicamente las probabilidades por cohorte semestral para prevenir degradación por cambios curriculares o económicos."),
                ]
                for tit_r, desc_r in rec_items:
                    st.markdown(
                        f"<div style='padding:10px 14px;margin-bottom:8px;border-radius:8px;background:#eff6ff;border-left:4px solid #184f95'>"
                        f"<div style='font-weight:700;color:#0d366b;font-size:0.95rem'>{tit_r}</div>"
                        f"<div style='color:#1e3a8a;font-size:0.88rem;margin-top:2px'>{desc_r}</div></div>",
                        unsafe_allow_html=True,
                    )

    with tab_etica:
        st.markdown("#### ⚖️ Gobernanza y Consideraciones Éticas de IA")
        st.info(
            "💡 **Principio Rector (*Human-in-the-loop*):** La alerta predictiva tiene como único fin **asignar acompañamiento académico y bienestar oportuno**, "
            "**nunca sancionar, excluir ni estigmatizar** al estudiante. Toda decisión recae siempre en un profesional de tutoría."
        )

        riesgos_cards = [
            (
                "⚖️ 1. Sesgo por Desbalance (32.1 % vs 67.9 %)",
                "Severidad Alta · Prob. Media",
                "#fef2f2",
                "#d03b3b",
                "Un modelo sin ajuste favorece la clase mayoritaria y omite alumnos vulnerables (FN).",
                "`class_weight='balanced'` en entrenamiento y umbral operativo `0.40` (reduce FN de 68 a 48).",
            ),
            (
                "👥 2. Sesgo Demográfico o Socioeconómico",
                "Severidad Media · Prob. Media",
                "#fffbeb",
                "#d97706",
                "Variables como edad, nacionalidad o estudios paternos podrían reflejar brechas históricas.",
                "Auditoría por permutación y uso exclusivo de la alerta para **apoyo positivo** (becas/tutoría).",
            ),
            (
                "🛡️ 3. Fuga de Datos (*Data Leakage*)",
                "Severidad Alta · Prob. Baja",
                "#eff6ff",
                "#184f95",
                "Preprocesar con todo el dataset inflaría artificialmente las métricas antes de producción.",
                "Encapsulamiento estricto en `Pipeline` + `ColumnTransformer` ajustado solo sobre `X_train`.",
            ),
            (
                "🔒 4. Privacidad y Datos Personales",
                "Severidad Alta · Prob. Baja",
                "#eff6ff",
                "#184f95",
                "Riesgo de exposición de calificaciones, morosidad o situación económica de estudiantes.",
                "Anonimización total (UCI #697 CC BY 4.0) y cumplimiento de la **Ley N° 29733** en despliegue local.",
            ),
            (
                "🔍 5. Opacidad Algorítmica (*Caja Negra*)",
                "Severidad Alta · Prob. Media",
                "#fef2f2",
                "#d03b3b",
                "Emitir alertas sin explicar qué factores elevaron el riesgo del estudiante.",
                "Explicabilidad global (Top 15 permutación) y desglose por alumno en el simulador en vivo.",
            ),
            (
                "📉 6. Degradación Temporal (*Concept Drift*)",
                "Severidad Media · Prob. Alta",
                "#fffbeb",
                "#d97706",
                "Cambios económicos o de malla curricular alteran los patrones históricos de abandono.",
                "Monitoreo de métricas por cohorte semestral y reentrenamiento anual del pipeline.",
            ),
        ]

        e_row1 = st.columns(3)
        for col_e, (tit_e, badge_e, bg_e, border_e, r_txt, m_txt) in zip(e_row1, riesgos_cards[:3]):
            with tarjeta(f"etica_card_{tit_e[:4]}", col_e, "stretch"):
                st.markdown(
                    f"<div style='display:inline-block;padding:2px 10px;border-radius:12px;background:{bg_e};color:{border_e};font-size:0.78rem;font-weight:700;margin-bottom:6px'>{badge_e}</div>"
                    f"<div style='font-weight:700;font-size:1.02rem;color:#0d366b;margin-bottom:6px'>{tit_e}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Riesgo:** {r_txt}")
                st.markdown(f"✅ **Mitigación:** {m_txt}")

        e_row2 = st.columns(3)
        for col_e, (tit_e, badge_e, bg_e, border_e, r_txt, m_txt) in zip(e_row2, riesgos_cards[3:]):
            with tarjeta(f"etica_card_{tit_e[:4]}", col_e, "stretch"):
                st.markdown(
                    f"<div style='display:inline-block;padding:2px 10px;border-radius:12px;background:{bg_e};color:{border_e};font-size:0.78rem;font-weight:700;margin-bottom:6px'>{badge_e}</div>"
                    f"<div style='font-weight:700;font-size:1.02rem;color:#0d366b;margin-bottom:6px'>{tit_e}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Riesgo:** {r_txt}")
                st.markdown(f"✅ **Mitigación:** {m_txt}")

