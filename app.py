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

from sklearn.metrics import precision_recall_curve

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
    initial_sidebar_state="expanded"
)

# Tarjetas blancas sobre el fondo tintado; solo aplica a la zona principal
st.html("""
<style>
  [data-testid="stMain"] [data-testid="stMetric"],
  [data-testid="stMain"] [data-testid="stExpander"] details,
  [class*="st-key-tarjeta"] { background: #ffffff; }
  [data-testid="stMain"] [data-testid="stExpander"] details { border-radius: .5rem; }
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
    <div style="padding:24px 28px;border-radius:14px;background:{COLOR_ENCABEZADO}">
      <h1 style="margin:0;padding:0;font-size:2.1rem;font-weight:700;line-height:1.25;color:#ffffff">{titulo}</h1>
      <div style="margin-top:8px;font-size:1.05rem;color:#e6effb">{subtitulo}</div>
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


def grafico_precision_recall(y_true, y_proba, m):
    """Curva Precision-Recall con el punto del umbral activo."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    curva = pd.DataFrame({"Recall": recalls, "Precision": precisions}).reset_index()
    escala = alt.Scale(domain=[0, 1])
    linea = alt.Chart(curva).mark_line(strokeWidth=2, color=COLOR_SERIE).encode(
        x=alt.X("Recall:Q", scale=escala, axis=alt.Axis(gridColor=COLOR_REJILLA)),
        y=alt.Y("Precision:Q", scale=escala, axis=alt.Axis(gridColor=COLOR_REJILLA)),
        order="index:Q",
    )
    punto = alt.Chart(pd.DataFrame([{"Recall": m["recall"], "Precision": m["precision"]}])).mark_point(
        filled=True, size=120, opacity=1, color=COLOR_TINTA, stroke=COLOR_SUPERFICIE, strokeWidth=2
    ).encode(
        x="Recall:Q", y="Precision:Q",
        tooltip=[alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")],
    )
    return (linea + punto).properties(height=320, background=COLOR_SUPERFICIE)


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
proba_test = predict_test(X_test)
model_name = metadata["modelo"]
medianas = X_train.median().to_dict()
tasa_abandono = (raw_df[TARGET_COL] == POSITIVE_LABEL).mean()

VISTA_INICIO = "🏠 Inicio"
VISTA_SIMULADOR = "🔮 Simulador"
VISTA_LOTES = "📂 Evaluar un grupo"
VISTA_EDA = "📊 Qué dicen los datos"
VISTA_MODELOS = "🏆 Comparación de modelos"
VISTA_RESULTADOS = "🎯 Resultados del modelo"
VISTA_DICCIONARIO = "📋 Diccionario de variables"

# ---------------------------------------------------------
# Barra Lateral (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.title("Alerta de Deserción")
    st.caption("Sistemas Inteligentes y Machine Learning — UPN")

    menu = st.radio(
        "Navegación",
        [VISTA_INICIO, VISTA_SIMULADOR, VISTA_LOTES, VISTA_EDA, VISTA_MODELOS, VISTA_RESULTADOS, VISTA_DICCIONARIO],
        index=0,
        label_visibility="collapsed"
    )

    with st.expander("⚙️ Ajustes avanzados"):
        umbral = st.slider(
            "Umbral de alerta",
            min_value=0.20, max_value=0.80, value=float(metadata["umbral_operativo"]), step=0.01,
            help="Probabilidad mínima para alertar. Más bajo detecta más casos, con más falsas alarmas."
        )

m_test = metricas(y_test, proba_test, umbral)

# ---------------------------------------------------------
# VISTA: INICIO
# ---------------------------------------------------------
if menu == VISTA_INICIO:
    encabezado(
        "¿Quién necesita apoyo para no abandonar la universidad?",
        "Este sistema estima el <b>riesgo de deserción</b> de cada estudiante para que tutoría y "
        "bienestar universitario puedan actuar a tiempo."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estudiantes analizados", f"{len(raw_df):,}", border=True)
    k2.metric("Abandonaron la carrera", f"{tasa_abandono * 100:.1f} %", border=True)
    k3.metric("Abandonos detectados", f"{m_test['recall'] * 100:.0f} de 100", border=True)
    k4.metric("Alertas acertadas", f"{m_test['precision'] * 100:.0f} de 100", border=True)
    st.caption(f"Detección medida con {len(X_test)} estudiantes que el modelo no usó para aprender.")

    st.subheader("Cómo funciona")
    p1, p2, p3 = st.columns(3)
    with tarjeta("paso_1", p1, "stretch"):
        st.markdown("**1. Aprende de casos reales**")
        st.markdown(f"Historial académico, pagos y datos personales de {len(raw_df):,} estudiantes de una institución de educación superior.")
    with tarjeta("paso_2", p2, "stretch"):
        st.markdown("**2. Compara cuatro modelos**")
        st.markdown(f"Se evaluaron cuatro algoritmos y se eligió **{model_name}** por su mejor F1 en la clase de deserción.")
    with tarjeta("paso_3", p3, "stretch"):
        st.markdown("**3. Emite una alerta**")
        st.markdown("Calcula una probabilidad y la traduce a un semáforo, individual o para un grupo completo.")

    st.subheader("Cómo leer el semáforo")
    corte = max(CORTE_ROJO, umbral)
    s1, s2, s3 = st.columns(3)
    s1.success(f"🟢 **Bajo:** menos de {umbral * 100:.0f} %. Seguimiento regular.")
    s2.warning(f"🟡 **Moderado:** desde {umbral * 100:.0f} %. Alerta y seguimiento cercano.")
    s3.error(f"🔴 **Alto:** desde {corte * 100:.0f} %. Priorizar intervención.")

    st.subheader("Alcance")
    st.markdown(
        "- El modelo usa notas hasta el **2do semestre**: acompaña el primer año, no predice el día de la matrícula.\n"
        "- La alerta sirve para **ofrecer apoyo**, nunca para sancionar. La decisión final es de una persona."
    )

    with st.expander("🎤 Guion para exponer (7 minutos)"):
        st.markdown(
            f"1. **Inicio (1 min):** el problema. {tasa_abandono * 100:.0f} % abandona; llegar a tiempo cambia el resultado.\n"
            f"2. **Qué dicen los datos (1 min):** los pagos atrasados y la edad marcan grandes diferencias.\n"
            f"3. **Comparación de modelos (1 min):** cuatro modelos; {model_name} gana por poco, y decirlo da credibilidad.\n"
            f"4. **Resultados (1 min):** de cada 100 que abandonan, detecta {m_test['recall'] * 100:.0f}.\n"
            "5. **Simulador (2 min):** pulsa los tres ejemplos y explica por qué cambia cada resultado.\n"
            "6. **Evaluar un grupo (30 s):** sube la plantilla y muestra la lista ordenada por riesgo.\n"
            "7. **Cierre (30 s):** apoya decisiones humanas; revisa sus límites en esta portada."
        )

# ---------------------------------------------------------
# VISTA: SIMULADOR DE PREDICCIÓN
# ---------------------------------------------------------
elif menu == VISTA_SIMULADOR:
    encabezado("🔮 Simulador de Riesgo", "Prueba un ejemplo o ingresa los datos del estudiante en la barra lateral.")

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

    # Rango observado en el dataset. Se valida al evaluar y no con min_value/max_value:
    # Streamlit descarta en silencio lo escrito fuera de esos límites y evalúa el valor anterior.
    rangos = {
        col: (int(raw_df[col].min()), int(raw_df[col].max()))
        for col in (COL_EDAD, COL_MATRIC_1, COL_APROB_1, COL_MATRIC_2, COL_APROB_2, COL_CONV_2)
    }

    def campo(col):
        return {"key": col, "step": 1, "help": f"Valores permitidos: {rangos[col][0]} a {rangos[col][1]}."}

    with st.sidebar:
        with st.form("perfil_estudiante"):
            st.markdown("#### Datos del estudiante")
            st.selectbox(ETIQUETAS[COL_CARRERA], options=list(CARRERAS), format_func=CARRERAS.get, key=COL_CARRERA)
            st.number_input(ETIQUETAS[COL_EDAD], **campo(COL_EDAD))

            st.markdown("#### 1er semestre")
            s1_matric, s1_aprob = st.columns(2)
            s1_matric.number_input("Matriculadas", **campo(COL_MATRIC_1))
            s1_aprob.number_input("Aprobadas", **campo(COL_APROB_1))

            st.markdown("#### 2do semestre")
            s2_matric, s2_aprob = st.columns(2)
            s2_matric.number_input("Matriculadas", **campo(COL_MATRIC_2))
            s2_aprob.number_input("Aprobadas", **campo(COL_APROB_2))
            st.number_input("Convalidadas", **campo(COL_CONV_2))

            st.markdown("#### Situación económica")
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

    perfil = st.session_state.get("perfil_evaluado")
    if perfil is None:
        if not errores:
            st.info("Elige un ejemplo arriba o completa los datos y pulsa **Evaluar Riesgo**.")
    else:
        input_df = construir_perfil(medianas, perfil)
        prob_dropout = float(model_pipeline.predict_proba(input_df)[0, 1])
        color = str(semaforo([prob_dropout], umbral)[0])
        st.html(tarjeta_resultado(prob_dropout, color, umbral))

        riesgo_tipico, factores = factores_riesgo(model_pipeline, medianas, perfil)
        suben = [(c, v) for c, v in factores if v >= 0.01][:3]
        bajan = [(c, v) for c, v in reversed(factores) if v <= -0.01][:3]

        st.subheader("¿Por qué este resultado?")
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

        st.subheader("Qué hacer")
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
            "usan el valor típico. Para evaluar datos completos, usa Evaluar un grupo."
        )

# ---------------------------------------------------------
# VISTA: EVALUACIÓN POR LOTES
# ---------------------------------------------------------
elif menu == VISTA_LOTES:
    encabezado(
        "📂 Evaluar un Grupo de Estudiantes",
        "Sube un CSV con las <b>36 variables</b> y obtén la lista ordenada por riesgo."
    )
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
# VISTA: DASHBOARD EXPLORATORIO (EDA)
# ---------------------------------------------------------
elif menu == VISTA_EDA:
    grupos = abandono_por_grupo(raw_df)
    mayor = grupos.iloc[0]
    encabezado(
        "📊 Qué Dicen los Datos",
        f"Abandonó <b>{tasa_abandono * 100:.1f} %</b> de los estudiantes, pero el riesgo no es igual para todos: "
        f"llega a <b>{mayor['valor'] * 100:.1f} %</b> entre quienes tienen <b>{mayor['Grupo'].lower()}</b>."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estudiantes", f"{len(raw_df):,}", border=True)
    k2.metric("Tasa de abandono", f"{tasa_abandono * 100:.1f} %", border=True)
    k3.metric("Valores faltantes", int(raw_df.isna().sum().sum()), border=True)
    k4.metric("Variables predictoras", len(ALL_FEATURE_COLS), border=True)

    st.subheader("Abandono por grupo de estudiantes")
    with tarjeta("grafico_grupos"):
        st.altair_chart(grafico_barras(grupos, [0, 1], ".0%", tasa_abandono), width="stretch")
    st.caption(f"Línea vertical: promedio general ({tasa_abandono * 100:.1f} %). Muestra asociación, no causa.")

    with st.expander("Ver tabla y explorar el dataset"):
        tabla = grupos.assign(Abandono=(grupos["valor"] * 100).round(1))[["Grupo", "Abandono", "Estudiantes"]]
        st.dataframe(tabla, width="stretch", hide_index=True, column_config={"Abandono": st.column_config.NumberColumn("Abandono (%)")})
        target_filter = st.multiselect("Filtrar por estado original:", options=raw_df[TARGET_COL].unique(), default=raw_df[TARGET_COL].unique())
        filtered_df = raw_df[raw_df[TARGET_COL].isin(target_filter)]
        st.dataframe(filtered_df.head(100), width="stretch")

# ---------------------------------------------------------
# VISTA: DICCIONARIO DE VARIABLES (P2)
# ---------------------------------------------------------
elif menu == VISTA_DICCIONARIO:
    encabezado("📋 Diccionario de Variables", "Las 36 variables predictoras y su tratamiento en el preprocesamiento.")

    tipo_filtro = st.radio("Filtrar por tipo de variable:", ["Todas (36)", "Nominales (9)", "Binarias (8)", "Numéricas (19)"], horizontal=True)

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

# ---------------------------------------------------------
# VISTA: COMPARACIÓN DE 4 MODELOS (P3)
# ---------------------------------------------------------
elif menu == VISTA_MODELOS:
    encabezado(
        "🏆 Comparación de Modelos",
        f"Se compararon cuatro algoritmos con los mismos datos. <b>{model_name}</b> obtuvo el mejor <b>F1</b>, "
        "la nota que equilibra detectar a quienes abandonan y evitar falsas alarmas."
    )

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
            "Estado Meta": "✔ Supera Meta (F1 > 0.78)"
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
            "Estado Meta": "✔ Supera Meta (F1 > 0.78)"
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
            "Estado Meta": "✔ Supera Meta (F1 > 0.78)"
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
            "Estado Meta": "❌ No cumple meta"
        }
    ])

    datos_f1 = pd.DataFrame({
        "valor": df_cv_results["F1 (CV)"],
        "etiqueta": [f"{m}  ·  {v:.4f}" for m, v in zip(df_cv_results["Modelo"], df_cv_results["F1 (CV)"])],
    })
    st.subheader("F1 en validación cruzada (0 a 1)")
    with tarjeta("grafico_modelos"):
        st.altair_chart(
            grafico_barras(datos_f1, [0, 1], ".2f", CRITERIOS["f1"], destacar=datos_f1["etiqueta"].iloc[0]),
            width="stretch"
        )
    st.caption(f"Línea vertical: meta del proyecto (F1 > {CRITERIOS['f1']}).")

    st.warning(
        "**Empate técnico entre los tres primeros:** la ventaja del ganador es menor a 0.005, dentro de la "
        "variación entre pruebas. Se eligió con la regla acordada antes de ver los resultados."
    )

    with st.expander("Ver detalle técnico"):
        st.markdown(
            "Validación cruzada estratificada de 5 pliegues, solo sobre train "
            f"(`X_train` = {len(X_train):,}). Split 80/20 estratificado, `SEED={SEED}`, clase positiva 1 = Dropout."
        )
        st.dataframe(df_cv_results, width="stretch", hide_index=True)
        st.markdown(
            "**Gradient Boosting** tuvo además la menor variación entre pliegues (± 0.0085) y la mayor "
            "precisión (84.96 %). **Regresión Logística** tuvo el mejor ROC-AUC (0.9209)."
        )

# ---------------------------------------------------------
# VISTA: RESULTADOS DEL GANADOR (P3)
# ---------------------------------------------------------
elif menu == VISTA_RESULTADOS:
    encabezado("🎯 Resultados del Modelo", f"Cómo le fue a <b>{model_name}</b> con {len(X_test)} estudiantes que no usó para aprender.")

    tn, fp, fn, tp = m_test["cm"].ravel()

    k1, k2, k3 = st.columns(3)
    k1.metric("Abandonos detectados", f"{m_test['recall'] * 100:.0f} de 100", border=True)
    k2.metric("Alertas acertadas", f"{m_test['precision'] * 100:.0f} de 100", border=True)
    k3.metric("Casos bien clasificados", f"{m_test['accuracy'] * 100:.0f} de 100", border=True)

    st.subheader(f"Qué pasó con los {len(X_test)} estudiantes")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("✅ Abandono detectado", int(tp), "Alerta a tiempo", delta_color="off", delta_arrow="off", border=True)
    r2.metric("❌ Abandono no detectado", int(fn), "Sin alerta", delta_color="off", delta_arrow="off", border=True)
    r3.metric("⚠️ Falsa alarma", int(fp), "Apoyo no necesario", delta_color="off", delta_arrow="off", border=True)
    r4.metric("👍 Continuó, sin alerta", int(tn), "Correcto", delta_color="off", delta_arrow="off", border=True)

    st.subheader("¿Cumple las metas del proyecto?")
    mostrar_metricas(m_test)
    st.caption(
        "En validación cruzada, el F1 fue 0.79: es la referencia más prudente. "
        "Las metas se evalúan sobre la clase Dropout."
    )

    st.info(
        f"**Lo que más pesa en la decisión:** {ETIQUETAS[COL_APROB_2].lower()}, "
        f"{ETIQUETAS[COL_PAGOS].lower()} y {ETIQUETAS[COL_APROB_1].lower()}."
    )

    with st.expander("Ver detalle técnico"):
        st.markdown(f"#### Efecto del umbral (activo: {umbral:.2f})")
        filas_umbral = []
        for valor, etiqueta in [(0.50, "Default"), (float(metadata["umbral_operativo"]), "Operativo P3"), (umbral, "Activo")]:
            m_u = metricas(y_test, proba_test, valor)
            filas_umbral.append({
                "Umbral": f"{etiqueta}: {valor:.2f}",
                "Accuracy": round(m_u["accuracy"], 4),
                "Precision": round(m_u["precision"], 4),
                "Recall": round(m_u["recall"], 4),
                "F1 (Dropout)": round(m_u["f1"], 4),
            })
        st.dataframe(pd.DataFrame(filas_umbral), width="stretch", hide_index=True)
        st.caption(
            "Mover el umbral aquí es exploratorio: el umbral oficial debe elegirse con predicciones "
            "de validación cruzada sobre train."
        )

        c_pr, c_imp = st.columns(2)
        with c_pr:
            st.markdown("#### Curva Precision-Recall")
            st.altair_chart(grafico_precision_recall(y_test, proba_test, m_test), width="stretch")
            st.caption("El punto marca el umbral activo.")
        with c_imp:
            st.markdown("#### Importancia por permutación (Top 15)")
            imp_path = FIGURES / "permutation_importance_top15.png"
            if imp_path.exists():
                st.image(str(imp_path), width="stretch")
            else:
                st.info("Gráfico de importancia no disponible.")

        st.markdown("#### Pipeline del proyecto (CRISP-DM)")
        pipe_path = FIGURES / "figura1_pipeline.png"
        if pipe_path.exists():
            st.image(str(pipe_path), width="stretch")
        else:
            st.info("Diagrama del pipeline disponible en el repositorio.")
