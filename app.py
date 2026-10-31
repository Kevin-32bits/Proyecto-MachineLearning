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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import precision_recall_curve

from src.config import FIGURES, POSITIVE_LABEL, SEED, TARGET_COL
from src.data_prep import load_dataset, make_binary_target, split_data
from src.preprocessor import ALL_FEATURE_COLS
from src.scoring import (
    CORTE_ROJO, CRITERIOS, categorias_desconocidas, construir_perfil,
    etiquetas_reales, leer_lote, metricas, preparar_lote, semaforo, validar_perfil,
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

# Top 10 por permutación a nivel de variable original, medida en validación cruzada
ENTRADAS_SIMULADOR = [
    COL_APROB_2, COL_PAGOS, COL_APROB_1, COL_CARRERA, COL_EDAD,
    COL_MATRIC_2, COL_DEUDA, COL_MATRIC_1, COL_BECA, COL_CONV_2,
]

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

SEMAFORO_UI = {
    "Rojo": ("🔴", st.error, "RIESGO ALTO: priorizar intervención"),
    "Amarillo": ("🟡", st.warning, "RIESGO MODERADO: alerta y seguimiento"),
    "Verde": ("🟢", st.success, "RIESGO BAJO: seguimiento regular"),
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
    """Carga un perfil en los campos del simulador y descarta el resultado anterior."""
    st.session_state.update(PERFILES[nombre])
    st.session_state.pop("perfil_evaluado", None)


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
        col.metric(nombre, valor, f"{'✔' if cumple else '✘'} Meta > {meta}", delta_color="off")
    columnas[4].metric("ROC-AUC", f"{m['roc_auc']:.4f}")


def grafico_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Resto (0)", "Dropout (1)"])
    ax.set_yticklabels(["Resto (0)", "Dropout (1)"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    plt.tight_layout()
    return fig


# Cargar datos y modelo
raw_df, X_train, X_test, y_train, y_test = load_all_data()
model_pipeline = load_winner_model()
metadata = load_metadata()
proba_test = predict_test(X_test)
model_name = metadata["modelo"]
medianas = X_train.median().to_dict()

VISTA_SIMULADOR = "🔮 Simulador de Riesgo"
VISTA_LOTES = "📂 Evaluación por Lotes"
VISTA_EDA = "📊 Dashboard Exploratorio (EDA)"
VISTA_DICCIONARIO = "📋 Diccionario de Variables (P2)"
VISTA_MODELOS = "🏆 Comparación de 4 Modelos (P3)"
VISTA_DIAGNOSTICO = "⚙️ Diagnóstico del Ganador y Explicabilidad (P3)"

# ---------------------------------------------------------
# Barra Lateral (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.title("Sistema de Alerta")
    st.markdown("**Proyecto de Sistemas Inteligentes — UPN**")
    st.caption(f"UCI Dataset 697 | Modelo: **{model_name}**")

    menu = st.radio(
        "Navegación del Sistema:",
        [VISTA_SIMULADOR, VISTA_LOTES, VISTA_EDA, VISTA_DICCIONARIO, VISTA_MODELOS, VISTA_DIAGNOSTICO],
        index=0
    )

    umbral = st.slider(
        "Umbral de decisión (probabilidad mínima de alerta)",
        min_value=0.20, max_value=0.80, value=float(metadata["umbral_operativo"]), step=0.01,
        help="Valores bajos detectan más casos (Recall) a cambio de más falsas alarmas."
    )

    if menu not in (VISTA_SIMULADOR, VISTA_LOTES):
        st.divider()
        st.markdown("### 📌 Parámetros Experimentales")
        st.markdown("- **Split:** 80% Train / 20% Test (Estratificado)")
        st.markdown(f"- **Semilla:** {SEED} (`SEED={SEED}`)")
        st.markdown(f"- **Clase Positiva:** 1 = Dropout ({y_train.mean() * 100:.1f}%)")
        st.markdown("- **Meta F1-Score:** > 0.78")
        st.markdown("- **Validación:** 5-Fold Stratified CV")

# ---------------------------------------------------------
# VISTA 1: SIMULADOR DE PREDICCIÓN EN TIEMPO REAL
# ---------------------------------------------------------
if menu == VISTA_SIMULADOR:
    st.title("🔮 Simulador de Riesgo de Deserción Estudiantil")
    st.markdown(
        f"Ingresa las **10 variables más influyentes** en la barra lateral y pulsa "
        f"**Evaluar Riesgo**. Modelo: **{model_name}**."
    )
    st.caption(
        "Las otras 26 variables se fijan en la mediana de train; para puntuar filas completas usa "
        "Evaluación por Lotes. El modelo necesita datos hasta el 2do semestre, así que no aplica "
        "al momento de la matrícula. Resultado orientativo para ofrecer apoyo, no para sancionar."
    )

    st.subheader("Perfiles de prueba")
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    col_demo1.button("🚨 Riesgo Crítico", width="stretch", on_click=aplicar_perfil, args=("riesgo",))
    col_demo2.button("⚠️ Regular Vulnerable", width="stretch", on_click=aplicar_perfil, args=("medio",))
    col_demo3.button("✅ Estudiante Destacado", width="stretch", on_click=aplicar_perfil, args=("seguro",))

    for col in ENTRADAS_SIMULADOR:
        st.session_state.setdefault(col, int(medianas[col]))

    with st.sidebar:
        st.divider()
        with st.form("perfil_estudiante"):
            st.markdown("#### Datos del estudiante")
            st.number_input("Materias aprobadas 2do semestre", min_value=0, max_value=30, key=COL_APROB_2)
            st.selectbox("¿Pagos de matrícula al día?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No", key=COL_PAGOS)
            st.number_input("Materias aprobadas 1er semestre", min_value=0, max_value=30, key=COL_APROB_1)
            st.selectbox("Carrera", options=list(CARRERAS), format_func=CARRERAS.get, key=COL_CARRERA)
            st.number_input("Edad al matricularse", min_value=17, max_value=70, key=COL_EDAD)
            st.number_input("Materias matriculadas 2do semestre", min_value=0, max_value=30, key=COL_MATRIC_2)
            st.selectbox("¿Tiene deudas pendientes?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No", key=COL_DEUDA)
            st.number_input("Materias matriculadas 1er semestre", min_value=0, max_value=30, key=COL_MATRIC_1)
            st.selectbox("¿Tiene beca?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No", key=COL_BECA)
            st.number_input("Materias convalidadas 2do semestre", min_value=0, max_value=30, key=COL_CONV_2)
            evaluar = st.form_submit_button("🚀 Evaluar Riesgo", type="primary", width="stretch")

    if evaluar:
        entradas = {col: st.session_state[col] for col in ENTRADAS_SIMULADOR}
        errores = validar_perfil({**entradas, COL_CONV_1: int(medianas[COL_CONV_1])})
        for error in errores:
            st.error(error)
        if errores:
            st.session_state.pop("perfil_evaluado", None)
        else:
            st.session_state["perfil_evaluado"] = entradas

    perfil = st.session_state.get("perfil_evaluado")
    if perfil is None:
        st.info("Completa los datos en la barra lateral y pulsa **Evaluar Riesgo**.")
    else:
        input_df = construir_perfil(medianas, perfil)
        prob_dropout = float(model_pipeline.predict_proba(input_df)[0, 1])
        color = str(semaforo([prob_dropout], umbral)[0])
        icono, aviso, texto = SEMAFORO_UI[color]

        st.subheader("Resultado de la Predicción")
        m_col1, m_col2 = st.columns([1, 2])
        m_col1.metric("Probabilidad de Deserción", f"{prob_dropout * 100:.1f} %")
        with m_col2:
            aviso(f"{icono} {texto}")
            st.progress(prob_dropout)
            st.caption(
                f"🟢 menor a {umbral:.2f} · 🟡 desde {umbral:.2f} · "
                f"🔴 desde {max(CORTE_ROJO, umbral):.2f}"
            )

        st.markdown("### 💡 Plan de Acción y Tutoría Sugerido")
        recomendaciones = []
        if perfil[COL_DEUDA] == 1 or perfil[COL_PAGOS] == 0:
            recomendaciones.append("💰 **Apoyo Financiero Inmediato:** El estudiante presenta deudas activas o pagos atrasados. Contactar con Bienestar Universitario para fraccionamiento o beca de emergencia.")
        if perfil[COL_APROB_2] < perfil[COL_MATRIC_2] / 2 or perfil[COL_APROB_1] < perfil[COL_MATRIC_1] / 2:
            recomendaciones.append("📚 **Tutoría Académica de Reforzamiento:** Aprobó menos de la mitad de sus materias en algún semestre. Las materias aprobadas son la señal académica más influyente del modelo.")
        if perfil[COL_BECA] == 0 and prob_dropout >= umbral:
            recomendaciones.append("🎯 **Evaluación Socioeconómica:** Evaluar elegibilidad para asignación de beca o subsidio alimenticio/transporte.")
        if perfil[COL_EDAD] >= 25:
            recomendaciones.append("⏰ **Flexibilidad Horaria:** Ingresó con 25 años o más; conviene revisar si compagina estudios con cargas familiares o laborales.")
        if not recomendaciones:
            recomendaciones.append("👏 **Seguimiento Preventivo Regular:** No se detectan factores de atención en los datos ingresados. Programar sesiones de seguimiento estándar.")

        for rec in recomendaciones:
            st.info(rec)

# ---------------------------------------------------------
# VISTA 2: EVALUACIÓN POR LOTES
# ---------------------------------------------------------
elif menu == VISTA_LOTES:
    st.title("📂 Evaluación por Lotes")
    st.markdown(
        "Sube un CSV con las **36 variables** del dataset (separado por `,` o `;`). Si incluye la "
        "columna `dropout` o `Target`, también se calculan las métricas contra el resultado real."
    )
    st.download_button(
        "⬇️ Descargar plantilla (conjunto de test)",
        data=X_test.assign(dropout=y_test).to_csv(index=False).encode("utf-8"),
        file_name="plantilla_lote.csv",
        mime="text/csv"
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
            st.warning(f"Códigos que el modelo no conoce (se puntúan igual, con menor fiabilidad): {detalle}.")

        proba_lote = model_pipeline.predict_proba(X_lote)[:, 1]
        resultado = lote[validas].copy()
        resultado.insert(0, "Semáforo", semaforo(proba_lote, umbral))
        resultado.insert(0, "Prob. Deserción", proba_lote.round(4))
        resultado = resultado.sort_values("Prob. Deserción", ascending=False)

        conteo = resultado["Semáforo"].value_counts()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estudiantes evaluados", f"{len(resultado):,}")
        k2.metric("🔴 Rojo", int(conteo.get("Rojo", 0)))
        k3.metric("🟡 Amarillo", int(conteo.get("Amarillo", 0)))
        k4.metric("🟢 Verde", int(conteo.get("Verde", 0)))

        y_lote = etiquetas_reales(lote[validas])
        if y_lote is not None:
            st.subheader("Métricas contra el resultado real")
            mostrar_metricas(metricas(y_lote, proba_lote, umbral))

        st.subheader("Estudiantes ordenados por riesgo")
        st.dataframe(resultado, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Descargar resultados",
            data=resultado.to_csv(index=False).encode("utf-8"),
            file_name="resultados_riesgo.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# VISTA 3: DASHBOARD EXPLORATORIO (EDA)
# ---------------------------------------------------------
elif menu == VISTA_EDA:
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown(f"Auditoría de calidad y distribución de variables sobre el dataset original de la UCI con **{len(raw_df):,} estudiantes**.")

    es_dropout = raw_df[TARGET_COL] == POSITIVE_LABEL

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Estudiantes", f"{len(raw_df):,}")
    kpi2.metric("Tasa de Deserción", f"{es_dropout.mean() * 100:.1f} %", f"{es_dropout.sum():,} desertores", delta_color="off")
    kpi3.metric("Valores Nulos", int(raw_df.isna().sum().sum()))
    kpi4.metric("Variables Predictoras", len(ALL_FEATURE_COLS))

    st.divider()

    c_eda1, c_eda2 = st.columns(2)

    with c_eda1:
        st.markdown("#### Deserción vs Estado de Pagos (Tuition Fees)")
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        counts = pd.crosstab(raw_df[COL_PAGOS], raw_df[TARGET_COL], normalize="index") * 100
        counts.plot(kind="bar", stacked=True, ax=ax1, color=["#e45756", "#f58518", "#4c78a8"])
        ax1.set_xlabel("Al día con tasas (0 = No, 1 = Sí)")
        ax1.set_ylabel("Porcentaje (%)")
        ax1.legend(["Dropout", "Enrolled", "Graduate"], loc="upper right")
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption(f"Con tasas atrasadas (0), la deserción es de {es_dropout[raw_df[COL_PAGOS] == 0].mean() * 100:.1f} %.")

    with c_eda2:
        st.markdown("#### Deserción vs Becas (Scholarship Holder)")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        counts_b = pd.crosstab(raw_df[COL_BECA], raw_df[TARGET_COL], normalize="index") * 100
        counts_b.plot(kind="bar", stacked=True, ax=ax2, color=["#e45756", "#f58518", "#4c78a8"])
        ax2.set_xlabel("Becado (0 = No, 1 = Sí)")
        ax2.set_ylabel("Porcentaje (%)")
        ax2.legend(["Dropout", "Enrolled", "Graduate"], loc="upper right")
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption(f"Entre becados, la deserción es de {es_dropout[raw_df[COL_BECA] == 1].mean() * 100:.1f} %.")

    st.subheader("Explorador Interactivo del Dataset")
    target_filter = st.multiselect("Filtrar por estado original:", options=raw_df[TARGET_COL].unique(), default=raw_df[TARGET_COL].unique())
    filtered_df = raw_df[raw_df[TARGET_COL].isin(target_filter)]
    st.dataframe(filtered_df.head(100), width="stretch")

# ---------------------------------------------------------
# VISTA 4: DICCIONARIO DE VARIABLES (P2)
# ---------------------------------------------------------
elif menu == VISTA_DICCIONARIO:
    st.title("📋 Diccionario de las 36 Variables Predictoras")
    st.markdown("Clasificación técnica y tratamiento metodológico implementado con `ColumnTransformer` (P2).")

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
# VISTA 5: COMPARACIÓN DE 4 MODELOS (P3)
# ---------------------------------------------------------
elif menu == VISTA_MODELOS:
    st.title("🏆 Comparación Experimental de 4 Modelos de Machine Learning")
    st.markdown(
        "Resultados obtenidos mediante **Validación Cruzada Estratificada de 5 Pliegues (5-Fold CV)** "
        "exclusivamente sobre el conjunto de entrenamiento (`X_train` = 3,539 registros)."
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

    st.subheader("1. Tabla Comparativa de Validación Cruzada (5-Fold CV)")
    st.dataframe(df_cv_results, width="stretch", hide_index=True)

    st.divider()

    col_cv_img, col_cv_text = st.columns([1.3, 1])

    with col_cv_img:
        st.markdown("#### Gráfico de Rendimiento Comparativo (F1 con Barras de Error)")
        fig_cv_path = FIGURES / "comparacion_modelos_cv.png"
        if fig_cv_path.exists():
            st.image(str(fig_cv_path), width="stretch")
        else:
            st.warning("Imagen comparativa no encontrada.")

    with col_cv_text:
        st.markdown("#### 🎯 Justificación Técnica del Ganador")
        st.markdown(
            """
            **Gradient Boosting fue seleccionado por la regla acordada: mayor F1 de la clase Dropout en CV.**
            1. **Mayor F1-Score Promedio:** 0.7907 en validación cruzada.
            2. **Menor variación entre pliegues:** desviación estándar de ± 0.0085.
            3. **Mayor precisión (84.96%):** menos falsas alarmas para los recursos de tutoría y bienestar.

            **Cautela:** la ventaja sobre Random Forest y Regresión Logística es menor a 0.005, dentro de
            la variación entre pliegues: es un empate técnico. Regresión Logística tiene el mejor ROC-AUC.
            """
        )

# ---------------------------------------------------------
# VISTA 6: DIAGNÓSTICO DEL GANADOR Y EXPLICABILIDAD (P3)
# ---------------------------------------------------------
elif menu == VISTA_DIAGNOSTICO:
    st.title("⚙️ Diagnóstico y Explicabilidad del Modelo Ganador (Test)")
    st.markdown(
        f"Evaluación del modelo **{model_name}** sobre el conjunto de prueba (`X_test` = {len(X_test)} alumnos), "
        "calculada en vivo con el umbral de la barra lateral."
    )

    m_activo = metricas(y_test, proba_test, umbral)
    tn, fp, fn, tp = m_activo["cm"].ravel()

    st.subheader(f"1. Rendimiento sobre el Conjunto de Prueba (umbral {umbral:.2f})")
    mostrar_metricas(m_activo)
    st.caption(
        f"Detectados {tp} de {tp + fn} desertores. Mover el umbral aquí es exploratorio: "
        "el umbral oficial debe elegirse con predicciones de validación cruzada sobre train."
    )

    filas_umbral = []
    for valor, etiqueta in [(0.50, "Default (0.50)"), (float(metadata["umbral_operativo"]), "Operativo P3"), (umbral, "Activo")]:
        m_u = metricas(y_test, proba_test, valor)
        filas_umbral.append({
            "Umbral": f"{etiqueta}: {valor:.2f}",
            "Accuracy": round(m_u["accuracy"], 4),
            "Precision": round(m_u["precision"], 4),
            "Recall": round(m_u["recall"], 4),
            "F1 (Dropout)": round(m_u["f1"], 4),
        })
    st.dataframe(pd.DataFrame(filas_umbral), width="stretch", hide_index=True)

    st.divider()

    c_cm, c_pr = st.columns(2)

    with c_cm:
        st.markdown(f"#### Matriz de Confusión en Test ({len(X_test)} Alumnos)")
        st.pyplot(grafico_confusion(m_activo["cm"]))
        st.caption(f"Verdaderos Negativos: {tn} | Falsos Positivos: {fp} | Falsos Negativos: {fn} | Verdaderos Positivos: {tp}")

    with c_pr:
        st.markdown("#### Curva Precision-Recall")
        precisions, recalls, _ = precision_recall_curve(y_test, proba_test)
        fig_pr, ax_pr = plt.subplots(figsize=(4.5, 3.5))
        ax_pr.plot(recalls, precisions, color="#4c78a8", lw=2, label="Curva P-R")
        ax_pr.scatter(m_activo["recall"], m_activo["precision"], color="#e45756", zorder=3, label=f"Umbral activo = {umbral:.2f}")
        ax_pr.set_xlabel("Recall (Sensibilidad)")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()
        plt.tight_layout()
        st.pyplot(fig_pr)

    st.divider()

    c_imp, c_pipe = st.columns(2)

    with c_imp:
        st.markdown("#### Importancia de Variables por Permutación (Top 15)")
        imp_path = FIGURES / "permutation_importance_top15.png"
        if imp_path.exists():
            st.image(str(imp_path), width="stretch")
        else:
            st.info("Gráfico de importancia no disponible.")
        st.caption(
            "La variable más determinante con amplia diferencia es **Curricular units 2nd sem (approved)** "
            "(caída de F1 = 0.2209), seguida del estado de pago de pensiones (**Tuition fees up to date** = 0.0763)."
        )

    with c_pipe:
        st.markdown("#### Diagrama del Pipeline Metodológico (CRISP-DM)")
        pipe_path = FIGURES / "figura1_pipeline.png"
        if pipe_path.exists():
            st.image(str(pipe_path), width="stretch")
        else:
            st.info("Diagrama del pipeline disponible en el repositorio.")
        st.caption("Flujo de 8 fases: Carga → Binarización → Partición → Preprocesador → Modelos → Evaluación → Permutación.")
