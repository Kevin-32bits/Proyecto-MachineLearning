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
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)

from src.config import DATA_RAW, DATA_PROCESSED, SEED, TARGET_COL, POSITIVE_LABEL, FIGURES
from src.preprocessor import (
    get_preprocessor, ALL_FEATURE_COLS, NOMINAL_COLS, BINARY_COLS, NUMERIC_COLS
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

# ---------------------------------------------------------
# Funciones cacheadas para carga de datos y modelo ganador
# ---------------------------------------------------------
@st.cache_data
def load_all_data():
    if not (DATA_PROCESSED / "train.csv").exists():
        from src.data_prep import get_train_test
        get_train_test()
    
    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    raw_df = pd.read_csv(DATA_RAW, sep=";")
    raw_df.columns = raw_df.columns.str.strip()
    return raw_df, train_df, test_df


@st.cache_resource
def load_winner_model(train_df, test_df):
    model_path = ROOT_DIR / "models" / "pipeline_ganador.joblib"
    X_train = train_df[ALL_FEATURE_COLS]
    y_train = train_df["dropout"]
    X_test = test_df[ALL_FEATURE_COLS]
    y_test = test_df["dropout"]

    if model_path.exists():
        pipeline = joblib.load(model_path)
    else:
        # Fallback si aún no existe el .joblib
        pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=SEED
            ))
        ])
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    clf_step = pipeline.named_steps["classifier"]
    clf_name = "Gradient Boosting" if "GradientBoosting" in clf_step.__class__.__name__ else clf_step.__class__.__name__

    metrics = {
        "model_name": clf_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cm": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    }
    return pipeline, metrics


# Cargar datos y modelo
raw_df, train_df, test_df = load_all_data()
model_pipeline, test_metrics = load_winner_model(train_df, test_df)

# ---------------------------------------------------------
# Barra Lateral (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=70)
    st.title("Sistema de Alerta Temprana")
    st.markdown("**Proyecto de Sistemas Inteligentes — UPN**")
    st.caption(f"UCI Dataset 697 | Modelo: **{test_metrics.get('model_name', 'Gradient Boosting')}**")

    menu = st.radio(
        "Navegación del Sistema:",
        [
            "🔮 Simulador de Riesgo",
            "📊 Dashboard Exploratorio (EDA)",
            "📋 Diccionario de Variables (P2)",
            "🏆 Comparación de 4 Modelos (P3)",
            "⚙️ Diagnóstico del Ganador y Explicabilidad (P3)"
        ],
        index=0
    )

    st.divider()
    st.markdown("### 📌 Parámetros Experimentales")
    st.markdown("- **Split:** 80% Train / 20% Test (Estratificado)")
    st.markdown("- **Semilla:** 42 (`SEED=42`)")
    st.markdown("- **Clase Positiva:** 1 = Dropout (32.1%)")
    st.markdown("- **Meta F1-Score:** > 0.78 (Superada: **0.8075**)")
    st.markdown("- **Validación:** 5-Fold Stratified CV")

# ---------------------------------------------------------
# VISTA 1: SIMULADOR DE PREDICCIÓN EN TIEMPO REAL
# ---------------------------------------------------------
if menu == "🔮 Simulador de Riesgo":
    st.title("🔮 Simulador de Riesgo de Deserción Estudiantil")
    st.markdown(
        f"Esta herramienta institucional utiliza el modelo ganador **{test_metrics.get('model_name', 'Gradient Boosting')}** "
        "para evaluar la probabilidad de abandono en tiempo real y emitir alertas preventivas."
    )

    # Configuración de Umbral
    with st.expander("🛠️ Ajuste de Umbral de Alerta Temprana (Threshold)", expanded=False):
        c_th1, c_th2 = st.columns([2, 1])
        with c_th1:
            umbral_sel = st.slider(
                "Umbral de Decisión (Probabilidad mínima para clasificar como Dropout)",
                min_value=0.20, max_value=0.80, value=0.40, step=0.01,
                help="Valores más bajos (ej. 0.40) priorizan la detección oportuna (Recall), minimizando los falsos negativos."
            )
        with c_th2:
            st.info(
                "**Recomendaciones:**\n"
                "- `0.50`: Estándar de Scikit-Learn\n"
                "- `0.40`: **Operativo Institucional** (F1=0.8266, Recall=83.1%)\n"
                "- `0.39`: **Óptimo Matemático** (F1=0.8272)"
            )

    st.subheader("1. Cargar Perfil de Prueba Rápido (Presets)")
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    default_vals = train_df[ALL_FEATURE_COLS].median().to_dict()

    preset = None
    if col_demo1.button("🚨 Estudiante en Riesgo Crítico", use_container_width=True):
        preset = "riesgo"
    if col_demo2.button("⚠️ Estudiante Regular Vulnerable", use_container_width=True):
        preset = "medio"
    if col_demo3.button("✅ Estudiante Destacado", use_container_width=True):
        preset = "seguro"

    if preset == "riesgo":
        default_vals.update({
            "Curricular units 1st sem (approved)": 0,
            "Curricular units 1st sem (grade)": 0.0,
            "Curricular units 2nd sem (approved)": 0,
            "Curricular units 2nd sem (grade)": 0.0,
            "Debtor": 1,
            "Tuition fees up to date": 0,
            "Scholarship holder": 0,
            "Age at enrollment": 32,
            "Admission grade": 105.0
        })
    elif preset == "medio":
        default_vals.update({
            "Curricular units 1st sem (approved)": 3,
            "Curricular units 1st sem (grade)": 11.0,
            "Curricular units 2nd sem (approved)": 2,
            "Curricular units 2nd sem (grade)": 10.5,
            "Debtor": 1,
            "Tuition fees up to date": 1,
            "Scholarship holder": 0,
            "Age at enrollment": 24,
            "Admission grade": 118.0
        })
    elif preset == "seguro":
        default_vals.update({
            "Curricular units 1st sem (approved)": 6,
            "Curricular units 1st sem (grade)": 15.5,
            "Curricular units 2nd sem (approved)": 6,
            "Curricular units 2nd sem (grade)": 16.0,
            "Debtor": 0,
            "Tuition fees up to date": 1,
            "Scholarship holder": 1,
            "Age at enrollment": 18,
            "Admission grade": 150.0
        })

    st.divider()
    st.subheader("2. Parámetros del Estudiante a Evaluar")

    col_ac1, col_ac2, col_soc = st.columns(3)

    with col_ac1:
        st.markdown("#### 📘 1er Semestre Académico")
        c1_enrolled = st.number_input("Materias matriculadas (1er Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 1st sem (enrolled)", 6)))
        c1_approved = st.number_input("Materias aprobadas (1er Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 1st sem (approved)", 5)))
        c1_grade = st.slider("Promedio de notas 1er Sem (0 a 20)", min_value=0.0, max_value=20.0, value=float(default_vals.get("Curricular units 1st sem (grade)", 12.5)), step=0.1)
        c1_evals = st.number_input("Evaluaciones rendidas 1er Sem", min_value=0, max_value=30, value=int(default_vals.get("Curricular units 1st sem (evaluations)", 6)))

    with col_ac2:
        st.markdown("#### 📗 2do Semestre Académico (Crítico)")
        c2_enrolled = st.number_input("Materias matriculadas (2do Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 2nd sem (enrolled)", 6)))
        c2_approved = st.number_input("Materias aprobadas (2do Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 2nd sem (approved)", 5)))
        c2_grade = st.slider("Promedio de notas 2do Sem (0 a 20)", min_value=0.0, max_value=20.0, value=float(default_vals.get("Curricular units 2nd sem (grade)", 12.5)), step=0.1)
        c2_evals = st.number_input("Evaluaciones rendidas 2do Sem", min_value=0, max_value=30, value=int(default_vals.get("Curricular units 2nd sem (evaluations)", 6)))

    with col_soc:
        st.markdown("#### 💳 Factores Socioeconómicos")
        debtor = st.selectbox("¿Tiene deudas pendientes? (Debtor)", options=[0, 1], format_func=lambda x: "Sí (Con deudas)" if x == 1 else "No (Sin deuda)", index=int(default_vals.get("Debtor", 0)))
        tuition_ok = st.selectbox("¿Al día en pagos de pensión? (Tuition)", options=[1, 0], format_func=lambda x: "Sí (Al día)" if x == 1 else "No (Atrasado)", index=0 if default_vals.get("Tuition fees up to date", 1) == 1 else 1)
        scholarship = st.selectbox("¿Tiene Beca de Estudios?", options=[0, 1], format_func=lambda x: "Sí (Becado)" if x == 1 else "No (Sin beca)", index=int(default_vals.get("Scholarship holder", 0)))
        age = st.number_input("Edad al matricularse", min_value=17, max_value=70, value=int(default_vals.get("Age at enrollment", 20)))
        gender = st.selectbox("Género", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino", index=int(default_vals.get("Gender", 1)))
        admission_grade = st.number_input("Nota del examen de admisión (95-190)", min_value=95.0, max_value=190.0, value=float(default_vals.get("Admission grade", 125.0)), step=0.5)

    st.markdown("---")
    if st.button("🚀 Evaluar Riesgo con Modelo Ganador", type="primary", use_container_width=True):
        input_data = default_vals.copy()
        input_data.update({
            "Curricular units 1st sem (enrolled)": c1_enrolled,
            "Curricular units 1st sem (approved)": c1_approved,
            "Curricular units 1st sem (grade)": c1_grade,
            "Curricular units 1st sem (evaluations)": c1_evals,
            "Curricular units 2nd sem (enrolled)": c2_enrolled,
            "Curricular units 2nd sem (approved)": c2_approved,
            "Curricular units 2nd sem (grade)": c2_grade,
            "Curricular units 2nd sem (evaluations)": c2_evals,
            "Debtor": debtor,
            "Tuition fees up to date": tuition_ok,
            "Scholarship holder": scholarship,
            "Age at enrollment": age,
            "Gender": gender,
            "Admission grade": admission_grade
        })

        input_df = pd.DataFrame([input_data])[ALL_FEATURE_COLS]
        prob_dropout = model_pipeline.predict_proba(input_df)[0, 1]
        es_desertor = prob_dropout >= umbral_sel

        st.subheader("Resultado de la Predicción")
        m_col1, m_col2, m_col3 = st.columns([1, 1.2, 1.5])

        with m_col1:
            st.metric("Probabilidad de Deserción", f"{prob_dropout*100:.1f} %")

        with m_col2:
            if es_desertor:
                st.error(f"🚨 ALERTA: RIESGO DE DESERCIÓN\n(Supera umbral de {umbral_sel*100:.0f}%)")
            else:
                st.success(f"✅ ESTUDIANTE REGULAR RETENIDO\n(Bajo umbral de {umbral_sel*100:.0f}%)")

        with m_col3:
            st.progress(min(prob_dropout, 1.0))
            st.caption(f"Umbral operativo activo: {umbral_sel*100:.1f}%")

        # Recomendaciones automáticas
        st.markdown("### 💡 Plan de Acción y Tutoría Sugerido")
        recomendaciones = []
        if debtor == 1 or tuition_ok == 0:
            recomendaciones.append("💰 **Apoyo Financiero Inmediato:** El estudiante presenta deudas activas o pensiones impagas (factor #2 más influyente en el modelo). Contactar con Bienestar Universitario para fraccionamiento o beca de emergencia.")
        if c2_approved < (c2_enrolled / 2) or c1_approved < (c1_enrolled / 2):
            recomendaciones.append("📚 **Tutoría Académica de Reforzamiento:** Desaprobó el 50% o más de las asignaturas. El rendimiento del 2do semestre es la variable con mayor peso predictivo según el análisis por permutación.")
        if scholarship == 0 and prob_dropout > 0.40:
            recomendaciones.append("🎯 **Evaluación Socioeconómica:** Evaluar elegibilidad para asignación de beca o subsidio alimenticio/transporte.")
        if age >= 25:
            recomendaciones.append("⏰ **Flexibilidad Horaria:** El estudiante ingresó con mayor edad promedio, lo que suele asociarse a cargas familiares y laborales simultáneas.")
        if not recomendaciones:
            recomendaciones.append("👏 **Seguimiento Preventivo Regular:** El estudiante mantiene un perfil regular saludable. Programar sesiones de seguimiento estándar.")

        for rec in recomendaciones:
            st.info(rec)

# ---------------------------------------------------------
# VISTA 2: DASHBOARD EXPLORATORIO (EDA)
# ---------------------------------------------------------
elif menu == "📊 Dashboard Exploratorio (EDA)":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Auditoría de calidad y distribución de variables sobre el dataset original de la UCI con **4,424 estudiantes**.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Estudiantes", f"{len(raw_df):,}")
    kpi2.metric("Tasa de Deserción", "32.1 %", "1,421 desertores")
    kpi3.metric("Valores Nulos", "0", "100% integridad")
    kpi4.metric("Variables Predictoras", "36")

    st.divider()

    c_eda1, c_eda2 = st.columns(2)

    with c_eda1:
        st.markdown("#### Deserción vs Estado de Pagos (Tuition Fees)")
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        counts = pd.crosstab(raw_df["Tuition fees up to date"], raw_df[TARGET_COL], normalize="index") * 100
        counts.plot(kind="bar", stacked=True, ax=ax1, color=["#e45756", "#f58518", "#4c78a8"])
        ax1.set_xlabel("Al día con tasas (0 = No, 1 = Sí)")
        ax1.set_ylabel("Porcentaje (%)")
        ax1.legend(["Dropout", "Enrolled", "Graduate"], loc="upper right")
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption("Los estudiantes con tasas atrasadas (0) presentan una tasa de deserción superior al 85%.")

    with c_eda2:
        st.markdown("#### Deserción vs Becas (Scholarship Holder)")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        counts_b = pd.crosstab(raw_df["Scholarship holder"], raw_df[TARGET_COL], normalize="index") * 100
        counts_b.plot(kind="bar", stacked=True, ax=ax2, color=["#e45756", "#f58518", "#4c78a8"])
        ax2.set_xlabel("Becado (0 = No, 1 = Sí)")
        ax2.set_ylabel("Porcentaje (%)")
        ax2.legend(["Dropout", "Enrolled", "Graduate"], loc="upper right")
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("Los estudiantes becados tienen una tasa de deserción significativamente menor (< 12%).")

    st.subheader("Explorador Interactivo del Dataset")
    target_filter = st.multiselect("Filtrar por estado original:", options=raw_df[TARGET_COL].unique(), default=raw_df[TARGET_COL].unique())
    filtered_df = raw_df[raw_df[TARGET_COL].isin(target_filter)]
    st.dataframe(filtered_df.head(100), use_container_width=True)

# ---------------------------------------------------------
# VISTA 3: DICCIONARIO DE VARIABLES (P2)
# ---------------------------------------------------------
elif menu == "📋 Diccionario de Variables (P2)":
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

    st.dataframe(df_d, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# VISTA 4: COMPARACIÓN DE 4 MODELOS (P3)
# ---------------------------------------------------------
elif menu == "🏆 Comparación de 4 Modelos (P3)":
    st.title("🏆 Comparación Experimental de 4 Modelos de Machine Learning")
    st.markdown(
        "Resultados obtenidos mediante **Validación Cruzada Estratificada de 5 Pliegues (5-Fold CV)** "
        "exclusivamente sobre el conjunto de entrenamiento (`X_train` = 3,539 registros). El conjunto de prueba (`X_test`) "
        "se mantuvo estrictamente aislado para evitar *data leakage*."
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
    st.dataframe(df_cv_results, use_container_width=True, hide_index=True)

    st.divider()

    col_cv_img, col_cv_text = st.columns([1.3, 1])

    with col_cv_img:
        st.markdown("#### Gráfico de Rendimiento Comparativo (F1 con Barras de Error)")
        fig_cv_path = FIGURES / "comparacion_modelos_cv.png"
        if fig_cv_path.exists():
            st.image(str(fig_cv_path), use_container_width=True)
        else:
            st.warning("Imagen comparativa no encontrada.")

    with col_cv_text:
        st.markdown("#### 🎯 Justificación Técnica del Ganador")
        st.markdown(
            """
            **Gradient Boosting fue seleccionado como el modelo ganador debido a:**
            1. **Mayor F1-Score Promedio:** Alcanzó **0.7907** en validación cruzada sobre la clase minoritaria (*Dropout*).
            2. **Mayor Estabilidad y Consistencia:** Tuvo la menor desviación estándar de todo el experimento (**± 0.0085**), lo que garantiza robustez ante diferentes subconjuntos de datos.
            3. **Mayor Precisión Institucional (84.96%):** Minimiza significativamente los falsos positivos, optimizando el uso de recursos de tutoría y bienestar.
            4. **Ventaja Arquitectónica:** A diferencia de los métodos de *bagging*, el aprendizaje secuencial con árboles superficiales (*boosting*) modela de manera óptima las complejas interacciones entre factores socioeconómicos y notas del 2do semestre sin memorizar ruido.
            """
        )

# ---------------------------------------------------------
# VISTA 5: DIAGNÓSTICO DEL GANADOR Y EXPLICABILIDAD (P3)
# ---------------------------------------------------------
elif menu == "⚙️ Diagnóstico del Ganador y Explicabilidad (P3)":
    st.title("⚙️ Diagnóstico y Explicabilidad del Modelo Ganador (Test)")
    st.markdown(
        "Evaluación definitiva del modelo **Gradient Boosting** sobre el conjunto de prueba (`X_test` = 885 alumnos), "
        "junto a su análisis de importancia de variables, curva Precision-Recall y pipeline metodológico."
    )

    # Cargar métricas exportadas si existen
    metricas_path = ROOT_DIR / "models" / "metricas_test.json"
    if metricas_path.exists():
        with open(metricas_path, "r", encoding="utf-8") as f:
            m_exp = json.load(f)
    else:
        m_exp = {
            "F1-Score (Dropout)": test_metrics['f1'],
            "Precision (Dropout)": test_metrics['precision'],
            "Recall (Dropout)": test_metrics['recall'],
            "Accuracy": test_metrics['accuracy'],
            "ROC-AUC": test_metrics['roc_auc']
        }

    st.subheader("1. Rendimiento Final sobre el Conjunto de Prueba (Test)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("F1-Score (Dropout)", f"{m_exp['F1-Score (Dropout)']:.4f}", "Meta > 0.78 ✔")
    m2.metric("ROC-AUC", f"{m_exp['ROC-AUC']:.4f}", "Excelente capacidad")
    m3.metric("Recall (Sensibilidad)", f"{m_exp['Recall (Dropout)']*100:.1f} %", "216/284 detectados")
    m4.metric("Precision", f"{m_exp['Precision (Dropout)']*100:.1f} %", "Baja tasa de falsa alarma")
    m5.metric("Accuracy Global", f"{m_exp['Accuracy']*100:.1f} %")

    st.divider()

    c_cm, c_pr = st.columns(2)

    with c_cm:
        st.markdown("#### Matriz de Confusión en Test (885 Alumnos)")
        cm_path = FIGURES / "matriz_confusion_ganador.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            cm = test_metrics["cm"]
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
            cax = ax_cm.matshow(cm, cmap="Blues", alpha=0.8)
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold")
            fig_cm.colorbar(cax)
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Resto (0)", "Dropout (1)"])
            ax_cm.set_yticklabels(["Resto (0)", "Dropout (1)"])
            ax_cm.set_xlabel("Predicción")
            ax_cm.set_ylabel("Valor Real")
            plt.tight_layout()
            st.pyplot(fig_cm)
        st.caption("Verdaderos Negativos: 566 | Falsos Positivos: 35 | Falsos Negativos: 68 | Verdaderos Positivos: 216")

    with c_pr:
        st.markdown("#### Curva Precision-Recall y Umbrales Óptimos")
        pr_path = FIGURES / "curva_precision_recall.png"
        if pr_path.exists():
            st.image(str(pr_path), use_container_width=True)
        else:
            st.info("Curva P-R disponible tras ejecución del script.")
        st.caption("El umbral operativo recomendado es **0.40** (eleva el Recall al 83.1% con F1 de 0.8266).")

    st.divider()

    c_imp, c_pipe = st.columns(2)

    with c_imp:
        st.markdown("#### Importancia de Variables por Permutación (Top 15)")
        imp_path = FIGURES / "permutation_importance_top15.png"
        if imp_path.exists():
            st.image(str(imp_path), use_container_width=True)
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
            st.image(str(pipe_path), use_container_width=True)
        else:
            st.info("Diagrama del pipeline disponible en el repositorio.")
        st.caption("Flujo de 8 fases: Carga → Binarización → Partición → Preprocesador → Modelos → Evaluación → Permutación.")
