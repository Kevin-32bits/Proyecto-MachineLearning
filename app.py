"""Aplicación Web Interactiva en Streamlit.
Predicción de la Deserción Universitaria (Sistemas Inteligentes - UPN).
"""
import sys
from pathlib import Path

# Configurar path raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from src.config import DATA_RAW, DATA_PROCESSED, SEED, TARGET_COL, POSITIVE_LABEL, FIGURES
from src.preprocessor import (
    get_preprocessor, ALL_FEATURE_COLS, NOMINAL_COLS, BINARY_COLS, NUMERIC_COLS
)

# ---------------------------------------------------------
# Configuración general de la página Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Deserción Universitaria",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Funciones cacheadas para carga de datos y entrenamiento
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
def train_model(train_df, test_df):
    X_train = train_df[ALL_FEATURE_COLS]
    y_train = train_df["dropout"]
    X_test = test_df[ALL_FEATURE_COLS]
    y_test = test_df["dropout"]

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
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
model_pipeline, test_metrics = train_model(train_df, test_df)

# ---------------------------------------------------------
# Barra Lateral (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=70)
    st.title("Sistema de Alerta de Deserción")
    st.markdown("**Proyecto de Sistemas Inteligentes (UPN)**")
    st.caption("UCI Dataset 697 | Modelo: Random Forest")

    menu = st.radio(
        "Navegación:",
        [
            "🔮 Simulador de Riesgo",
            "📊 Dashboard Exploratorio (EDA)",
            "📋 Diccionario de Variables",
            "⚙️ Métricas y Modelo (Test)"
        ],
        index=0
    )

    st.divider()
    st.markdown("### 📌 Acuerdos del Equipo")
    st.markdown("- **Split:** 80% Train / 20% Test")
    st.markdown("- **Semilla:** 42 (`SEED=42`)")
    st.markdown("- **Objetivo:** 1 = Dropout, 0 = Resto")
    st.markdown("- **Métrica Clave:** F1 > 0.78")

# ---------------------------------------------------------
# VISTA 1: SIMULADOR DE PREDICCIÓN EN TIEMPO REAL
# ---------------------------------------------------------
if menu == "🔮 Simulador de Riesgo":
    st.title("🔮 Simulador de Riesgo de Deserción Estudiantil")
    st.markdown(
        "Esta herramienta permite al tutor o consejero académico ingresar la información "
        "de un estudiante para estimar su **probabilidad de abandono** y activar alertas tempranas."
    )

    st.subheader("1. Cargar Perfil de Prueba Rápido (Opcional)")
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    # Perfil por defecto (mediana de train)
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
    st.subheader("2. Datos del Estudiante")

    col_ac1, col_ac2, col_soc = st.columns(3)

    with col_ac1:
        st.markdown("#### 📘 1er Semestre Académico")
        c1_enrolled = st.number_input("Materias matriculadas (1er Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 1st sem (enrolled)", 6)))
        c1_approved = st.number_input("Materias aprobadas (1er Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 1st sem (approved)", 5)))
        c1_grade = st.slider("Promedio de notas 1er Sem (0 a 20)", min_value=0.0, max_value=20.0, value=float(default_vals.get("Curricular units 1st sem (grade)", 12.5)), step=0.1)
        c1_evals = st.number_input("Evaluaciones rendidas 1er Sem", min_value=0, max_value=30, value=int(default_vals.get("Curricular units 1st sem (evaluations)", 6)))

    with col_ac2:
        st.markdown("#### 📗 2do Semestre Académico")
        c2_enrolled = st.number_input("Materias matriculadas (2do Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 2nd sem (enrolled)", 6)))
        c2_approved = st.number_input("Materias aprobadas (2do Sem)", min_value=0, max_value=25, value=int(default_vals.get("Curricular units 2nd sem (approved)", 5)))
        c2_grade = st.slider("Promedio de notas 2do Sem (0 a 20)", min_value=0.0, max_value=20.0, value=float(default_vals.get("Curricular units 2nd sem (grade)", 12.5)), step=0.1)
        c2_evals = st.number_input("Evaluaciones rendidas 2do Sem", min_value=0, max_value=30, value=int(default_vals.get("Curricular units 2nd sem (evaluations)", 6)))

    with col_soc:
        st.markdown("#### 💳 Factores Económicos y Personales")
        debtor = st.selectbox("¿Tiene deudas pendientes? (Debtor)", options=[0, 1], format_func=lambda x: "Sí (Con deudas)" if x == 1 else "No (Sin deuda)", index=int(default_vals.get("Debtor", 0)))
        tuition_ok = st.selectbox("¿Al día en pagos de pensión? (Tuition)", options=[1, 0], format_func=lambda x: "Sí (Al día)" if x == 1 else "No (Atrasado)", index=0 if default_vals.get("Tuition fees up to date", 1) == 1 else 1)
        scholarship = st.selectbox("¿Tiene Beca de Estudios?", options=[0, 1], format_func=lambda x: "Sí (Becado)" if x == 1 else "No (Sin beca)", index=int(default_vals.get("Scholarship holder", 0)))
        age = st.number_input("Edad al matricularse", min_value=17, max_value=70, value=int(default_vals.get("Age at enrollment", 20)))
        gender = st.selectbox("Género", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino", index=int(default_vals.get("Gender", 1)))
        admission_grade = st.number_input("Nota previa o de admisión (95-190)", min_value=95.0, max_value=190.0, value=float(default_vals.get("Admission grade", 125.0)), step=0.5)

    st.markdown("---")
    if st.button("🚀 Evaluar Riesgo del Estudiante", type="primary", use_container_width=True):
        # Crear vector con todas las 36 variables usando la base de medianas
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
        pred_label = model_pipeline.predict(input_df)[0]

        st.subheader("Resultado de la Predicción")
        m_col1, m_col2, m_col3 = st.columns([1, 1, 1.5])

        with m_col1:
            st.metric("Probabilidad de Deserción", f"{prob_dropout*100:.1f} %")

        with m_col2:
            if prob_dropout >= 0.60:
                st.error("🚨 RIESGO ALTO (CRÍTICO)")
            elif prob_dropout >= 0.35:
                st.warning("⚠️ RIESGO MODERADO")
            else:
                st.success("✅ RIESGO BAJO (SEGURO)")

        with m_col3:
            st.progress(prob_dropout)
            st.caption(f"Semáforo institucional: Umbral de alerta temprana en 35.0%")

        # Recomendaciones automáticas según los factores
        st.markdown("### 💡 Plan de Acción Recomendado")
        recomendaciones = []
        if debtor == 1 or tuition_ok == 0:
            recomendaciones.append("💰 **Apoyo Financiero:** El estudiante presenta deudas activas o cuotas impagas. Contactar con Bienestar Estudiantil para financiamiento o beca de emergencia.")
        if c1_approved < (c1_enrolled / 2) or c2_approved < (c2_enrolled / 2):
            recomendaciones.append("📚 **Tutoría Académica Urgente:** El estudiante ha desaprobado más del 50% de las asignaturas matriculadas. Asignar un tutor de reforzamiento.")
        if scholarship == 0 and prob_dropout > 0.40:
            recomendaciones.append("🎯 **Evaluación Socioeconómica:** Evaluar elegibilidad para asignación de beca o descuento por vulnerabilidad.")
        if age >= 25:
            recomendaciones.append("⏰ **Flexibilidad Horaria:** El estudiante ingresó con mayor edad promedio, lo que suele asociarse a responsabilidades laborales simultáneas.")
        if not recomendaciones:
            recomendaciones.append("👏 **Seguimiento Regular:** El estudiante mantiene un perfil regular y saludable. No requiere intervención extraordinaria.")

        for rec in recomendaciones:
            st.info(rec)

# ---------------------------------------------------------
# VISTA 2: DASHBOARD EXPLORATORIO (EDA)
# ---------------------------------------------------------
elif menu == "📊 Dashboard Exploratorio (EDA)":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración estadística del dataset original de la UCI con **4,424 estudiantes**.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Estudiantes", f"{len(raw_df):,}")
    kpi2.metric("Tasa de Deserción", "32.1 %", "1,421 desertores")
    kpi3.metric("Valores Nulos", "0", "100% completo")
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
        st.caption("Los estudiantes con tasas atrasadas (0) tienen una tasa de deserción superior al 85%.")

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
        st.caption("Los estudiantes con beca tienen una tasa de deserción significativamente menor (< 12%).")

    st.subheader("Explorador Interactivo del Dataset")
    target_filter = st.multiselect("Filtrar por estado:", options=raw_df[TARGET_COL].unique(), default=raw_df[TARGET_COL].unique())
    filtered_df = raw_df[raw_df[TARGET_COL].isin(target_filter)]
    st.dataframe(filtered_df.head(100), use_container_width=True)

# ---------------------------------------------------------
# VISTA 3: DICCIONARIO DE VARIABLES
# ---------------------------------------------------------
elif menu == "📋 Diccionario de Variables":
    st.title("📋 Diccionario de las 36 Variables Predictoras")
    st.markdown("Clasificación técnica y tratamiento para el preprocesamiento con Scikit-Learn.")

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
        ('Daytime/evening attendance', 'Binaria', 'Turno de asistencia (1=Diurno, 0=Nocturno)', 'Passthrough (0/1)'),
        ('Displaced', 'Binaria', 'Estudiante foráneo o desplazado (1=Sí, 0=No)', 'Passthrough (0/1)'),
        ('Educational special needs', 'Binaria', 'Necesidades educativas especiales (1=Sí, 0=No)', 'Passthrough (0/1)'),
        ('Debtor', 'Binaria', 'Pagos atrasados o deudas pendientes (1=Sí, 0=No)', 'Passthrough (0/1)'),
        ('Tuition fees up to date', 'Binaria', 'Al día en el pago de tasas de matrícula (1=Sí, 0=No)', 'Passthrough (0/1)'),
        ('Gender', 'Binaria', 'Género reportado (1=Masculino, 0=Femenino)', 'Passthrough (0/1)'),
        ('Scholarship holder', 'Binaria', 'Beneficiario de beca estudiantil (1=Sí, 0=No)', 'Passthrough (0/1)'),
        ('International', 'Binaria', 'Estudiante internacional extranjero (1=Sí, 0=No)', 'Passthrough (0/1)'),
        ('Application order', 'Numérica', 'Preferencia de orden de carrera elegida (0 a 9)', 'StandardScaler'),
        ('Age at enrollment', 'Numérica', 'Edad al momento de matricularse (17 a 70)', 'StandardScaler'),
        ('Previous qualification (grade)', 'Numérica', 'Calificación de educación previa (0-200)', 'StandardScaler'),
        ('Admission grade', 'Numérica', 'Nota del examen de admisión (0-200)', 'StandardScaler'),
        ('Curricular units 1st sem (credited)', 'Numérica', 'Créditos convalidados en 1er semestre', 'StandardScaler'),
        ('Curricular units 1st sem (enrolled)', 'Numérica', 'Cursos matriculados en 1er semestre', 'StandardScaler'),
        ('Curricular units 1st sem (evaluations)', 'Numérica', 'Evaluaciones rendidas en 1er semestre', 'StandardScaler'),
        ('Curricular units 1st sem (approved)', 'Numérica', 'Cursos aprobados en 1er semestre', 'StandardScaler'),
        ('Curricular units 1st sem (grade)', 'Numérica', 'Promedio de notas en 1er semestre (0-20)', 'StandardScaler'),
        ('Curricular units 1st sem (without evaluations)', 'Numérica', 'Cursos sin evaluación en 1er semestre', 'StandardScaler'),
        ('Curricular units 2nd sem (credited)', 'Numérica', 'Créditos convalidados en 2do semestre', 'StandardScaler'),
        ('Curricular units 2nd sem (enrolled)', 'Numérica', 'Cursos matriculados en 2do semestre', 'StandardScaler'),
        ('Curricular units 2nd sem (evaluations)', 'Numérica', 'Evaluaciones rendidas en 2do semestre', 'StandardScaler'),
        ('Curricular units 2nd sem (approved)', 'Numérica', 'Cursos aprobados en 2do semestre', 'StandardScaler'),
        ('Curricular units 2nd sem (grade)', 'Numérica', 'Promedio de notas en 2do semestre (0-20)', 'StandardScaler'),
        ('Curricular units 2nd sem (without evaluations)', 'Numérica', 'Cursos sin evaluación en 2do semestre', 'StandardScaler'),
        ('Unemployment rate', 'Numérica', 'Tasa de desempleo nacional al cursar (%)', 'StandardScaler'),
        ('Inflation rate', 'Numérica', 'Tasa de inflación anual (%)', 'StandardScaler'),
        ('GDP', 'Numérica', 'Crecimiento del PBI nacional (%)', 'StandardScaler')
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
# VISTA 4: MÉTRICAS Y MODELO (TEST EVALUATION)
# ---------------------------------------------------------
elif menu == "⚙️ Métricas y Modelo (Test)":
    st.title("⚙️ Evaluación del Modelo en Test (885 Estudiantes)")
    st.markdown("Rendimiento del clasificador con preprocesador integrado en el conjunto de prueba nunca antes visto.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("F1-Score (Dropout)", f"{test_metrics['f1']:.4f}", "Meta > 0.78")
    m2.metric("ROC-AUC", f"{test_metrics['roc_auc']:.4f}")
    m3.metric("Recall (Sensibilidad)", f"{test_metrics['recall']*100:.1f} %")
    m4.metric("Precision", f"{test_metrics['precision']*100:.1f} %")
    m5.metric("Accuracy", f"{test_metrics['accuracy']*100:.1f} %")

    st.divider()

    c_cm1, c_cm2 = st.columns([1, 1.2])

    with c_cm1:
        st.markdown("#### Matriz de Confusión")
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

    with c_cm2:
        st.markdown("#### Pipeline y Diagrama Metodológico")
        fig_p = FIGURES / "figura1_pipeline.png"
        if not fig_p.exists():
            from src.pipeline_diagram import main as make_diagram
            make_diagram()
        st.image(str(fig_p), use_container_width=True)

