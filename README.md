# Predicción de la deserción universitaria (Dropout vs resto)

Proyecto final – Sistemas Inteligentes y Machine Learning (UPN).
Dataset: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI, CC BY 4.0).

📢 ACUERDOS CLAVE DEL EQUIPO (P1 ➔ P2, P3, P4 y P5)

Para que todos los modelos compitan en igualdad de condiciones y sin fuga de datos, se fijan estas 8 decisiones:

1. 🎯 OBJETIVO BINARIO
   • Variable: dropout
   • 1 = abandonó (Target == 'Dropout')
   • 0 = continúa o se graduó (Enrolled + Graduate)

2. 📊 PARTICIÓN
   • 80 % train (3,539) / 20 % test (885)
   • Estratificada: conserva el 32.1 % de dropout en ambos sets

3. 🎲 SEMILLA ÚNICA
   • SEED = 42 en la partición y en todos los modelos (random_state=42)

4. ⚖️ BALANCEO DE CLASES
   • Se usa class_weight="balanced" (en Gradient Boosting, pesos por muestra equivalentes)
   • Se aplica SOLO al entrenar. El test no se balancea ni se modifica

5. 🏆 ELECCIÓN DEL MODELO GANADOR
   • Métrica principal: F1-score de la clase dropout=1
   • Desempate: ROC-AUC (se reporta también en 4.3)
   • Método: validación cruzada de 5 folds estratificada, SOLO sobre train
   • Accuracy, precision y recall se reportan como métricas complementarias

6. 🎚️ UMBRAL DE DECISIÓN
   • El umbral del modelo ganador se elige con train (predicciones de la validación cruzada)
   • No se ajusta mirando el test

7. ✅ CRITERIOS DE ÉXITO (sección 1.4, sobre test, clase Dropout)
   • Accuracy > 75 %  |  Precision > 75 %  |  Recall > 80 %  |  F1 > 0.78
   • Si no se cumplen, se reporta tal cual y se analiza en 4.3.c. No se modifican los criterios

8. 📂 Dónde está el Dataset:
   • Para usarlo directo en Python:
       from src.data_prep import get_train_test
       X_train, X_test, y_train, y_test = get_train_test()
   • Archivos CSV listos: `data/processed/train.csv` y `data/processed/test.csv`
   • Dataset original: `data/raw/data.csv` (UCI Repo ID: 697)

⚠️ REGLA DE ORO:
El preprocesamiento (escalado, one-hot encoding) debe ajustarse ÚNICAMENTE con los datos de train (usar `Pipeline` de scikit-learn). El conjunto de test se evalúa UNA SOLA VEZ al final.

## Cómo usar (P2 y P3)
```python
from src.data_prep import get_train_test
X_train, X_test, y_train, y_test = get_train_test()
```

## Puesta en marcha
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.data_prep         # descarga el dataset, hace el split y genera la evidencia
python -m src.pipeline_diagram  # genera reports/figures/figura1_pipeline.png
streamlit run app.py            # inicia la aplicación interactiva en el navegador
```

## Estructura
```
app.py                   aplicación web interactiva en Streamlit
src/config.py            parámetros compartidos
src/preprocessor.py      objeto ColumnTransformer y clasificación de variables
src/data_prep.py         carga, objetivo binario y split
src/pipeline_diagram.py  Figura 1 del informe
notebooks/               notebooks de P2 y P3
reports/                 evidencias y figuras para el informe
```
