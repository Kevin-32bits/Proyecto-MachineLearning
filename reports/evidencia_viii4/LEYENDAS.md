# Evidencias VIII.4: resultado final

Capturas reales del producto mínimo viable, ejecutado en local con el modelo `models/pipeline_ganador.joblib` y umbral de alerta 0.40.

## Producto mínimo viable: aplicación web

| Archivo | Leyenda sugerida |
|---|---|
| `01_inicio.png` | Pantalla de inicio: propósito del sistema, indicadores principales, funcionamiento y lectura del semáforo. |
| `02_simulador_riesgo_alto.png` | Simulador con perfil de riesgo alto: probabilidad de 96.2 %, semáforo rojo, factores influyentes y acciones sugeridas. |
| `03_simulador_riesgo_moderado.png` | Simulador con perfil de riesgo moderado: probabilidad de 45.8 % y semáforo amarillo. |
| `04_simulador_riesgo_bajo.png` | Simulador con perfil de riesgo bajo: probabilidad de 2.7 % y semáforo verde. |
| `05_simulador_validacion.png` | Validación de datos: la aplicación rechaza valores fuera de rango o incoherentes y explica el error. |
| `06_lote_885_estudiantes.png` | Evaluación por lotes de los 885 estudiantes de prueba: 227 en rojo, 60 en amarillo y 598 en verde. |
| `08_datos.png` | Análisis exploratorio: tasa de abandono por grupo de estudiantes frente al promedio general de 32.1 %. |
| `09_comparacion_modelos.png` | Comparación de los cuatro modelos por F1 en validación cruzada, con la advertencia de empate técnico. |
| `10_resultados_modelo.png` | Resultados sobre el conjunto de prueba, cumplimiento de metas y detalle técnico. |
| `11_diccionario.png` | Diccionario de las 36 variables predictoras y su preprocesamiento. |

## Prueba: mismas filas en la app y en el notebook

| Archivo | Leyenda sugerida |
|---|---|
| `07_app_10_filas_test.png` | Las 10 filas de prueba evaluadas en la aplicación, identificadas por ID. |
| `14_notebook_10_filas_vs_app.png` | Las mismas 10 filas en el notebook: probabilidad y semáforo coinciden con la aplicación en todos los casos. |
| `15_notebook_885_filas_vs_app.png` | Comparación completa: las 885 filas de prueba dan igual probabilidad y semáforo en la aplicación y en el notebook. |

## Modelo y algoritmo

| Archivo | Leyenda sugerida |
|---|---|
| `12_notebook_modelo.png` | Modelo cargado por la aplicación: pipeline de preprocesamiento y Gradient Boosting con sus hiperparámetros. |
| `13_notebook_metricas_test.png` | Métricas en el conjunto de prueba con umbrales 0.50 y 0.40, calculadas directamente en el notebook. |

## Notas para el informe

- Las 10 filas se fijaron sin mirar probabilidades: primeras 5 con `dropout = 1` y primeras 5 con `dropout = 0`. El 100 % de `07_app_10_filas_test.png` corresponde solo a esa muestra y no mide el rendimiento del modelo. Para eso, usar `10_resultados_modelo.png`.
- Las probabilidades se comparan redondeadas a 4 decimales, como las exporta la aplicación.
- Los resultados de la aplicación están en `resultados_app_10.csv` y `resultados_app_885.csv`. Se reproducen con `notebooks/03_p5_verificacion_app.ipynb`.
- Si cambia el umbral operativo, hay que repetir estas capturas: dependen de él el semáforo y las métricas con umbral activo.
