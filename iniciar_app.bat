@echo off
title Sistema de Prediccion de Desercion - Streamlit
echo ========================================================
echo   Iniciando Aplicacion Web Streamlit...
echo ========================================================
echo.

cd /d "%~dp0"
python -m streamlit run app.py
pause

