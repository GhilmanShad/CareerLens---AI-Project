@echo off
setlocal
cd /d "%~dp0"
title CareerLens

where py >nul 2>&1
if not errorlevel 1 (
    py -3.14 -m streamlit run app.py
) else (
    python -m streamlit run app.py
)

if errorlevel 1 (
    echo.
    echo CareerLens could not start. Install requirements once with:
    echo py -3.14 -m pip install -r requirements.txt
    pause
)
