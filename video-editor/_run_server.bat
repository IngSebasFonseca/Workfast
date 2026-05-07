@echo off
title Evse Server
cd /d "%~dp0"

echo.
echo ============================================
echo   EVSE SERVER
echo ============================================
echo.
echo Servidor de Evse corriendo en http://127.0.0.1:5000
echo Para apagar Evse: cierra esta ventana.
echo.
echo --------------------------------------------
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: No encuentro venv\Scripts\python.exe
    echo Cierra esta ventana y vuelve a abrir Abrir WorkFast.bat
    pause
    exit /b 1
)

if not exist "backend\main.py" (
    echo ERROR: No encuentro backend\main.py
    pause
    exit /b 1
)

:: Borrar TODOS los .pyc por las dudas (a veces rmdir falla)
del /s /q backend\*.pyc >nul 2>nul
del /s /q backend\__pycache__\*.* >nul 2>nul
del /s /q backend\video_processor\__pycache__\*.* >nul 2>nul

:: -B = no escribir bytecode (evita .pyc fantasma para siempre)
:: PYTHONDONTWRITEBYTECODE como segundo seguro
set PYTHONDONTWRITEBYTECODE=1
"%CD%\venv\Scripts\python.exe" -B backend\main.py

echo.
echo --------------------------------------------
echo.
echo El servidor se cerro o crasheo.
echo Revisa el mensaje de error que aparece arriba.
echo.
pause
exit /b 0
