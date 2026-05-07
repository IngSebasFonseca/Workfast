@echo off
setlocal enabledelayedexpansion
title Evse Video Studio - Lanzador

cd /d "%~dp0"

echo.
echo ============================================
echo   EVSE - VIDEO STUDIO
echo ============================================
echo.

:: --- 1. Cerrar servidores Python anteriores en el puerto 5000 ---
echo [1/5] Cerrando servidores anteriores y limpiando cache...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
    echo   - Cerrando PID %%a
    taskkill /f /pid %%a >nul 2>nul
)
timeout /t 1 /nobreak >nul

:: Borrar __pycache__ para forzar recompilacion (evita errores fantasma de versiones viejas)
if exist "backend\__pycache__" rmdir /s /q "backend\__pycache__" >nul 2>nul
if exist "backend\video_processor\__pycache__" rmdir /s /q "backend\video_processor\__pycache__" >nul 2>nul

:: --- 2. Verificar Python ---
echo [2/5] Verificando Python...
where py >nul 2>nul
if not errorlevel 1 (
    py -3.10 -V >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BASE=py -3.10"
        echo   Usando Python 3.10 via launcher.
    ) else (
        set "PYTHON_BASE=python"
        echo   Python 3.10 no encontrado via launcher, usando predeterminado.
    )
) else (
    set "PYTHON_BASE=python"
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo ERROR: Python no esta instalado o no esta en PATH.
        echo Descarga desde https://www.python.org/downloads/
        echo IMPORTANTE: marca "Add Python to PATH" durante la instalacion.
        echo.
        pause
        exit /b 1
    )
)

:: --- 3. Verificar FFmpeg ---
echo [3/5] Verificando FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: FFmpeg no esta instalado o no esta en PATH.
    pause
    exit /b 1
)
where ffprobe >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: ffprobe no esta instalado o no esta en PATH.
    pause
    exit /b 1
)

:: --- 4. Preparar entorno virtual ---
echo [4/5] Preparando entorno...
if not exist "venv\Scripts\python.exe" (
    echo   Primera vez: creando entorno virtual...
    %PYTHON_BASE% -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "ENV_KEY=%%A"
        set "ENV_VALUE=%%B"
        if not "!ENV_KEY!"=="" if not "!ENV_KEY:~0,1!"=="#" set "!ENV_KEY!=!ENV_VALUE!"
    )
)

echo   Verificando dependencias...
"%PYTHON_EXE%" -c "import flask, flask_cors, yt_dlp, faster_whisper, deep_translator, whisperx, df" >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Faltan dependencias. Instalando ^(2-5 min la primera vez^)...
    echo.
    rem Upgrade solo pip; setuptools y wheel quedan compatibles con torch/deepfilternet
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo ERROR: No se pudo actualizar pip.
        pause
        exit /b 1
    )
    rem Pre-fijar versiones criticas para evitar el ping-pong de packaging/setuptools
    "%PYTHON_EXE%" -m pip install "setuptools<82" "packaging>=23.0,<24.0" "wheel<0.45"
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: No se pudieron instalar las dependencias.
        pause
        exit /b 1
    )
    echo   Dependencias instaladas.
)

:: --- 5. Arrancar servidor en su propia ventana ---
echo [5/5] Arrancando servidor...
echo.
echo Se abrira una ventana llamada "Evse Server".
echo Para apagar Evse: cierra esa ventana.
echo.

:: Lanzar el worker en una ventana NUEVA. El worker se encarga del Python.
start "Evse Server" "%CD%\_run_server.bat"

:: Esperar a que el server responda
echo Esperando a que el servidor arranque...
set /a tries=0
:waitloop
set /a tries+=1
timeout /t 1 /nobreak >nul
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:5000/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch { exit 1 }; exit 1" >nul 2>nul
if not errorlevel 1 goto :ready
if !tries! lss 25 goto :waitloop

echo.
echo ============================================
echo ADVERTENCIA: El servidor no respondio.
echo ============================================
echo.
echo Revisa la ventana "Evse Server" para ver el error.
echo Si dice algo como "ModuleNotFoundError" o similar,
echo borra la carpeta venv y vuelve a abrir este archivo.
echo.
pause
exit /b 1

:ready
echo.
echo Servidor listo en http://127.0.0.1:5000
echo Abriendo navegador...
start "" "http://127.0.0.1:5000"
timeout /t 2 /nobreak >nul
exit /b 0
