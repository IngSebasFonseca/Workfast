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
echo [1/5] Cerrando servidores anteriores y procesos zombis...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
    echo   - Liberando puerto 5000 (PID %%a)
    taskkill /f /pid %%a >nul 2>nul
)
echo   - Limpiando procesos en segundo plano...
wmic process where "ExecutablePath like '%%video-editor\\\\venv%%'" call terminate >nul 2>nul
wmic process where "CommandLine like '%%--remote-debugging-port=9222%%'" call terminate >nul 2>nul
timeout /t 1 /nobreak >nul

:: Borrar __pycache__ para forzar recompilacion (evita errores fantasma de versiones viejas)
if exist "backend\__pycache__" rmdir /s /q "backend\__pycache__" >nul 2>nul
if exist "backend\video_processor\__pycache__" rmdir /s /q "backend\video_processor\__pycache__" >nul 2>nul

:: --- 2. Verificar Python ---
echo [2/5] Verificando Python...
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -V >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BASE=py -3.11"
        echo   Usando Python 3.11 via launcher.
    ) else (
        py -3.10 -V >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_BASE=py -3.10"
            echo   Usando Python 3.10 via launcher.
        ) else (
            set "PYTHON_BASE=python"
            echo   Python 3.11/3.10 no encontrado via launcher, usando predeterminado.
        )
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

%PYTHON_BASE% -V >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python aparece en PATH pero no arranca bien.
    echo En esta PC suele pasar cuando Windows usa el acceso directo de Microsoft Store.
    echo Instala Python 3.10 o 3.11 desde https://www.python.org/downloads/
    echo y marca "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)

:: --- 3. Verificar FFmpeg ---
echo [3/5] Verificando FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    :: No esta en PATH global. Si el instalador dejo ffmpeg bundleado en la
    :: carpeta del programa, lo agregamos a PATH solo para esta sesion.
    if exist "ffmpeg\ffmpeg.exe" (
        echo   - Usando FFmpeg bundleado ^(carpeta ffmpeg\^)
        set "PATH=%CD%\ffmpeg;%PATH%"
    ) else (
        echo.
        echo ERROR: FFmpeg no esta instalado o no esta en PATH.
        echo Descarga desde https://www.gyan.dev/ffmpeg/builds/
        echo.
        pause
        exit /b 1
    )
)
where ffprobe >nul 2>nul
if errorlevel 1 (
    if exist "ffmpeg\ffprobe.exe" (
        :: ffmpeg.exe encontrado pero no ffprobe en PATH - usar el bundle
        set "PATH=%CD%\ffmpeg;%PATH%"
    ) else (
        echo.
        echo ERROR: ffprobe no esta instalado o no esta en PATH.
        pause
        exit /b 1
    )
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
) else (
    "venv\Scripts\python.exe" -V >nul 2>nul
    if errorlevel 1 (
        echo   Entorno virtual roto o copiado desde otra PC. Recreando venv...
        rmdir /s /q venv >nul 2>nul
        %PYTHON_BASE% -m venv venv
        if errorlevel 1 (
            echo.
            echo ERROR: No se pudo recrear el entorno virtual.
            echo Instala Python 3.10/3.11 real desde python.org y marca Add Python to PATH.
            pause
            exit /b 1
        )
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
"%PYTHON_EXE%" -c "import flask, flask_cors, yt_dlp, faster_whisper, ctranslate2, deep_translator, df, soundfile, pedalboard, parselmouth, PIL" >nul 2>nul
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

:: --- 5. Abrir ventana nativa con la app (cierra el server al cerrar) ---
echo [5/5] Abriendo Evse Video Studio...
echo.

:: launcher_window.py:
::   - levanta el server Flask como subproceso sin ventana (log a evse_server.log)
::   - espera /api/health
::   - abre ventana nativa WebView2 (pywebview). Si WebView2 no esta, cae a navegador.
::   - al cerrar la ventana, mata el server.
::
:: Bloquea esta CMD hasta que el usuario cierre la ventana. Si Evse.vbs
:: la lanzo oculta, el usuario solo ve la ventana nativa - ni una CMD.
"%PYTHON_EXE%" -B launcher_window.py
exit /b %errorlevel%
