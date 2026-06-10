@echo off
:: ===================================================================
:: _setup_env.bat - Se ejecuta durante el setup (Inno, runhidden).
:: ===================================================================
:: Crea/usa el venv y instala dependencias. Salida -> install.log.
::
:: CLAVE: el venv vive en %LOCALAPPDATA%\EvseVideoStudio\venv (ubicacion
:: ESTABLE, fuera de la carpeta del programa). Asi SOBREVIVE a
:: desinstalar/reinstalar: se descarga torch+CUDA UNA sola vez por PC.
:: Ademas, si las dependencias ya estan instaladas, OMITE la descarga
:: por completo (chequeo de import). Reinstalar = instantaneo.
:: ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

:: venv estable por-usuario (sobrevive reinstalaciones)
set "VENV_DIR=%LOCALAPPDATA%\EvseVideoStudio\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo. > install.log
echo === Evse Video Studio - Setup de entorno === >> install.log
echo Fecha: %date% %time% >> install.log
echo App dir: %CD% >> install.log
echo Venv dir: %VENV_DIR% >> install.log
echo. >> install.log

:: --- 0. Si el venv ya existe y tiene TODAS las dependencias, salir ya ---
if exist "%VENV_PY%" (
    echo [0] venv existente detectado, verificando dependencias... >> install.log
    "%VENV_PY%" -c "import torch,torchaudio,flask,flask_cors,yt_dlp,faster_whisper,ctranslate2,deep_translator,soundfile,pedalboard,parselmouth,PIL,webview" >> install.log 2>&1
    if not errorlevel 1 (
        echo TODAS las dependencias ya estan instaladas. Omitiendo descarga. >> install.log
        echo === SETUP OK (reuso de venv, sin descarga) === >> install.log
        exit /b 0
    )
    echo Faltan dependencias o cambiaron; se completara la instalacion. >> install.log
)

:: --- 1. Encontrar Python ---
echo [1/4] Buscando Python... >> install.log
set "PYTHON_BASE="

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -V >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BASE=py -3.11"
        py -3.11 -V >> install.log 2>&1
        goto :py_found
    )
    py -3.10 -V >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BASE=py -3.10"
        py -3.10 -V >> install.log 2>&1
        goto :py_found
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python -V >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BASE=python"
        python -V >> install.log 2>&1
        goto :py_found
    )
)

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_BASE=%LocalAppData%\Programs\Python\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe" -V >> install.log 2>&1
    goto :py_found
)

echo ERROR: No encontre Python ni en PATH ni en el bundle. >> install.log
exit /b 10

:py_found
echo Usando: %PYTHON_BASE% >> install.log
echo. >> install.log

:: --- 2. Crear venv estable (si no existe) ---
echo [2/4] Preparando venv en %VENV_DIR% ... >> install.log
if not exist "%LOCALAPPDATA%\EvseVideoStudio" mkdir "%LOCALAPPDATA%\EvseVideoStudio" >nul 2>nul
if exist "%VENV_PY%" (
    echo venv ya existe, reutilizando. >> install.log
) else (
    %PYTHON_BASE% -m venv "%VENV_DIR%" >> install.log 2>&1
    if errorlevel 1 (
        echo ERROR creando venv. >> install.log
        exit /b 20
    )
)
if not exist "%VENV_PY%" (
    echo ERROR: %VENV_PY% no aparecio tras crear el venv. >> install.log
    exit /b 21
)
echo venv OK. >> install.log
echo. >> install.log

:: --- 3. pip + base packages ---
echo [3/4] Actualizando pip + setuptools + packaging + wheel... >> install.log
"%VENV_PY%" -m pip install --upgrade pip >> install.log 2>&1
if errorlevel 1 (
    echo ERROR actualizando pip. >> install.log
    exit /b 30
)
"%VENV_PY%" -m pip install "setuptools<82" "packaging>=23.0,<24.0" "wheel<0.45" >> install.log 2>&1
if errorlevel 1 (
    echo ERROR instalando base packages. >> install.log
    exit /b 31
)
echo. >> install.log

:: --- 4. requirements.txt (torch+CUDA: solo si falta algo. pip usa su cache) ---
echo [4/4] Instalando dependencias (solo descarga lo que falte; usa cache de pip)... >> install.log
"%VENV_PY%" -m pip install -r requirements.txt >> install.log 2>&1
if errorlevel 1 (
    echo ERROR instalando requirements.txt. >> install.log
    exit /b 40
)
echo. >> install.log

echo === SETUP DE ENTORNO COMPLETADO OK === >> install.log
exit /b 0
