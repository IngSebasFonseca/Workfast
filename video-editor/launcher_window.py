"""Evse Video Studio - launcher con ventana nativa (WebView2).

Flow optimizado para sentirse rapido:
  1. Abre la ventana INSTANTANEAMENTE con un splash de carga.
  2. En background lanza el server Flask y espera a que responda.
  3. Cuando esta listo, navega la ventana a la app real.

Tambien:
  - Setea AppUserModelID para que la barra de tareas no diga "Python".
  - Setea el icono de la ventana (turtle.ico) via win32 API.
  - Anade el FFmpeg bundleado al PATH antes de spawnar el server.
  - Mata el server cuando se cierra la ventana.

pythonw.exe (sin consola) corre este script, asi que no se ve ninguna
CMD. Errores se muestran como MessageBox de Windows + log a archivo.
"""
from __future__ import annotations

import atexit
import ctypes
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _resolve_python() -> Path:
    """venv local (copia de desarrollo) si existe; sino el estable por-usuario
    que crea el instalador en %LOCALAPPDATA%\\EvseVideoStudio\\venv."""
    local = ROOT / "venv" / "Scripts" / "python.exe"
    if local.exists():
        return local
    stable = Path(os.environ.get("LOCALAPPDATA", "")) / "EvseVideoStudio" / "venv" / "Scripts" / "python.exe"
    return stable


PYTHON_EXE = _resolve_python()
SERVER_SCRIPT = ROOT / "backend" / "main.py"
SERVER_LOG = ROOT / "evse_server.log"
LAUNCHER_LOG = ROOT / "evse_launcher.log"
ICON_PATH = ROOT / "turtle.ico"
URL = "http://127.0.0.1:5000"
WINDOW_TITLE = "Evse Video Studio"
APP_USER_MODEL_ID = "SebastianFonseca.EvseVideoStudio.1"

SPLASH_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; background: #0b0b0d; }
  body {
    color: #fff;
    font-family: -apple-system, "Segoe UI", "Inter", sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    user-select: none;
  }
  .spinner {
    width: 56px; height: 56px;
    border: 4px solid rgba(139,92,246,.18);
    border-top-color: #8b5cf6;
    border-radius: 50%;
    animation: spin .9s linear infinite;
    margin-bottom: 28px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .title {
    font-size: 20px; font-weight: 700; letter-spacing: -.01em;
    color: #fff;
  }
  .sub {
    font-size: 12.5px; opacity: .55; margin-top: 10px;
    letter-spacing: .02em;
  }
</style>
</head>
<body>
  <div class="spinner"></div>
  <div class="title">Evse Video Studio</div>
  <div class="sub">Cargando...</div>
</body>
</html>"""

_server_proc: subprocess.Popen | None = None
_server_log_fp = None  # handle del log del server; se cierra al apagar


# ---------------------------------------------------------------------
#  Logging y errores visibles (pythonw no muestra stdout)
# ---------------------------------------------------------------------
def show_error(title: str, msg: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, msg, title, 0x10)
    except Exception:
        pass


def log(msg: str) -> None:
    try:
        with open(LAUNCHER_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------
#  Identidad de app en Windows (taskbar + alt-tab)
# ---------------------------------------------------------------------
def setup_app_identity() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        log(f"AppUserModelID seteado: {APP_USER_MODEL_ID}")
    except Exception as exc:
        log(f"No pude setear AppUserModelID: {exc}")


def set_window_icon_async() -> None:
    """En thread aparte: encuentra la ventana y le mete turtle.ico via WM_SETICON."""
    if sys.platform != "win32" or not ICON_PATH.exists():
        return

    def worker() -> None:
        LR_LOADFROMFILE = 0x00000010
        IMAGE_ICON = 1
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        ico_str = str(ICON_PATH)
        hwnd = 0
        # La ventana puede tardar hasta ~2s en aparecer
        for _ in range(60):
            hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)
            if hwnd:
                break
            time.sleep(0.1)
        if not hwnd:
            log("No encontre la ventana para meterle el icono")
            return
        try:
            hi_small = ctypes.windll.user32.LoadImageW(
                None, ico_str, IMAGE_ICON, 16, 16, LR_LOADFROMFILE
            )
            hi_big = ctypes.windll.user32.LoadImageW(
                None, ico_str, IMAGE_ICON, 32, 32, LR_LOADFROMFILE
            )
            if hi_small:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hi_small)
            if hi_big:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hi_big)
            log("Icono custom aplicado a la ventana")
        except Exception as exc:
            log(f"Fallo seteando icono: {exc}")

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------
#  FFmpeg bundleado en PATH
# ---------------------------------------------------------------------
def prepend_bundled_ffmpeg_to_path() -> None:
    bundled = ROOT / "ffmpeg"
    if bundled.is_dir() and (bundled / "ffmpeg.exe").exists():
        current = os.environ.get("PATH", "")
        if str(bundled).lower() not in current.lower():
            os.environ["PATH"] = f"{bundled}{os.pathsep}{current}"
            log(f"FFmpeg bundleado anadido al PATH: {bundled}")


# ---------------------------------------------------------------------
#  Server Flask como subproceso oculto
# ---------------------------------------------------------------------
def start_server() -> None:
    global _server_proc, _server_log_fp
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"No encuentro Python en {PYTHON_EXE}")
    if not SERVER_SCRIPT.exists():
        raise FileNotFoundError(f"No encuentro {SERVER_SCRIPT}")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    _server_log_fp = open(SERVER_LOG, "w", encoding="utf-8", errors="replace")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    _server_proc = subprocess.Popen(
        [str(PYTHON_EXE), "-B", str(SERVER_SCRIPT)],
        cwd=str(ROOT),
        stdout=_server_log_fp,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=creationflags,
    )
    log(f"Server arrancado, PID {_server_proc.pid}")


def stop_server() -> None:
    global _server_log_fp
    if _server_proc is not None and _server_proc.poll() is None:
        try:
            _server_proc.terminate()
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        except Exception:
            pass
    # Cerrar el handle del log para no dejarlo abierto.
    if _server_log_fp is not None:
        try:
            _server_log_fp.close()
        except Exception:
            pass
        _server_log_fp = None


def wait_for_server(timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _server_proc and _server_proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(URL + "/api/health", timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------------
#  Main flow
# ---------------------------------------------------------------------
def main() -> int:
    try:
        atexit.register(stop_server)
        setup_app_identity()
        prepend_bundled_ffmpeg_to_path()

        try:
            import webview  # type: ignore
        except ImportError:
            log("pywebview no instalado, abriendo en navegador")
            start_server()
            if wait_for_server():
                webbrowser.open(URL)
                log("Servidor corriendo en segundo plano. Cierra esta ventana (o presiona Ctrl+C) para detenerlo.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            return 0

        # ── Habilitar descargas en WebView2 ──────────────────────────
        # Por defecto WebView2 bloquea descargas silenciosamente. Esto hace
        # que aparezca el dialogo nativo de Windows "Guardar como" cuando
        # el usuario clickea Descargar en la app.
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True
        except Exception as exc:
            log(f"No pude habilitar ALLOW_DOWNLOADS: {exc}")

        # Crear ventana con splash (aparece instantaneamente)
        window = webview.create_window(
            title=WINDOW_TITLE,
            html=SPLASH_HTML,
            width=1400,
            height=900,
            min_size=(1000, 600),
            background_color="#0b0b0d",
        )

        # Mientras se muestra el splash: setear icono + arrancar server en thread
        # aparte (sino bloquea el event loop de WebView2 y el spinner se congela).
        def background_work() -> None:
            try:
                start_server()
                if wait_for_server():
                    window.load_url(URL)
                    log("Server listo, ventana navegada al app")
                else:
                    window.load_html(
                        '<html><body style="background:#0b0b0d;color:#f87171;'
                        'font-family:sans-serif;padding:60px">'
                        '<h1>Error iniciando el servidor</h1>'
                        '<p>Revisa evse_server.log</p>'
                        '</body></html>'
                    )
            except Exception as exc:
                log(f"Error en background_work: {exc}\n{traceback.format_exc()}")
                window.load_html(
                    f'<html><body style="background:#0b0b0d;color:#f87171;'
                    f'font-family:sans-serif;padding:60px">'
                    f'<h1>Error iniciando</h1><pre>{exc}</pre>'
                    f'</body></html>'
                )

        def on_window_ready() -> None:
            set_window_icon_async()
            threading.Thread(target=background_work, daemon=True).start()

        webview.start(on_window_ready)
        return 0

    except Exception as exc:
        log(f"FATAL: {exc}\n{traceback.format_exc()}")
        show_error(
            "Evse Video Studio - Error",
            f"Error iniciando el programa:\n\n{exc}\n\nDetalle en:\n{LAUNCHER_LOG}",
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
