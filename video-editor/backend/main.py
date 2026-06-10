from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True  # No generar .pyc (evita errores fantasma de versiones viejas)

import threading
import time
import uuid
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import re
import socket
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse
import shutil
import urllib.request

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from yt_dlp import YoutubeDL


def _preload_env_file() -> None:
    """Load .env before importing video/audio processors that read env at import."""
    base_dir = Path(__file__).resolve().parents[1]
    for env_path in (base_dir.parent / ".env", base_dir / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.getenv(key):
                os.environ[key] = value


_preload_env_file()

from video_processor import (
    RenderCancelled,
    TranscriberError,
    VideoEditor,
    cleanup_cache,
    generate_subtitles,
)
from video_processor.enhancer import enhance_video, EnhancerError, normalize_enhancer_profile
from video_processor.transcriber import describe_backend
from video_processor.audio_enhancer import voice_backend_status


BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_FOLDER = BASE_DIR / "assets" / "uploads"
OUTPUT_FOLDER = BASE_DIR / "assets" / "outputs"
LIBRARY_FOLDER = BASE_DIR / "assets" / "library"
SUBTITLE_CACHE_FOLDER = LIBRARY_FOLDER / "subtitle_cache"
YOUTUBE_COOKIES_FILE = LIBRARY_FOLDER / "youtube_cookies.txt"
FACEBOOK_COOKIES_FILE = LIBRARY_FOLDER / "facebook_cookies.txt"
YOUTUBE_BROWSER_PROFILE = LIBRARY_FOLDER / "youtube_chrome_profile"
PROFILES_FILE = LIBRARY_FOLDER / "profiles.json"
FB_ACCOUNTS_FILE = LIBRARY_FOLDER / "fb_accounts.json"
FB_QUEUE_FILE = LIBRARY_FOLDER / "fb_queue.json"
HUNTER_CHANNELS_FILE = LIBRARY_FOLDER / "hunter_channels.json"
HUNTER_SEED_FILE = LIBRARY_FOLDER / "hunter_seed.json"

# Slots diarios fijos para publicación en Facebook (hora local)
FB_DAILY_SLOTS = [(9, 0), (13, 0), (19, 0)]
YOUTUBE_DEBUG_PORT = 9222

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "png", "jpg", "jpeg", "webp"}
ASSET_TYPES = {"logo", "follow", "ending"}
YOUTUBE_COOKIE_BROWSERS = {
    "auto": None,
    "none": None,
    "file": "file",
    "edge": "edge",
    "chrome": "chrome",
    "firefox": "firefox",
    "brave": "brave",
}
YOUTUBE_AUTO_COOKIE_SEQUENCE = [None, "file"]
YOUTUBE_DOWNLOAD_FORMATS = [
    # Cada selector exige audio: bv*+ba ya garantiza audio porque "+ba"
    # falla si no hay audio. Los fallbacks "b" y "best" llevan [acodec!=none]
    # para evitar quedarse con un stream video-only (problema tipico de Shorts
    # cuya version HEVC viene sin pista de audio).
    "bv*[height<=1080][fps<=60][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080][fps<=60]+ba/bv*[height<=1080]+ba/b[height<=1080][acodec!=none]/best[height<=1080][acodec!=none]/best[acodec!=none]",
    "bv*+ba/b[acodec!=none]",
    "best[acodec!=none]",
    None,
]

OUTPUT_LRU_KEEP = 500  # Antes 60 -- borraba renders agendados en FB que no
                       # estaban localmente. Subido a 500 para que un mes de
                       # programacion (4 vids/dia x 30 dias = 120) no se pierda.
JOB_RETENTION_SECONDS = 60 * 60 * 12  # keep finished jobs for 12 hours


def load_env_file(override: bool = False) -> None:
    for env_path in (BASE_DIR.parent / ".env", BASE_DIR / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or not os.getenv(key):
                os.environ[key] = value


load_env_file()

# Silenciar el warning de Flask sobre python-dotenv (ya tenemos parser propio)
os.environ.setdefault("FLASK_SKIP_DOTENV", "1")

app = Flask(__name__, static_folder=None)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 2_000 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
LIBRARY_FOLDER.mkdir(parents=True, exist_ok=True)
SUBTITLE_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
YOUTUBE_BROWSER_PROFILE.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
debug_browser_process: subprocess.Popen | None = None

# ── Cancelacion de renders ───────────────────────────────────────────
# _job_cancel: por job_id, un Event que se setea cuando el usuario cancela.
# _job_procs: por job_id, los subprocesos FFmpeg activos (para matarlos).
# El worker y VideoEditor consultan/registran via los helpers de abajo.
_job_cancel: dict[str, threading.Event] = {}
_job_procs: dict[str, list] = {}
_job_proc_lock = threading.Lock()

# Serializa los renders: hasta 2 en paralelo comparten la GPU (NVENC + Whisper).
# Con RTX 5070 (12GB VRAM) dos sesiones h264_nvenc de 1080p usan ~2-3GB de VRAM
# en total — caben bien. Si Whisper tambien corre en paralelo puede haber presion
# pero no fallo. Reducir a 1 con WORKFAST_MAX_PARALLEL_RENDERS=1 en .env si hay
# problemas de VRAM, o subir a 3 si la GPU es mas grande.
_MAX_PARALLEL_RENDERS = max(1, int(os.getenv("WORKFAST_MAX_PARALLEL_RENDERS", "2")))
_RENDER_SEM = threading.BoundedSemaphore(_MAX_PARALLEL_RENDERS)

# Limita descargas de YouTube/TikTok/FB concurrentes. Con muchos videos
# seleccionados al mismo tiempo, yt-dlp lanza muchos procesos y Flask puede
# saturarse causando "Failed to fetch" en el cliente. 3 en paralelo es el
# balance optimo entre velocidad y estabilidad.
_MAX_PARALLEL_DOWNLOADS = max(1, int(os.getenv("WORKFAST_MAX_PARALLEL_DOWNLOADS", "3")))
_DOWNLOAD_SEM = threading.BoundedSemaphore(_MAX_PARALLEL_DOWNLOADS)


def _register_job_process(job_id: str, proc) -> None:
    """VideoEditor llama esto cuando arranca un FFmpeg, para poder matarlo."""
    with _job_proc_lock:
        _job_procs.setdefault(job_id, []).append(proc)


def _job_is_cancelled(job_id: str) -> bool:
    ev = _job_cancel.get(job_id)
    return ev is not None and ev.is_set()


def _begin_cancellable_job(job_id: str) -> None:
    _job_cancel[job_id] = threading.Event()


def _clear_job_tracking(job_id: str) -> None:
    with _job_proc_lock:
        _job_procs.pop(job_id, None)
    _job_cancel.pop(job_id, None)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    for prefix in ("video_", "youtube_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem.replace("_", " ").replace("-", " ").strip() or "Mi Video"


def output_name_from_title(title: str, fallback: str = "video") -> str:
    clean = re.sub(r"[^\w\s.-]", " ", title or "", flags=re.UNICODE)
    clean = re.sub(r"\s+", " ", clean).strip()
    filename = secure_filename(clean)[:90].strip("._-")
    if not filename:
        filename = secure_filename(fallback)[:90].strip("._-") or "video"
    return filename


def asset_dir(asset_type: str) -> Path:
    safe_type = secure_filename(asset_type)
    if safe_type not in ASSET_TYPES:
        raise ValueError("Tipo de recurso no valido.")
    folder = LIBRARY_FOLDER / safe_type
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def allowed_asset_file(filename: str, asset_type: str) -> bool:
    if not allowed_file(filename):
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    if asset_type in {"logo", "follow"}:
        return ext in {"png", "jpg", "jpeg", "webp"}
    if asset_type == "ending":
        return ext in {"mp4", "mov", "mkv", "avi"}
    return False


def serialize_asset(path: Path, asset_type: str) -> dict:
    return {
        "type": asset_type,
        "filename": path.name,
        "filepath": str(path),
        "preview_url": f"/api/assets/file/{asset_type}/{quote(path.name)}",
        "size": path.stat().st_size,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "is_default": path.name.startswith("default_"),
    }


def list_library_assets(asset_type: str | None = None) -> dict:
    types = [asset_type] if asset_type else sorted(ASSET_TYPES)
    result: dict[str, dict] = {}
    for current_type in types:
        folder = asset_dir(current_type)
        assets = [
            serialize_asset(path, current_type)
            for path in folder.iterdir()
            if path.is_file() and allowed_asset_file(path.name, current_type)
        ]
        assets.sort(key=lambda item: (not item["is_default"], item["updated_at"]))
        default = next((item for item in assets if item["is_default"]), assets[0] if assets else None)
        result[current_type] = {"default": default, "items": assets}
    return result


def load_profiles() -> dict:
    for path in (PROFILES_FILE, PROFILES_FILE.with_suffix(".json.bak")):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:
            continue
    return {}


def save_profiles(profiles: dict) -> None:
    if PROFILES_FILE.exists():
        PROFILES_FILE.replace(PROFILES_FILE.with_suffix(".json.bak"))
    PROFILES_FILE.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_profile_assets(profile: dict) -> dict:
    resolved = {}
    assets = profile.get("assets") or {}
    for asset_type in sorted(ASSET_TYPES):
        filename = secure_filename(str(assets.get(asset_type) or ""))
        path = asset_dir(asset_type) / filename
        if filename and path.exists() and allowed_asset_file(path.name, asset_type):
            resolved[asset_type] = serialize_asset(path, asset_type)
    return resolved


def list_profiles() -> list[dict]:
    profiles = load_profiles()
    result = []
    for name, profile in sorted(profiles.items()):
        result.append(
            {
                "name": name,
                "assets": resolve_profile_assets(profile),
                "youtube_imports": profile.get("youtube_imports") or {},
                "updated_at": profile.get("updated_at"),
            }
        )
    return result


def youtube_video_key_from_entry(entry: dict) -> str:
    video_id = str(entry.get("id") or "").strip()
    if video_id:
        return video_id
    url = str(entry.get("webpage_url") or entry.get("url") or "").strip()
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc and parsed.query:
        from urllib.parse import parse_qs

        video_ids = parse_qs(parsed.query).get("v")
        if video_ids:
            return video_ids[0]
    return url or str(entry.get("title") or "").strip()


def profile_imported_youtube(profile_name: str) -> dict:
    profiles = load_profiles()
    profile = profiles.get(profile_name) or {}
    imports = profile.get("youtube_imports") or {}
    return imports if isinstance(imports, dict) else {}


def mark_profile_youtube_import(profile_name: str, video: dict, local_filename: str) -> None:
    profile_name = " ".join(str(profile_name or "").split()).strip()
    if not profile_name:
        return
    key = youtube_video_key_from_entry(video)
    if not key:
        return
    profiles = load_profiles()
    profile = profiles.setdefault(profile_name, {"assets": {}})
    imports = profile.setdefault("youtube_imports", {})
    imports[key] = {
        "id": key,
        "title": video.get("title") or "",
        "url": video.get("webpage_url") or video.get("url") or "",
        "filename": local_filename,
        "imported_at": datetime.utcnow().isoformat() + "Z",
    }
    profile["updated_at"] = datetime.utcnow().isoformat() + "Z"
    save_profiles(profiles)


def seed_library_defaults() -> None:
    patterns = {
        "logo": "logo_*",
        "follow": "follow_*",
        "ending": "ending_*",
    }
    for asset_type, pattern in patterns.items():
        folder = asset_dir(asset_type)
        if any(path.is_file() and path.name != ".gitkeep" for path in folder.iterdir()):
            continue
        candidates = [
            path
            for path in UPLOAD_FOLDER.glob(pattern)
            if path.is_file() and allowed_asset_file(path.name, asset_type)
        ]
        if not candidates:
            continue
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        target = folder / f"default_{secure_filename(latest.name)}"
        shutil.copy2(latest, target)


def cleanup_outputs(keep: int = OUTPUT_LRU_KEEP) -> None:
    """LRU pruning of generated MP4/SRT/ASS so the disk doesn't fill up."""
    if not OUTPUT_FOLDER.exists():
        return
    files = sorted(
        (path for path in OUTPUT_FOLDER.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files[keep:]:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def cleanup_orphan_uploads() -> int:
    """Borra basura de uploads/: .part (descargas incompletas de yt-dlp) y .ytdl.

    Retorna numero de archivos borrados. Se ejecuta al arrancar el server.
    """
    if not UPLOAD_FOLDER.exists():
        return 0
    deleted = 0
    for path in UPLOAD_FOLDER.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".part") or name.endswith(".ytdl") or ".mp4.part-" in name:
            try:
                path.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted


def _delayed_unlink(path: Path, delay: float = 7200.0) -> None:
    def _worker():
        time.sleep(delay)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def reap_old_jobs() -> None:
    cutoff = time.time() - JOB_RETENTION_SECONDS
    with jobs_lock:
        stale = [
            job_id
            for job_id, job in jobs.items()
            if job.get("status") in {"completed", "error"}
            and _job_timestamp(job) < cutoff
        ]
        for job_id in stale:
            jobs.pop(job_id, None)


def _job_timestamp(job: dict) -> float:
    raw = job.get("updated_at") or ""
    if not raw:
        return 0.0
    try:
        cleaned = raw.replace("Z", "")
        return datetime.fromisoformat(cleaned).timestamp()
    except Exception:
        return 0.0


def set_job(job_id: str, **updates) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(updates)
        jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"


def set_job_progress(job_id: str, progress: int, step: str) -> None:
    """Update render progress without letting parallel phases move it backwards."""
    with jobs_lock:
        job = jobs.setdefault(job_id, {})
        current = int(job.get("progress") or 0)
        job.update(status="processing", progress=max(current, int(progress)), step=step)
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"


def is_supported_video_url(url: str) -> bool:
    parsed = urlparse(url or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _video_file_has_audio(path: Path) -> bool:
    """ffprobe rapido para verificar que el video tenga un audio stream con datos.

    No basta con que el container declare un stream de audio; tambien debe
    tener packets reales. Algunos downloads salen con stream "fantasma".
    """
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type,duration",
             "-of", "default=noprint_wrappers=1:nokey=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        out = result.stdout or ""
        if "codec_type=audio" not in out:
            return False
        # Audio bytes per second sanity check
        packets = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "packet=size", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        total = sum(int(line) for line in packets.stdout.splitlines() if line.strip().isdigit())
        # Necesitamos al menos 5KB de datos (audio real -- silencio sintetico
        # de anullsrc es <100 bytes)
        return total > 5120
    except Exception:
        return False


def detect_platform(url: str) -> str:
    """Devuelve 'youtube', 'tiktok', 'facebook' o 'other' segun el dominio de la URL."""
    netloc = urlparse(url or "").netloc.lower()
    if "youtube.com" in netloc or "youtu.be" in netloc:
        return "youtube"
    if "tiktok.com" in netloc or "vm.tiktok.com" in netloc or "vt.tiktok.com" in netloc:
        return "tiktok"
    if "facebook.com" in netloc or "fb.com" in netloc or "fb.watch" in netloc or "m.facebook.com" in netloc:
        return "facebook"
    return "other"


def tiktok_options(base_options: dict) -> dict:
    """Opciones de yt-dlp optimizadas para TikTok.

    Prioriza el stream 'download_addr' (sin marca de agua) sobre 'play_addr'.
    TikTok sirve dos versiones del mismo video:
      - play_addr       -> CON marca de agua incrustada
      - download_addr   -> SIN marca de agua (formato de descarga nativa)
    yt-dlp elige automaticamente la version limpia cuando el formato lo permite.
    """
    options = dict(base_options)
    # No cookies necesarias para TikTok publico
    options.pop("cookiesfrombrowser", None)
    options.pop("cookiefile", None)
    # Extractor args para preferir el stream sin watermark
    options["extractor_args"] = {
        "tiktok": {
            "api_hostname": ["api16-normal-c-useast1a.tiktokv.com"],
        }
    }
    return options


def tiktok_download_formats() -> list[str | None]:
    """Formatos de descarga para TikTok, en orden de preferencia.

    El primer formato pide el stream sin watermark (download_addr).
    Los siguientes son fallbacks progresivos hasta 'best'.
    """
    return [
        # PRIORIDAD: streams que TIENEN audio (Tiktok no-watermark a veces es
        # video-only y los videos llegan mudos). Probamos las versiones con
        # audio garantizado primero, despues fallbacks.
        # 1) Mejor combo video+audio que TikTok sirva
        "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio",
        # 2) Streams combinados con audio (el container ya incluye audio)
        "best[height<=1080][acodec!=none]/best[acodec!=none]",
        # 3) bytevc1 sin watermark + cualquier audio (si TikTok lo permite)
        "bytevc1[height<=1080]+bestaudio/bytevc1+bestaudio",
        # 4) Default de yt-dlp como ultimo recurso
        None,
    ]


def facebook_options(base_options: dict, browser: str | None) -> dict:
    """Opciones de yt-dlp para Facebook.

    Facebook requiere autenticacion para la mayoria del contenido.
    Fuentes de cookies soportadas (en orden de prioridad):
      1. facebook_cookies.txt: exportado con extension 'Get cookies.txt LOCALLY'
      2. youtube_cookies.txt: el mismo archivo general (si tiene cookies de FB)
      3. cookiesfrombrowser: Edge/Chrome/Firefox (Brave falla por DPAPI en v127+)
    """
    options = dict(base_options)
    if browser == "file":
        # Prioridad: facebook_cookies.txt > youtube_cookies.txt
        if FACEBOOK_COOKIES_FILE.exists():
            options["cookiefile"] = str(FACEBOOK_COOKIES_FILE)
        elif YOUTUBE_COOKIES_FILE.exists():
            options["cookiefile"] = str(YOUTUBE_COOKIES_FILE)
        else:
            raise FileNotFoundError(
                "No hay cookies.txt de Facebook. "
                "Exporta las cookies desde Brave con la extension 'Get cookies.txt LOCALLY' "
                "en facebook.com y subelas en la seccion Facebook de la app."
            )
    elif browser:
        options["cookiesfrombrowser"] = (browser, None, None, None)
    return options


def facebook_download_formats() -> list[str | None]:
    """Formatos de descarga para Facebook en orden de preferencia."""
    return [
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080][acodec!=none]",
        "bestvideo[height<=720]+bestaudio/best[height<=720][acodec!=none]",
        "best[acodec!=none]",
        None,
    ]


def facebook_error_message(error: Exception, attempted: list[str | None]) -> str:
    """Mensaje de error amigable y accionable para fallos de Facebook."""
    raw = clean_error_text(str(error))
    tried = ", ".join(youtube_browser_label(b) for b in attempted)
    if "login" in raw.lower() or "log in" in raw.lower() or "checkpoint" in raw.lower():
        return (
            "Facebook requiere inicio de sesion para ese contenido. "
            "Sube un facebook_cookies.txt exportado desde Brave con la extension "
            "'Get cookies.txt LOCALLY'. "
            f"Intentos realizados: {tried}."
        )
    if "private" in raw.lower() or "not available" in raw.lower():
        return (
            "El video o perfil de Facebook es privado o no esta disponible publicamente. "
            f"Intentos realizados: {tried}."
        )
    if "dpapi" in raw.lower() or "decrypt" in raw.lower():
        return (
            "Brave bloquea la lectura directa de cookies (DPAPI). "
            "Exporta las cookies manualmente: abre Brave, ve a facebook.com, "
            "usa la extension 'Get cookies.txt LOCALLY' y sube el archivo en la app."
        )
    if "reel" in raw.lower() and "extract" in raw.lower():
        return (
            "No pude extraer ese Reel de Facebook. "
            "Prueba pegando el link del video individual en lugar del perfil. "
            f"Detalle: {raw[:200]}"
        )
    if "unsupported url" in raw.lower() or "no video formats" in raw.lower():
        return (
            "Facebook no permite extraer todos los videos de una fanpage o perfil de una sola vez. "
            "Debes copiar y pegar el link de un video individual o Reel (ej: facebook.com/.../videos/12345)."
        )
    return f"No pude leer Facebook ({tried}): {raw[:300]}"


def normalize_cookie_browser(value: str | None) -> str:
    browser = secure_filename((value or "auto").lower())
    return browser if browser in YOUTUBE_COOKIE_BROWSERS else "auto"


def youtube_cookie_attempts(cookie_browser: str) -> list[str | None]:
    if cookie_browser == "auto":
        return [
            browser
            for browser in YOUTUBE_AUTO_COOKIE_SEQUENCE
            if browser != "file" or YOUTUBE_COOKIES_FILE.exists()
        ]
    return [YOUTUBE_COOKIE_BROWSERS[cookie_browser]]


def youtube_options(base_options: dict, browser: str | None) -> dict:
    options = dict(base_options)
    options.setdefault("js_runtimes", {"node": {}})
    if browser == "file":
        if not YOUTUBE_COOKIES_FILE.exists():
            raise FileNotFoundError("No hay cookies.txt guardado. Sube el archivo en la seccion YouTube.")
        options["cookiefile"] = str(YOUTUBE_COOKIES_FILE)
    elif browser:
        options["cookiesfrombrowser"] = (browser, None, None, None)
    return options


def youtube_browser_label(browser: str | None) -> str:
    labels = {
        None: "sin cookies",
        "file": "cookies.txt",
        "edge": "Edge",
        "chrome": "Chrome",
        "firefox": "Firefox",
        "brave": "Brave",
    }
    return labels.get(browser, str(browser))


def youtube_error_message(error: Exception, attempted: list[str | None]) -> str:
    raw = clean_error_text(str(error))
    tried = ", ".join(youtube_browser_label(browser) for browser in attempted)
    if "not a bot" in raw or "Sign in to confirm" in raw:
        return (
            "YouTube pidio confirmar que no eres bot. Abre YouTube en el navegador donde tienes sesion, "
            "sube un cookies.txt o elige manualmente ese navegador en 'Sesion YouTube'. "
            f"Intentos realizados: {tried}."
        )
    if "failed to load cookies" in raw or "could not copy" in raw.lower():
        return (
            "No pude leer las cookies del navegador seleccionado. Cierra ese navegador por unos segundos, "
            "prueba otro navegador manualmente o sube un archivo cookies.txt. "
            f"Intentos realizados: {tried}."
        )
    if "no devolvio informacion" in raw:
        return (
            "YouTube no devolvio informacion sin sesion. Si antes funcionaba, puede ser un bloqueo temporal "
            "de YouTube para esta IP o este video. Sube un cookies.txt para hacerlo estable. "
            f"Intentos realizados: {tried}."
        )
    return f"No pude leer YouTube ({tried}): {raw}"


def clean_error_text(message: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", message or "").strip()


def find_chrome_executable() -> str | None:
    candidates = [
        os.environ.get("WORKFAST_CHROME_PATH", ""),
        str(Path(os.environ.get("PROGRAMFILES", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        str(Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
    ]
    return next((path for path in candidates if path and Path(path).exists()), None)


def open_youtube_login_browser() -> None:
    global debug_browser_process
    chrome_path = find_chrome_executable()
    if not chrome_path:
        raise RuntimeError("No encontre Chrome o Edge instalado.")

    debug_browser_process = subprocess.Popen(
        [
            chrome_path,
            f"--remote-debugging-port={YOUTUBE_DEBUG_PORT}",
            f"--user-data-dir={YOUTUBE_BROWSER_PROFILE}",
            "--no-first-run",
            "--disable-features=DialMediaRouteProvider",
            "https://www.youtube.com/account",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def close_debug_browser() -> None:
    global debug_browser_process
    if debug_browser_process is None:
        return
    try:
        debug_browser_process.terminate()
        debug_browser_process.wait(timeout=4)
    except Exception:
        try:
            debug_browser_process.kill()
        except Exception:
            pass
    debug_browser_process = None


def read_debug_json(path: str) -> dict | list:
    with urllib.request.urlopen(f"http://127.0.0.1:{YOUTUBE_DEBUG_PORT}{path}", timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def websocket_request(ws_url: str, method: str, params: dict | None = None) -> dict:
    parsed = urlparse(ws_url)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path
    if parsed.query:
        path += f"?{parsed.query}"

    with socket.create_connection((host, port), timeout=5) as sock:
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(handshake.encode("ascii"))
        response = sock.recv(4096)
        if b" 101 " not in response:
            raise RuntimeError("No pude conectar con Chrome para leer la sesion.")

        payload = json.dumps({"id": 1, "method": method, "params": params or {}}).encode("utf-8")
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.extend([0x80 | 126, (length >> 8) & 255, length & 255])
        else:
            header.append(0x80 | 127)
            header.extend(length.to_bytes(8, "big"))
        mask = os.urandom(4)
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        sock.sendall(bytes(header) + masked)

        while True:
            frame_header = sock.recv(2)
            if len(frame_header) < 2:
                raise RuntimeError("Chrome cerro la conexion antes de responder.")
            opcode = frame_header[0] & 0x0F
            length = frame_header[1] & 0x7F
            if length == 126:
                length = int.from_bytes(sock.recv(2), "big")
            elif length == 127:
                length = int.from_bytes(sock.recv(8), "big")
            if frame_header[1] & 0x80:
                server_mask = sock.recv(4)
                raw_payload = sock.recv(length)
                frame_payload = bytes(byte ^ server_mask[index % 4] for index, byte in enumerate(raw_payload))
            else:
                frame_payload = sock.recv(length)
            if opcode == 8:
                raise RuntimeError("Chrome cerro la conexion.")
            if opcode != 1:
                continue
            message = json.loads(frame_payload.decode("utf-8"))
            if message.get("id") == 1:
                if "error" in message:
                    raise RuntimeError(message["error"].get("message", "Chrome no entrego cookies."))
                return message.get("result", {})


def export_youtube_cookies_from_debug_browser() -> int:
    try:
        version = read_debug_json("/json/version")
    except Exception as exc:
        raise RuntimeError("Primero pulsa 'Conectar acceso', inicia sesion en esa ventana y dejala abierta.") from exc

    ws_url = version.get("webSocketDebuggerUrl")
    if not ws_url:
        targets = read_debug_json("/json")
        pages = [target for target in targets if target.get("webSocketDebuggerUrl")]
        if not pages:
            raise RuntimeError("No encontre una pestana de Chrome abierta para WorkFast.")
        ws_url = pages[0]["webSocketDebuggerUrl"]

    result = websocket_request(ws_url, "Storage.getCookies")
    cookies = [
        cookie
        for cookie in result.get("cookies", [])
        if "youtube.com" in cookie.get("domain", "") or "google.com" in cookie.get("domain", "")
    ]
    if not cookies:
        raise RuntimeError("No encontre sesion de YouTube. Inicia sesion en la ventana que abrio WorkFast.")

    lines = ["# Netscape HTTP Cookie File", "# Generated by WorkFast Video Editor"]
    for cookie in cookies:
        domain = cookie.get("domain", "")
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        path = cookie.get("path") or "/"
        secure = "TRUE" if cookie.get("secure") else "FALSE"
        expires = int(cookie.get("expires") or 0)
        name = cookie.get("name", "")
        value = cookie.get("value", "")
        lines.append("\t".join([domain, include_subdomains, path, secure, str(expires), name, value]))

    YOUTUBE_COOKIES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(cookies)


def normalize_youtube_entry(entry: dict, platform: str = "youtube") -> dict:
    """Normaliza metadata de un video de cualquier plataforma (YouTube, TikTok, etc.)."""
    if not isinstance(entry, dict):
        entry = {"url": str(entry), "title": str(entry)}

    video_id = entry.get("id") or ""
    webpage_url = entry.get("webpage_url") or entry.get("url") or ""

    # Completar URLs relativas segun plataforma
    if video_id and not webpage_url.startswith("http"):
        if platform == "tiktok":
            webpage_url = f"https://www.tiktok.com/@unknown/video/{video_id}"
        else:
            webpage_url = f"https://www.youtube.com/watch?v={video_id}"

    thumbnails = entry.get("thumbnails") or []
    thumbnail = entry.get("thumbnail") or ""
    if thumbnails and not thumbnail:
        # TikTok pone los thumbs en orden inverso de calidad — tomar el ultimo
        thumbnail = thumbnails[-1].get("url") or ""

    # Autor: TikTok usa 'uploader' o 'creator', YouTube usa 'channel'
    channel = (
        entry.get("channel")
        or entry.get("uploader")
        or entry.get("creator")
        or entry.get("uploader_id")
        or ""
    )

    return {
        "id": video_id,
        "key": youtube_video_key_from_entry(entry),
        "title": entry.get("title") or entry.get("description") or "Video sin titulo",
        "url": webpage_url,
        "duration": entry.get("duration"),
        "channel": channel,
        "uploader": entry.get("uploader") or channel,
        "view_count": entry.get("view_count") or entry.get("play_count") or 0,
        "upload_date": entry.get("upload_date") or "",
        "thumbnail": thumbnail,
        "platform": platform,
    }


seed_library_defaults()
cleanup_outputs()
cleanup_cache(SUBTITLE_CACHE_FOLDER)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/favicon.ico")
def favicon():
    """Sirve la tortuga como favicon. Si no existe el PNG, devuelve 204."""
    logo_path = FRONTEND_DIR / "assets" / "turtle-logo.png"
    if logo_path.exists():
        return send_file(logo_path, mimetype="image/png")
    return ("", 204)


@app.get("/<path:path>")
def static_files(path: str):
    return send_from_directory(FRONTEND_DIR, path)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "message": "WorkFast server running"})


@app.get("/api/capabilities")
def capabilities():
    whisper_info = describe_backend()
    nvenc_status = VideoEditor.get_nvenc_status()
    nvenc_ok = nvenc_status["available"]
    rtx_profile = nvenc_ok and VideoEditor._high_end_nvidia_gpu()
    voice_info = voice_backend_status()
    return jsonify(
        {
            "success": True,
            "subtitles_local": True,
            "translation_languages": ["es", "en", "pt"],
            "max_queue": 20,
            "subtitle_engine": "faster-whisper",
            "video_encoder": "h264_nvenc" if nvenc_ok else "libx264",
            "video_backend_label": "GPU RTX (NVENC HQ)" if rtx_profile else ("GPU (NVENC)" if nvenc_ok else "CPU (libx264)"),
            "nvenc": nvenc_status,
            "whisper": whisper_info,
            "enhancer": {
                "profile": normalize_enhancer_profile(),
                "gpu": os.getenv("WORKFAST_ENHANCER_GPU", "0"),
                "ncnn_jobs": os.getenv("WORKFAST_NCNN_JOBS", "1:2:1"),
                "rife_balanced": os.getenv("WORKFAST_ENABLE_RIFE_BALANCED", "0") == "1",
            },
            "npu": {
                "used": False,
                "reason": "Las librerias actuales (FFmpeg, Whisper, Real-ESRGAN/RIFE) no exponen backend NPU estable en Windows.",
            },
            "voice": voice_info,
        }
    )


@app.get("/api/voice/models")
def voice_models():
    return jsonify({"success": True, **voice_backend_status()})


@app.post("/api/upload")
def upload_file():
    file = request.files.get("file")
    file_type = secure_filename(request.form.get("type", "file"))

    if not file or not file.filename:
        return jsonify({"error": "No seleccionaste ningun archivo."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Tipo de archivo no permitido."}), 400

    original_name = secure_filename(file.filename)
    unique_name = f"{file_type}_{uuid.uuid4().hex[:10]}_{original_name}"
    filepath = UPLOAD_FOLDER / unique_name
    file.save(filepath)

    return jsonify(
        {
            "success": True,
            "filename": unique_name,
            "filepath": str(filepath),
            "type": file_type,
            "title": title_from_filename(original_name) if file_type == "video" else "",
        }
    )


@app.get("/api/assets")
def get_assets():
    return jsonify({"success": True, "assets": list_library_assets()})


@app.get("/api/profiles")
def get_profiles():
    return jsonify({"success": True, "profiles": list_profiles()})


@app.post("/api/profiles")
def save_profile():
    data = request.get_json(silent=True) or {}
    name = " ".join(str(data.get("name") or "").split()).strip()
    previous_name = " ".join(str(data.get("previous_name") or "").split()).strip()
    if not name:
        return jsonify({"error": "Escribe un nombre para el perfil."}), 400
    if len(name) > 40:
        name = name[:40].strip()

    incoming_assets = data.get("assets") or {}
    saved_assets = {}
    for asset_type in sorted(ASSET_TYPES):
        raw_value = incoming_assets.get(asset_type) or ""
        filename = secure_filename(Path(str(raw_value)).name)
        path = asset_dir(asset_type) / filename
        if filename and path.exists() and allowed_asset_file(path.name, asset_type):
            saved_assets[asset_type] = filename

    profiles = load_profiles()
    existing_profile = profiles.get(previous_name or name, {}) if isinstance(profiles, dict) else {}
    if previous_name and previous_name != name:
        profiles.pop(previous_name, None)
    profiles[name] = {
        **existing_profile,
        "assets": saved_assets,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_profiles(profiles)
    return jsonify({"success": True, "profiles": list_profiles()})


@app.delete("/api/profiles/<path:profile_name>")
def delete_profile(profile_name: str):
    name = " ".join(str(profile_name or "").split()).strip()
    profiles = load_profiles()
    if name not in profiles:
        return jsonify({"error": "Perfil no encontrado."}), 404
    profiles.pop(name, None)
    save_profiles(profiles)
    return jsonify({"success": True, "profiles": list_profiles()})


@app.get("/api/assets/file/<asset_type>/<path:filename>")
def get_asset_file(asset_type: str, filename: str):
    safe_type = secure_filename(asset_type)
    safe_filename = secure_filename(filename)

    if safe_type not in ASSET_TYPES or not safe_filename:
        return jsonify({"error": "Recurso no valido."}), 400

    filepath = asset_dir(safe_type) / safe_filename
    if not filepath.exists() or not allowed_asset_file(filepath.name, safe_type):
        return jsonify({"error": "Recurso no encontrado."}), 404

    return send_file(filepath)


@app.post("/api/assets/upload")
def upload_asset():
    file = request.files.get("file")
    asset_type = secure_filename(request.form.get("type", ""))
    make_default = request.form.get("make_default", "true").lower() != "false"

    if asset_type not in ASSET_TYPES:
        return jsonify({"error": "Tipo de recurso no valido."}), 400

    if not file or not file.filename:
        return jsonify({"error": "No seleccionaste ningun archivo."}), 400

    if not allowed_asset_file(file.filename, asset_type):
        return jsonify({"error": "Archivo no permitido para ese recurso."}), 400

    folder = asset_dir(asset_type)
    original_name = secure_filename(file.filename)
    prefix = "default" if make_default else asset_type
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}_{original_name}"
    filepath = folder / filename
    file.save(filepath)

    if make_default:
        for other in folder.glob("default_*"):
            if other != filepath:
                other.rename(folder / other.name.replace("default_", f"{asset_type}_", 1))

    return jsonify({"success": True, "asset": serialize_asset(filepath, asset_type), "assets": list_library_assets()})


@app.post("/api/assets/default")
def set_default_asset():
    data = request.get_json(silent=True) or {}
    asset_type = secure_filename(data.get("type", ""))
    filename = secure_filename(data.get("filename", ""))

    if asset_type not in ASSET_TYPES:
        return jsonify({"error": "Tipo de recurso no valido."}), 400

    folder = asset_dir(asset_type)
    selected = folder / filename
    if not selected.exists() or not allowed_asset_file(selected.name, asset_type):
        return jsonify({"error": "Recurso no encontrado."}), 404

    for other in folder.glob("default_*"):
        if other != selected:
            other.rename(folder / other.name.replace("default_", f"{asset_type}_", 1))

    if not selected.name.startswith("default_"):
        selected = selected.rename(folder / f"default_{selected.name}")

    return jsonify({"success": True, "asset": serialize_asset(selected, asset_type), "assets": list_library_assets()})


@app.post("/api/youtube/list")
def list_youtube_videos():
    """Lista videos de YouTube, TikTok o Facebook. Detecta la plataforma automaticamente."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    cookie_browser = normalize_cookie_browser(data.get("cookie_browser"))
    profile_name = " ".join(str(data.get("profile_name") or "").split()).strip()

    if not is_supported_video_url(url):
        return jsonify({"error": "Pega una URL valida de YouTube, TikTok, Facebook, canal o playlist."}), 400

    platform = detect_platform(url)

    base_options = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    attempted: list[str | None] = []
    last_error: Exception | None = None

    try:
        info = None

        if platform == "tiktok":
            # TikTok: sin cookies, sin cookie_browser, opciones especificas
            attempted.append(None)
            try:
                with YoutubeDL(tiktok_options(base_options)) as ydl:
                    info = ydl.extract_info(url, download=False)
                if not info:
                    raise RuntimeError("TikTok no devolvio informacion. Verifica que el perfil sea publico.")
            except Exception as exc:
                last_error = exc

        elif platform == "facebook":
            # Facebook: intentar primero con cookies.txt de Facebook si existe,
            # luego sin cookies (paginas publicas), luego browser cookies.
            fb_browsers: list[str | None] = []
            if FACEBOOK_COOKIES_FILE.exists():
                fb_browsers.append("file")  # facebook_cookies.txt tiene prioridad
            fb_browsers.append(None)  # sin cookies (publico)
            fb_browsers.extend(youtube_cookie_attempts(cookie_browser))  # navegador

            for browser in fb_browsers:
                attempted.append(browser)
                try:
                    fb_opts = facebook_options(base_options, browser)
                    with YoutubeDL(fb_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                    if not info:
                        raise RuntimeError("Facebook no devolvio informacion en este intento.")
                    break
                except FileNotFoundError as exc:
                    last_error = exc
                    break
                except Exception as exc:
                    last_error = exc
                    err_low = str(exc).lower()
                    if "login" in err_low or "log in" in err_low or "dpapi" in err_low or "decrypt" in err_low:
                        continue  # intentar con siguiente fuente de cookies
                    break

            if info is None and last_error:
                raise RuntimeError(facebook_error_message(last_error, attempted))

        else:
            # YouTube y otras plataformas: logica de cookies existente
            for browser in youtube_cookie_attempts(cookie_browser):
                attempted.append(browser)
                try:
                    with YoutubeDL(youtube_options(base_options, browser)) as ydl:
                        info = ydl.extract_info(url, download=False)
                    if not info:
                        raise RuntimeError("YouTube no devolvio informacion en este intento.")
                    break
                except Exception as exc:
                    last_error = exc
                    continue

            if info is None and last_error:
                raise RuntimeError(youtube_error_message(last_error, attempted))

        if not info:
            platform_labels = {"tiktok": "TikTok", "facebook": "Facebook"}
            platform_label = platform_labels.get(platform, "YouTube")
            return jsonify({"error": f"{platform_label} no devolvio informacion para ese link."}), 404

        entries = info.get("entries") if isinstance(info, dict) else None
        if entries:
            videos = [normalize_youtube_entry(entry, platform) for entry in entries if entry]
        else:
            videos = [normalize_youtube_entry(info, platform)]

        videos = [video for video in videos if video.get("url")]
        imported = profile_imported_youtube(profile_name) if profile_name else {}
        for video in videos:
            key = video.get("key") or video.get("id") or video.get("url")
            if key in imported:
                video["already_imported"] = True
                video["imported_at"] = imported[key].get("imported_at")
                video["imported_filename"] = imported[key].get("filename")
        return jsonify({"success": True, "videos": videos, "count": len(videos), "platform": platform})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/youtube/cookies")
def get_youtube_cookies_status():
    if not YOUTUBE_COOKIES_FILE.exists():
        return jsonify({"success": True, "configured": False})
    return jsonify(
        {
            "success": True,
            "configured": True,
            "filename": YOUTUBE_COOKIES_FILE.name,
            "size": YOUTUBE_COOKIES_FILE.stat().st_size,
            "updated_at": datetime.fromtimestamp(YOUTUBE_COOKIES_FILE.stat().st_mtime).isoformat(),
        }
    )


@app.post("/api/youtube/cookies")
def upload_youtube_cookies():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No seleccionaste ningun archivo cookies.txt."}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".txt"):
        return jsonify({"error": "Sube un archivo .txt en formato Netscape cookies."}), 400

    file.save(YOUTUBE_COOKIES_FILE)
    return jsonify(
        {
            "success": True,
            "configured": True,
            "filename": YOUTUBE_COOKIES_FILE.name,
            "size": YOUTUBE_COOKIES_FILE.stat().st_size,
        }
    )


@app.get("/api/facebook/cookies")
def get_facebook_cookies_status():
    """Estado de las cookies de Facebook subidas manualmente."""
    if not FACEBOOK_COOKIES_FILE.exists():
        return jsonify({"success": True, "configured": False})
    return jsonify(
        {
            "success": True,
            "configured": True,
            "filename": FACEBOOK_COOKIES_FILE.name,
            "size": FACEBOOK_COOKIES_FILE.stat().st_size,
            "updated_at": datetime.fromtimestamp(FACEBOOK_COOKIES_FILE.stat().st_mtime).isoformat(),
        }
    )


@app.post("/api/facebook/cookies")
def upload_facebook_cookies():
    """Sube un archivo facebook_cookies.txt exportado desde Brave/Chrome."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No seleccionaste ningun archivo cookies.txt."}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".txt"):
        return jsonify({"error": "Sube un archivo .txt en formato Netscape cookies."}), 400

    file.save(FACEBOOK_COOKIES_FILE)
    return jsonify(
        {
            "success": True,
            "configured": True,
            "filename": FACEBOOK_COOKIES_FILE.name,
            "size": FACEBOOK_COOKIES_FILE.stat().st_size,
        }
    )



@app.post("/api/youtube/login-browser")
def youtube_login_browser():
    try:
        open_youtube_login_browser()
        return jsonify({"success": True, "message": "Chrome abierto para iniciar sesion en YouTube."})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/youtube/capture-session")
def youtube_capture_session():
    try:
        cookie_count = export_youtube_cookies_from_debug_browser()
        # Close the unauthenticated debug Chrome instance once cookies are captured
        # so we never leave port 9222 dangling.
        close_debug_browser()
        return jsonify(
            {
                "success": True,
                "configured": True,
                "cookie_count": cookie_count,
                "filename": YOUTUBE_COOKIES_FILE.name,
                "size": YOUTUBE_COOKIES_FILE.stat().st_size,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/youtube/import")
def import_youtube_video():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    cookie_browser = normalize_cookie_browser(data.get("cookie_browser"))
    profile_name = " ".join(str(data.get("profile_name") or "").split()).strip()
    video_key = str(data.get("video_key") or "").strip()
    video_title = str(data.get("title") or "").strip()

    if not is_supported_video_url(url):
        return jsonify({"error": "URL de video no valida."}), 400

    job_id = uuid.uuid4().hex
    set_job(job_id, status="queued", progress=0, step="En cola para importar", kind="import")

    def download_hook(payload: dict) -> None:
        status = payload.get("status")
        if status == "downloading":
            total = payload.get("total_bytes") or payload.get("total_bytes_estimate") or 0
            downloaded = payload.get("downloaded_bytes") or 0
            progress = 5
            if total:
                progress = 5 + int((downloaded / total) * 85)
            set_job(job_id, status="processing", progress=min(progress, 90), step="Descargando video")
        elif status == "finished":
            set_job(job_id, status="processing", progress=92, step="Preparando archivo")

    def worker() -> None:
        try:
            platform = detect_platform(url)
            platform_labels = {"tiktok": "TikTok", "facebook": "Facebook"}
            platform_label = platform_labels.get(platform, "YouTube")
            set_job(job_id, status="processing", progress=1, step=f"Conectando con {platform_label}")
            output_template = str(UPLOAD_FOLDER / f"youtube_{job_id}_%(title).80s.%(ext)s")
            base_options = {
                "merge_output_format": "mp4",
                "outtmpl": output_template,
                "quiet": True,
                "noprogress": True,
                "no_warnings": True,
                "noplaylist": True,
                "progress_hooks": [download_hook],
                "postprocessor_args": {"ffmpeg": ["-movflags", "+faststart"]},
            }
            info = None
            downloaded = None
            attempted: list[str | None] = []
            last_error: Exception | None = None

            if platform == "tiktok":
                # TikTok: intentar formatos sin watermark en orden de preferencia
                set_job(job_id, status="processing", progress=2, step="Descargando de TikTok (sin marca de agua)")
                for format_spec in tiktok_download_formats():
                    try:
                        options = tiktok_options(dict(base_options))
                        if format_spec:
                            options["format"] = format_spec
                        with YoutubeDL(options) as ydl:
                            info = ydl.extract_info(url, download=True) or {}
                            downloaded = Path(ydl.prepare_filename(info))
                            if downloaded.suffix.lower() != ".mp4":
                                downloaded = downloaded.with_suffix(".mp4")
                        break
                    except Exception as exc:
                        last_error = exc
                        if "Requested format is not available" in str(exc) or "No video formats found" in str(exc):
                            continue
                        break
                if info is None or downloaded is None:
                    err_raw = clean_error_text(str(last_error or ""))
                    raise RuntimeError(
                        f"No pude descargar el video de TikTok. "
                        f"Verifica que el link sea publico y no sea un Live. "
                        f"Detalle: {err_raw}"
                    )

            elif platform == "facebook":
                # Facebook: intentar sin cookies primero (videos publicos de paginas),
                # luego escalar a cookies si es necesario.
                set_job(job_id, status="processing", progress=2, step="Descargando de Facebook")
                browsers_to_try = [None] + list(youtube_cookie_attempts(cookie_browser))
                for browser in browsers_to_try:
                    attempted.append(browser)
                    for format_spec in facebook_download_formats():
                        try:
                            options = facebook_options(dict(base_options), browser)
                            if format_spec:
                                options["format"] = format_spec
                            with YoutubeDL(options) as ydl:
                                info = ydl.extract_info(url, download=True) or {}
                                downloaded = Path(ydl.prepare_filename(info))
                                if downloaded.suffix.lower() != ".mp4":
                                    downloaded = downloaded.with_suffix(".mp4")
                            break
                        except FileNotFoundError as exc:
                            last_error = exc
                            break  # no hay cookies.txt
                        except Exception as exc:
                            last_error = exc
                            if "Requested format is not available" in str(exc):
                                continue
                            if "login" in str(exc).lower() or "log in" in str(exc).lower():
                                break  # necesita cookies, pasar al siguiente browser
                            break
                    if info is not None and downloaded is not None:
                        break

                if info is None or downloaded is None:
                    raise RuntimeError(facebook_error_message(last_error or RuntimeError("Facebook no respondio."), attempted))

            else:
                # YouTube y otras plataformas: logica de cookies existente
                base_options["format_sort"] = ["res:1080", "fps:60", "vcodec:avc1", "acodec:mp4a", "ext:mp4"]
                for browser in youtube_cookie_attempts(cookie_browser):
                    attempted.append(browser)
                    for format_spec in YOUTUBE_DOWNLOAD_FORMATS:
                        set_job(
                            job_id,
                            status="processing",
                            progress=2,
                            step="Conectando con YouTube",
                        )
                        try:
                            options = dict(base_options)
                            if format_spec:
                                options["format"] = format_spec
                            with YoutubeDL(youtube_options(options, browser)) as ydl:
                                info = ydl.extract_info(url, download=True) or {}
                                downloaded = Path(ydl.prepare_filename(info))
                                if downloaded.suffix.lower() != ".mp4":
                                    downloaded = downloaded.with_suffix(".mp4")
                            break
                        except Exception as exc:
                            last_error = exc
                            if "Requested format is not available" in str(exc):
                                continue
                            break
                    if info is not None and downloaded is not None:
                        break

                if info is None or downloaded is None:
                    raise RuntimeError(youtube_error_message(last_error or RuntimeError("YouTube no respondio."), attempted))

            if not downloaded.exists():
                candidates = sorted(
                    UPLOAD_FOLDER.glob(f"youtube_{job_id}_*"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if not candidates:
                    raise RuntimeError("No encontre el archivo descargado.")
                downloaded = candidates[0]

            # Verificacion critica: el archivo bajado debe tener audio.
            # TikTok no-watermark stream a veces es video-only; YouTube Shorts HEVC
            # tambien. Sin esta verificacion, los videos llegan mudos a la cola.
            if not _video_file_has_audio(downloaded):
                downloaded.unlink(missing_ok=True)
                raise RuntimeError(
                    f"El video descargado de {platform} llego sin audio. "
                    "yt-dlp eligio un stream video-only (comun en versiones "
                    "sin marca de agua de TikTok o en HEVC de Shorts). "
                    "Re-intenta la importacion -- el proximo intento usara "
                    "un formato distinto."
                )

            safe_name = f"video_{job_id}_{secure_filename(downloaded.name)}"
            final_path = UPLOAD_FOLDER / safe_name
            if downloaded != final_path:
                downloaded.replace(final_path)

            youtube_record = {
                "id": video_key or (info.get("id") if isinstance(info, dict) else ""),
                "title": video_title or info.get("title") or "",
                "url": url,
                "webpage_url": url,
            }
            mark_profile_youtube_import(profile_name, youtube_record, safe_name)

            set_job(
                job_id,
                status="completed",
                progress=100,
                step="Video importado",
                filename=safe_name,
                filepath=str(final_path),
                type="video",
                title=info.get("title") or title_from_filename(final_path.name),
            )
        except Exception as exc:
            set_job(job_id, status="error", progress=0, step="Error", error=clean_error_text(str(exc)))

    def _download_worker() -> None:
        # Intentar obtener turno de descarga sin bloquear
        acquired = _DOWNLOAD_SEM.acquire(blocking=False)
        if not acquired:
            # Todas las ranuras ocupadas: informar al usuario y esperar
            set_job(job_id, status="queued", progress=0, step="Esperando turno de descarga")
            _DOWNLOAD_SEM.acquire()
        try:
            worker()
        finally:
            _DOWNLOAD_SEM.release()

    threading.Thread(target=_download_worker, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


def render_subtitle_preview(input_video: Path, subtitle_path: Path, output_path: Path, duration: float) -> None:
    """Render a short vertical preview focused on subtitle timing/style."""
    subtitle_file = VideoEditor._escape_filter_path(subtitle_path)
    fonts_dir = VideoEditor._escape_filter_path(VideoEditor.SUBTITLE_FONTS_DIR)
    video_filter = (
        "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"ass=filename='{subtitle_file}':fontsdir='{fonts_dir}'"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_video),
        "-t",
        f"{duration:.2f}",
        "-vf",
        video_filter,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        *VideoEditor._video_encoder_args(),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not output_path.exists():
        tail = (result.stderr or result.stdout or "").strip()[-1400:]
        raise RuntimeError(f"No pude renderizar preview de subtitulos:\n{tail}")


@app.post("/api/subtitles/preview")
def preview_subtitles():
    data = request.get_json(silent=True) or {}
    input_video = Path(data.get("input_video", ""))
    if not input_video.exists():
        return jsonify({"error": "No encontre el video para preview."}), 400

    job_id = uuid.uuid4().hex
    preview_filename = f"preview_subs_{job_id[:8]}.mp4"
    preview_path = OUTPUT_FOLDER / preview_filename
    duration = max(5.0, min(30.0, float(data.get("duration", 18.0))))
    offset_ms = max(-750.0, min(750.0, float(data.get("subtitle_offset_ms", 0.0))))
    language_value = secure_filename(str(data.get("subtitle_language") or "original")) or "original"

    set_job(
        job_id,
        status="queued",
        progress=0,
        step="En cola para preview",
        kind="subtitle_preview",
        output_filename=preview_filename,
        output_path=str(preview_path),
    )

    def progress_callback(progress: int, step: str) -> None:
        mapped = 5 + int((max(0, min(16, progress)) / 16) * 55)
        set_job(job_id, status="processing", progress=mapped, step=step)

    def worker() -> None:
        temp_subtitle = OUTPUT_FOLDER / ".workfast_tmp" / f"preview_sub_{job_id}.ass"
        temp_subtitle.parent.mkdir(parents=True, exist_ok=True)
        try:
            set_job(job_id, status="processing", progress=2, step="Generando subtitulos de preview")
            generate_subtitles(
                input_video=input_video,
                output_ass=temp_subtitle,
                cache_dir=SUBTITLE_CACHE_FOLDER,
                target_language=language_value,
                progress_callback=progress_callback,
                subtitle_offset_ms=offset_ms,
            )
            set_job(job_id, status="processing", progress=70, step="Renderizando preview")
            render_subtitle_preview(input_video, temp_subtitle, preview_path, duration)
            set_job(
                job_id,
                status="completed",
                progress=100,
                step="Preview listo",
                preview_filename=preview_filename,
                preview_url=f"/api/download/{preview_filename}",
                download_url=f"/api/download/{preview_filename}",
            )
        except Exception as exc:
            set_job(job_id, status="error", progress=0, step="Error", error=clean_error_text(str(exc)))
        finally:
            temp_subtitle.unlink(missing_ok=True)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id, "preview_filename": preview_filename})


@app.post("/api/process")
def process_video():
    data = request.get_json(silent=True) or {}
    input_video = Path(data.get("input_video", ""))

    if not input_video.exists():
        return jsonify({"error": "No encontre el video de entrada."}), 400

    provided_title = data.get("title_text")
    title_text = provided_title if provided_title is not None else title_from_filename(input_video.name)
    output_base = output_name_from_title(title_text, input_video.stem)
    output_filename = f"{output_base}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = OUTPUT_FOLDER / output_filename
    job_id = uuid.uuid4().hex

    set_job(
        job_id,
        status="queued",
        progress=0,
        step="En cola",
        output_filename=output_filename,
        output_path=str(output_path),
        kind="render",
    )
    _begin_cancellable_job(job_id)

    def progress_callback(progress: int, step: str) -> None:
        set_job_progress(job_id, progress, step)

    def raise_if_cancelled(stage: str = "") -> None:
        """Aborta el worker si el usuario cancelo. Se llama entre etapas."""
        if _job_is_cancelled(job_id):
            raise RenderCancelled(stage or "cancelado")

    def worker() -> None:
        subtitle_path: Path | None = None
        subtitle_filename: str | None = None
        subtitle_warning: str | None = None
        enhanced_video: Path = input_video
        working_input = input_video
        trimmed_video: Path | None = None

        # Serializar el render: solo uno toca la GPU a la vez. Si hay otro
        # activo, este job espera aca mostrandose "En cola" al usuario.
        sem_acquired = _RENDER_SEM.acquire(blocking=False)
        if not sem_acquired:
            set_job(job_id, status="queued", progress=0, step=f"Esperando turno (límite de {_MAX_PARALLEL_RENDERS} renders activo)")
            _RENDER_SEM.acquire()
            sem_acquired = True
        try:
            # Si lo cancelaron mientras esperaba el semaforo, salir limpio.
            if _job_is_cancelled(job_id):
                raise RenderCancelled("cancelado mientras esperaba el turno")
            set_job(job_id, status="processing", progress=1, step="Iniciando")

            remove_silences = bool(data.get("remove_silences", False))
            silence_threshold_db = float(data.get("silence_threshold_db", -32.0))
            silence_min_duration = float(data.get("silence_min_duration", 0.45))
            enhance_profile = normalize_enhancer_profile(str(data.get("enhance_profile") or ""))

            if remove_silences:
                set_job(job_id, status="processing", progress=2, step="Recortando silencios")
                trim_probe_output = OUTPUT_FOLDER / ".workfast_tmp" / f"trim_probe_{job_id}.mp4"
                trim_editor = VideoEditor(
                    input_video=input_video,
                    output_path=trim_probe_output,
                    progress_callback=progress_callback,
                )
                try:
                    if trim_editor.has_audio:
                        candidate = trim_editor._strip_silences(
                            input_video,
                            threshold_db=silence_threshold_db,
                            min_duration=silence_min_duration,
                        )
                        if candidate != input_video and candidate.exists():
                            trimmed_video = OUTPUT_FOLDER / ".workfast_tmp" / f"trim_{job_id}.mp4"
                            trimmed_video.parent.mkdir(parents=True, exist_ok=True)
                            if trimmed_video.exists():
                                trimmed_video.unlink()
                            shutil.move(str(candidate), str(trimmed_video))
                            working_input = trimmed_video
                finally:
                    shutil.rmtree(trim_editor.temp_dir, ignore_errors=True)

            # ── Auto-deteccion de subtitulos quemados en la fuente ──────────
            # Muchos shorts ya traen subtitulos incrustados. Si vamos a poner los
            # nuestros, taparlos evita el doble subtitulo. Deteccion liviana (~1s):
            # solo se activa si el usuario NO marco el toggle manual y SI vamos a
            # generar subtitulos. Desactivable con auto_cover_existing_subs=false.
            effective_remove_subs = bool(data.get("remove_existing_subtitles", False))
            if (not effective_remove_subs
                    and data.get("generate_subtitles")
                    and bool(data.get("auto_cover_existing_subs", True))):
                try:
                    from video_processor.subtitle_detect import detect_burned_in_subtitles
                    set_job_progress(job_id, 3, "Revisando si el original ya trae subtitulos")
                    det = detect_burned_in_subtitles(working_input)
                    if det.get("present"):
                        effective_remove_subs = True
                        conf = det.get("confidence")
                        set_job_progress(job_id, 4, f"Subtitulos detectados en el original (conf {conf}); tapandolos")
                except Exception:
                    pass

            def subtitle_progress(progress: int, step: str) -> None:
                mapped = 4 + int((max(0, min(16, progress)) / 16) * 26)
                set_job_progress(job_id, mapped, step)

            def enhancer_progress(_progress: int, step: str) -> None:
                mapped = 10 + int((max(0, min(100, _progress)) / 100) * 60)
                set_job_progress(job_id, mapped, step)

            def render_progress(progress: int, step: str) -> None:
                mapped = 70 + int((max(0, min(100, progress)) / 100) * 29)
                set_job_progress(job_id, mapped, step)

            def build_subtitles() -> tuple[Path | None, str | None, str | None]:
                if not data.get("generate_subtitles"):
                    return None, None, None
                subtitle_target = output_path.with_suffix(".ass")
                temp_subtitle = OUTPUT_FOLDER / ".workfast_tmp" / f"sub_{job_id}.ass"
                temp_subtitle.parent.mkdir(parents=True, exist_ok=True)
                language_value = secure_filename(str(data.get("subtitle_language") or "original")) or "original"
                try:
                    created = generate_subtitles(
                        input_video=working_input,
                        output_ass=temp_subtitle,
                        cache_dir=SUBTITLE_CACHE_FOLDER,
                        target_language=language_value,
                        progress_callback=subtitle_progress,
                        subtitle_offset_ms=float(data.get("subtitle_offset_ms", 0.0)),
                    )
                    shutil.copy2(created, OUTPUT_FOLDER / subtitle_target.name)
                    return created, subtitle_target.name, None
                except (TranscriberError, Exception) as sub_exc:
                    return None, None, str(sub_exc)

            def build_enhanced_video() -> Path:
                if not data.get("enhance_with_ai", True):
                    return working_input
                enhanced_target = OUTPUT_FOLDER / ".workfast_tmp" / f"enh_{job_id}.mp4"
                enhanced_target.parent.mkdir(parents=True, exist_ok=True)
                print(f"[ENHANCER] Procesando {Path(working_input).name} con perfil {enhance_profile}...")
                try:
                    result = enhance_video(
                        input_video=working_input,
                        output_video=enhanced_target,
                        profile=enhance_profile,
                        progress_callback=enhancer_progress,
                    )
                    if result and Path(result).exists():
                        size_mb = Path(result).stat().st_size / 1024 / 1024
                        print(f"[ENHANCER] OK: {Path(result).name} ({size_mb:.1f} MB)")
                        return Path(result)
                    print("[ENHANCER] No se genero salida, usando entrada preparada")
                except EnhancerError as enh_exc:
                    print(f"[ENHANCER] WARNING: {enh_exc}")
                except Exception as enh_exc:
                    print(f"[ENHANCER] EXCEPCION: {enh_exc}")
                    import traceback
                    traceback.print_exc()
                return working_input

            parallel_tasks: dict = {}
            can_parallel_gpu = os.getenv("WORKFAST_GPU_PARALLEL", "0") == "1" or enhance_profile == "fast"
            if data.get("generate_subtitles") and data.get("enhance_with_ai", True) and can_parallel_gpu:
                set_job(job_id, status="processing", progress=4, step="Subtitulos y mejora IA en paralelo")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    parallel_tasks[executor.submit(build_subtitles)] = "subtitles"
                    parallel_tasks[executor.submit(build_enhanced_video)] = "enhance"
                    for future in as_completed(parallel_tasks):
                        task_name = parallel_tasks[future]
                        if task_name == "subtitles":
                            subtitle_path, subtitle_filename, subtitle_warning = future.result()
                            if subtitle_warning:
                                set_job_progress(job_id, 10, "Sin subtitulos (fallo IA)")
                        else:
                            enhanced_video = future.result()
            else:
                if data.get("generate_subtitles"):
                    subtitle_path, subtitle_filename, subtitle_warning = build_subtitles()
                    if subtitle_warning:
                        set_job_progress(job_id, 10, "Sin subtitulos (fallo IA)")
                enhanced_video = build_enhanced_video()

            raise_if_cancelled("antes de renderizar")

            editor = VideoEditor(
                input_video=enhanced_video,
                output_path=output_path,
                logo_path=data.get("logo_path") or None,
                ending_path=data.get("ending_path") or None,
                follow_image_path=data.get("follow_image_path") or None,
                subtitle_path=subtitle_path,
                progress_callback=render_progress,
                cancel_check=lambda: _job_is_cancelled(job_id),
                on_process_start=lambda proc: _register_job_process(job_id, proc),
            )
            editor.process_complete(
                title_text=title_text,
                speed=float(data.get("speed", 1.05)),
                zoom_bottom=float(data.get("zoom_bottom", 1.96)),
                zoom_top=float(data.get("zoom_top", 0.96)),
                saturation=float(data.get("saturation", 100)),
                volume_db=float(data.get("volume_db", 5.4)),
                pitch_factor=float(data.get("pitch_factor", 1.0)),
                formant_factor=float(data.get("formant_factor", 1.0)),
                voice_preset=str(data.get("voice_preset", "")),
                filter_intensity=float(data.get("filter_intensity", 0.55)),
                title_interval=float(data.get("title_interval", 10)),
                remove_existing_subtitles=effective_remove_subs,
                burn_subtitles=bool(data.get("burn_subtitles", True)),
                remove_silences=False,
                silence_threshold_db=silence_threshold_db,
                silence_min_duration=silence_min_duration,
            )

            cleanup_outputs()
            cleanup_cache(SUBTITLE_CACHE_FOLDER)

            # Auto-cleanup: deshabilitado para permitir multiples renders del mismo video.
            # try:
            #     if input_video and input_video.exists() and input_video.is_relative_to(UPLOAD_FOLDER):
            #         input_video.unlink()
            # except Exception:
            #     pass  # no fatal si falla

            set_job(
                job_id,
                status="completed",
                progress=100,
                step="Completado",
                download_url=f"/api/download/{output_filename}",
                subtitle_filename=subtitle_filename,
                subtitle_warning=subtitle_warning,
            )
        except RenderCancelled:
            # El usuario detuvo el render. Borrar el MP4 parcial (incompleto)
            # para que no aparezca en Resultados.
            set_job(job_id, status="cancelled", progress=0, step="Cancelado por el usuario")
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass
        except Exception as exc:
            # Si fue cancelado mientras corria otra cosa (ej: FFmpeg killeado por
            # el endpoint produjo un RuntimeError generico), tratarlo como cancel.
            if _job_is_cancelled(job_id):
                set_job(job_id, status="cancelled", progress=0, step="Cancelado por el usuario")
                try:
                    if output_path.exists():
                        output_path.unlink()
                except Exception:
                    pass
            else:
                set_job(job_id, status="error", progress=0, step="Error", error=clean_error_text(str(exc)))
        finally:
            if sem_acquired:
                _RENDER_SEM.release()

            def _delayed_clear() -> None:
                time.sleep(3600)  # Mantener 1 hora por si la pestana del frontend esta inactiva
                with jobs_lock:
                    if job_id in jobs:
                        del jobs[job_id]
            threading.Thread(target=_delayed_clear, daemon=True).start()

            _clear_job_tracking(job_id)
            if subtitle_path:
                subtitle_path.unlink(missing_ok=True)
            if enhanced_video != input_video and enhanced_video != working_input and enhanced_video.exists():
                enhanced_video.unlink(missing_ok=True)
            if trimmed_video and trimmed_video.exists():
                trimmed_video.unlink(missing_ok=True)

    threading.Thread(target=worker, daemon=True).start()

    return jsonify(
        {
            "success": True,
            "job_id": job_id,
            "output_filename": output_filename,
        }
    )


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado."}), 404
    return jsonify(job)


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job_endpoint(job_id: str):
    """Detiene un render en curso: setea el flag de cancelacion y mata el
    FFmpeg activo de ese job. El worker lo ve, limpia y marca 'cancelled'."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado."}), 404
    if job.get("status") not in {"queued", "processing"}:
        return jsonify({"error": "El job ya termino, no hay nada que detener."}), 400

    # 1. Señalar cancelacion (el worker la consulta entre etapas)
    ev = _job_cancel.get(job_id)
    if ev is not None:
        ev.set()

    # 2. Matar cualquier FFmpeg activo de este job (corta el render YA)
    with _job_proc_lock:
        procs = list(_job_procs.get(job_id, []))
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    # 3. Marcar el job como cancelado (el worker tambien lo hara, pero
    #    respondemos rapido para que la UI reaccione al instante).
    set_job(job_id, status="cancelled", progress=0, step="Cancelado por el usuario")
    return jsonify({"success": True})


@app.get("/api/jobs")
def list_jobs_endpoint():
    """Bulk job status endpoint so the frontend uses a single poll instead of N."""
    raw_ids = request.args.get("ids", "")
    ids = [item for item in raw_ids.split(",") if item]
    reap_old_jobs()
    with jobs_lock:
        if ids:
            data = {job_id: jobs[job_id] for job_id in ids if job_id in jobs}
        else:
            data = {job_id: job for job_id, job in jobs.items() if job.get("status") in {"queued", "processing"}}
    return jsonify({"success": True, "jobs": data})


@app.post("/api/session/reset")
def reset_session():
    """Limpia la sesion de trabajo actual: outputs renderizados, uploads
    de YouTube/TikTok/FB pendientes y temporales. NO toca perfiles ni la
    biblioteca de assets. Util al cambiar de canal sin reiniciar el server.

    Rechaza con 409 si hay un job en curso (queued/processing) para no
    borrar archivos que un render esta usando.
    """
    with jobs_lock:
        active = [
            job_id for job_id, job in jobs.items()
            if job.get("status") in {"queued", "processing"}
        ]
    if active:
        return jsonify({
            "error": (
                "Hay un render o importacion en curso. Espera a que termine "
                "o cancelalo antes de reiniciar la sesion."
            ),
            "active_jobs": len(active),
        }), 409

    deleted = {"outputs": 0, "uploads": 0, "tmp": 0}
    errors: list[str] = []

    def _wipe(folder: Path, key: str) -> None:
        if not folder.exists():
            return
        for item in folder.iterdir():
            if item.name == ".gitkeep":
                continue
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                    deleted[key] += 1
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    deleted[key] += 1
            except Exception as exc:
                errors.append(f"{item.name}: {exc}")

    _wipe(OUTPUT_FOLDER, "outputs")
    _wipe(UPLOAD_FOLDER, "uploads")
    _wipe(BASE_DIR / ".tmp", "tmp")

    with jobs_lock:
        terminal = {"completed", "error", "cancelled", "done"}
        for job_id in [jid for jid, job in jobs.items() if job.get("status") in terminal]:
            jobs.pop(job_id, None)

    return jsonify({
        "success": True,
        "deleted": deleted,
        "errors": errors[:10],
    })


@app.get("/api/preview/<filename>")
def preview_upload(filename: str):
    """Serve an uploaded video for in-app preview (supports range requests for seeking)."""
    filepath = UPLOAD_FOLDER / secure_filename(filename)
    if not filepath.exists():
        return jsonify({"error": "Archivo no encontrado."}), 404
    return send_file(filepath, conditional=True)


@app.get("/api/download/<filename>")
def download_file(filename: str):
    filepath = OUTPUT_FOLDER / secure_filename(filename)
    if not filepath.exists():
        return jsonify({"error": "Archivo no encontrado."}), 404
    return send_file(filepath, as_attachment=True, download_name=filepath.name)


@app.post("/api/download-bundle")
def download_bundle():
    data = request.get_json(silent=True) or {}
    filenames = data.get("filenames") or []
    safe_files = []
    for filename in filenames[:50]:
        filepath = OUTPUT_FOLDER / secure_filename(str(filename))
        if filepath.exists() and filepath.is_file():
            safe_files.append(filepath)

    if not safe_files:
        return jsonify({"error": "No hay videos listos para descargar."}), 400

    # ZIP_STORED: sin compresion (los MP4 ya estan comprimidos — deflate no ayuda y tarda mucho).
    # Se escribe en disco para no cargar GB en RAM.
    tmp_dir = OUTPUT_FOLDER / ".workfast_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"bundle_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for filepath in safe_files:
            archive.write(filepath, arcname=filepath.name)

    _delayed_unlink(tmp_path, delay=7200.0)
    return send_file(
        tmp_path,
        as_attachment=True,
        download_name=f"evse_videos_{datetime.utcnow().strftime('%Y%m%d')}.zip",
        mimetype="application/zip",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FACEBOOK PUBLISHER — accounts, queue, scheduler, routes
# ═══════════════════════════════════════════════════════════════════════════════

_fb_lock = threading.Lock()


def _load_fb_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_fb_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fb_accounts_load() -> list[dict]:
    return _load_fb_json(FB_ACCOUNTS_FILE).get("accounts", [])


def fb_accounts_save(accounts: list[dict]) -> None:
    _save_fb_json(FB_ACCOUNTS_FILE, {"accounts": accounts})


def fb_queue_load() -> list[dict]:
    return _load_fb_json(FB_QUEUE_FILE).get("items", [])


def fb_queue_save(items: list[dict]) -> None:
    _save_fb_json(FB_QUEUE_FILE, {"items": items})


def fb_next_slot(page_id: str) -> datetime:
    """Return next available 9am/1pm/7pm slot (local time) for this page."""
    from datetime import date as _date
    items = fb_queue_load()
    occupied: set[str] = {
        it["scheduled_time"]
        for it in items
        if it.get("page_id") == page_id and it.get("status") in {"pending", "uploading", "published"}
    }
    now = datetime.now()
    # Search up to 60 days ahead
    for day_offset in range(60):
        day = now.date() if day_offset == 0 else (_date.fromordinal(now.date().toordinal() + day_offset))
        for h, m in FB_DAILY_SLOTS:
            candidate = datetime(day.year, day.month, day.day, h, m)
            if candidate <= now:
                continue
            iso = candidate.strftime("%Y-%m-%dT%H:%M:00")
            if iso not in occupied:
                return candidate
    return now  # fallback (should never reach here)


def _fb_scheduler_loop() -> None:
    """Background thread: publishes queue items when their scheduled time arrives."""
    import time as _time
    while True:
        _time.sleep(60)
        try:
            _fb_scheduler_tick()
        except Exception:
            pass


def _fb_scheduler_tick() -> None:
    now_ts = datetime.now().timestamp()
    with _fb_lock:
        items = fb_queue_load()
        due = [it for it in items if it.get("status") == "pending"
               and it.get("scheduled_ts", 9e18) <= now_ts]
        if not due:
            return
        for it in due:
            it["status"] = "uploading"
        fb_queue_save(items)

    for item in due:
        threading.Thread(target=_fb_publish_item, args=(item["id"],), daemon=True).start()


def _fb_publish_item(item_id: str) -> None:
    from facebook_publisher import upload_reel, check_video_status, FacebookAPIError

    with _fb_lock:
        items = fb_queue_load()
        item = next((it for it in items if it["id"] == item_id), None)
        if not item:
            return

    accounts = fb_accounts_load()
    account = next((a for a in accounts if a["id"] == item.get("account_id")), None)
    page = None
    if account:
        page = next((p for p in account.get("pages", []) if p["id"] == item.get("page_id")), None)

    if not account or not page:
        _fb_update_item(item_id, status="error", error="Cuenta o página no encontrada")
        return

    page_token = page.get("access_token", "")
    if not page_token:
        _fb_update_item(item_id, status="error", error="Token de página no configurado")
        return

    video_path = Path(item.get("video_path", ""))
    if not video_path.exists():
        _fb_update_item(item_id, status="error", error="Archivo de video no encontrado")
        return

    description = item.get("description", "")
    hashtags = " ".join(item.get("hashtags", []))
    full_desc = f"{description}\n\n{hashtags}".strip()

    try:
        result = upload_reel(
            page_id=item["page_id"],
            page_token=page_token,
            video_path=video_path,
            description=full_desc,
            title=item.get("video_title", ""),
        )
        video_id = result.get("video_id", "")
        _fb_update_item(item_id, status="published",
                        fb_video_id=video_id,
                        fb_post_id=result.get("post_id", ""),
                        published_at=datetime.utcnow().isoformat() + "Z")

        # Check copyright ~5 min after upload
        def _check_copyright():
            import time as _t
            _t.sleep(300)
            try:
                status_data = check_video_status(video_id, page_token)
                cr = status_data.get("copyright_check_status", {})
                if cr.get("status") == "complete" and cr.get("matches_found"):
                    _fb_update_item(item_id, status="copyright",
                                    copyright_info=str(cr.get("matches_found", [])))
            except Exception:
                pass
        threading.Thread(target=_check_copyright, daemon=True).start()

    except FacebookAPIError as exc:
        _fb_update_item(item_id, status="error", error=str(exc))
    except Exception as exc:
        _fb_update_item(item_id, status="error", error=f"Error inesperado: {exc}")


def _fb_update_item(item_id: str, **kwargs) -> None:
    with _fb_lock:
        items = fb_queue_load()
        for it in items:
            if it["id"] == item_id:
                it.update(kwargs)
                it["updated_at"] = datetime.utcnow().isoformat() + "Z"
                break
        fb_queue_save(items)


# Start scheduler thread
threading.Thread(target=_fb_scheduler_loop, daemon=True).start()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/fb/ollama-status")
def fb_ollama_status():
    from description_ai import check_ollama
    return jsonify(check_ollama())


@app.route("/api/fb/completed-videos")
def fb_completed_videos():
    files = sorted(
        (p for p in OUTPUT_FOLDER.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:100]
    return jsonify([
        {"filename": p.name, "size": p.stat().st_size,
         "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat()}
        for p in files
    ])


@app.route("/api/fb/accounts", methods=["GET"])
def fb_accounts_get():
    return jsonify(fb_accounts_load())


@app.route("/api/fb/accounts", methods=["POST"])
def fb_accounts_add():
    from facebook_publisher import get_pages, FacebookAPIError
    body = request.get_json(force=True) or {}
    name = str(body.get("name", "")).strip()
    token = str(body.get("user_token", "")).strip()
    if not name or not token:
        return jsonify({"error": "name y user_token son requeridos"}), 400

    try:
        pages = get_pages(token)
    except FacebookAPIError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"No se pudo conectar con Facebook: {exc}"}), 502

    with _fb_lock:
        accounts = fb_accounts_load()
        account = {
            "id": uuid.uuid4().hex,
            "name": name,
            "user_token": token,
            "added_at": datetime.utcnow().isoformat() + "Z",
            "pages": [
                {"id": p["id"], "name": p["name"],
                 "access_token": p.get("access_token", ""),
                 "picture": p.get("picture", {}).get("data", {}).get("url", "") if isinstance(p.get("picture"), dict) else ""}
                for p in pages
            ],
        }
        accounts.append(account)
        fb_accounts_save(accounts)

    account_safe = {k: v for k, v in account.items() if k != "user_token"}
    account_safe["pages_count"] = len(account["pages"])
    return jsonify(account_safe), 201


@app.route("/api/fb/accounts/<account_id>", methods=["DELETE"])
def fb_accounts_delete(account_id):
    with _fb_lock:
        accounts = fb_accounts_load()
        accounts = [a for a in accounts if a["id"] != account_id]
        fb_accounts_save(accounts)
    return jsonify({"ok": True})


@app.route("/api/fb/accounts/<account_id>/refresh", methods=["POST"])
def fb_accounts_refresh(account_id):
    from facebook_publisher import get_pages, FacebookAPIError
    with _fb_lock:
        accounts = fb_accounts_load()
        account = next((a for a in accounts if a["id"] == account_id), None)
        if not account:
            return jsonify({"error": "Cuenta no encontrada"}), 404
        try:
            pages = get_pages(account["user_token"])
        except FacebookAPIError as exc:
            return jsonify({"error": str(exc)}), 400
        account["pages"] = [
            {"id": p["id"], "name": p["name"],
             "access_token": p.get("access_token", ""),
             "picture": p.get("picture", {}).get("data", {}).get("url", "") if isinstance(p.get("picture"), dict) else ""}
            for p in pages
        ]
        fb_accounts_save(accounts)
    return jsonify({"pages_count": len(account["pages"]), "pages": account["pages"]})


@app.route("/api/fb/queue", methods=["GET"])
def fb_queue_get():
    return jsonify(fb_queue_load())


@app.route("/api/fb/queue", methods=["POST"])
def fb_queue_add():
    body = request.get_json(force=True) or {}
    account_id = str(body.get("account_id", "")).strip()
    page_id = str(body.get("page_id", "")).strip()
    video_filename = secure_filename(str(body.get("video_filename", "")).strip())
    description = str(body.get("description", "")).strip()
    hashtags = [str(h).strip() for h in (body.get("hashtags") or []) if str(h).strip().startswith("#")]

    if not all([account_id, page_id, video_filename, description]):
        return jsonify({"error": "account_id, page_id, video_filename y description son requeridos"}), 400
    if len(hashtags) < 5:
        return jsonify({"error": "Se requieren mínimo 5 hashtags"}), 400

    video_path = OUTPUT_FOLDER / video_filename
    if not video_path.exists():
        return jsonify({"error": "Archivo de video no encontrado en outputs"}), 404

    accounts = fb_accounts_load()
    account = next((a for a in accounts if a["id"] == account_id), None)
    if not account:
        return jsonify({"error": "Cuenta no encontrada"}), 404
    page = next((p for p in account.get("pages", []) if p["id"] == page_id), None)
    if not page:
        return jsonify({"error": "Página no encontrada en esta cuenta"}), 404

    slot = fb_next_slot(page_id)
    import calendar
    scheduled_ts = int(calendar.timegm(slot.timetuple()))  # local→UTC not ideal; using as-is for scheduling

    item = {
        "id": uuid.uuid4().hex,
        "account_id": account_id,
        "account_name": account["name"],
        "page_id": page_id,
        "page_name": page["name"],
        "video_path": str(video_path),
        "video_filename": video_filename,
        "video_title": video_filename.replace("_", " ").replace(".mp4", ""),
        "description": description,
        "hashtags": hashtags,
        "scheduled_time": slot.strftime("%Y-%m-%dT%H:%M:00"),
        "scheduled_ts": scheduled_ts,
        "status": "pending",
        "fb_video_id": None,
        "fb_post_id": None,
        "error": None,
        "copyright_info": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "published_at": None,
    }

    with _fb_lock:
        items = fb_queue_load()
        items.append(item)
        fb_queue_save(items)

    return jsonify(item), 201


@app.route("/api/fb/queue/<item_id>", methods=["DELETE"])
def fb_queue_delete(item_id):
    with _fb_lock:
        items = fb_queue_load()
        items = [it for it in items if it["id"] != item_id]
        fb_queue_save(items)
    return jsonify({"ok": True})


@app.route("/api/fb/generate-description", methods=["POST"])
def fb_generate_description():
    from description_ai import generate_description
    body = request.get_json(force=True) or {}
    title = str(body.get("title", "")).strip()
    transcript = str(body.get("transcript", "")).strip()
    model = str(body.get("model", "")).strip() or None

    if not title:
        return jsonify({"error": "title es requerido"}), 400

    kwargs = {}
    if model:
        kwargs["model"] = model

    try:
        result = generate_description(title, transcript, **kwargs)
        return jsonify(result)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Error generando descripción: {exc}"}), 500


@app.route("/api/fb/next-slot")
def fb_next_slot_route():
    page_id = request.args.get("page_id", "")
    if not page_id:
        return jsonify({"error": "page_id requerido"}), 400
    slot = fb_next_slot(page_id)
    return jsonify({"scheduled_time": slot.strftime("%Y-%m-%dT%H:%M:00"),
                    "display": slot.strftime("%a %d/%m · %I:%M %p")})


# ─────────────────────────────────────────────────────────────────────────────
# SHORTS HUNTER — Gestion de canales para reciclar contenido
# ─────────────────────────────────────────────────────────────────────────────

def _load_hunter_channels() -> list[dict]:
    if not HUNTER_CHANNELS_FILE.exists():
        return []
    try:
        data = json.loads(HUNTER_CHANNELS_FILE.read_text(encoding="utf-8"))
        return data.get("channels", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _save_hunter_channels(channels: list[dict]) -> None:
    HUNTER_CHANNELS_FILE.write_text(
        json.dumps({"channels": channels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_hunter_seed() -> list[dict]:
    """Lista pre-curada de canales sugeridos por idioma+nicho."""
    if not HUNTER_SEED_FILE.exists():
        return []
    try:
        data = json.loads(HUNTER_SEED_FILE.read_text(encoding="utf-8"))
        return data.get("channels", []) if isinstance(data, dict) else []
    except Exception:
        return []


@app.get("/api/hunter/channels")
def hunter_list_channels():
    """Devuelve canales guardados por el usuario."""
    return jsonify({"channels": _load_hunter_channels()})


@app.post("/api/hunter/channels")
def hunter_add_channel():
    """Guardar un canal favorito. Body: {url, name, language, niche, platform}."""
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not is_supported_video_url(url):
        return jsonify({"error": "URL invalida."}), 400

    channels = _load_hunter_channels()
    # No duplicar
    if any(c.get("url") == url for c in channels):
        return jsonify({"error": "Ese canal ya esta guardado."}), 400

    channel = {
        "id": uuid.uuid4().hex,
        "url": url,
        "name": str(data.get("name", "")).strip()[:80] or url[:80],
        "language": str(data.get("language", "")).strip()[:5].lower(),
        "niche": str(data.get("niche", "")).strip()[:40],
        "platform": detect_platform(url),
        "added_at": datetime.utcnow().isoformat() + "Z",
    }
    channels.append(channel)
    _save_hunter_channels(channels)
    return jsonify({"success": True, "channel": channel}), 201


@app.delete("/api/hunter/channels/<channel_id>")
def hunter_delete_channel(channel_id: str):
    channels = _load_hunter_channels()
    new_channels = [c for c in channels if c.get("id") != channel_id]
    if len(new_channels) == len(channels):
        return jsonify({"error": "Canal no encontrado."}), 404
    _save_hunter_channels(new_channels)
    return jsonify({"success": True})


@app.get("/api/hunter/suggested")
def hunter_suggested():
    """Devuelve canales sugeridos (lista pre-curada). Filtros por query string."""
    language = request.args.get("language", "").lower().strip()
    niche = request.args.get("niche", "").lower().strip()
    seed = _load_hunter_seed()
    results = []
    for ch in seed:
        if language and ch.get("language", "").lower() != language:
            continue
        if niche and niche not in ch.get("niche", "").lower():
            continue
        results.append(ch)
    return jsonify({"channels": results, "count": len(results)})


# Mapeo de nichos -> queries de busqueda por idioma. Construido a partir de
# lenguaje natural que usan los creadores en cada plataforma/idioma.
HUNTER_NICHE_QUERIES = {
    "reddit":        {"en": "reddit stories drama",       "de": "reddit geschichten",      "pt": "histórias do reddit drama",  "es": "historias reddit drama",       "fr": "histoires reddit"},
    "curiosities":   {"en": "did you know facts",         "de": "wusstest du fakten",      "pt": "você sabia curiosidades",    "es": "sabías que curiosidades",      "fr": "le saviez-vous"},
    "true_crime":    {"en": "true crime short",           "de": "wahre verbrechen",        "pt": "casos reais crime",          "es": "casos reales crimen",          "fr": "chroniques criminelles"},
    "sitcom":        {"en": "tbbt friends sitcom clips",  "de": "sitcom clips deutsch",    "pt": "sitcom clipes",              "es": "sitcom mejores momentos",      "fr": "sitcom moments"},
    "motivational":  {"en": "motivational speech short",  "de": "motivation deutsch",      "pt": "motivacional curto",         "es": "motivacional discurso",        "fr": "motivation discours"},
    "cooking":       {"en": "quick recipe short",         "de": "schnelles rezept",        "pt": "receita rápida",             "es": "receta rápida",                "fr": "recette rapide"},
    "animals":       {"en": "cute animals funny",         "de": "lustige tiere",           "pt": "animais engraçados",         "es": "animales graciosos",           "fr": "animaux drôles"},
    "history":       {"en": "history facts short",        "de": "geschichte fakten",       "pt": "fatos históricos",           "es": "datos históricos",             "fr": "faits historiques"},
}


@app.post("/api/hunter/live-search")
def hunter_live_search():
    """Busca shorts en vivo en YouTube/TikTok usando yt-dlp.

    Body: {language, niche, custom_query?, max_results?, platform?}
    Returns: {videos: [...], query: str}
    """
    data = request.get_json(silent=True) or {}
    language = str(data.get("language", "en")).lower().strip()
    niche = str(data.get("niche", "")).lower().strip()
    custom_query = str(data.get("custom_query", "")).strip()
    max_results = max(5, min(50, int(data.get("max_results", 30))))
    platform = str(data.get("platform", "youtube")).lower().strip()

    # Construir query
    if custom_query:
        query = custom_query
    elif niche and niche in HUNTER_NICHE_QUERIES:
        query = HUNTER_NICHE_QUERIES[niche].get(language) or HUNTER_NICHE_QUERIES[niche].get("en", niche)
    else:
        return jsonify({"error": "Selecciona un nicho o escribe una busqueda."}), 400

    # yt-dlp soporta "ytsearch<N>:<query>" como pseudo-URL para buscar en YouTube
    # Para shorts, agregamos "#shorts" al query (truco para sesgar a videos verticales)
    search_query = f"ytsearch{max_results}:{query} #shorts"

    try:
        with YoutubeDL({
            "extract_flat": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "default_search": "ytsearch",
        }) as ydl:
            info = ydl.extract_info(search_query, download=False)

        entries = info.get("entries", []) if isinstance(info, dict) else []
        videos = []
        for entry in entries:
            if not entry:
                continue
            normalized = normalize_youtube_entry(entry, "youtube")
            if normalized.get("url"):
                videos.append(normalized)

        return jsonify({
            "success": True,
            "videos": videos,
            "count": len(videos),
            "query": query,
        })
    except Exception as exc:
        return jsonify({"error": f"Error en búsqueda: {clean_error_text(str(exc))}"}), 500


if __name__ == "__main__":
    # Limpieza de temporales y basura en cada inicio
    import shutil
    for folder in [".tmp", "assets/outputs"]:
        p = Path(folder)
        if p.exists():
            for item in p.iterdir():
                if item.is_file() and item.name != ".gitkeep":
                    try: item.unlink()
                    except: pass
                elif item.is_dir():
                    try: shutil.rmtree(item)
                    except: pass

    # Basura de uploads/: .part de yt-dlp interrumpidos, .ytdl manifests
    orphans_removed = cleanup_orphan_uploads()
    if orphans_removed:
        print(f"  Limpieza: {orphans_removed} descargas incompletas borradas de uploads/")

    # Banner de startup en BACKGROUND: get_nvenc_status() corre un encode de
    # prueba con FFmpeg (~1-2s) y describe_backend() consulta CUDA. Antes corrian
    # ANTES de app.run(), retrasando el arranque del server ~3-5s (y la ventana
    # del launcher esperaba a /api/health). Ahora arrancan en un thread aparte:
    # el server responde de inmediato y el banner se imprime cuando termine.
    def _print_startup_banner() -> None:
        try:
            nvenc_status = VideoEditor.get_nvenc_status()
            whisper_info = describe_backend()
            print()
            print("  Evse Video Studio  -  http://127.0.0.1:5000")
            print("  " + "-" * 48)
            if nvenc_status.get("available"):
                print("  Video    : GPU NVENC (rapido, GPU encoding)")
            else:
                print("  Video    : CPU libx264 (mas lento, CPU)")
                err = nvenc_status.get("last_error")
                if err:
                    print(f"             motivo: {err[:60]}")
            device = whisper_info.get("device", "cpu")
            model = whisper_info.get("model", "small")
            if device == "cuda":
                print(f"  Whisper  : GPU CUDA  (modelo {model}, rapido)")
            else:
                print(f"  Whisper  : CPU       (modelo {model})")
            print("  " + "-" * 48)
            print()
        except Exception as exc:
            print(f"Evse server: banner failed: {exc}")

    print("Evse Video Studio  -  http://127.0.0.1:5000 (iniciando...)")
    threading.Thread(target=_print_startup_banner, daemon=True).start()
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
