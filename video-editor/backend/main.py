from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True  # No generar .pyc (evita errores fantasma de versiones viejas)

import threading
import time
import uuid
import base64
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

from video_processor import (
    TranscriberError,
    VideoEditor,
    cleanup_cache,
    generate_subtitles,
)
from video_processor.enhancer import enhance_video, EnhancerError
from video_processor.transcriber import describe_backend


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
    "bv*[height<=1080][fps<=60][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080][fps<=60]+ba/bv*[height<=1080]+ba/b[height<=1080]/best[height<=1080]/best",
    "bv*+ba/b",
    "best",
    None,
]

OUTPUT_LRU_KEEP = 30
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
    if not PROFILES_FILE.exists():
        return {}
    try:
        data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_profiles(profiles: dict) -> None:
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


def is_supported_video_url(url: str) -> bool:
    parsed = urlparse(url or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
        # Formato sin marca de agua: bytevc1 es el codec nativo de descarga de TikTok
        "bytevc1[height<=1080]+bestaudio/bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio",
        # Fallback: mejor video disponible sin especificar codec
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "best",
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
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "best",
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
    return jsonify(
        {
            "success": True,
            "subtitles_local": True,
            "translation_languages": ["es", "en", "pt"],
            "max_queue": 5,
            "subtitle_engine": "faster-whisper",
            "video_encoder": "h264_nvenc" if nvenc_ok else "libx264",
            "video_backend_label": "GPU (NVENC)" if nvenc_ok else "CPU (libx264)",
            "nvenc": nvenc_status,
            "whisper": whisper_info,
        }
    )


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

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@app.post("/api/process")
def process_video():
    data = request.get_json(silent=True) or {}
    input_video = Path(data.get("input_video", ""))

    if not input_video.exists():
        return jsonify({"error": "No encontre el video de entrada."}), 400

    title_text = data.get("title_text") or title_from_filename(input_video.name)
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

    def progress_callback(progress: int, step: str) -> None:
        set_job(job_id, status="processing", progress=progress, step=step)

    def worker() -> None:
        subtitle_path: Path | None = None
        subtitle_filename: str | None = None
        subtitle_warning: str | None = None
        try:
            set_job(job_id, status="processing", progress=1, step="Iniciando")
            if data.get("generate_subtitles"):
                set_job(job_id, status="processing", progress=2, step="Generando subtitulos")
                subtitle_target = output_path.with_suffix(".ass")
                temp_subtitle = OUTPUT_FOLDER / ".workfast_tmp" / f"sub_{job_id}.ass"
                temp_subtitle.parent.mkdir(parents=True, exist_ok=True)
                language_value = secure_filename(str(data.get("subtitle_language") or "original")) or "original"
                try:
                    subtitle_path = generate_subtitles(
                        input_video=input_video,
                        output_ass=temp_subtitle,
                        cache_dir=SUBTITLE_CACHE_FOLDER,
                        target_language=language_value,
                        progress_callback=progress_callback,
                    )
                    subtitle_filename = subtitle_target.name
                    shutil.copy2(subtitle_path, OUTPUT_FOLDER / subtitle_filename)
                except TranscriberError as sub_exc:
                    subtitle_warning = str(sub_exc)
                    subtitle_path = None
                    set_job(job_id, status="processing", progress=10, step="Sin subtitulos (fallo IA)")

            enhanced_video = input_video
            if data.get("enhance_with_ai", True):
                set_job(job_id, status="processing", progress=15, step="Mejorando calidad con IA")
                try:
                    enhanced_target = OUTPUT_FOLDER / ".workfast_tmp" / f"enh_{job_id}.mp4"
                    enhanced_target.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[ENHANCER] Procesando {Path(input_video).name} con Real-ESRGAN...")
                    enhanced_video = enhance_video(
                        input_video=input_video,
                        output_video=enhanced_target,
                        progress_callback=progress_callback,
                    )
                    if enhanced_video and Path(enhanced_video).exists():
                        size_mb = Path(enhanced_video).stat().st_size / 1024 / 1024
                        print(f"[ENHANCER] OK: {Path(enhanced_video).name} ({size_mb:.1f} MB)")
                    else:
                        print(f"[ENHANCER] No se genero salida, usando original")
                        enhanced_video = input_video
                except EnhancerError as enh_exc:
                    print(f"[ENHANCER] WARNING: {enh_exc}")
                    enhanced_video = input_video
                except Exception as enh_exc:
                    print(f"[ENHANCER] EXCEPCION: {enh_exc}")
                    import traceback
                    traceback.print_exc()
                    enhanced_video = input_video

            editor = VideoEditor(
                input_video=enhanced_video,
                output_path=output_path,
                logo_path=data.get("logo_path") or None,
                ending_path=data.get("ending_path") or None,
                follow_image_path=data.get("follow_image_path") or None,
                subtitle_path=subtitle_path,
                progress_callback=progress_callback,
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
                remove_existing_subtitles=bool(data.get("remove_existing_subtitles", False)),
                burn_subtitles=bool(data.get("burn_subtitles", True)),
                remove_silences=bool(data.get("remove_silences", False)),
                silence_threshold_db=float(data.get("silence_threshold_db", -32.0)),
                silence_min_duration=float(data.get("silence_min_duration", 0.45)),
            )

            cleanup_outputs()
            cleanup_cache(SUBTITLE_CACHE_FOLDER)

            set_job(
                job_id,
                status="completed",
                progress=100,
                step="Completado",
                download_url=f"/api/download/{output_filename}",
                subtitle_filename=subtitle_filename,
                subtitle_warning=subtitle_warning,
            )
        except Exception as exc:
            set_job(job_id, status="error", progress=0, step="Error", error=clean_error_text(str(exc)))
        finally:
            if subtitle_path:
                subtitle_path.unlink(missing_ok=True)
            if 'enhanced_video' in locals() and enhanced_video != input_video and enhanced_video.exists():
                enhanced_video.unlink(missing_ok=True)

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
    for filename in filenames[:5]:
        filepath = OUTPUT_FOLDER / secure_filename(str(filename))
        if filepath.exists() and filepath.is_file():
            safe_files.append(filepath)

    if not safe_files:
        return jsonify({"error": "No hay videos listos para descargar."}), 400

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filepath in safe_files:
            archive.write(filepath, arcname=filepath.name)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"workfast_videos_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip",
        mimetype="application/zip",
    )


if __name__ == "__main__":
    # Banner de startup: muestra que backend de video y de subtitulos esta activo
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
        print(f"WorkFast server: http://127.0.0.1:5000  (banner failed: {exc})")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
