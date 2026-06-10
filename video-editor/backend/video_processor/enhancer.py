"""IA-based video enhancement: Real-ESRGAN upscale + RIFE 60fps interpolation.

Usa los binarios ncnn-vulkan oficiales que corren en CUALQUIER GPU via Vulkan
(no requiere CUDA Toolkit ni PyTorch). Se descargan en el primer uso a
`assets/tools/`. Diseñados para ejecutarse local en Windows.

API publica:
  enhance_video(input_video, output_video, target_w, target_h, ...) -> Path
    - En modo balanced aplica upscale solo si la fuente es realmente pequena.
    - En modo ultra puede aplicar Real-ESRGAN y RIFE a 60fps.
    - Devuelve la ruta al video mejorado (o el mismo input si no hizo falta).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[int, str], None]

# Tools directory: <repo_root>/assets/tools
TOOLS_DIR = Path(__file__).resolve().parents[2] / "assets" / "tools"

REAL_ESRGAN_DIR_NAME = "realesrgan-ncnn-vulkan-20220424"
REAL_ESRGAN_ZIP_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"
)

RIFE_DIR_NAME = "rife-ncnn-vulkan-20221029-windows"
RIFE_ZIP_URL = (
    "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
    "rife-ncnn-vulkan-20221029-windows.zip"
)


class EnhancerError(RuntimeError):
    """Raised when enhancement cannot proceed (download failed, GPU missing, etc.)."""


def _ncnn_gpu_args() -> list[str]:
    """Force the discrete GPU by default; set WORKFAST_ENHANCER_GPU=auto to let ncnn choose."""
    gpu_id = os.getenv("WORKFAST_ENHANCER_GPU", "0").strip()
    if not gpu_id or gpu_id.lower() == "auto":
        return []
    return ["-g", gpu_id]


def normalize_enhancer_profile(profile: str | None = None) -> str:
    """Return the enhancement profile used by the queue.

    balanced is intentionally conservative: it keeps the RTX available for
    Whisper/NVENC and avoids RIFE unless the user explicitly chooses ultra.
    """
    raw = (profile or os.getenv("WORKFAST_ENHANCER_PROFILE") or "balanced").strip().lower()
    aliases = {
        "auto": "balanced",
        "smart": "balanced",
        "normal": "balanced",
        "inteligente": "balanced",
        "pro": "ultra",
        "quality": "ultra",
        "calidad": "ultra",
        "rapido": "fast",
        "speed": "fast",
        "off": "fast",
        "none": "fast",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in {"fast", "balanced", "ultra"} else "balanced"


def _ncnn_jobs_args(profile: str | None = None) -> list[str]:
    """ncnn load:process:save workers.

    The previous 2:4:2 default could pin the RTX at 100% for minutes. Balanced
    mode leaves thermal headroom; ultra can still be pushed through .env.
    """
    configured = os.getenv("WORKFAST_NCNN_JOBS", "").strip()
    if configured:
        return ["-j", configured]
    jobs = "2:4:2" if normalize_enhancer_profile(profile) == "ultra" else "1:2:1"
    return ["-j", jobs] if jobs else []


def _has_encoder(name: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return False
    return result.returncode == 0 and name in result.stdout


def _intermediate_video_encoder_args() -> list[str]:
    """Fast high-quality intermediate encode; final export still controls delivery settings."""
    if os.getenv("WORKFAST_FORCE_CPU") != "1" and _has_encoder("h264_nvenc"):
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p7",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "15",
            "-b:v", "0",
            "-spatial-aq", "1",
            "-temporal-aq", "1",
            "-aq-strength", "8",
            "-pix_fmt", "yuv420p",
        ]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p"]


def _download_and_extract(url: str, dest_dir: Path, progress_callback: Optional[ProgressCallback]) -> Path:
    """Descarga un zip y extrae al destino. Devuelve dest_dir."""
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir.parent / (dest_dir.name + ".zip")
    if progress_callback:
        progress_callback(0, f"Descargando {dest_dir.name} (~30 MB)")
    LOGGER.info("Descargando %s -> %s", url, zip_path)
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:
        raise EnhancerError(f"No pude descargar {url}: {exc}") from exc

    if progress_callback:
        progress_callback(0, f"Extrayendo {dest_dir.name}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir.parent)
    except Exception as exc:
        raise EnhancerError(f"No pude extraer {zip_path}: {exc}") from exc
    finally:
        zip_path.unlink(missing_ok=True)

    if not dest_dir.exists():
        # Algunos zip extraen en un nombre diferente; buscar la carpeta resultante
        candidates = [p for p in dest_dir.parent.iterdir() if p.is_dir() and p.name.startswith(dest_dir.name.split("-")[0])]
        if candidates:
            return candidates[0]
        direct_exe = next(dest_dir.parent.glob(f"{dest_dir.name.split('-')[0]}*.exe"), None)
        if direct_exe:
            return dest_dir.parent
        raise EnhancerError(f"Extraccion exitosa pero no encuentro {dest_dir}")
    return dest_dir


def ensure_realesrgan(progress_callback: Optional[ProgressCallback] = None) -> Path:
    """Asegura el binario realesrgan-ncnn-vulkan.exe disponible. Lo descarga si falta."""
    install_dir = TOOLS_DIR / REAL_ESRGAN_DIR_NAME
    binary = install_dir / "realesrgan-ncnn-vulkan.exe"
    if binary.exists():
        return binary
    direct_binary = TOOLS_DIR / "realesrgan-ncnn-vulkan.exe"
    if direct_binary.exists():
        return direct_binary
    install_dir = _download_and_extract(REAL_ESRGAN_ZIP_URL, install_dir, progress_callback)
    binary = install_dir / "realesrgan-ncnn-vulkan.exe"
    if not binary.exists():
        raise EnhancerError(f"realesrgan-ncnn-vulkan.exe no encontrado tras extraer en {install_dir}")
    return binary


def ensure_rife(progress_callback: Optional[ProgressCallback] = None) -> Path:
    install_dir = TOOLS_DIR / RIFE_DIR_NAME
    binary = install_dir / "rife-ncnn-vulkan.exe"
    if binary.exists():
        return binary
    direct_binary = TOOLS_DIR / "rife-ncnn-vulkan.exe"
    if direct_binary.exists():
        return direct_binary
    install_dir = _download_and_extract(RIFE_ZIP_URL, install_dir, progress_callback)
    binary = install_dir / "rife-ncnn-vulkan.exe"
    if not binary.exists():
        raise EnhancerError(f"rife-ncnn-vulkan.exe no encontrado tras extraer en {install_dir}")
    return binary


def _probe_video(video: Path) -> tuple[int, int, float, float]:
    """Devuelve (width, height, fps, duration_seconds) usando ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
        "-of",
        "default=nw=1",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise EnhancerError(f"ffprobe fallo: {result.stderr}")
    width = height = 0
    fps = 30.0
    duration = 0.0
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "width":
            width = int(val)
        elif key == "height":
            height = int(val)
        elif key in ("r_frame_rate", "avg_frame_rate") and "/" in val:
            num, den = val.split("/")
            try:
                num_f, den_f = float(num), float(den)
                if den_f > 0:
                    fps = num_f / den_f
            except ValueError:
                pass
        elif key == "duration":
            try:
                parsed = float(val)
                if parsed > duration:
                    duration = parsed
            except ValueError:
                pass
    return width, height, fps, duration


def _probe_resolution_and_fps(video: Path) -> tuple[int, int, float]:
    """Devuelve (width, height, fps) usando ffprobe."""
    width, height, fps, _duration = _probe_video(video)
    return width, height, fps


def _count_frames(folder: Path) -> int:
    try:
        return sum(1 for _ in folder.glob("*.png"))
    except OSError:
        return 0


def _run(
    cmd: list[str],
    desc: str,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 0,
    progress_end: int = 0,
    progress_step: str | None = None,
    frame_dir: Path | None = None,
    expected_frames: int = 0,
) -> None:
    LOGGER.info("[%s] %s", desc, " ".join(cmd))
    if not progress_callback or not frame_dir or expected_frames <= 0 or progress_end <= progress_start:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise EnhancerError(f"{desc} fallo: {result.stderr.strip()[:500]}")
        return

    step = progress_step or desc
    last_percent = progress_start - 1
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        while process.poll() is None:
            produced = _count_frames(frame_dir)
            ratio = max(0.0, min(1.0, produced / max(expected_frames, 1)))
            percent = progress_start + int(ratio * (progress_end - progress_start))
            if percent > last_percent:
                progress_callback(percent, f"{step} ({produced}/{expected_frames})")
                last_percent = percent
            time.sleep(1.0)

        return_code = process.wait()
        produced = _count_frames(frame_dir)
        if return_code == 0:
            progress_callback(progress_end, f"{step} ({produced}/{expected_frames})")
            return

        log_file.seek(0)
        log_tail = log_file.read().strip()[-1200:]
        raise EnhancerError(f"{desc} fallo: {log_tail}")


def _map_progress(
    callback: Optional[ProgressCallback],
    start: int,
    end: int,
) -> Optional[ProgressCallback]:
    if callback is None:
        return None

    def mapped(progress: int, step: str) -> None:
        clamped = max(0, min(100, int(progress)))
        callback(start + int((clamped / 100.0) * (end - start)), step)

    return mapped


def upscale_with_realesrgan(
    input_video: Path,
    output_video: Path,
    *,
    scale: int = 4,
    model: str = "realesrgan-x4plus",
    ncnn_profile: str | None = None,
    binary: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Upscale 2x/3x/4x con Real-ESRGAN.

    Modelos disponibles (incluidos en el .zip oficial):
        realesrgan-x4plus       -> general / fotorealismo (DEFAULT, recomendado para vlog/talking head)
        realesr-animevideov3    -> rapido pero suave en cara real (era el viejo default)
        realesrgan-x4plus-anime -> animacion
        realesrnet-x4plus       -> red base sin GAN (mas conservadora, menos detalle pero menos artefactos)

    Pipeline: extrae frames PNG -> upscale c/u con Real-ESRGAN -> reensambla con audio original.
    """
    if binary is None:
        binary = ensure_realesrgan(progress_callback)

    work_dir = output_video.parent / f".enhance_{output_video.stem}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    frames_in = work_dir / "in"
    frames_out = work_dir / "out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Detectar fps original para reensamblar
        _, _, fps, duration = _probe_video(input_video)
        expected_frames = max(1, int(round(duration * fps))) if duration > 0 else 0

        if progress_callback:
            progress_callback(2, "Extrayendo frames para upscale IA")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_video),
                "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                str(frames_in / "%08d.png"),
            ],
            "extract frames",
            progress_callback=progress_callback,
            progress_start=3,
            progress_end=14,
            progress_step="Extrayendo frames para upscale IA",
            frame_dir=frames_in,
            expected_frames=expected_frames,
        )
        n_input = _count_frames(frames_in)
        if n_input <= 0:
            raise EnhancerError("No se extrajeron frames para Real-ESRGAN.")

        if progress_callback:
            progress_callback(15, f"Upscale IA {scale}x con Real-ESRGAN")
        # Real-ESRGAN procesa carpeta entera con paralelismo interno (ncnn batch).
        _run(
            [
                str(binary),
                "-i", str(frames_in),
                "-o", str(frames_out),
                "-n", model,
                "-s", str(scale),
                "-f", "png",
                *_ncnn_gpu_args(),
                *_ncnn_jobs_args(ncnn_profile),
            ],
            "realesrgan",
            progress_callback=progress_callback,
            progress_start=16,
            progress_end=82,
            progress_step=f"Upscale IA {scale}x con Real-ESRGAN",
            frame_dir=frames_out,
            expected_frames=n_input,
        )

        if progress_callback:
            progress_callback(86, "Reensamblando video upscaleado")
        # Reensambla con audio del original
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", f"{fps:.5f}",
                "-i", str(frames_out / "%08d.png"),
                "-i", str(input_video),
                "-map", "0:v:0", "-map", "1:a?",
                *_intermediate_video_encoder_args(),
                "-c:a", "copy",
                "-shortest",
                "-movflags", "+faststart",
                str(output_video),
            ],
            "reassemble",
        )
        if progress_callback:
            progress_callback(100, "Upscale IA listo")
        return output_video
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def interpolate_to_60fps_with_rife(
    input_video: Path,
    output_video: Path,
    *,
    target_fps: int = 60,
    ncnn_profile: str | None = None,
    binary: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Interpola frames con RIFE para llegar a target_fps reales (no duplicados).

    RIFE genera frames intermedios usando flow estimation neural. Resultado:
    movimiento fluido 60fps desde fuentes 30fps reales.
    """
    if binary is None:
        binary = ensure_rife(progress_callback)

    width, height, src_fps, duration = _probe_video(input_video)
    if src_fps >= target_fps - 0.5:
        # Ya esta a 60+ fps o por encima
        shutil.copy2(input_video, output_video)
        if progress_callback:
            progress_callback(100, "Video ya esta a 60fps")
        return output_video

    # Cuantos frames necesitamos por cada par original
    multiplier = max(2, round(target_fps / max(src_fps, 1)))

    work_dir = output_video.parent / f".rife_{output_video.stem}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    frames_in = work_dir / "in"
    frames_out = work_dir / "out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    try:
        expected_frames = max(1, int(round(duration * src_fps))) if duration > 0 else 0
        if progress_callback:
            progress_callback(2, "Extrayendo frames para RIFE")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_video),
                "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                str(frames_in / "%08d.png"),
            ],
            "extract frames rife",
            progress_callback=progress_callback,
            progress_start=3,
            progress_end=14,
            progress_step="Extrayendo frames para RIFE",
            frame_dir=frames_in,
            expected_frames=expected_frames,
        )

        n_input = _count_frames(frames_in)
        if n_input <= 0:
            raise EnhancerError("No se extrajeron frames para RIFE.")
        n_output = n_input * multiplier

        if progress_callback:
            progress_callback(15, f"Interpolando IA {src_fps:.0f}->{target_fps}fps con RIFE")
        _run(
            [
                str(binary),
                "-i", str(frames_in),
                "-o", str(frames_out),
                "-n", str(n_output),
                "-m", "rife-v4.6",  # modelo balanced
                "-f", "%08d.png",
                *_ncnn_gpu_args(),
                *_ncnn_jobs_args(ncnn_profile),
            ],
            "rife interpolate",
            progress_callback=progress_callback,
            progress_start=16,
            progress_end=84,
            progress_step=f"Interpolando IA {src_fps:.0f}->{target_fps}fps con RIFE",
            frame_dir=frames_out,
            expected_frames=n_output,
        )

        if progress_callback:
            progress_callback(88, "Reensamblando 60fps")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(target_fps),
                "-i", str(frames_out / "%08d.png"),
                "-i", str(input_video),
                "-map", "0:v:0", "-map", "1:a?",
                *_intermediate_video_encoder_args(),
                "-c:a", "copy",
                "-shortest",
                "-movflags", "+faststart",
                str(output_video),
            ],
            "reassemble rife",
        )
        if progress_callback:
            progress_callback(100, "Interpolacion IA lista")
        return output_video
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _estimate_bitrate_kbps(input_video: Path) -> float:
    """Estima el bitrate del video en kbps. Util para decidir si una fuente
    1080p esta comprimida (vale la pena pasar IA) o ya es nitida (no tocar)."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate",
            "-show_entries", "format=bit_rate,duration,size",
            "-of", "default=nw=1",
            str(input_video),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            return 0.0
        stream_br = format_br = duration = size = 0.0
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            try:
                v = float(val.strip())
            except ValueError:
                continue
            if key.strip() == "bit_rate":
                if stream_br == 0:
                    stream_br = v
                else:
                    format_br = v
            elif key.strip() == "duration":
                duration = v
            elif key.strip() == "size":
                size = v
        if stream_br > 0:
            return stream_br / 1000.0
        if format_br > 0:
            return format_br / 1000.0
        if duration > 0 and size > 0:
            return (size * 8.0 / duration) / 1000.0
    except Exception:
        pass
    return 0.0


def enhance_video(
    input_video: Path,
    output_video: Path,
    *,
    target_fps: int = 60,
    do_upscale: bool = True,
    do_interpolate: bool = True,
    model: str = "realesrgan-x4plus",
    profile: str | None = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Pipeline completo de mejora IA.

    Perfiles:
      - fast: no corre redes neuronales de video; el render final igual usa
        NVENC, escalado Lanczos y filtros de nitidez.
      - balanced: default para cola. Omite redes neuronales de video para no
        clavar la RTX; el render final aplica Lanczos/NVENC/nitidez.
      - ultra: modo antiguo/profundo. Usa Real-ESRGAN/RIFE cuando aplica.
    """
    profile = normalize_enhancer_profile(profile)
    width, height, fps, duration = _probe_video(input_video)
    bitrate_kbps = _estimate_bitrate_kbps(input_video)
    short_edge = min(width, height)
    LOGGER.info(
        "Source: %dx%d @ %.2f fps, %.1fs, bitrate ~%.0f kbps, enhancer=%s",
        width, height, fps, duration, bitrate_kbps, profile,
    )

    if profile in {"fast", "balanced"}:
        label = "rapido" if profile == "fast" else "inteligente"
        if progress_callback:
            progress_callback(100, f"Modo {label}: IA pesada de video omitida")
        if input_video != output_video:
            shutil.copy2(input_video, output_video)
        return output_video

    # Decidir scale honestamente por el borde corto. En vertical, 720x1280 es
    # 720p aunque height sea 1280; usar solo height haria trabajar de mas.
    if short_edge < 540:
        scale = 4
        reason = f"muy baja resolucion ({width}x{height})"
    elif short_edge < 900:
        scale = 2
        reason = f"fuente 720p/vertical ({width}x{height})"
    elif short_edge < 1300 and bitrate_kbps > 0 and bitrate_kbps < 3500:
        scale = 2
        reason = f"1080p comprimido ({bitrate_kbps:.0f} kbps)"
    else:
        scale = 0
        reason = (
            f"ya esta nitido ({width}x{height}, {bitrate_kbps:.0f} kbps) - "
            "IA aqui agregaria artefactos"
        )
    needs_interp = do_interpolate and fps < target_fps - 0.5

    needs_upscale = do_upscale and scale > 0

    if not needs_upscale and not needs_interp:
        LOGGER.info("Skip enhance: %s; fps=%.1f", reason, fps)
        if progress_callback:
            progress_callback(100, f"IA de video omitida: {reason}")
        if input_video != output_video:
            shutil.copy2(input_video, output_video)
        return output_video

    current = input_video
    work_intermediate = output_video.parent / f"_enh_intermediate_{output_video.stem}.mp4"

    if needs_upscale:
        LOGGER.info("Aplicando Real-ESRGAN modelo=%s scale=%dx (motivo: %s)",
                    model, scale, reason)
        if progress_callback:
            progress_callback(0, f"Mejorando IA {scale}x: {reason}")
        target = work_intermediate if needs_interp else output_video
        upscale_with_realesrgan(
            current, target,
            scale=scale,
            model=model,
            ncnn_profile=profile,
            progress_callback=_map_progress(
                progress_callback,
                3,
                47 if needs_interp else 92,
            ),
        )
        current = target
        LOGGER.info("Real-ESRGAN OK -> %s", target)
    elif needs_interp:
        LOGGER.info("Skip upscale (%s) pero aplicando RIFE 60fps", reason)

    if needs_interp:
        interpolate_to_60fps_with_rife(
            current, output_video,
            target_fps=target_fps,
            ncnn_profile=profile,
            progress_callback=_map_progress(progress_callback, 48 if needs_upscale else 3, 92),
        )
        if work_intermediate.exists():
            work_intermediate.unlink(missing_ok=True)

    if progress_callback:
        progress_callback(100, "Mejora IA lista")
    return output_video
