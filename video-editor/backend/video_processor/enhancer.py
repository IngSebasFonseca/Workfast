"""IA-based video enhancement: Real-ESRGAN upscale + RIFE 60fps interpolation.

Usa los binarios ncnn-vulkan oficiales que corren en CUALQUIER GPU via Vulkan
(no requiere CUDA Toolkit ni PyTorch). Se descargan en el primer uso a
`assets/tools/`. Diseñados para ejecutarse local en Windows.

API publica:
  enhance_video(input_video, output_video, target_w, target_h, ...) -> Path
    - Detecta automaticamente si el video es <1080p y aplica upscale 2x/4x
      con Real-ESRGAN si hace falta.
    - Si fps < 60, interpola con RIFE a 60fps (movimiento suave real,
      no duplicacion de frames).
    - Devuelve la ruta al video mejorado (o el mismo input si no hizo falta).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
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
    "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"
)

RIFE_DIR_NAME = "rife-ncnn-vulkan-20221029-windows"
RIFE_ZIP_URL = (
    "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
    "rife-ncnn-vulkan-20221029-windows.zip"
)


class EnhancerError(RuntimeError):
    """Raised when enhancement cannot proceed (download failed, GPU missing, etc.)."""


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
        raise EnhancerError(f"Extraccion exitosa pero no encuentro {dest_dir}")
    return dest_dir


def ensure_realesrgan(progress_callback: Optional[ProgressCallback] = None) -> Path:
    """Asegura el binario realesrgan-ncnn-vulkan.exe disponible. Lo descarga si falta."""
    install_dir = TOOLS_DIR / REAL_ESRGAN_DIR_NAME
    binary = install_dir / "realesrgan-ncnn-vulkan.exe"
    if binary.exists():
        return binary
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
    install_dir = _download_and_extract(RIFE_ZIP_URL, install_dir, progress_callback)
    binary = install_dir / "rife-ncnn-vulkan.exe"
    if not binary.exists():
        raise EnhancerError(f"rife-ncnn-vulkan.exe no encontrado tras extraer en {install_dir}")
    return binary


def _probe_resolution_and_fps(video: Path) -> tuple[int, int, float]:
    """Devuelve (width, height, fps) usando ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "default=nw=1",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise EnhancerError(f"ffprobe fallo: {result.stderr}")
    width = height = 0
    fps = 30.0
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
    return width, height, fps


def _run(cmd: list[str], desc: str) -> None:
    LOGGER.info("[%s] %s", desc, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise EnhancerError(f"{desc} fallo: {result.stderr.strip()[:500]}")


def upscale_with_realesrgan(
    input_video: Path,
    output_video: Path,
    *,
    scale: int = 4,
    model: str = "realesrgan-x4plus",
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
        _, _, fps = _probe_resolution_and_fps(input_video)

        if progress_callback:
            progress_callback(0, "Extrayendo frames para upscale IA")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_video),
                "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                str(frames_in / "%08d.png"),
            ],
            "extract frames",
        )

        if progress_callback:
            progress_callback(0, f"Upscale IA {scale}x con Real-ESRGAN")
        # Real-ESRGAN procesa carpeta entera con paralelismo interno (ncnn batch).
        _run(
            [
                str(binary),
                "-i", str(frames_in),
                "-o", str(frames_out),
                "-n", model,
                "-s", str(scale),
                "-f", "png",
            ],
            "realesrgan",
        )

        if progress_callback:
            progress_callback(0, "Reensamblando video upscaleado")
        # Reensambla con audio del original
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", f"{fps:.5f}",
                "-i", str(frames_out / "%08d.png"),
                "-i", str(input_video),
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-shortest",
                str(output_video),
            ],
            "reassemble",
        )
        return output_video
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def interpolate_to_60fps_with_rife(
    input_video: Path,
    output_video: Path,
    *,
    target_fps: int = 60,
    binary: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Interpola frames con RIFE para llegar a target_fps reales (no duplicados).

    RIFE genera frames intermedios usando flow estimation neural. Resultado:
    movimiento fluido 60fps desde fuentes 30fps reales.
    """
    if binary is None:
        binary = ensure_rife(progress_callback)

    width, height, src_fps = _probe_resolution_and_fps(input_video)
    if src_fps >= target_fps - 0.5:
        # Ya esta a 60+ fps o por encima
        shutil.copy2(input_video, output_video)
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
        if progress_callback:
            progress_callback(0, "Extrayendo frames para RIFE")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_video),
                "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                str(frames_in / "%08d.png"),
            ],
            "extract frames rife",
        )

        n_input = len(list(frames_in.glob("*.png")))
        n_output = n_input * multiplier

        if progress_callback:
            progress_callback(0, f"Interpolando IA {src_fps:.0f}->{target_fps}fps con RIFE")
        _run(
            [
                str(binary),
                "-i", str(frames_in),
                "-o", str(frames_out),
                "-n", str(n_output),
                "-m", "rife-v4.6",  # modelo balanced
                "-f", "%08d.png",
            ],
            "rife interpolate",
        )

        if progress_callback:
            progress_callback(0, "Reensamblando 60fps")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(target_fps),
                "-i", str(frames_out / "%08d.png"),
                "-i", str(input_video),
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-shortest",
                str(output_video),
            ],
            "reassemble rife",
        )
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
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Pipeline completo de mejora IA. Solo aplica cuando hay win REAL.

    Reglas honestas (no forzar mejoras que se vean peor):
      - <720p (480p, 360p):           4x upscale (gran win, fuente low-res)
      - 720p:                         4x upscale (win claro, downsize a 1080p preserva detalle)
      - 1080p comprimido (<3 Mbps):   2x detail pass (recupera detalle perdido por compresion)
      - 1080p limpio (>=3 Mbps):      SKIP (ya esta nitido, IA agrega artefactos)
      - >=1440p:                      SKIP (mejor que 1080p ya, IA solo lo empeora)

    fps:
      - <60fps:  RIFE 60fps (movimiento suave real)
      - >=60fps: SKIP
    """
    width, height, fps = _probe_resolution_and_fps(input_video)
    bitrate_kbps = _estimate_bitrate_kbps(input_video)
    LOGGER.info("Source: %dx%d @ %.2f fps, bitrate ~%.0f kbps",
                width, height, fps, bitrate_kbps)

    # Decidir scale honestamente
    if height < 720:
        scale = 4
        reason = f"low-res ({height}p)"
    elif height < 1000:
        scale = 4
        reason = f"720p->4K detail boost"
    elif height < 1300 and bitrate_kbps > 0 and bitrate_kbps < 3500:
        scale = 2
        reason = f"1080p comprimido ({bitrate_kbps:.0f} kbps)"
    else:
        scale = 0
        reason = (f"ya esta nitido ({height}p, {bitrate_kbps:.0f} kbps) - "
                  f"IA aqui agregaria artefactos")

    needs_upscale = do_upscale and scale > 0
    needs_interp = do_interpolate and fps < target_fps - 0.5

    if not needs_upscale and not needs_interp:
        LOGGER.info("Skip enhance: %s; fps=%.1f", reason, fps)
        if progress_callback:
            progress_callback(0, "Video ya en buena calidad, IA omitida (no la fuerzo)")
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
            progress_callback=progress_callback,
        )
        current = target
        LOGGER.info("Real-ESRGAN OK -> %s", target)
    elif needs_interp:
        LOGGER.info("Skip upscale (%s) pero aplicando RIFE 60fps", reason)

    if needs_interp:
        interpolate_to_60fps_with_rife(
            current, output_video,
            target_fps=target_fps,
            progress_callback=progress_callback,
        )
        if work_intermediate.exists():
            work_intermediate.unlink(missing_ok=True)

    return output_video
