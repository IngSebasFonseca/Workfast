"""Deteccion liviana de subtitulos QUEMADOS en el video fuente (sin OCR pesado).

Problema: muchos shorts de TikTok/YT ya traen subtitulos incrustados en los
pixeles. Si encima ponemos los nuestros, quedan dobles. Esto detecta si la
fuente ya trae subtitulos y a que altura, para poder taparlos.

Heuristica (rapida, corre en CPU en <1s):
  Los subtitulos quemados son texto MUY claro (blanco) con borde OSCURO, en una
  banda horizontal de la zona baja, centrado, y ESTABLE entre frames. Ese patron
  (pixel casi-blanco pegado a uno casi-negro, agrupado en filas) es raro en
  contenido natural. Muestreamos varios frames, medimos ese "text-score" por
  banda vertical y decidimos.

API:
  detect_burned_in_subtitles(video_path) -> dict con:
    present: bool
    confidence: float (0..1)
    band_top_ratio: float | None   # altura de la banda con texto (0=arriba,1=abajo)
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _probe_duration(video: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def detect_burned_in_subtitles(
    video_path: str | Path,
    *,
    sample_count: int = 10,
    bottom_region: float = 0.5,       # analiza el 50% inferior del frame
    n_bands: int = 16,                # divide la zona baja en 16 bandas horizontales
    peak_ratio: float = 2.6,          # banda pico vs mediana para decir "hay subs"
    min_edge: float = 0.03,           # densidad de borde minima en la banda pico
    center_frac: float = 0.62,        # fraccion de borde que debe estar centrada
) -> dict:
    """Devuelve si la fuente trae subtitulos quemados y a que altura.

    Discriminador: los subtitulos forman una banda horizontal de ALTA densidad
    de bordes que SOBRESALE del resto (ratio pico/mediana alto) y esta CENTRADA.
    Un patron uniforme (mucho detalle en todos lados) tiene ratio bajo -> no
    dispara. Texto disperso natural no esta centrado -> no dispara.
    """
    video = Path(video_path)
    result = {"present": False, "confidence": 0.0, "band_top_ratio": None}
    if not video.exists():
        return result

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        LOGGER.warning("numpy/PIL no disponibles; salteando deteccion de subs")
        return result

    duration = _probe_duration(video)
    if duration <= 0:
        return result

    work = Path(tempfile.mkdtemp(prefix="subdetect_"))
    try:
        # Frames espaciados, en gris, 480 de ancho con bicubico (preserva bordes
        # del texto; 'area' los difuminaba y perdiamos la deteccion).
        fps = max(0.2, sample_count / duration)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video),
            "-vf", f"fps={fps:.5f},scale=480:-1:flags=bicubic,format=gray",
            "-frames:v", str(sample_count + 2),
            str(work / "f_%03d.png"),
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        frames = sorted(work.glob("f_*.png"))
        if len(frames) < 3:
            return result

        band_edge = np.zeros(n_bands, dtype=np.float64)   # densidad de borde por banda (zona baja)
        band_center = np.zeros(n_bands, dtype=np.float64)  # fraccion centrada del borde
        n_used = 0

        for fp in frames:
            arr = np.asarray(Image.open(fp).convert("L"), dtype=np.float32)
            h, w = arr.shape
            if h < 8 or w < 8:
                continue
            # gradiente horizontal (los bordes verticales del texto destacan)
            gx = np.abs(np.diff(arr, axis=1))          # (h, w-1)
            edge = gx >= 55                            # pixel "borde fuerte"
            y0 = int(h * (1.0 - bottom_region))
            sub = edge[y0:, :]                          # zona baja
            sh = sub.shape[0]
            cx0, cx1 = int(w * 0.19), int(w * 0.81)     # franja central horizontal
            for b in range(n_bands):
                r0 = int(sh * b / n_bands)
                r1 = int(sh * (b + 1) / n_bands)
                if r1 <= r0:
                    continue
                strip = sub[r0:r1, :]
                total = float(strip.mean())             # densidad de borde de la banda
                band_edge[b] += total
                centered = float(strip[:, cx0:cx1].sum())
                allg = float(strip.sum()) or 1.0
                band_center[b] += centered / allg
            n_used += 1

        if n_used == 0:
            return result
        band_edge /= n_used
        band_center /= n_used

        median = float(np.median(band_edge)) or 1e-6
        peak_idx = int(band_edge.argmax())
        peak = float(band_edge[peak_idx])
        ratio = peak / median
        centered_ok = band_center[peak_idx] >= center_frac

        present = (ratio >= peak_ratio) and (peak >= min_edge) and centered_ok
        # Confianza combina cuanto sobresale y cuan centrado esta
        confidence = min(1.0, (ratio / (peak_ratio * 1.8)) * (0.5 + 0.5 * band_center[peak_idx]))

        if present:
            within = (peak_idx + 0.5) / n_bands
            band_top_ratio = (1.0 - bottom_region) + within * bottom_region
            result.update(present=True, confidence=round(confidence, 3),
                          band_top_ratio=round(band_top_ratio, 3))
        else:
            result.update(present=False, confidence=round(confidence, 3))
        return result
    except Exception as exc:
        LOGGER.warning("Deteccion de subs fallo: %s", exc)
        return result
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
