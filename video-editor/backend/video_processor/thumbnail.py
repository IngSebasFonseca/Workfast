"""Generador de miniaturas profesionales a partir del propio video.

Filosofia: la miniatura es un FRAME REAL del video (nada inventado), elegido
por su IMPACTO VISUAL (nitido, colorido, con un foco claro = la "esencia" del
clip). NO se priorizan caras de personas; se busca el cuadro mas llamativo.
Luego se compone con calidad pro: grade vibrante, vignette para foco, y el
titulo en grande con tipografia bold (estilo de las miniaturas top de YouTube).

Solo usa numpy + PIL + ffmpeg (sin dependencias extra ni modelos).

API:
  generate_thumbnail(video_path, output_path, title_text="") -> Path | None
"""
from __future__ import annotations

import logging
import math
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

# Color de acento para la palabra clave del hook (amarillo viral, super legible).
ACCENT_COLOR = (255, 214, 0, 255)

LOGGER = logging.getLogger(__name__)

FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Montserrat-Black.ttf"
U2NET_PATH = Path(__file__).resolve().parents[2] / "assets" / "tools" / "u2net.onnx"
THUMB_W, THUMB_H = 1080, 1920

_ORT_SESSION = None  # cache de la sesion onnxruntime (cargar el modelo 1 vez)


def _get_u2net_session():
    """Carga U2-Net (Apache 2.0) via onnxruntime, cacheado. None si no se puede.
    Usa CPU a proposito: la inferencia 320x320 tarda ~0.3s, no vale activar CUDA
    (que ademas pide DLLs de cuDNN en el PATH que onnxruntime no siempre halla)."""
    global _ORT_SESSION
    if _ORT_SESSION is not None:
        return _ORT_SESSION or None
    if not U2NET_PATH.exists():
        _ORT_SESSION = False
        return None
    try:
        import onnxruntime as ort
        _ORT_SESSION = ort.InferenceSession(str(U2NET_PATH), providers=["CPUExecutionProvider"])
        return _ORT_SESSION
    except Exception as exc:
        LOGGER.warning("No pude cargar U2-Net: %s", exc)
        _ORT_SESSION = False
        return None


def _ai_cutout(img):
    """Devuelve una mascara PIL 'L' (alpha del sujeto) del tamano de img, o None.
    Segmentacion de saliencia con U2-Net."""
    sess = _get_u2net_session()
    if sess is None:
        return None
    try:
        import numpy as np
        from PIL import Image
        im = img.convert("RGB").resize((320, 320), Image.LANCZOS)
        arr = np.array(im).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
        m = out[0, 0]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return Image.fromarray((m * 255).astype("uint8")).resize(img.size, Image.LANCZOS)
    except Exception as exc:
        LOGGER.warning("Cutout fallo: %s", exc)
        return None


def _compose_pop(base):
    """Composicion estilo miniatura pro: sujeto recortado que 'salta' de un
    fondo oscurecido+desenfocado, con glow. Devuelve la imagen o None si no hay
    un sujeto claro (en cuyo caso el caller usa la composicion plana)."""
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    mask = _ai_cutout(base)
    if mask is None:
        return None
    arr = np.asarray(mask)
    frac = float((arr > 120).mean())
    # Necesitamos un sujeto presente pero que NO llene todo el cuadro.
    if not (0.06 <= frac <= 0.62):
        return None

    # Fondo dramatico: desenfoque + oscuro + saturado + vignette fuerte
    bg = base.filter(ImageFilter.GaussianBlur(22))
    bg = ImageEnhance.Brightness(bg).enhance(0.52)
    bg = ImageEnhance.Color(bg).enhance(1.5)
    bg = _apply_vignette(bg, strength=0.6)

    # Sujeto: nitido y vibrante
    fg = ImageEnhance.Color(base).enhance(1.42)
    fg = ImageEnhance.Contrast(fg).enhance(1.2)
    fg = ImageEnhance.Sharpness(fg).enhance(1.7)

    # Glow: dilatar la mascara, difuminar -> halo claro detras del sujeto
    glow_mask = mask.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(20))
    glow_mask = glow_mask.point(lambda p: int(min(255, p * 0.6)))
    glow = Image.new("RGB", base.size, (255, 255, 255))

    out = Image.composite(glow, bg, glow_mask)   # halo sobre el fondo
    out = Image.composite(fg, out, mask)          # sujeto nitido encima
    return out


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


def _extract_frames(video: Path, work: Path, count: int) -> list[Path]:
    duration = _probe_duration(video)
    if duration <= 0:
        return []
    # Evitar el primer y ultimo 8% (intros/cierres flojos para portada)
    fps = max(0.2, count / (duration * 0.84))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{duration * 0.08:.2f}",
        "-i", str(video),
        "-t", f"{duration * 0.84:.2f}",
        "-vf", f"fps={fps:.5f},scale={THUMB_W}:-2:flags=bicubic",
        "-frames:v", str(count + 2),
        str(work / "c_%03d.png"),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120)
    return sorted(work.glob("c_*.png"))


def _laplacian_var(gray) -> float:
    """Varianza del Laplaciano (nitidez) con numpy puro."""
    import numpy as np
    g = gray.astype(np.float64)
    lap = (
        -4 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0)
        + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )
    return float(lap[1:-1, 1:-1].var())


def _score_frame(arr_gray, arr_rgb) -> tuple[float, float]:
    """Devuelve (score, busy_top_ratio).

    score: que tan buena portada es (impacto visual).
    busy_top_ratio: cuanta "actividad" tiene la mitad superior vs total
                    (para decidir donde poner el texto, en la zona mas calma).
    """
    import numpy as np
    h, w = arr_gray.shape

    # Nitidez
    sharp = min(1.0, _laplacian_var(arr_gray) / 700.0)

    # Colorido (Hasler-Susstrunk): portadas vivas rinden mas
    r = arr_rgb[..., 0].astype(np.float64)
    g = arr_rgb[..., 1].astype(np.float64)
    b = arr_rgb[..., 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorful = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    colorful = min(1.0, colorful / 80.0)

    # Contraste (rango dinamico)
    contrast = min(1.0, float(arr_gray.std()) / 70.0)

    # Brillo en zona util (ni oscuro ni quemado)
    bright = float(arr_gray.mean()) / 255.0
    bright_score = max(0.0, min(1.0, 1.0 - abs(bright - 0.52) * 1.6))

    # Foco central: bordes concentrados en el centro = hay un sujeto/objeto claro
    gy = np.abs(np.diff(arr_gray.astype(np.float64), axis=0))
    gx = np.abs(np.diff(arr_gray.astype(np.float64), axis=1))
    edge_rows = gy.mean(axis=1) if gy.size else np.zeros(1)
    total_edge = float(gx.mean() + gy.mean()) or 1e-6
    cy0, cy1 = int(h * 0.22), int(h * 0.82)
    cx0, cx1 = int(w * 0.16), int(w * 0.84)
    center_edge = float(np.abs(np.diff(arr_gray[cy0:cy1, cx0:cx1].astype(np.float64), axis=1)).mean())
    center_focus = min(1.0, (center_edge / total_edge) * 0.9)

    score = (0.28 * sharp + 0.26 * colorful + 0.20 * contrast
             + 0.18 * center_focus + 0.08 * bright_score)

    # actividad de la mitad superior (para ubicar texto en la zona mas calma)
    half = max(1, len(edge_rows) // 2)
    top_act = float(edge_rows[:half].mean())
    bot_act = float(edge_rows[half:].mean())
    busy_top_ratio = top_act / (top_act + bot_act + 1e-6)

    return score, busy_top_ratio


def _cover_zoom(img, tw: int, th: int, zoom: float = 1.06):
    """Cover + leve zoom para una composicion mas intencional/pro."""
    from PIL import Image
    sw, sh = img.size
    scale = max(tw / sw, th / sh) * zoom
    nw, nh = int(math.ceil(sw * scale)), int(math.ceil(sh * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _apply_vignette(img, strength: float = 0.42):
    """Oscurece bordes, centro brillante: foco profesional en el sujeto."""
    from PIL import Image
    import numpy as np
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h * 0.5
    d = np.sqrt(((xx - cx) / (w * 0.64)) ** 2 + ((yy - cy) / (h * 0.64)) ** 2)
    mask = np.clip(1.0 - strength * np.clip(d - 0.55, 0, None) * 2.2, 0.4, 1.0)
    arr = np.asarray(img).astype(np.float32) * mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def _wrap_title(text: str, max_chars: int) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    return textwrap.wrap(text, width=max_chars, break_long_words=False) or [text]


def _parse_accent(text: str) -> tuple[str, set]:
    """Detecta la palabra clave a resaltar. Si el hook viene con *palabra*,
    esa es; si no, resalta la ultima palabra significativa. Devuelve
    (texto_limpio_sin_asteriscos, set_de_palabras_acento_en_minuscula)."""
    text = " ".join((text or "").split())
    accent: set[str] = set()
    marks = re.findall(r"\*([^*]+)\*", text)
    if marks:
        for phrase in marks:
            for w in phrase.split():
                accent.add(w.strip(".,:!?¡¿\"'").lower())
        clean = text.replace("*", "")
    else:
        clean = text
        words = clean.split()
        # ultima palabra de >=3 letras (evita resaltar "DE", "EL", etc.)
        for w in reversed(words):
            if len(w.strip(".,:!?¡¿\"'")) >= 3:
                accent.add(w.strip(".,:!?¡¿\"'").lower())
                break
    return clean, accent


def _wrap_px(words: list[str], draw, font, usable: float) -> list[list[str]]:
    """Envuelve por ancho real en pixeles (greedy)."""
    space_w = draw.textlength(" ", font=font)
    lines: list[list[str]] = []
    cur: list[str] = []
    cur_w = 0.0
    for w in words:
        ww = draw.textlength(w, font=font)
        add = ww if not cur else cur_w + space_w + ww
        if cur and add > usable:
            lines.append(cur)
            cur, cur_w = [w], ww
        else:
            cur.append(w)
            cur_w = add
    if cur:
        lines.append(cur)
    return lines


def _draw_hook_line(draw, words: list[str], accent: set, font, y: float, stroke: int) -> None:
    """Dibuja una linea centrada; la palabra clave en amarillo, el resto blanco.
    Cada palabra con sombra dura + borde negro grueso (estilo miniatura viral)."""
    space_w = draw.textlength(" ", font=font)
    widths = [draw.textlength(w, font=font) for w in words]
    total = sum(widths) + space_w * (len(words) - 1)
    x = (THUMB_W - total) / 2.0
    for w, ww in zip(words, widths):
        is_acc = w.strip(".,:!?¡¿\"'").lower() in accent
        color = ACCENT_COLOR if is_acc else (255, 255, 255, 255)
        draw.text((x + 6, y + 7), w, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), w, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        x += ww + space_w


def generate_thumbnail(
    video_path: str | Path,
    output_path: str | Path,
    title_text: str = "",
    *,
    sample_count: int = 26,
) -> Optional[Path]:
    """Genera la miniatura 1080x1920 desde el video. Devuelve la ruta o None."""
    video = Path(video_path)
    out = Path(output_path)
    if not video.exists():
        return None

    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageEnhance
    except ImportError:
        LOGGER.warning("PIL/numpy no disponibles; sin miniatura")
        return None

    work = Path(tempfile.mkdtemp(prefix="thumb_"))
    try:
        frames = _extract_frames(video, work, sample_count)
        if not frames:
            return None

        best = None  # (score, frame_path, busy_top_ratio)
        for fp in frames:
            try:
                rgb = Image.open(fp).convert("RGB")
            except Exception:
                continue
            arr_rgb = np.asarray(rgb)
            arr_gray = np.asarray(rgb.convert("L"))
            score, busy_top = _score_frame(arr_gray, arr_rgb)
            if best is None or score > best[0]:
                best = (score, fp, busy_top)
        if best is None:
            return None
        _, best_fp, busy_top_ratio = best

        base = Image.open(best_fp).convert("RGB")
        base = _cover_zoom(base, THUMB_W, THUMB_H, zoom=1.06)

        # IA: si hay un sujeto claro, lo recortamos y lo hacemos "saltar" del
        # fondo (look pro). Si no, composicion plana con grade + vignette.
        popped = _compose_pop(base)
        if popped is not None:
            base = popped
        else:
            base = ImageEnhance.Color(base).enhance(1.38)
            base = ImageEnhance.Contrast(base).enhance(1.18)
            base = ImageEnhance.Brightness(base).enhance(1.04)
            base = ImageEnhance.Sharpness(base).enhance(1.5)
            base = _apply_vignette(base, strength=0.42)

        # Texto en la zona MAS CALMA (menos actividad) para que se lea pro.
        text_at_top = busy_top_ratio <= 0.5

        draw = ImageDraw.Draw(base, "RGBA")
        clean_text, accent = _parse_accent(title_text)
        words = clean_text.split()
        if words:
            margin = 64
            usable = THUMB_W - 2 * margin
            # Buscar el font_size mas grande que entre en <=3 lineas
            font_size = 156
            font = None
            wrapped: list[list[str]] = [words]
            while font_size >= 54:
                try:
                    font = ImageFont.truetype(str(FONT_PATH), font_size)
                except Exception:
                    font = ImageFont.load_default()
                    break
                wrapped = _wrap_px(words, draw, font, usable)
                if len(wrapped) <= 3 and font_size * len(wrapped) <= THUMB_H * 0.40:
                    break
                font_size -= 6
            if font is None:
                font = ImageFont.load_default()

            line_h = int(font_size * 1.12)
            block_h = line_h * len(wrapped)
            if text_at_top:
                y0 = int(THUMB_H * 0.06)
                grad_top, grad_bot = 0, y0 + block_h + 90
            else:
                y0 = int(THUMB_H * 0.94) - block_h
                grad_top, grad_bot = y0 - 90, THUMB_H

            # Banda de gradiente oscuro detras del texto (legibilidad pro)
            gh = max(1, grad_bot - grad_top)
            grad = Image.new("L", (1, gh), 0)
            for i in range(gh):
                a = int(185 * (1 - abs((i / gh) - 0.5) * 2) ** 0.6)
                grad.putpixel((0, i), a)
            grad = grad.resize((THUMB_W, gh))
            shade = Image.new("RGBA", (THUMB_W, gh), (0, 0, 0, 0))
            shade.putalpha(grad)
            base.paste(Image.new("RGBA", shade.size, (0, 0, 0, 255)), (0, grad_top), shade)
            draw = ImageDraw.Draw(base, "RGBA")

            stroke = max(9, font_size // 11)
            for i, line_words in enumerate(wrapped):
                y = y0 + i * line_h
                _draw_hook_line(draw, line_words, accent, font, y, stroke)

        out.parent.mkdir(parents=True, exist_ok=True)
        base.convert("RGB").save(out, "JPEG", quality=92)
        return out
    except Exception as exc:
        LOGGER.warning("Generacion de miniatura fallo: %s", exc)
        return None
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
