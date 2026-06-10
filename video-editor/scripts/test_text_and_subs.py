"""Tests rapidos para los fixes de titulos e .ass.

Corren sin FFmpeg ni Whisper. Verifican:
  1. Titulos no se desbordan a los lados (con/sin Pillow).
  2. Subtitulos no quedan colgados despues del habla.
  3. Audio chain switchable por env var WORKFAST_AUDIO_PROFILE.

Uso:
    python scripts\\test_text_and_subs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from video_processor.editor import VideoEditor
from video_processor.subtitles import (
    Word,
    _tighten_word_ends,
    group_words_into_phrases,
    build_ass,
)


def title_overflow_cases() -> int:
    """Verifica que ningun titulo supere el ancho seguro."""
    print("\n=== Test 1: titulos no desbordan ===")
    safe = VideoEditor.WIDTH - 2 * VideoEditor.TITLE_SAFE_MARGIN
    cases = [
        "Como ganar dinero en internet desde casa",
        "WhatsApp Web Trucos Secretos 2026",
        "Mi Video",
        "M" * 30,
        "En la calle, cualquiera puede tener una historia que te marque la vida para siempre",
        "Wow! IA en 2026 cambia todo",
        "MUNDO MUNDO MUNDO MUNDO MUNDO MUNDO",
        "AAAAAAAAAAAAAAAAAAAAAAAAAA",
    ]
    failed = 0
    for txt in cases:
        lines, fs = VideoEditor._wrap_title(VideoEditor._clean_overlay_text(txt))
        max_px = max(VideoEditor._estimate_title_pixel_width(line, fs) for line in lines)
        ok = max_px <= safe and len(lines) <= 4
        flag = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{flag}] fs={fs:>2} px={max_px:>5.0f}/{safe} lines={lines}")
    return failed


def subtitle_lag_cases() -> int:
    """Verifica que los subtitulos no se queden colgados despues del habla."""
    print("\n=== Test 2: subtitulos sin lag ===")
    # Simulamos overshoot tipico de Whisper (cada word.end se va 250ms al silencio).
    sim_words = [
        Word("Hola",  0.10, 0.95),  # real ~0.55
        Word("como",  1.00, 1.85),  # real ~1.30
        Word("estas", 1.90, 2.95),  # real ~2.45
    ]
    cleaned = _tighten_word_ends(sim_words)
    failed = 0
    for original, fixed in zip(sim_words, cleaned):
        gain = original.end - fixed.end
        ok = gain >= 0.10  # esperamos al menos 100ms recortados
        flag = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(
            f"  [{flag}] {original.text:>6} {original.end:.2f}s -> {fixed.end:.2f}s  "
            f"(recorte: {gain*1000:.0f}ms)"
        )

    # Verifica que las frases no se solapen
    phrases = group_words_into_phrases(sim_words)
    for i in range(len(phrases) - 1):
        end = phrases[i].end
        nxt = phrases[i + 1].start
        ok = end <= nxt
        flag = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{flag}] frase {i}.end={end:.2f}s <= frase {i+1}.start={nxt:.2f}s")

    # Verifica que el .ass no tenga +0.05 padding
    ass = build_ass(phrases)
    if "fad(25,25)" in ass:
        print("  [OK] fade minimo 25ms para habla rapida")
    else:
        print("  [FAIL] no se encontro fad(25,25) en .ass")
        failed += 1
    return failed


def audio_chain_cases() -> int:
    """Verifica que la cadena de audio default ya trae voice_pro."""
    print("\n=== Test 3: cadena de audio profesional por defecto ===")
    failed = 0
    saved = os.environ.get("WORKFAST_AUDIO_PROFILE")
    try:
        # Default sin env var = voice_pro
        os.environ.pop("WORKFAST_AUDIO_PROFILE", None)
        chain = VideoEditor._build_audio_chain(speed=1.05, volume_db=5.4)
        for substr, label in [
            ("anlmdn", "denoise non-local-means"),
            ("f=50", "notch 50Hz"),
            ("f=60", "notch 60Hz"),
            ("f=3000", "EQ presencia 3kHz"),
            ("f=7200", "de-esser 7.2kHz"),
            ("loudnorm=I=-14", "loudnorm streaming"),
        ]:
            ok = substr in chain
            flag = "OK" if ok else "FAIL"
            if not ok:
                failed += 1
            print(f"  [{flag}] default chain contiene {label}")

        # legacy fallback
        os.environ["WORKFAST_AUDIO_PROFILE"] = "legacy"
        chain = VideoEditor._build_audio_chain(speed=1.05, volume_db=5.4)
        ok = "afftdn" in chain
        flag = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{flag}] legacy fallback contiene afftdn")
    finally:
        if saved is None:
            os.environ.pop("WORKFAST_AUDIO_PROFILE", None)
        else:
            os.environ["WORKFAST_AUDIO_PROFILE"] = saved
    return failed


def main() -> int:
    failed = 0
    failed += title_overflow_cases()
    failed += subtitle_lag_cases()
    failed += audio_chain_cases()
    print(f"\n{'='*40}")
    if failed == 0:
        print("Todos los tests OK")
        return 0
    print(f"{failed} test(s) FALLARON")
    return 1


if __name__ == "__main__":
    sys.exit(main())
