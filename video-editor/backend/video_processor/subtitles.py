"""ASS subtitle generator with Opus Clips style word-by-word karaoke highlighting."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Word:
    """A single transcribed word with timing in seconds."""

    text: str
    start: float
    end: float


@dataclass
class Phrase:
    """A short subtitle line composed of 1-3 words."""

    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:01d}:{minutes:02d}:{secs:05.2f}"


def _tighten_word_ends(
    words: Sequence[Word],
    *,
    max_overshoot: float = 0.18,
    end_gap: float = 0.03,
) -> list[Word]:
    """Recorta el `end` de cada palabra para que no se cuele en la siguiente.

    Whisper tiende a sobrestimar el end de palabra y empujarlo hasta el siguiente
    silencio (overshoot). Si la siguiente palabra arranca en T, capamos el end
    actual en T - end_gap. Tambien limitamos el ancho maximo de cada palabra
    a `start + word_len*0.085 + max_overshoot` para evitar palabras "infinitas".
    """
    cleaned: list[Word] = []
    for index, word in enumerate(words):
        new_end = word.end
        if index + 1 < len(words):
            next_start = words[index + 1].start
            if new_end > next_start - end_gap:
                new_end = max(word.start + 0.05, next_start - end_gap)
        # Cap por longitud razonable de palabra
        natural = word.start + 0.18 + len(word.text) * 0.085 + max_overshoot
        new_end = min(new_end, natural)
        # Garantiza que end > start
        new_end = max(new_end, word.start + 0.05)
        cleaned.append(Word(text=word.text, start=word.start, end=new_end))
    return cleaned


def _subtitle_lead_seconds() -> float:
    """Small negative display offset so captions feel locked to fast speech."""
    raw = os.getenv("WORKFAST_SUBTITLE_LEAD_MS", "140").strip()
    try:
        lead_ms = float(raw)
    except ValueError:
        lead_ms = 140.0
    return max(0.0, min(350.0, lead_ms)) / 1000.0


def _shift_word_timings(words: Sequence[Word], offset_seconds: float) -> list[Word]:
    """Shift display timing. Negative = earlier, positive = later."""
    if abs(offset_seconds) <= 0.001:
        return list(words)
    shifted: list[Word] = []
    for word in words:
        start = max(0.0, word.start + offset_seconds)
        end = max(start + 0.05, word.end + offset_seconds)
        shifted.append(Word(text=word.text, start=start, end=end))
    return shifted


def group_words_into_phrases(
    words: Sequence[Word],
    *,
    max_words: int = 2,
    max_chars: int = 20,
    max_duration: float = 1.05,
    pause_split: float = 0.32,
    tighten: bool = True,
    lead_seconds: float | None = None,
    timing_offset_seconds: float | None = None,
    manual_offset_seconds: float = 0.0,
) -> list[Phrase]:
    """Group whisper word-level timestamps into short phrases for the screen.

    Uses Opus Clips heuristics: 1-2 words per phrase, break on short pauses,
    keep each phrase under ~20 chars and ~1.05s of duration.
    Si `tighten=True` recorta los ends de palabra (anti-lag). Por defecto el
    texto se adelanta con `WORKFAST_SUBTITLE_LEAD_MS`; `manual_offset_seconds`
    suma un ajuste fino: negativo adelanta, positivo atrasa.
    """
    if not words:
        return []
    if tighten:
        words = _tighten_word_ends(words)
    if timing_offset_seconds is None:
        if lead_seconds is None:
            lead_seconds = _subtitle_lead_seconds()
        timing_offset_seconds = -lead_seconds
    words = _shift_word_timings(words, timing_offset_seconds + manual_offset_seconds)
    phrases: list[Phrase] = []
    current: list[Word] = []
    for word in words:
        if not word.text.strip():
            continue
        if not current:
            current.append(word)
            continue
        prospective_text = " ".join(w.text for w in current + [word])
        prospective_duration = word.end - current[0].start
        gap = word.start - current[-1].end
        too_many = len(current) >= max_words
        too_long_chars = len(prospective_text) > max_chars
        too_long_time = prospective_duration > max_duration
        long_pause = gap > pause_split
        if too_many or too_long_chars or too_long_time or long_pause:
            phrases.append(Phrase(words=current))
            current = [word]
        else:
            current.append(word)
    if current:
        phrases.append(Phrase(words=current))

    # Ultimo paso: recorta el end de cada frase para no pisarse con la siguiente
    for index, phrase in enumerate(phrases[:-1]):
        next_start = phrases[index + 1].start
        if phrase.words and phrase.words[-1].end > next_start - 0.04:
            last = phrase.words[-1]
            new_end = max(last.start + 0.06, next_start - 0.04)
            phrase.words[-1] = Word(text=last.text, start=last.start, end=new_end)
    return phrases


def _ass_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
    )


def build_ass(
    phrases: Iterable[Phrase],
    *,
    font_name: str = "Montserrat Black",
    font_size: int = 84,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    margin_v: int = 520,
    primary_color: str = "&H00FFFFFF",
    highlight_color: str = "&H0000F4FF",
    outline_color: str = "&H00000000",
    outline: int = 5,
    shadow: int = 2,
) -> str:
    """Render an ASS file with Opus Clips style karaoke captions.

    Each phrase is one event; the active word is enlarged 12% and re-coloured
    with `highlight_color`; inactive words use `primary_color`.
    """
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Opus,{font_name},{font_size},{primary_color},{highlight_color},"
        f"{outline_color},&H64000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},"
        f"2,80,80,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for phrase in phrases:
        if not phrase.words:
            continue
        events.append(_render_phrase_event(phrase, highlight_color, primary_color))

    return header + "\n".join(events) + "\n"


# Color de palabra aun-no-dicha: blanco translucido (alpha alto = mas transparente).
# Hace que el ojo siga el ritmo: lo no dicho se ve atenuado, lo dicho queda solido.
_PENDING_ALPHA = "&H70&"   # ~44% visible


def _render_phrase_event(
    phrase: Phrase,
    highlight_color: str,
    primary_color: str,
) -> str:
    start_ts = _format_timestamp(phrase.start)
    # Sin pad extra: termina exactamente cuando termina la ultima palabra.
    # Asi los subtitulos se cierran junto con la voz (anti-lag).
    end_ts = _format_timestamp(phrase.end)

    # Caption viral estilo TikTok/Opus Clip:
    #   - La frase entra con un pop (\fad + escala 90->100).
    #   - Cada palabra aun-no-dicha se ve atenuada (alpha).
    #   - Al decirse: POP con rebote (132% -> 113%) + color highlight + opaca.
    #   - Al terminar: vuelve a blanco solido tamano normal (queda legible).
    base_start = phrase.start
    # Entrada de la frase: fade corto + micro pop de escala global.
    line_parts: list[str] = ["{\\fad(60,30)\\fscx90\\fscy90\\t(0,90,\\fscx100\\fscy100)}"]

    for index, word in enumerate(phrase.words):
        word_start_cs = max(0, int((word.start - base_start) * 100))
        word_end_cs = max(word_start_cs + 1, int((word.end - base_start) * 100))
        pop_top = word_start_cs + 8       # cima del salto
        pop_settle = word_start_cs + 18   # asentado tras el rebote
        # Estado inicial: atenuada (si aun no es la primera palabra ya activa).
        # Pop con rebote al decirse, y vuelta a blanco solido al terminar.
        line_parts.append(
            "{"
            f"\\alpha{_PENDING_ALPHA}"
            # aparece nitida justo antes de su pop (anti-flash en habla rapida)
            f"\\t({max(0, word_start_cs - 4)},{word_start_cs},\\alpha&H00&)"
            # POP: salta a 132% + color highlight
            f"\\t({word_start_cs},{pop_top},\\fscx132\\fscy132\\1c{highlight_color}\\alpha&H00&)"
            # REBOTE: asienta a 113%
            f"\\t({pop_top},{pop_settle},\\fscx113\\fscy113)"
            # FIN: vuelve a 100% blanco solido (queda legible el resto de la frase)
            f"\\t({word_end_cs},{word_end_cs + 7},\\fscx100\\fscy100\\1c{primary_color})"
            "}"
        )
        line_parts.append(_ass_escape(word.text))
        if index < len(phrase.words) - 1:
            line_parts.append(" ")

    text = "".join(line_parts)
    return (
        f"Dialogue: 0,{start_ts},{end_ts},Opus,,0,0,0,,{text}"
    )


def write_ass_file(
    words: Sequence[Word],
    output_path: Path,
    *,
    font_name: str = "Montserrat Black",
    font_size: int = 84,
    margin_v: int = 520,
) -> Path:
    """Convenience: build phrases from words and write an .ass file to disk."""
    phrases = group_words_into_phrases(words)
    ass_text = build_ass(
        phrases,
        font_name=font_name,
        font_size=font_size,
        margin_v=margin_v,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass_text, encoding="utf-8")
    return output_path
