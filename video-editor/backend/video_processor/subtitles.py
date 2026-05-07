"""ASS subtitle generator with Opus Clips style word-by-word karaoke highlighting."""
from __future__ import annotations

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


def group_words_into_phrases(
    words: Sequence[Word],
    *,
    max_words: int = 3,
    max_chars: int = 22,
    max_duration: float = 1.4,
    pause_split: float = 0.45,
    tighten: bool = True,
) -> list[Phrase]:
    """Group whisper word-level timestamps into short phrases for the screen.

    Uses Opus Clips heuristics: 1-3 words per phrase, break on long pauses,
    keep each phrase under ~22 chars and ~1.4s of duration.
    Si `tighten=True` recorta los ends de palabra (anti-lag).
    """
    if not words:
        return []
    if tighten:
        words = _tighten_word_ends(words)
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


def _render_phrase_event(
    phrase: Phrase,
    highlight_color: str,
    primary_color: str,
) -> str:
    start_ts = _format_timestamp(phrase.start)
    # Sin pad extra: termina exactamente cuando termina la ultima palabra.
    # Asi los subtitulos se cierran junto con la voz (anti-lag).
    end_ts = _format_timestamp(phrase.end)

    # Build a karaoke-style line where each word lights up at its own timestamp.
    # Fade-in 60ms / fade-out 40ms: entrada nitida, salida casi instantanea.
    line_parts: list[str] = ["{\\fad(60,40)}"]
    base_start = phrase.start
    for index, word in enumerate(phrase.words):
        word_start_cs = max(0, int((word.start - base_start) * 100))
        word_end_cs = max(word_start_cs + 1, int((word.end - base_start) * 100))
        # Inactive: white. At word_start: switch color + scale up. At word_end: revert.
        line_parts.append(
            "{\\t("
            f"{word_start_cs},{word_start_cs + 5},"
            f"\\1c{highlight_color}\\fscx112\\fscy112)"
            "\\t("
            f"{word_end_cs},{word_end_cs + 5},"
            f"\\1c{primary_color}\\fscx100\\fscy100)"
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
