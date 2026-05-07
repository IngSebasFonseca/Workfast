"""Local audio transcription + translation using faster-whisper and argostranslate.

Replaces the previous OpenAI-based pipeline. No API keys required.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from .subtitles import Phrase, Word, build_ass, group_words_into_phrases

ProgressCallback = Callable[[int, str], None]

LOGGER = logging.getLogger(__name__)


def _setup_cuda_dll_paths() -> None:
    """En Windows, agrega los paths de los wheels NVIDIA al DLL search.
    
    Los paquetes pip nvidia-cudnn-cu12 y nvidia-cublas-cu12 traen sus DLLs en
    site-packages/nvidia/cudnn/bin y site-packages/nvidia/cublas/bin.
    Si las agregamos al PATH de DLLs antes de importar faster-whisper,
    CTranslate2 las encuentra y CUDA funciona sin instalacion manual."""
    import sys
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):  # Python 3.7-
        return
    candidates: list[Path] = []
    for module_name, subdir in [("nvidia.cudnn", "bin"), ("nvidia.cublas", "bin")]:
        try:
            mod = __import__(module_name, fromlist=["__file__"])
            if mod.__file__:
                dll_dir = Path(mod.__file__).parent / subdir
                if dll_dir.exists():
                    candidates.append(dll_dir)
        except ImportError:
            pass
        except Exception as exc:
            LOGGER.debug("No pude resolver %s: %s", module_name, exc)
    for path in candidates:
        try:
            os.add_dll_directory(str(path))
            LOGGER.debug("Agregado al DLL path: %s", path)
        except (OSError, FileNotFoundError):
            pass


_setup_cuda_dll_paths()


# Modelos disponibles: tiny (39MB), base (74MB), small (244MB), medium (769MB), large-v3 (1550MB)
# Para GTX 1650 Ti 4GB: small en CPU, medium en GPU (si CUDA disponible)
DEFAULT_MODEL = os.getenv("WORKFAST_WHISPER_MODEL", "small")
# auto: usa CUDA si las DLLs estan cargables, sino CPU. Cambia a "cpu" para forzar CPU
DEFAULT_DEVICE = os.getenv("WORKFAST_WHISPER_DEVICE", "auto")
DEFAULT_COMPUTE = os.getenv("WORKFAST_WHISPER_COMPUTE", "auto")


def describe_backend() -> dict:
    """Retorna info del backend que se usaria. Util para mostrar al usuario."""
    device = _resolve_device(DEFAULT_DEVICE)
    compute = _resolve_compute_type(device, DEFAULT_COMPUTE)
    cuda_ok = _cuda_available()
    cuda13_warning = _has_incompatible_cuda13()
    if device == "cuda":
        hint = "Usando CUDA (rápido)"
    elif cuda13_warning:
        hint = (
            "Detecté CUDA 13 instalado pero faster-whisper requiere CUDA 12. "
            "Desinstala CUDA 13 desde Panel de control. Las DLLs de CUDA 12 "
            "vienen incluidas en los wheels pip nvidia-cudnn-cu12 y nvidia-cublas-cu12."
        )
    elif cuda_ok is False and DEFAULT_DEVICE == "auto":
        hint = (
            "Usando CPU. Para activar GPU: el proyecto incluye los wheels pip "
            "nvidia-cudnn-cu12 + nvidia-cublas-cu12. Cierra y reabre Abrir WorkFast.bat "
            "para que se instalen automaticamente. NO necesitas instalar CUDA Toolkit."
        )
    else:
        hint = "Usando CPU."
    return {
        "model": DEFAULT_MODEL,
        "device": device,
        "compute_type": compute,
        "cuda_detected": cuda_ok,
        "cuda13_conflict": cuda13_warning,
        "hint": hint,
    }


def _has_incompatible_cuda13() -> bool:
    """Detecta si el usuario instalo CUDA Toolkit 13.x a nivel sistema, lo cual
    rompe la compatibilidad con faster-whisper (que necesita CUDA 12)."""
    import sys
    if sys.platform != "win32":
        return False
    # CUDA 13.x trae cublas64_13.dll. Si lo encontramos en PATH y no hay cublas64_12, hay conflicto.
    try:
        import ctypes
        try:
            ctypes.CDLL("cublas64_13.dll")
            cuda13 = True
        except OSError:
            cuda13 = False
        if not cuda13:
            return False
        try:
            ctypes.CDLL("cublas64_12.dll")
            return False  # ambos disponibles, no hay conflicto
        except OSError:
            return True  # cuda13 instalado pero no cuda12 → conflicto
    except Exception:
        return False


class TranscriberError(RuntimeError):
    """Raised when transcription cannot complete."""


def _resolve_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda":
        return "float16"
    return "int8"


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        # Verificar que las DLLs de CUDA realmente cargan
        if _cuda_available():
            return "cuda"
        LOGGER.warning("CUDA pedido pero no disponible. Usando CPU.")
        return "cpu"
    # auto: solo cuda si todo carga limpio
    if _cuda_available():
        return "cuda"
    return "cpu"


def _cuda_available() -> bool:
    try:
        import ctranslate2  # type: ignore
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        # Verificar cublas64 que es la libreria que falla mas seguido
        import ctypes
        for dll in ("cublas64_12.dll", "cublas64_11.dll", "cudnn_ops_infer64_8.dll"):
            try:
                ctypes.CDLL(dll)
                return True
            except (OSError, AttributeError):
                continue
        return False
    except Exception:
        return False


def extract_audio(input_video: Path, output_path: Path) -> Path:
    """Extract a 16kHz mono wav for whisper."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not output_path.exists():
        raise TranscriberError(f"No pude preparar audio para subtitulos: {result.stderr.strip()}")
    return output_path


def _stable_ts_available() -> bool:
    """stable-ts mejora la precision de timestamps post-Whisper. Es opcional.

    Para activar: pip install stable-ts. Si no esta, usamos faster-whisper puro
    (que ya tiene VAD estricto + condition_on_previous_text=False configurado).
    """
    try:
        import stable_whisper  # type: ignore # noqa: F401
        return True
    except Exception:
        return False

def _whisperx_available() -> bool:
    try:
        import whisperx  # type: ignore # noqa: F401
        return True
    except Exception:
        return False


def _transcribe_audio(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    transcribe_kwargs: dict,
    use_stable_ts: bool,
    use_whisperx: bool,
    progress_callback: Optional[ProgressCallback],
) -> tuple[list[Word], str]:
    """Transcribe usando whisperx (sota), stable-ts o faster-whisper puro."""
    if use_whisperx:
        try:
            return _transcribe_with_whisperx(
                audio_path,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                transcribe_kwargs=transcribe_kwargs,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            LOGGER.warning("whisperX fallo (%s). Cayendo a fallback.", exc)
    if use_stable_ts:
        try:
            return _transcribe_with_stable_ts(
                audio_path,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                transcribe_kwargs=transcribe_kwargs,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            LOGGER.warning("stable-ts fallo (%s). Usando faster-whisper puro.", exc)
    return _transcribe_with_faster_whisper(
        audio_path,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        transcribe_kwargs=transcribe_kwargs,
        progress_callback=progress_callback,
    )


def _transcribe_with_whisperx(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    transcribe_kwargs: dict,
    progress_callback: Optional[ProgressCallback],
) -> tuple[list[Word], str]:
    import whisperx # type: ignore
    
    if progress_callback:
        progress_callback(4, f"Cargando WhisperX {model_name} ({device})")
        
    try:
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
    except Exception as exc:
        LOGGER.warning("WhisperX GPU init fallo (%s): %s. Cayendo a CPU.", device, exc)
        device = "cpu"
        compute_type = "int8"
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        
    if progress_callback:
        progress_callback(6, "Transcribiendo audio (WhisperX)")
        
    audio = whisperx.load_audio(str(audio_path))
    # whisperx doesnt take all the same kwargs directly, so clean up transcribe_kwargs if needed
    batch_size = 16 if device == "cuda" else 4
    result = model.transcribe(audio, batch_size=batch_size, language=transcribe_kwargs.get("language"))
    detected_language = result["language"]
    
    if progress_callback:
        progress_callback(8, "Alineando subtitulos (WhisperX)")
        
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as exc:
        LOGGER.warning("WhisperX align fallo: %s", exc)
        
    words: list[Word] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            text = w.get("word", "").strip()
            if not text:
                continue
            start = w.get("start", segment.get("start", 0.0))
            end = w.get("end", segment.get("end", start + 0.2))
            words.append(Word(text=text, start=start, end=end))
            
    if not words:
        raise TranscriberError("No detecte voz en el audio.")
    return words, detected_language


def _transcribe_with_stable_ts(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    transcribe_kwargs: dict,
    progress_callback: Optional[ProgressCallback],
) -> tuple[list[Word], str]:
    """stable-ts realinea timestamps por silencios reales del audio. ~50ms mas precisos."""
    import stable_whisper  # type: ignore

    try:
        model = stable_whisper.load_faster_whisper(
            model_name, device=device, compute_type=compute_type
        )
    except Exception as exc:
        LOGGER.warning("stable-ts GPU init fallo (%s, %s): %s. Cayendo a CPU.",
                       device, compute_type, exc)
        model = stable_whisper.load_faster_whisper(model_name, device="cpu", compute_type="int8")

    if progress_callback:
        progress_callback(6, "Transcribiendo audio (stable-ts)")

    # stable-ts acepta los mismos kwargs y ademas regroups palabras por silencios.
    result = model.transcribe_stable(str(audio_path), regroup=True, **transcribe_kwargs)

    detected_language = (getattr(result, "language", None) or "en").lower()
    words: list[Word] = []
    segments = getattr(result, "segments", None) or []
    for segment in segments:
        seg_words = getattr(segment, "words", None) or []
        for w in seg_words:
            text = (getattr(w, "word", None) or getattr(w, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(w, "start", None) or getattr(segment, "start", 0.0))
            end = float(getattr(w, "end", None) or getattr(segment, "end", start + 0.2))
            words.append(Word(text=text, start=start, end=end))

    if not words:
        raise TranscriberError("No detecte voz en el audio. Revisa que el video tenga audio claro.")
    return words, detected_language


def _transcribe_with_faster_whisper(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    transcribe_kwargs: dict,
    progress_callback: Optional[ProgressCallback],
) -> tuple[list[Word], str]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time
        raise TranscriberError(
            "Falta instalar faster-whisper. Cierra y vuelve a abrir Workfast para que start.bat instale dependencias."
        ) from exc

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        LOGGER.warning("Whisper init fallo (%s, %s): %s. Fallback a CPU.",
                       device, compute_type, exc)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

    if progress_callback:
        progress_callback(6, "Transcribiendo audio")

    segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)
    detected_language = (info.language or "en").lower()

    words: list[Word] = []
    for segment in segments_iter:
        if not getattr(segment, "words", None):
            continue
        for word_info in segment.words:
            text = (word_info.word or "").strip()
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(word_info.start or segment.start),
                    end=float(word_info.end or segment.end),
                )
            )

    if not words:
        raise TranscriberError("No detecte voz en el audio. Revisa que el video tenga audio claro.")
    return words, detected_language


def _video_signature(input_video: Path) -> str:
    stat = input_video.stat()
    # v2: nuevo VAD + tightening de word ends. Romper cache viejo.
    raw = f"v2|{input_video.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_paths(cache_dir: Path, signature: str, language: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    words_path = cache_dir / f"{signature}_{language}.words.json"
    return words_path, cache_dir / f"{signature}_{language}.lang.txt"


def _load_cached_words(words_path: Path) -> Optional[list[Word]]:
    if not words_path.exists():
        return None
    try:
        data = json.loads(words_path.read_text(encoding="utf-8"))
        return [Word(text=item["text"], start=float(item["start"]), end=float(item["end"])) for item in data]
    except Exception:
        return None


def _save_cached_words(words_path: Path, words: Sequence[Word]) -> None:
    words_path.write_text(
        json.dumps([asdict(word) for word in words], ensure_ascii=False),
        encoding="utf-8",
    )


def transcribe_words(
    input_video: Path,
    *,
    cache_dir: Path,
    language: str = "auto",
    progress_callback: Optional[ProgressCallback] = None,
    model_name: str = DEFAULT_MODEL,
) -> tuple[list[Word], str]:
    """Transcribe a video to word-level timestamps using faster-whisper.

    Returns (words, detected_language). Caches results so re-renders are free.
    """
    if progress_callback:
        progress_callback(2, "Preparando audio para subtitulos")

    signature = _video_signature(input_video)
    cache_lang = language if language != "auto" else "src"
    words_path, lang_path = _cache_paths(cache_dir, signature, cache_lang)

    cached = _load_cached_words(words_path)
    if cached is not None:
        detected = lang_path.read_text(encoding="utf-8").strip() if lang_path.exists() else "en"
        if progress_callback:
            progress_callback(8, "Subtitulos cacheados")
        return cached, detected

    audio_path = cache_dir / f"{signature}.wav"
    extract_audio(input_video, audio_path)

    device = _resolve_device(DEFAULT_DEVICE)
    compute_type = _resolve_compute_type(device, DEFAULT_COMPUTE)

    use_whisperx = _whisperx_available()
    use_stable_ts = _stable_ts_available() and not use_whisperx
    
    if progress_callback and not use_whisperx:
        backend = "stable-ts" if use_stable_ts else "faster-whisper"
        progress_callback(4, f"Cargando modelo {backend} {model_name} ({device})")

    transcribe_kwargs: dict = {
        "word_timestamps": True,
        "vad_filter": True,
        # VAD mas estricto: pausas de 250ms y minimos 200ms de habla.
        # Reduce el "overshoot" tipico de Whisper en finales de palabra.
        "vad_parameters": dict(
            min_silence_duration_ms=250,
            speech_pad_ms=80,
            min_speech_duration_ms=200,
        ),
        "beam_size": 5,
        # Evita que el modelo arrastre contexto erroneo (mejora alineacion temporal).
        "condition_on_previous_text": False,
        # Subir el threshold reduce subtitulos "fantasma" en silencios.
        "no_speech_threshold": 0.55,
    }
    if language and language != "auto":
        transcribe_kwargs["language"] = language

    words, detected_language = _transcribe_audio(
        audio_path,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        transcribe_kwargs=transcribe_kwargs,
        use_stable_ts=use_stable_ts,
        use_whisperx=use_whisperx,
        progress_callback=progress_callback,
    )

    audio_path.unlink(missing_ok=True)

    if not words:
        raise TranscriberError("No detecte voz en el audio. Revisa que el video tenga audio claro.")

    _save_cached_words(words_path, words)
    lang_path.write_text(detected_language, encoding="utf-8")

    if progress_callback:
        progress_callback(10, "Subtitulos transcritos")

    return words, detected_language


def translate_words(
    words: Sequence[Word],
    *,
    source_language: str,
    target_language: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[Word]:
    """Translate a word list while keeping per-word timing.

    Strategy: join into phrases, translate the phrase via argostranslate, then
    redistribute the translated text proportionally across the original words
    so karaoke timing keeps working. Pure offline once language packages are
    installed; falls back to deep-translator (Google web, no API key) if argos
    fails.
    """
    if not words or source_language == target_language:
        return list(words)

    phrases = group_words_into_phrases(words)
    if not phrases:
        return list(words)

    if progress_callback:
        progress_callback(12, f"Traduciendo a {target_language}")

    translator = _get_translator(source_language, target_language)
    translated_phrases: list[list[Word]] = []
    for phrase in phrases:
        translated_text = translator(phrase.text).strip() or phrase.text
        translated_phrases.append(_redistribute_words(phrase, translated_text))

    return [word for phrase_words in translated_phrases for word in phrase_words]


def _redistribute_words(phrase: Phrase, translated_text: str) -> list[Word]:
    """Spread the translated text across the original word slots evenly."""
    new_tokens = [token for token in translated_text.split() if token]
    if not new_tokens:
        return list(phrase.words)

    duration = max(0.05, phrase.end - phrase.start)
    per = duration / len(new_tokens)
    return [
        Word(
            text=token,
            start=phrase.start + per * index,
            end=phrase.start + per * (index + 1),
        )
        for index, token in enumerate(new_tokens)
    ]


def _get_translator(source: str, target: str) -> Callable[[str], str]:
    """Return a callable text->text translator using local Argos first."""
    try:
        return _argos_translator(source, target)
    except Exception as exc:
        LOGGER.warning("Argos translator unavailable (%s -> %s): %s", source, target, exc)

    try:
        from deep_translator import GoogleTranslator  # type: ignore

        translator = GoogleTranslator(source=source, target=target)
        return translator.translate  # type: ignore[return-value]
    except Exception as exc:
        LOGGER.warning("deep-translator unavailable: %s", exc)

    def passthrough(text: str) -> str:
        return text

    return passthrough


def _argos_translator(source: str, target: str) -> Callable[[str], str]:
    import argostranslate.package as ap  # type: ignore
    import argostranslate.translate as at  # type: ignore

    installed_languages = at.get_installed_languages()
    src = next((lang for lang in installed_languages if lang.code == source), None)
    tgt = next((lang for lang in installed_languages if lang.code == target), None)

    if not src or not tgt or not src.get_translation(tgt):
        ap.update_package_index()
        available = ap.get_available_packages()
        package = next((pkg for pkg in available if pkg.from_code == source and pkg.to_code == target), None)
        if package is None:
            raise RuntimeError(f"No hay paquete Argos {source}->{target}")
        download_path = package.download()
        ap.install_from_path(download_path)
        installed_languages = at.get_installed_languages()
        src = next((lang for lang in installed_languages if lang.code == source), None)
        tgt = next((lang for lang in installed_languages if lang.code == target), None)
        if not src or not tgt:
            raise RuntimeError("Argos no instalo el paquete correctamente")

    translation = src.get_translation(tgt)
    if not translation:
        raise RuntimeError("Argos no entrego una traduccion utilizable")

    def translate(text: str) -> str:
        return translation.translate(text) or text

    return translate


def generate_subtitles(
    input_video: Path,
    output_ass: Path,
    *,
    cache_dir: Path,
    target_language: str = "original",
    progress_callback: Optional[ProgressCallback] = None,
    font_name: str = "Montserrat Black",
    font_size: int = 84,
    margin_v: int = 520,
) -> Path:
    """End-to-end: video -> word timestamps -> (optional translation) -> .ass file."""
    words, source_language = transcribe_words(
        input_video,
        cache_dir=cache_dir,
        progress_callback=progress_callback,
    )

    if target_language and target_language not in {"original", source_language}:
        words = translate_words(
            words,
            source_language=source_language,
            target_language=target_language,
            progress_callback=progress_callback,
        )

    if progress_callback:
        progress_callback(14, "Renderizando archivo ASS")

    phrases = group_words_into_phrases(words)
    ass_text = build_ass(
        phrases,
        font_name=font_name,
        font_size=font_size,
        margin_v=margin_v,
    )
    output_ass.parent.mkdir(parents=True, exist_ok=True)
    output_ass.write_text(ass_text, encoding="utf-8")
    if progress_callback:
        progress_callback(16, "Subtitulos listos")
    return output_ass


def cleanup_cache(cache_dir: Path, max_files: int = 60) -> None:
    """Mantener el cache de subtítulos a un tamaño razonable.

    Borra los archivos más viejos cuando hay más de max_files.
    """
    try:
        if not cache_dir.exists():
            return
        files = [
            (f, f.stat().st_mtime)
            for f in cache_dir.iterdir()
            if f.is_file()
        ]
        if len(files) <= max_files:
            return
        files.sort(key=lambda item: item[1])
        for path, _ in files[: len(files) - max_files]:
            try:
                path.unlink()
            except OSError:
                pass
    except Exception as exc:
        LOGGER.debug("cleanup_cache fallo: %s", exc)
