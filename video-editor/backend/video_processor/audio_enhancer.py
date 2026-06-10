"""Audio enhancement: DeepFilterNet 3 (denoise) + pedalboard PitchShift (sin chipmunk)."""
import hashlib
import functools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

LOGGER = logging.getLogger(__name__)
_PRAAT_LOCK = threading.Lock()
BASE_DIR = Path(__file__).resolve().parents[2]
RVC_MODELS_DIR = BASE_DIR / "assets" / "library" / "voice_models" / "rvc"
QWEN3_RUNNER = BASE_DIR / "scripts" / "qwen3_tts_render.py"
QWEN3_DEFAULT_PYTHON = BASE_DIR / "assets" / "tools" / "qwen3_tts_venv" / "Scripts" / "python.exe"
QWEN3_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN3_PRESETS: dict[str, dict[str, str]] = {
    "mujer_latina_suave": {
        "id": "mujer_latina_suave",
        "preset": "qwen3:mujer_latina_suave",
        "name": "Mujer latina suave",
        "description": "Voz femenina latina, suave, calida y natural.",
    },
    "hombre_30": {
        "id": "hombre_30",
        "preset": "qwen3:hombre_30",
        "name": "Hombre 30 anos",
        "description": "Voz masculina de unos 30 anos, cercana y clara.",
    },
}
QWEN3_LANGUAGE_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
SOX_WINGET_DIR = (
    Path(os.getenv("LOCALAPPDATA", ""))
    / "Microsoft"
    / "WinGet"
    / "Packages"
    / "ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "sox-14.4.2"
)


def _ensure_sox_on_path() -> str | None:
    """Make winget's portable SoX visible to qwen_tts in fresh subprocesses."""
    found = shutil.which("sox")
    if found:
        return found
    sox_exe = SOX_WINGET_DIR / "sox.exe"
    if sox_exe.exists():
        current = os.environ.get("PATH", "")
        sox_dir = str(SOX_WINGET_DIR)
        if sox_dir.lower() not in current.lower().split(";"):
            os.environ["PATH"] = f"{current};{sox_dir}" if current else sox_dir
        return str(sox_exe)
    return None


def _configured_path(value: str | None, default: Path) -> Path:
    raw = (value or "").strip().strip('"').strip("'")
    if not raw:
        return default
    path = Path(raw)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("._-") or "voice"


def _format_command(template: str, values: dict[str, str]) -> str:
    class _Values(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_Values(values))


def _qwen3_python_path() -> Path:
    return _configured_path(os.getenv("WORKFAST_QWEN3_PYTHON"), QWEN3_DEFAULT_PYTHON)


def _qwen3_model_id() -> str:
    return (os.getenv("WORKFAST_QWEN3_MODEL") or QWEN3_DEFAULT_MODEL).strip()


def _qwen3_preset(preset_name: str | None) -> dict[str, str] | None:
    if not preset_name or not preset_name.startswith("qwen3:"):
        return None
    return QWEN3_PRESETS.get(preset_name.split(":", 1)[1])


@functools.lru_cache(maxsize=8)
def _python_can_import_qwen3(python_exe: str, runner_mtime: int) -> tuple[bool, str]:
    _ensure_sox_on_path()
    python_path = Path(python_exe)
    if not python_path.exists():
        if python_path == QWEN3_DEFAULT_PYTHON:
            fallback = Path(sys.executable)
            if fallback.exists():
                python_path = fallback
            else:
                return False, f"No existe Python para Qwen3-TTS: {python_exe}"
        else:
            return False, f"No existe Python para Qwen3-TTS: {python_exe}"
    try:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                (
                    "import contextlib, io, shutil\n"
                    "buf = io.StringIO()\n"
                    "with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n"
                    "    import qwen_tts, torch\n"
                    "print(getattr(qwen_tts, '__version__', 'qwen_tts'))\n"
                    "print(torch.cuda.is_available())\n"
                    "print(shutil.which('sox') or 'sox-not-found')\n"
                ),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(os.getenv("WORKFAST_QWEN3_IMPORT_TIMEOUT_SECONDS", "60")),
        )
    except Exception as exc:
        return False, str(exc)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        return False, msg[:300] or "qwen_tts no disponible"
    return True, (result.stdout or "").strip()


def _qwen3_status() -> dict:
    python_path = _qwen3_python_path()
    if not python_path.exists() and QWEN3_DEFAULT_PYTHON == python_path:
        python_path = Path(sys.executable)
    runner_mtime = int(QWEN3_RUNNER.stat().st_mtime) if QWEN3_RUNNER.exists() else 0
    ready, detail = _python_can_import_qwen3(str(python_path), runner_mtime)
    return {
        "qwen3_configured": bool(ready and QWEN3_RUNNER.exists()),
        "qwen3_python": str(python_path),
        "qwen3_runner": str(QWEN3_RUNNER),
        "qwen3_model": _qwen3_model_id(),
        "qwen3_device": os.getenv("WORKFAST_QWEN3_DEVICE", "cuda:0"),
        "qwen3_detail": detail,
        "qwen3_presets": list(QWEN3_PRESETS.values()),
    }


def _voice_language_for_qwen(detected_language: str) -> str:
    forced = (os.getenv("WORKFAST_QWEN3_LANGUAGE") or "").strip()
    if forced:
        return forced
    lang = (detected_language or "").split("-", 1)[0].lower()
    return QWEN3_LANGUAGE_MAP.get(lang, "Auto")


def _words_to_tts_text(words) -> str:
    text = " ".join(str(getattr(word, "text", "")).strip() for word in words)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cache_key_for_audio(input_wav: Path) -> str:
    """Returns a hash of the input audio to use for caching."""
    hasher = hashlib.md5()
    hasher.update(str(input_wav.stat().st_size).encode())
    hasher.update(str(int(input_wav.stat().st_mtime)).encode())
    return hasher.hexdigest()


def _rvc_models_root() -> Path:
    return _configured_path(os.getenv("WORKFAST_RVC_MODELS_DIR"), RVC_MODELS_DIR)


def _read_voice_metadata(folder: Path) -> dict:
    for name in ("voice.json", "metadata.json"):
        path = folder / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            LOGGER.warning("No pude leer metadata RVC %s: %s", path, exc)
    return {}


def list_rvc_voice_models() -> list[dict]:
    """Return local RVC voice models dropped under assets/library/voice_models/rvc.

    Expected layout:
      rvc/
        mi_voz/
          model.pth
          added.index        # optional
          voice.json         # optional: {"name": "...", "pitch": 0}
    """
    root = _rvc_models_root()
    if not root.exists():
        return []

    models: list[dict] = []
    seen: set[str] = set()
    for model_path in sorted(root.rglob("*.pth")):
        if not model_path.is_file():
            continue
        folder = model_path.parent
        metadata = _read_voice_metadata(folder)
        rel_parent = folder.relative_to(root) if folder != root else Path(model_path.stem)
        raw_id = str(metadata.get("id") or rel_parent.as_posix() or model_path.stem)
        voice_id = _safe_token(raw_id.replace("/", "_"))
        if voice_id in seen:
            voice_id = _safe_token(f"{voice_id}_{model_path.stem}")
        seen.add(voice_id)

        index_path = next(folder.glob("*.index"), None)
        name = str(metadata.get("name") or folder.name or model_path.stem).strip()
        try:
            pitch = int(metadata.get("pitch", os.getenv("WORKFAST_RVC_DEFAULT_PITCH", "0")))
        except (TypeError, ValueError):
            pitch = 0
        models.append(
            {
                "id": voice_id,
                "preset": f"rvc:{voice_id}",
                "name": name,
                "model_path": str(model_path),
                "index_path": str(index_path) if index_path else "",
                "pitch": pitch,
                "f0_method": str(metadata.get("f0_method") or os.getenv("WORKFAST_RVC_F0_METHOD", "rmvpe")),
            }
        )
    return models


def _rvc_model_for_preset(preset_name: str | None) -> dict | None:
    if not preset_name or not preset_name.startswith("rvc:"):
        return None
    wanted = preset_name.split(":", 1)[1]
    return next((model for model in list_rvc_voice_models() if model["id"] == wanted), None)


def is_external_voice_preset(preset_name: str | None) -> bool:
    if not preset_name or preset_name == "normal":
        return False
    if preset_name.startswith("qwen3:"):
        return True
    if preset_name.startswith("rvc:"):
        return True
    return bool(os.getenv("WORKFAST_VOICE_CONVERTER_CMD", "").strip())


def voice_cache_fingerprint(preset_name: str | None) -> str:
    if not preset_name:
        return "manual"
    if preset_name.startswith("qwen3:"):
        parts = [
            _safe_token(preset_name),
            _safe_token(_qwen3_model_id()),
            _safe_token(os.getenv("WORKFAST_QWEN3_LANGUAGE", "auto")),
        ]
        if QWEN3_RUNNER.exists():
            stat = QWEN3_RUNNER.stat()
            parts.append(str(int(stat.st_mtime)))
        return "_".join(parts)
    model = _rvc_model_for_preset(preset_name)
    if not model:
        return _safe_token(preset_name)
    parts = [_safe_token(model["id"])]
    for key in ("model_path", "index_path"):
        path = Path(model.get(key) or "")
        if path.exists():
            stat = path.stat()
            parts.append(str(stat.st_size))
            parts.append(str(int(stat.st_mtime)))
    return "_".join(parts)


def voice_backend_status() -> dict:
    rvc_command = os.getenv("WORKFAST_RVC_COMMAND", "").strip()
    generic_command = os.getenv("WORKFAST_VOICE_CONVERTER_CMD", "").strip()
    rvc_models = list_rvc_voice_models()
    qwen3_info = _qwen3_status()
    engine = "praat_formant"
    if qwen3_info["qwen3_configured"]:
        engine = "qwen3_tts"
    elif rvc_command and rvc_models:
        engine = "rvc"
    elif generic_command:
        engine = "external_neural"
    return {
        "engine": engine,
        "neural_converter_configured": bool(qwen3_info["qwen3_configured"] or rvc_command or generic_command),
        **qwen3_info,
        "rvc_configured": bool(rvc_command),
        "rvc_models_dir": str(_rvc_models_root()),
        "rvc_models": rvc_models,
        "builtin_presets": [
            {"id": "normal", "name": "Original"},
            {"id": "mujer_latina_suave", "name": "Mujer latina suave (rapida/local)"},
            {"id": "hombre_30", "name": "Hombre 30 anos (rapida/local)"},
        ],
    }


# Presets de voz: cada uno define (pitch_factor, formant_factor, color).
# pitch_factor: tono fundamental. >1 = agudo, <1 = grave.
# formant_factor: tracto vocal aparente. >1 = boca/garganta mas pequenia
#   (joven/femenino), <1 = mas grande (adulto/masculino/profundo).
# color: nombre del post-EQ aplicado para dar caracter (sensual, dulce, etc).
#
# Solo cambiando pitch suena Alvin/ardilla. Para que suene MUJER de verdad
# hay que subir AMBOS: pitch y formant. La diferencia entre 'adulta' y 'joven'
# es justamente cuanto formant: mas formant = vocal tract mas chico = joven.
#
# Calibracion conservadora: buscamos cambio usable y natural. Valores extremos
# dan la sensacion de ardilla, aunque el formant ayude.
VOICE_PRESETS: dict = {
    "normal":            (1.00, 1.00, None),
    "ligeramente_grave": (0.92, 0.97, None),
    "grave":             (0.82, 0.92, None),
    "anonimo":           (0.78, 0.88, None),
    "ligeramente_agudo": (1.10, 1.05, None),
    "mujer_adulta":      (1.30, 1.18, None),     # ~40-50 anios (lo viejo)
    "mujer_joven":       (1.34, 1.18, "young_female"),  # ~20 anios, clara
    "hombre_joven":      (1.06, 1.03, "young_male"),    # ~20 anios, limpio
    "mujer_latina_suave": (1.24, 1.13, "latina_soft"),   # suave, calida, natural
    "hombre_30":          (0.92, 0.96, "male_30"),       # adulto joven, cercano
    "mujer_sensual":     (1.45, 1.24, "warm"),   # ~25 anios, calida e intima
    "mujer_dulce":       (1.50, 1.30, "soft"),   # ~22 anios, dulce/inocente
    "mujer_alta":        (1.60, 1.30, "fresh"),  # joven aguda
    "mujer":             (1.30, 1.18, None),     # alias de adulta (compat)
    "robot":             (1.00, 0.75, None),
}


def voice_preset_from_pitch(pitch_factor: float) -> tuple[float, float, str | None]:
    """Compatibilidad con el dropdown viejo que solo manda pitch_factor."""
    pf = round(pitch_factor, 2)
    mapping = {
        0.80: VOICE_PRESETS["anonimo"],
        0.90: VOICE_PRESETS["grave"],
        1.10: VOICE_PRESETS["ligeramente_agudo"],
        1.30: VOICE_PRESETS["mujer_adulta"],
        1.40: VOICE_PRESETS["mujer_adulta"],
        1.50: VOICE_PRESETS["mujer_joven"],
    }
    return mapping.get(pf, (pitch_factor, 1.0, None))


def _apply_voice_color(input_wav: Path, output_wav: Path, color: str) -> Path:
    """Aplica un post-EQ de pedalboard para dar caracter a la voz transformada.

    Colores disponibles:
      'young_female': presencia y aire sin exagerar sibilancia.
      'young_male':   presencia limpia, menos caja/grave turbio.
      'latina_soft':  voz femenina suave, calida, con sibilancia contenida.
      'male_30':      hombre adulto joven, grave controlado y cercano.
      'fresh':        mas brillo (aire), claridad alta. Voz juvenil fresca.
      'warm':         mids calidos (200-500Hz), highs suaves. Voz intima.
      'soft':         compresion suave + ligeramente breathy.
    """
    try:
        from pedalboard import (  # type: ignore
            Pedalboard, LowShelfFilter, HighShelfFilter, PeakFilter,
            Compressor,
        )
        from pedalboard.io import AudioFile  # type: ignore
    except ImportError:
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav

    if color == "warm":
        # Calida/intima: cuerpo en mids, highs suaves, compresion para cercania
        plugins = [
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=-2.0),  # limpieza grave
            PeakFilter(cutoff_frequency_hz=280, gain_db=2.5, q=0.9),  # warmth
            PeakFilter(cutoff_frequency_hz=6500, gain_db=-2.0, q=1.4),  # de-ess
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=-1.5),  # suaviza brillo
            Compressor(threshold_db=-22, ratio=2.6, attack_ms=8, release_ms=180),
        ]
    elif color in {"fresh", "young_female"}:
        # Joven/fresca: claridad alta, aire, ligeramente brillante
        plugins = [
            LowShelfFilter(cutoff_frequency_hz=100, gain_db=-1.5),
            PeakFilter(cutoff_frequency_hz=3200, gain_db=1.8, q=1.0),  # presencia
            PeakFilter(cutoff_frequency_hz=7200, gain_db=-0.8, q=1.5),  # de-ess leve
            HighShelfFilter(cutoff_frequency_hz=9500, gain_db=1.6),  # aire
            Compressor(threshold_db=-20, ratio=2.2),
        ]
    elif color == "latina_soft":
        # Femenina suave: menos filo que young_female, mas calidez y cercania.
        plugins = [
            LowShelfFilter(cutoff_frequency_hz=115, gain_db=-1.8),
            PeakFilter(cutoff_frequency_hz=260, gain_db=1.2, q=0.9),
            PeakFilter(cutoff_frequency_hz=3100, gain_db=1.4, q=1.0),
            PeakFilter(cutoff_frequency_hz=6900, gain_db=-1.8, q=1.5),
            HighShelfFilter(cutoff_frequency_hz=9800, gain_db=0.8),
            Compressor(threshold_db=-23, ratio=2.0, attack_ms=12, release_ms=210),
        ]
    elif color == "young_male":
        # Hombre joven: limpio y cercano sin hacerlo cavernoso.
        plugins = [
            LowShelfFilter(cutoff_frequency_hz=90, gain_db=-1.5),
            PeakFilter(cutoff_frequency_hz=220, gain_db=-1.2, q=0.9),
            PeakFilter(cutoff_frequency_hz=2800, gain_db=1.8, q=1.0),
            HighShelfFilter(cutoff_frequency_hz=9000, gain_db=0.9),
            Compressor(threshold_db=-21, ratio=2.1, attack_ms=10, release_ms=160),
        ]
    elif color == "male_30":
        # Hombre de 30: cuerpo moderado, voz estable, sin sonar demasiado grave.
        plugins = [
            LowShelfFilter(cutoff_frequency_hz=85, gain_db=-1.0),
            PeakFilter(cutoff_frequency_hz=180, gain_db=1.4, q=0.85),
            PeakFilter(cutoff_frequency_hz=420, gain_db=-1.0, q=1.0),
            PeakFilter(cutoff_frequency_hz=2700, gain_db=1.5, q=1.0),
            HighShelfFilter(cutoff_frequency_hz=8800, gain_db=0.5),
            Compressor(threshold_db=-21, ratio=2.3, attack_ms=9, release_ms=170),
        ]
    elif color == "soft":
        # Dulce/inocente: highs delicados, compresion suave, casi-breathy
        plugins = [
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=-1.0),
            PeakFilter(cutoff_frequency_hz=500, gain_db=1.5, q=1.0),  # cuerpo suave
            PeakFilter(cutoff_frequency_hz=7500, gain_db=-1.5, q=1.5),  # de-ess
            HighShelfFilter(cutoff_frequency_hz=11000, gain_db=1.5),  # leve aire
            Compressor(threshold_db=-24, ratio=1.8, attack_ms=15, release_ms=220),
        ]
    else:
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav

    try:
        with AudioFile(str(input_wav)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
        board = Pedalboard(plugins)
        colored = board(audio, sample_rate=sr)
        with AudioFile(str(output_wav), "w", sr, colored.shape[0]) as f:
            f.write(colored)
        return output_wav
    except Exception as exc:
        LOGGER.warning("Voice color '%s' fallo: %s", color, exc)
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav


def transform_voice(
    input_wav: Path,
    output_wav: Path,
    pitch_factor: float = 1.0,
    formant_factor: float = 1.0,
    color: str | None = None,
    preset_name: str | None = None,
) -> Path:
    """Cambia tono/formantes localmente; no reemplaza un clonador neuronal.

    Estrategia (en orden de calidad, primero el mejor disponible):
      0. WORKFAST_RVC_COMMAND + modelos locales RVC (Applio/RVC externo).
      0b. WORKFAST_VOICE_CONVERTER_CMD: motor externo neuronal generico.
      1. Praat 'Change gender' via parselmouth: modifica pitch/formantes.
      2. pyworld (WORLD vocoder): separa F0/spectral envelope/aperiodicity
         y cada uno se modifica por separado. Calidad profesional.
      3. pedalboard: solo pitch shift (preserva formantes pero no las shiftea).
         Si solo se cambia pitch, suena natural. Si se quiere efecto femenino,
         da el resultado mas "limpio" pero sin caracter de mujer.
      4. librosa: ultimo fallback.
    """
    if preset_name and preset_name.startswith("qwen3:"):
        external = _run_qwen3_tts_converter(input_wav, output_wav, preset_name=preset_name)
        if external is not None:
            return external

        fallback_name = preset_name.split(":", 1)[1]
        fallback = VOICE_PRESETS.get(fallback_name)
        if fallback:
            LOGGER.warning(
                "Preset Qwen3-TTS seleccionado pero no esta listo; usando preset local '%s'.",
                fallback_name,
            )
            return transform_voice(
                input_wav,
                output_wav,
                pitch_factor=fallback[0],
                formant_factor=fallback[1],
                color=fallback[2] if len(fallback) > 2 else None,
                preset_name=fallback_name,
            )
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav

    if preset_name and preset_name.startswith("rvc:"):
        external = _run_rvc_voice_converter(input_wav, output_wav, preset_name=preset_name)
        if external is not None:
            return external
        LOGGER.warning("Preset RVC seleccionado pero no esta configurado/listo; usando audio limpio original.")
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav

    external = _run_external_voice_converter(input_wav, output_wav, preset_name=preset_name)
    if external is not None:
        return external

    if abs(pitch_factor - 1.0) < 0.01 and abs(formant_factor - 1.0) < 0.01:
        # Sin shift de pitch/formant pero quizas hay color
        if color:
            return _apply_voice_color(input_wav, output_wav, color)
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav

    # Si hay color, redirigimos la salida del shift a un archivo temporal
    # y luego le aplicamos el post-EQ. Si no hay color, guardamos directo.
    shift_target = output_wav
    if color:
        shift_target = output_wav.with_name(output_wav.stem + "__noColor.wav")

    # 1. PRAAT (parselmouth): real voice gender / age / character change
    try:
        import parselmouth  # type: ignore

        LOGGER.info("Voice transform con Praat: pitch=%.2fx formant=%.2fx",
                    pitch_factor, formant_factor)
        with _PRAAT_LOCK:
            sound = parselmouth.Sound(str(input_wav))

            # Praat "Change gender" solo funciona en mono. Extraemos canal 1.
            mono_sound = parselmouth.praat.call(sound, "Extract one channel", 1)
            pitch_floor = 75.0
            pitch_ceiling = 600.0
            new_pitch_median = 0.0
            if abs(pitch_factor - 1.0) > 0.01:
                try:
                    pitch_obj = parselmouth.praat.call(
                        mono_sound,
                        "To Pitch",
                        0.0,
                        pitch_floor,
                        pitch_ceiling,
                    )
                    median_hz = float(
                        parselmouth.praat.call(
                            pitch_obj,
                            "Get quantile",
                            0.0,
                            0.0,
                            0.5,
                            "Hertz",
                        )
                    )
                    if math.isfinite(median_hz) and median_hz > 0:
                        new_pitch_median = max(70.0, min(300.0, median_hz * pitch_factor))
                except Exception as exc:
                    LOGGER.warning("No pude estimar pitch mediano para Praat: %s", exc)

            # Praat 'Change gender' parametros:
            #   pitch_floor: 75 Hz (default, voces masculinas)
            #   pitch_ceiling: 600 Hz (cubre voces femeninas)
            #   formant_shift_ratio: >1 sube formantes (efecto femenino)
            #   new_pitch_median: Hz objetivo; 0 mantiene la mediana original
            #   pitch_range_factor: 1.0 = no cambia el rango melodico
            #   duration_factor: 1.0 = no cambia la duracion (atempo lo hace despues)
            new_mono = parselmouth.praat.call(
                mono_sound,
                "Change gender",
                pitch_floor,
                pitch_ceiling,
                formant_factor,
                new_pitch_median,
                1.0,
                1.0,
            )

            # Duplicar el canal mono a estereo
            new_sound = parselmouth.praat.call([new_mono, new_mono], "Combine to stereo")

            # Normalizar amplitud para evitar clipping de Praat.
            try:
                parselmouth.praat.call(new_sound, "Scale peak", 0.99)
            except Exception:
                pass

            new_sound.save(str(shift_target), parselmouth.SoundFileFormat.WAV)
        if color:
            _apply_voice_color(shift_target, output_wav, color)
            shift_target.unlink(missing_ok=True)
        return output_wav
    except ImportError:
        LOGGER.warning("parselmouth (Praat) no instalado, probando pyworld...")
    except Exception as exc:
        LOGGER.warning("Praat fallo (%s), probando pyworld", exc)

    # 2. pyworld - WORLD vocoder: separa F0, spectral envelope y aperiodicity
    try:
        import pyworld as pw  # type: ignore
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore

        LOGGER.info("Voice transform con pyworld: pitch=%.2fx formant=%.2fx",
                    pitch_factor, formant_factor)
        audio, sr = sf.read(str(input_wav))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        audio = audio.astype(np.float64)

        f0, t = pw.dio(audio, sr)
        f0 = pw.stonemask(audio, f0, t, sr)
        sp = pw.cheaptrick(audio, f0, t, sr)
        ap = pw.d4c(audio, f0, t, sr)

        # Shift pitch
        f0_new = f0 * pitch_factor

        # Shift formant: warp spectral envelope along frequency axis
        if abs(formant_factor - 1.0) > 0.01:
            sp_new = np.zeros_like(sp)
            n_freq = sp.shape[1]
            for i in range(sp.shape[0]):
                # Resample por columna: stretch axis frecuencia por formant_factor
                idx = np.linspace(0, n_freq - 1, n_freq) / formant_factor
                idx = np.clip(idx, 0, n_freq - 1)
                sp_new[i] = np.interp(np.arange(n_freq), idx, sp[i])
            sp = sp_new

        synth = pw.synthesize(f0_new, sp, ap, sr)
        sf.write(str(shift_target), synth, sr)
        if color:
            _apply_voice_color(shift_target, output_wav, color)
            shift_target.unlink(missing_ok=True)
        return output_wav
    except ImportError:
        LOGGER.warning("pyworld no instalado, probando pedalboard...")
    except Exception as exc:
        LOGGER.warning("pyworld fallo (%s), probando pedalboard", exc)

    # 3. pedalboard - solo pitch (sin formant shift, suena ardilla con valores altos)
    semitones = 12.0 * math.log2(max(pitch_factor, 0.01))
    try:
        from pedalboard import Pedalboard, PitchShift  # type: ignore
        from pedalboard.io import AudioFile  # type: ignore

        LOGGER.warning(
            "Sin Praat/pyworld: solo pitch shift, sin formant. "
            "Para voz de mujer real, instala 'praat-parselmouth' o 'pyworld'."
        )
        with AudioFile(str(input_wav)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
        board = Pedalboard([PitchShift(semitones=semitones)])
        shifted = board(audio, sample_rate=sr)
        with AudioFile(str(shift_target), "w", sr, shifted.shape[0]) as f:
            f.write(shifted)
        if color:
            _apply_voice_color(shift_target, output_wav, color)
            shift_target.unlink(missing_ok=True)
        return output_wav
    except Exception as exc:
        LOGGER.error("Voice transform fallo completamente: %s", exc)
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav


def pitch_shift_speech(
    input_wav: Path,
    output_wav: Path,
    pitch_factor: float,
) -> Path:
    """Backwards-compat wrapper: usa transform_voice con preset deducido."""
    pitch, formant, color = voice_preset_from_pitch(pitch_factor)
    return transform_voice(input_wav, output_wav, pitch, formant, color)


def _run_qwen3_tts_converter(
    input_wav: Path,
    output_wav: Path,
    *,
    preset_name: str | None,
) -> Path | None:
    preset = _qwen3_preset(preset_name)
    if preset is None:
        return None

    template = os.getenv("WORKFAST_QWEN3_COMMAND", "").strip()
    status = _qwen3_status()
    if not template and not status["qwen3_configured"]:
        LOGGER.warning("Qwen3-TTS no esta configurado: %s", status.get("qwen3_detail", ""))
        return None

    try:
        from .transcriber import transcribe_words

        cache_dir = BASE_DIR / "assets" / "library" / "qwen3_tts_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        words, detected_language = transcribe_words(
            input_wav,
            cache_dir=cache_dir,
            model_name=os.getenv("WORKFAST_QWEN3_TRANSCRIBE_MODEL", "small"),
        )
        text = _words_to_tts_text(words)
        if not text:
            LOGGER.warning("Qwen3-TTS no recibio texto de Whisper.")
            return None
        language = _voice_language_for_qwen(detected_language)
    except Exception as exc:
        LOGGER.warning("No pude transcribir audio para Qwen3-TTS: %s", exc)
        return None

    target_duration = _media_duration(input_wav)
    text_path = output_wav.with_name(output_wav.stem + "__qwen3_text.txt")
    raw_wav = output_wav.with_name(output_wav.stem + "__qwen3_raw.wav")
    text_path.write_text(text, encoding="utf-8")

    if template:
        command = _format_command(
            template,
            {
                "input": str(input_wav),
                "output": str(raw_wav),
                "text_file": str(text_path),
                "text": text,
                "preset": preset["id"],
                "language": language,
                "duration": f"{target_duration:.3f}",
            },
        )
        rendered = _run_voice_command(command, raw_wav, f"Qwen3-TTS {preset['name']}")
    else:
        _ensure_sox_on_path()
        python_path = Path(status["qwen3_python"])
        command = [
            str(python_path),
            str(QWEN3_RUNNER),
            "--text-file",
            str(text_path),
            "--output",
            str(raw_wav),
            "--preset",
            preset["id"],
            "--language",
            language,
            "--model",
            status["qwen3_model"],
            "--device",
            os.getenv("WORKFAST_QWEN3_DEVICE", "cuda:0"),
            "--dtype",
            os.getenv("WORKFAST_QWEN3_DTYPE", "auto"),
            "--attention",
            os.getenv("WORKFAST_QWEN3_ATTENTION", "sdpa"),
        ]
        rendered = _run_voice_subprocess(command, raw_wav, f"Qwen3-TTS {preset['name']}")

    text_path.unlink(missing_ok=True)
    if rendered is None:
        raw_wav.unlink(missing_ok=True)
        return None

    try:
        fitted = _fit_audio_duration(raw_wav, output_wav, target_duration)
        raw_wav.unlink(missing_ok=True)
        return fitted
    except Exception as exc:
        LOGGER.warning("No pude ajustar duracion Qwen3-TTS: %s", exc)
        if raw_wav.exists():
            shutil.move(str(raw_wav), str(output_wav))
            return output_wav
    return None


def _run_external_voice_converter(
    input_wav: Path,
    output_wav: Path,
    *,
    preset_name: str | None = None,
) -> Path | None:
    """Optional neural VC hook.

    Set WORKFAST_VOICE_CONVERTER_CMD to a command template with {input} and {output}.
    Optional {reference} is filled from WORKFAST_VOICE_REFERENCE.
    Optional {preset} is filled with the UI preset name.

    Example:
      WORKFAST_VOICE_CONVERTER_CMD=python rvc_infer.py --input "{input}" --output "{output}" --model voice.pth
    """
    template = os.getenv("WORKFAST_VOICE_CONVERTER_CMD", "").strip()
    if not template:
        return None
    reference = os.getenv("WORKFAST_VOICE_REFERENCE", "").strip()
    command = _format_command(
        template,
        {
            "input": str(input_wav),
            "output": str(output_wav),
            "reference": reference,
            "preset": preset_name or "",
        },
    )
    return _run_voice_command(command, output_wav, "motor externo neuronal")


def _run_rvc_voice_converter(
    input_wav: Path,
    output_wav: Path,
    *,
    preset_name: str | None,
) -> Path | None:
    template = os.getenv("WORKFAST_RVC_COMMAND", "").strip()
    model = _rvc_model_for_preset(preset_name)
    if not template or model is None:
        return None
    values = {
        "input": str(input_wav),
        "output": str(output_wav),
        "model": model["model_path"],
        "index": model.get("index_path") or "",
        "pitch": str(model.get("pitch", 0)),
        "f0_method": model.get("f0_method") or os.getenv("WORKFAST_RVC_F0_METHOD", "rmvpe"),
        "device": os.getenv("WORKFAST_RVC_DEVICE", "cuda:0"),
        "preset": preset_name or "",
        "voice_id": model["id"],
    }
    command = _format_command(template, values)
    return _run_voice_command(command, output_wav, f"RVC {model['name']}")


def _run_voice_command(command: str, output_wav: Path, label: str) -> Path | None:
    LOGGER.info("Voice transform con %s.", label)
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        LOGGER.warning("%s fallo: %s", label, (result.stderr or result.stdout).strip()[:800])
        return None
    if output_wav.exists() and output_wav.stat().st_size > 0:
        return output_wav
    LOGGER.warning("%s termino sin generar salida: %s", label, output_wav)
    return None


def _run_voice_subprocess(command: list[str], output_wav: Path, label: str) -> Path | None:
    LOGGER.info("Voice transform con %s.", label)
    try:
        timeout = int(os.getenv("WORKFAST_QWEN3_TIMEOUT_SECONDS", "1800"))
    except ValueError:
        timeout = 1800
    result = subprocess.run(
        command,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        LOGGER.warning("%s fallo: %s", label, (result.stderr or result.stdout).strip()[:1200])
        return None
    if output_wav.exists() and output_wav.stat().st_size > 0:
        return output_wav
    LOGGER.warning("%s termino sin generar salida: %s", label, output_wav)
    return None


def _media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        return max(0.05, float((result.stdout or "").strip()))
    except ValueError:
        return 0.05


def _atempo_filters(tempo: float) -> list[str]:
    tempo = max(0.05, min(100.0, tempo))
    filters: list[str] = []
    while tempo < 0.5:
        filters.append("atempo=0.5")
        tempo /= 0.5
    while tempo > 2.0:
        filters.append("atempo=2.0")
        tempo /= 2.0
    filters.append(f"atempo={tempo:.6f}")
    return filters


def _fit_audio_duration(input_wav: Path, output_wav: Path, target_duration: float) -> Path:
    generated_duration = _media_duration(input_wav)
    if target_duration <= 0 or generated_duration <= 0:
        shutil.copy2(input_wav, output_wav)
        return output_wav
    tempo = generated_duration / target_duration
    filters = [
        "aresample=48000",
        *_atempo_filters(tempo),
        f"apad=whole_dur={target_duration:.3f}",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "aformat=sample_fmts=s16:channel_layouts=stereo",
    ]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_wav),
        "-af",
        ",".join(filters),
        "-t",
        f"{target_duration:.3f}",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not output_wav.exists():
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg fallo").strip()[:800])
    return output_wav


_df_cache: tuple | None = None


def _get_df_model():
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    from df.enhance import init_df
    import torch
    model, df_state, _ = init_df(log_file=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model.to(device)
        LOGGER.info("DeepFilterNet cargado en %s", device)
    except Exception as exc:
        LOGGER.warning("No pude mover DeepFilterNet a GPU (%s), usando CPU", exc)
        device = torch.device("cpu")
    _df_cache = (model, df_state, device)
    return _df_cache


def enhance_speech(input_wav: Path, output_wav: Path) -> Path:
    """Cleans up the audio by removing background noise and enhancing speech.
    Returns the path to the cleaned audio.
    """
    try:
        from df.enhance import enhance, load_audio, save_audio
        import torch
    except ImportError:
        LOGGER.warning("DeepFilterNet not installed, skipping audio enhancement.")
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav

    LOGGER.info("Starting DeepFilterNet enhancement for %s", input_wav.name)
    try:
        model, df_state, _ = _get_df_model()
        audio, _ = load_audio(str(input_wav), sr=df_state.sr())
        # audio permanece en CPU: df_features() mueve los features a GPU internamente.
        # enhance() ya devuelve el tensor en CPU (hace .cpu() tras la inferencia).
        enhanced = enhance(model, df_state, audio)
        save_audio(str(output_wav), enhanced, df_state.sr())
        LOGGER.info("DeepFilterNet enhancement complete: %s", output_wav.name)
        return output_wav
    except Exception as exc:
        LOGGER.error("Failed to enhance audio: %s", exc)
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav
