"""Audio enhancement: DeepFilterNet 3 (denoise) + pedalboard PitchShift (sin chipmunk)."""
import hashlib
import logging
import math
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)

def cache_key_for_audio(input_wav: Path) -> str:
    """Returns a hash of the input audio to use for caching."""
    hasher = hashlib.md5()
    hasher.update(str(input_wav.stat().st_size).encode())
    hasher.update(str(int(input_wav.stat().st_mtime)).encode())
    return hasher.hexdigest()


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
# Calibracion (asume voz fuente masculina F0~120Hz, formantes adulto):
#   adulta 35-50 anios: pitch +30% formant +18% -> F0~156Hz
#   joven 20-30 anios:  pitch +55% formant +28% -> F0~186Hz, vocal tract chico
#   sensual joven:      pitch +45% formant +24% + post-EQ calido/intimo
#   aguda dulce:        pitch +60% formant +30% (chica fresca, no sensual)
VOICE_PRESETS: dict = {
    "normal":            (1.00, 1.00, None),
    "ligeramente_grave": (0.92, 0.97, None),
    "grave":             (0.82, 0.92, None),
    "anonimo":           (0.78, 0.88, None),
    "ligeramente_agudo": (1.10, 1.05, None),
    "mujer_adulta":      (1.30, 1.18, None),     # ~40-50 anios (lo viejo)
    "mujer_joven":       (1.55, 1.28, "fresh"),  # ~25 anios, fresca/clara
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
      'fresh': mas brillo (aire), claridad alta. Voz juvenil fresca.
      'warm':  mids calidos (200-500Hz), highs suaves. Voz intima/sensual.
      'soft':  compresion suave + ligeramente breathy. Voz dulce/inocente.
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
    elif color == "fresh":
        # Joven/fresca: claridad alta, aire, ligeramente brillante
        plugins = [
            LowShelfFilter(cutoff_frequency_hz=100, gain_db=-1.5),
            PeakFilter(cutoff_frequency_hz=3500, gain_db=2.0, q=1.0),  # presencia
            HighShelfFilter(cutoff_frequency_hz=9000, gain_db=2.0),  # aire
            Compressor(threshold_db=-20, ratio=2.2),
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
) -> Path:
    """Cambia tono Y formantes por separado para conseguir voz humana real.

    Estrategia (en orden de calidad, primero el mejor disponible):
      1. Praat 'Change gender' via parselmouth: el gold standard para voice
         gender modification. Shiftea pitch + formant + duration de manera
         coherente. Resultado: voz de mujer suena como MUJER, no como ardilla.
      2. pyworld (WORLD vocoder): separa F0/spectral envelope/aperiodicity
         y cada uno se modifica por separado. Calidad profesional.
      3. pedalboard: solo pitch shift (preserva formantes pero no las shiftea).
         Si solo se cambia pitch, suena natural. Si se quiere efecto femenino,
         da el resultado mas "limpio" pero sin caracter de mujer.
      4. librosa: ultimo fallback.
    """
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
        sound = parselmouth.Sound(str(input_wav))
        # Praat 'Change gender' parametros:
        #   pitch_floor: 75 Hz (default, voces masculinas)
        #   pitch_ceiling: 600 Hz (cubre voces femeninas)
        #   formant_shift_ratio: >1 sube formantes (efecto femenino)
        #   new_pitch_median: 0 = mantiene la mediana, escalada por pitch_factor
        #   pitch_range_factor: 1.0 = no cambia el rango de pitch
        #   duration_factor: 1.0 = no cambia la duracion (atempo lo hace despues)
        new_sound = parselmouth.praat.call(
            sound,
            "Change gender",
            75,  # pitch_floor Hz
            600,  # pitch_ceiling Hz
            formant_factor,
            0,  # new_pitch_median (0 = relativo)
            pitch_factor,  # pitch_range_factor (escalas la mediana actual)
            1.0,  # duration_factor
        )
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
    pitch, formant = voice_preset_from_pitch(pitch_factor)
    return transform_voice(input_wav, output_wav, pitch, formant)

def enhance_speech(input_wav: Path, output_wav: Path) -> Path:
    """Cleans up the audio by removing background noise and enhancing speech.
    Returns the path to the cleaned audio.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError:
        LOGGER.warning("DeepFilterNet not installed, skipping audio enhancement.")
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav

    LOGGER.info("Starting DeepFilterNet enhancement for %s", input_wav.name)
    try:
        # Load model only once or init it.
        # DeepFilterNet handles caching the model weights.
        model, df_state, _ = init_df()
        audio, _ = load_audio(str(input_wav), sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(str(output_wav), enhanced, df_state.sr())
        LOGGER.info("DeepFilterNet enhancement complete: %s", output_wav.name)
        return output_wav
    except Exception as exc:
        LOGGER.error("Failed to enhance audio: %s", exc)
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return output_wav
