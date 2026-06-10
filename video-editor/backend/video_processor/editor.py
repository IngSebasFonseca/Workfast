from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import unicodedata
from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[int, str], None]


class RenderCancelled(RuntimeError):
    """Se levanta cuando el usuario detiene el render a mitad de proceso.

    Subclase de RuntimeError para que el retry-wrapper la distinga de un
    fallo real de NVENC (no debe reintentar con libx264 si fue cancelado)."""


# Windows: aislamos FFmpeg del process group del padre (Flask) para que
# Ctrl+C, reload del dev server, o cierres del console host no envien SIGTERM
# a FFmpeg a mitad de un render (causaba exit code 3221225786 / 0xC000013A
# y audio truncado en el output).
def _subprocess_isolation_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


class VideoEditor:
    """FFmpeg-based renderer for the WorkFast TikTok editing preset.

    Tuned for true 1080x1920 60fps FullHD output:
      * Lanczos upscale (avoids the bilinear softness of the default scale).
      * NVENC two-pass + lookahead when a NVIDIA GPU is present.
      * libx264 preset slow + CRF 16 + advanced x264 params on CPU fallback.
      * Sharper top layer (unsharp ramp), cleaner blur on the mirrored bottom.
      * Optional ASS subtitle burn-in (Opus Clips style) with custom font dir.
    """

    WIDTH = 1080
    HEIGHT = 1920
    FPS = 60
    # Fuente del título: Montserrat Black es más impactante que Arial Bold
    TITLE_FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Montserrat-Black.ttf"
    TITLE_FONT = TITLE_FONT_PATH.as_posix().replace(":", "\\:")
    TITLE_DISPLAY_SECONDS = 6  # Cuántos segundos se ve cada repetición del título (antes 3s)
    SUBTITLE_FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
    _ENCODER_CACHE: dict[str, bool] = {}

    def __init__(
        self,
        input_video: str | Path,
        output_path: str | Path,
        logo_path: str | Path | None = None,
        ending_path: str | Path | None = None,
        follow_image_path: str | Path | None = None,
        subtitle_path: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_check: Callable[[], bool] | None = None,
        on_process_start: Callable[[subprocess.Popen], None] | None = None,
    ) -> None:
        self.input_video = Path(input_video).resolve()
        self.output_path = Path(output_path).resolve()
        self.logo_path = Path(logo_path).resolve() if logo_path else None
        self.ending_path = Path(ending_path).resolve() if ending_path else None
        self.follow_image_path = Path(follow_image_path).resolve() if follow_image_path else None
        self.subtitle_path = Path(subtitle_path).resolve() if subtitle_path else None
        self.progress_callback = progress_callback
        # Cancelacion: cancel_check() -> True si el usuario detuvo el render.
        # on_process_start(proc) registra el FFmpeg para que se pueda matar.
        self.cancel_check = cancel_check
        self.on_process_start = on_process_start
        temp_root = self.output_path.parent / ".workfast_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = temp_root / f"job_{self.output_path.stem}"
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_video.exists():
            raise FileNotFoundError(f"Video no encontrado: {self.input_video}")

        self.video_info = self._probe(self.input_video)
        self.duration = float(self.video_info.get("duration") or 0)
        self.has_audio = self._has_audio(self.video_info)
        if self.duration <= 0:
            raise RuntimeError("No pude detectar la duracion del video.")

    def process_complete(
        self,
        title_text: str = "",
        speed: float = 1.05,
        zoom_bottom: float = 1.96,
        zoom_top: float = 0.96,
        saturation: float = 100,
        volume_db: float = 5.4,
        pitch_factor: float = 1.0,
        formant_factor: float = 1.0,
        voice_preset: str = "",
        filter_intensity: float = 0.55,  # subido 0.42 -> 0.55 para mas nitidez por defecto
        title_interval: float = 10,
        remove_existing_subtitles: bool = False,
        burn_subtitles: bool = True,
        remove_silences: bool = False,
        silence_threshold_db: float = -32.0,
        silence_min_duration: float = 0.45,
    ) -> Path:
        """Render the complete preset and return the final MP4 path."""
        self._ensure_tools()
        # Si vino un voice_preset por nombre, sobrescribe pitch/formant/color.
        self._voice_color = None
        self._voice_preset_name = voice_preset if voice_preset and voice_preset != "normal" else ""
        if voice_preset:
            try:
                from .audio_enhancer import VOICE_PRESETS
                if voice_preset in VOICE_PRESETS:
                    preset = VOICE_PRESETS[voice_preset]
                    pitch_factor = preset[0]
                    formant_factor = preset[1]
                    self._voice_color = preset[2] if len(preset) > 2 else None
            except Exception:
                pass
        speed = self._clamp(speed, 0.5, 2.0)
        zoom_bottom = self._clamp(zoom_bottom, 1.0, 3.0)
        zoom_top = self._clamp(zoom_top, 0.5, 1.5)
        pitch_factor = self._clamp(pitch_factor, 0.5, 2.0)
        formant_factor = self._clamp(formant_factor, 0.5, 2.0)
        filter_intensity = self._clamp(filter_intensity, 0.0, 1.0)
        title_interval = self._clamp(title_interval, 3.0, 60.0)
        silence_threshold_db = self._clamp(silence_threshold_db, -60.0, -10.0)
        silence_min_duration = self._clamp(silence_min_duration, 0.2, 3.0)

        try:
            if remove_silences and self.has_audio:
                self._progress(2, "Detectando silencios")
                trimmed = self._strip_silences(
                    self.input_video,
                    threshold_db=silence_threshold_db,
                    min_duration=silence_min_duration,
                )
                if trimmed and trimmed != self.input_video:
                    self.input_video = trimmed
                    self.video_info = self._probe(self.input_video)
                    self.duration = float(self.video_info.get("duration") or 0) or self.duration
                    self.has_audio = self._has_audio(self.video_info)
            self._progress(5, "Preparando render FFmpeg")
            main_video = self.temp_dir / "main.mp4"
            self._render_main_video(
                output_path=main_video,
                title_text=title_text,
                speed=speed,
                zoom_bottom=zoom_bottom,
                zoom_top=zoom_top,
                saturation=saturation,
                volume_db=volume_db,
                pitch_factor=pitch_factor,
                formant_factor=formant_factor,
                filter_intensity=filter_intensity,
                title_interval=title_interval,
                remove_existing_subtitles=remove_existing_subtitles,
                burn_subtitles=burn_subtitles,
            )

            if self.ending_path and self.ending_path.exists():
                self._progress(88, "Normalizando ending")
                final_video = self._append_ending(main_video)
            else:
                final_video = main_video

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_path.exists():
                self.output_path.unlink()
            shutil.move(str(final_video), str(self.output_path))

            # Verificacion final: el output debe tener audio con datos reales
            # (no solo el stream vacio que producia anullsrc o un stream
            # truncado por SIGTERM). Verificamos packets+bytes del stream.
            if self.has_audio:
                try:
                    audio_bytes_per_sec = self._audio_data_rate(self.output_path)
                    # < 1 KB/s = stream silencioso o roto. Audio normal son
                    # 12-32 KB/s a 96-256 kbps.
                    if audio_bytes_per_sec < 1024:
                        raise RuntimeError(
                            f"El render produjo un stream de audio con datos "
                            f"insuficientes ({int(audio_bytes_per_sec)} bytes/s). "
                            "Probable interrupcion mid-render o filtro que "
                            "silencio la pista. Reintenta el render."
                        )
                except RuntimeError:
                    raise
                except Exception:
                    pass  # probe error no fatal, solo es verificacion

            self._progress(100, "Video listo")
            return self.output_path
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _render_main_video(
        self,
        output_path: Path,
        title_text: str,
        speed: float,
        zoom_bottom: float,
        zoom_top: float,
        saturation: float,
        volume_db: float,
        pitch_factor: float,
        formant_factor: float,
        filter_intensity: float,
        title_interval: float,
        remove_existing_subtitles: bool,
        burn_subtitles: bool,
    ) -> None:
        inputs = ["-i", str(self.input_video)]
        next_input_index = 1
        audio_index = 0
        logo_index = None
        follow_index = None
        audio_already_denoised = False

        if not self.has_audio:
            # No silenciamos en silencio. Si el source vino sin audio (yt-dlp
            # bajo solo el video stream, o un Short sin pista de audio),
            # avisamos al usuario. Si quiere, puede re-importar el video.
            raise RuntimeError(
                "El video de origen no tiene audio. Esto suele pasar cuando "
                "yt-dlp baja solo el stream de video (formato HEVC sin audio "
                "asociado). Vuelve a importar el video o usa otra fuente. "
                f"Archivo: {self.input_video.name}"
            )
        else:
            try:
                from .audio_enhancer import (
                    cache_key_for_audio, transform_voice,
                    is_external_voice_preset, voice_cache_fingerprint,
                )

                # Siempre extraemos el audio como WAV limpio para el chain de FFmpeg.
                # DeepFilterNet desactivado (quitaba la música de fondo).
                extracted_wav = self.temp_dir / "extracted_audio.wav"
                # Algunos MP4 de YouTube tienen audio en index 0 y video en index 1.
                # Usamos -map 0:a:0 para seleccionar explicitamente el primer
                # audio stream (no depender del orden).
                wav_result = subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-y", "-i", str(self.input_video),
                        "-map", "0:a:0", "-ac", "2", "-ar", "48000",
                        "-c:a", "pcm_s16le", str(extracted_wav)
                    ],
                    capture_output=True,
                    **_subprocess_isolation_kwargs(),
                )

                wav_ok = extracted_wav.exists() and extracted_wav.stat().st_size > 10240  # > 10KB
                if not wav_ok:
                    err_tail = wav_result.stderr.decode('utf-8', errors='replace')[-400:] if wav_result.stderr else ""
                    raise RuntimeError(
                        f"Audio extraction produced invalid WAV (code {wav_result.returncode}, "
                        f"size {extracted_wav.stat().st_size if extracted_wav.exists() else 0}). "
                        f"FFmpeg stderr: {err_tail}"
                    )

                voice_color = getattr(self, "_voice_color", None)
                voice_preset_name = getattr(self, "_voice_preset_name", "")
                external_voice = is_external_voice_preset(voice_preset_name)
                needs_voice_transform = (
                    external_voice
                    or abs(pitch_factor - 1.0) > 0.01
                    or abs(formant_factor - 1.0) > 0.01
                    or voice_color is not None
                )

                final_audio = extracted_wav

                if needs_voice_transform and extracted_wav.exists():
                    cache_dir = self.output_path.parent / ".workfast_tmp" / "audio_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    audio_hash = cache_key_for_audio(extracted_wav)
                    color_tag = voice_color or "none"
                    preset_tag = voice_cache_fingerprint(voice_preset_name) if external_voice else (voice_preset_name or "manual")
                    voice_wav = cache_dir / (
                        f"{audio_hash}_voice_{preset_tag}_p{pitch_factor:.2f}_f{formant_factor:.2f}_{color_tag}.wav"
                    )
                    if not voice_wav.exists():
                        msg_color = f" [{voice_color}]" if voice_color else ""
                        if external_voice:
                            self._progress(7, f"Transformando voz neuronal ({voice_preset_name})")
                        else:
                            self._progress(
                                7,
                                f"Transformando voz (pitch {pitch_factor:.2f}x formant {formant_factor:.2f}x){msg_color}",
                            )
                        transform_voice(
                            extracted_wav, voice_wav,
                            pitch_factor=pitch_factor,
                            formant_factor=formant_factor,
                            color=voice_color,
                            preset_name=voice_preset_name,
                        )
                    if voice_wav.exists():
                        final_audio = voice_wav
                        audio_already_denoised = True

                if final_audio.exists():
                    inputs.extend(["-i", str(final_audio)])
                    audio_index = next_input_index
                    next_input_index += 1

            except Exception as e:
                import logging
                logging.getLogger(__name__).error("Audio extraction failed: %s", e)
                # NO silent fallback - antes caia a audio_index=0 que producia
                # videos sin audio. Si la extraccion falla, queremos saberlo.
                raise RuntimeError(
                    f"No se pudo extraer el audio del video. Detalle: {e}"
                ) from e

        if self.logo_path and self.logo_path.exists():
            logo_index = next_input_index
            next_input_index += 1
            inputs.extend(["-i", str(self.logo_path)])

        if self.follow_image_path and self.follow_image_path.exists():
            follow_index = next_input_index
            next_input_index += 1
            inputs.extend(["-i", str(self.follow_image_path)])

        filter_complex = self._build_filter_complex(
            title_text=title_text,
            speed=speed,
            zoom_bottom=zoom_bottom,
            zoom_top=zoom_top,
            saturation=saturation,
            volume_db=volume_db,
            pitch_factor=pitch_factor,
            filter_intensity=filter_intensity,
            title_interval=title_interval,
            remove_existing_subtitles=remove_existing_subtitles,
            burn_subtitles=burn_subtitles,
            audio_index=audio_index,
            logo_index=logo_index,
            follow_index=follow_index,
            audio_already_denoised=audio_already_denoised,
        )

        # Originalidad: timestamp aleatorio en metadata para que cada render
        # tenga huella distinta. Sin -map_metadata -1 + creation_time random,
        # Facebook puede fingerprintear por metadatos del container.
        import random as _random
        from datetime import datetime as _dt, timedelta as _td
        random_creation = (_dt.utcnow() - _td(days=_random.randint(1, 90),
                                              hours=_random.randint(0, 23),
                                              minutes=_random.randint(0, 59))
                          ).strftime("%Y-%m-%dT%H:%M:%S.000000Z")

        command = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            *self._video_encoder_args(),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-shortest",
            # ── Strip metadata + timestamp aleatorio (originalidad) ─────
            "-map_metadata", "-1",
            "-metadata", f"creation_time={random_creation}",
            "-metadata", "encoder=",
            "-progress",
            "pipe:1",
            "-nostats",
            str(output_path),
        ]

        target_duration = self.duration / speed
        self._run_ffmpeg(command, target_duration=target_duration, start=8, end=86)

    def _build_filter_complex(
        self,
        title_text: str,
        speed: float,
        zoom_bottom: float,
        zoom_top: float,
        saturation: float,
        volume_db: float,
        filter_intensity: float,
        title_interval: float,
        remove_existing_subtitles: bool,
        burn_subtitles: bool,
        audio_index: int,
        logo_index: int | None,
        follow_index: int | None,
        pitch_factor: float = 1.0,
        audio_already_denoised: bool = False,
    ) -> str:
        bottom_crop_w = self._even(math.floor(self.WIDTH / zoom_bottom))
        bottom_crop_h = self._even(math.floor(self.HEIGHT / zoom_bottom))
        top_scale_w = self._even(math.ceil(self.WIDTH * zoom_top))
        top_scale_h = self._even(math.ceil(self.HEIGHT * zoom_top))
        bottom_saturation = 1.0 + self._clamp(saturation, 0.0, 100.0) / 100.0
        contrast = 1.0 + (filter_intensity * 0.16)
        # Sharpen progresivo: 0.50 base -> 1.10 max para fuentes 1080p reales
        sharpen = 0.50 + (filter_intensity * 0.60)
        title_lines, title_font_size = self._wrap_title(self._clean_overlay_text(title_text.strip()))
        interval = f"{title_interval:.2f}"
        output_duration = max(self.duration / speed, 1.0)

        # Lanczos kernel keeps fine detail when upscaling 720p sources to 1080p
        # and avoids the soft "almost-720p" look users see with the default
        # bilinear scaler. Accurate rounding + full chroma improve gradients.
        scale_flags = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"

        # ── ORIGINALIDAD (video): micro-jitter imperceptible por render ──────
        # Meta/Facebook fingerprintea por hash perceptual de frames. Variar
        # brillo/saturacion/contraste/tono en cantidades minimas (que el ojo no
        # nota) hace que dos renders del MISMO clip den hashes distintos.
        import random as _rnd
        orig_bright = _rnd.uniform(-0.012, 0.012)
        orig_sat = 1.04 + _rnd.uniform(-0.03, 0.03)
        orig_contrast = contrast + _rnd.uniform(-0.010, 0.010)
        orig_hue = _rnd.uniform(-1.8, 1.8)          # grados, imperceptible
        orig_sharpen = sharpen + _rnd.uniform(-0.04, 0.04)

        parts = [
            (
                f"[0:v]fps={self.FPS},scale={self.WIDTH}:{self.HEIGHT}:"
                f"flags={scale_flags}:force_original_aspect_ratio=increase,"
                f"crop={self.WIDTH}:{self.HEIGHT},split=2[bottom_src][top_src]"
            ),
            (
                f"[bottom_src]hflip,crop={bottom_crop_w}:{bottom_crop_h}:"
                f"({self.WIDTH}-{bottom_crop_w})/2:({self.HEIGHT}-{bottom_crop_h})/2,"
                f"scale={self.WIDTH}:{self.HEIGHT}:flags={scale_flags},"
                f"eq=saturation={bottom_saturation:.2f}:contrast=1.08:brightness=0.04,"
                "boxblur=22:2,format=rgba,colorchannelmixer=aa=0.58[bottom_raw]"
            ),
            (
                f"color=c=white@0.32:s={self.WIDTH}x{self.HEIGHT}:d={output_duration:.3f},"
                "format=rgba[bottom_veil]"
            ),
            (
                "[bottom_raw][bottom_veil]overlay=0:0:format=auto[bottom]"
            ),
            (
                f"[top_src]scale={top_scale_w}:{top_scale_h}:"
                f"flags={scale_flags}:force_original_aspect_ratio=increase,"
                f"crop={top_scale_w}:{top_scale_h},"
                f"eq=contrast={orig_contrast:.3f}:saturation={orig_sat:.3f}:brightness={orig_bright:.3f},"
                f"hue=h={orig_hue:.2f},"
                f"unsharp=5:5:{orig_sharpen:.2f}:3:3:0.30,format=rgba[top]"
            ),
            "[bottom][top]overlay=(W-w)/2:(H-h)/2:format=auto[stage0]",
        ]

        stage = "stage0"
        stage_count = 1

        if remove_existing_subtitles:
            next_stage = f"stage{stage_count}"
            stage_count += 1
            parts.extend(
                [
                    f"[{stage}]split[subtitle_clean_base][subtitle_clean_src]",
                    (
                        "[subtitle_clean_src]crop=w=iw:h=360:x=0:y=ih-560,"
                        "boxblur=24:2,eq=brightness=-0.03:saturation=0.75[subtitle_clean_band]"
                    ),
                    (
                        f"[subtitle_clean_base][subtitle_clean_band]overlay=0:H-560:"
                        f"format=auto[{next_stage}]"
                    ),
                ]
            )
            stage = next_stage

        if logo_index is not None:
            parts.append(
                f"[{logo_index}:v]scale=190:-1:flags={scale_flags},format=rgba,"
                "colorchannelmixer=aa=0.24[logo]"
            )
            next_stage = f"stage{stage_count}"
            stage_count += 1
            parts.append(
                f"[{stage}][logo]overlay="
                f"x='(W-w)*t/{output_duration:.3f}':y=(H-h)/2:"
                f"format=auto:eof_action=repeat:repeatlast=1[{next_stage}]"
            )
            stage = next_stage

        if follow_index is not None:
            parts.append(
                f"[{follow_index}:v]scale=320:-1:flags={scale_flags},format=rgba,"
                "colorchannelmixer=aa=0.88[follow]"
            )
            next_stage = f"stage{stage_count}"
            stage_count += 1
            parts.append(
                f"[{stage}][follow]overlay="
                "x=W-w-48:y=H-h-210:enable='between(mod(t\\,20)\\,12\\,17)':"
                "format=auto:eof_action=repeat:repeatlast=1"
                f"[{next_stage}]"
            )
            stage = next_stage

        if burn_subtitles and self.subtitle_path and self.subtitle_path.exists():
            next_stage = f"stage{stage_count}"
            stage_count += 1
            subtitle_path = self._escape_filter_path(self.subtitle_path)
            fonts_dir = self._escape_filter_path(self.SUBTITLE_FONTS_DIR)
            extension = self.subtitle_path.suffix.lower()
            if extension == ".ass":
                parts.append(
                    f"[{stage}]ass=filename='{subtitle_path}':fontsdir='{fonts_dir}'[{next_stage}]"
                )
            else:
                style = (
                    "FontName=Montserrat Black,FontSize=44,Bold=0,"
                    "PrimaryColour=&H00FFFFFF,SecondaryColour=&H0000F4FF,"
                    "OutlineColour=&H00000000,BackColour=&H64000000,"
                    "BorderStyle=1,Outline=4,Shadow=2,"
                    "Alignment=2,MarginV=320"
                )
                parts.append(
                    f"[{stage}]subtitles=filename='{subtitle_path}':fontsdir='{fonts_dir}':"
                    f"force_style='{style}'[{next_stage}]"
                )
            stage = next_stage

        title_start_y = 165
        title_line_height = title_font_size + 14
        for index, title_line in enumerate(title_lines):
            title = self._escape_drawtext(title_line)
            next_stage = f"stage{stage_count}"
            stage_count += 1
            y_position = title_start_y + index * title_line_height
            parts.append(
                f"[{stage}]drawtext=text='{title}':"
                f"fontcolor=white:fontsize={title_font_size}:fontfile='{self.TITLE_FONT}':"
                f"borderw=7:bordercolor=black:"
                f"shadowcolor=black@0.38:shadowx=2:shadowy=3:"
                f"x=(w-text_w)/2:y={y_position}:line_spacing=8:"
                f"enable='lt(mod(t\\,{interval})\\,{self.TITLE_DISPLAY_SECONDS})'[{next_stage}]"
            )
            stage = next_stage

        parts.append(f"[{stage}]setpts=PTS/{speed:.5f},setsar=1[vout]")
        parts.append(
            f"[{audio_index}:a:0]" + self._build_audio_chain(speed, volume_db, pitch_factor, audio_already_denoised) + "[aout]"
        )

        return ";".join(parts)

    def _append_ending(self, main_video: Path) -> Path:
        normalized_ending = self.temp_dir / "ending_normalized.mp4"
        output = self.temp_dir / "with_ending.mp4"
        concat_list = self.temp_dir / "concat.txt"
        ending_info = self._probe(self.ending_path)
        ending_duration = float(ending_info.get("duration") or 1)
        ending_has_audio = self._has_audio(ending_info)

        normalize_command = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(self.ending_path),
        ]
        if not ending_has_audio:
            normalize_command.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{ending_duration:.3f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
            )

        normalize_command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0" if ending_has_audio else "1:a:0",
            "-vf",
            (
                f"fps={self.FPS},scale={self.WIDTH}:{self.HEIGHT}:"
                "flags=lanczos:force_original_aspect_ratio=decrease,"
                f"pad={self.WIDTH}:{self.HEIGHT}:"
                "(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
            ),
            "-af",
            "aresample=48000",
            *self._video_encoder_args(),
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            "-ar",
            "48000",
            "-shortest",
            str(normalized_ending),
            ]
        )
        self._run_ffmpeg(normalize_command, target_duration=None, start=88, end=93)

        concat_list.write_text(
            f"file '{main_video.as_posix()}'\nfile '{normalized_ending.as_posix()}'\n",
            encoding="utf-8",
        )
        concat_command = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            "-map_metadata", "-1",
            "-metadata", "encoder=",
            str(output),
        ]
        self._run_ffmpeg(concat_command, target_duration=None, start=94, end=99)
        return output

    def _run_ffmpeg(
        self,
        command: list[str],
        target_duration: float | None,
        start: int,
        end: int,
    ) -> None:
        """Ejecuta FFmpeg y reintenta con libx264 si NVENC falla en runtime."""
        try:
            return self._run_ffmpeg_inner(command, target_duration, start, end)
        except RenderCancelled:
            raise  # cancelacion del usuario, no es un fallo de NVENC: no reintentar
        except RuntimeError as exc:
            err = str(exc)
            nvenc_error = (
                "No capable devices found" in err
                or "Multiple reference frames are not supported" in err
                or "h264_nvenc" in err and "Error while opening encoder" in err
                or "InitializeEncoder failed" in err
                or "Invalid encoder type" in err
            )
            if not nvenc_error or "h264_nvenc" not in command:
                raise
            # Reintentar reemplazando los args de NVENC por libx264
            self._progress(start, "GPU no compatible, usando CPU")
            VideoEditor._ENCODER_CACHE["nvenc_real"] = False
            new_command = self._swap_nvenc_for_libx264(command)
            return self._run_ffmpeg_inner(new_command, target_duration, start, end)

    @staticmethod
    def _swap_nvenc_for_libx264(command: list[str]) -> list[str]:
        """Reemplaza los args de h264_nvenc por libx264 manteniendo el resto."""
        # Identificar el rango de args NVENC: desde "-c:v h264_nvenc" hasta el siguiente arg que no es nvenc-specific
        nvenc_args_keys = {
            "-c:v", "-preset", "-rc", "-cq", "-b:v", "-maxrate", "-bufsize",
            "-bf", "-pix_fmt", "-tune", "-multipass", "-rc-lookahead",
            "-spatial-aq", "-temporal-aq", "-aq-strength", "-refs",
            "-profile:v", "-level",
        }
        out = []
        i = 0
        replaced = False
        while i < len(command):
            arg = command[i]
            # Si encontramos -c:v h264_nvenc, salteamos los args NVENC y los reemplazamos por libx264
            if not replaced and arg == "-c:v" and i + 1 < len(command) and command[i + 1] == "h264_nvenc":
                # Insertar args de libx264
                out.extend([
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "17",
                    "-maxrate", "60M",
                    "-bufsize", "80M",
                    "-profile:v", "high",
                    "-level", "4.2",
                    "-pix_fmt", "yuv420p",
                ])
                replaced = True
                # Saltar todos los args NVENC consecutivos
                i += 2
                while i < len(command):
                    a = command[i]
                    if a in nvenc_args_keys and i + 1 < len(command) and not command[i + 1].startswith("-"):
                        i += 2
                    elif a in {"-spatial-aq", "-temporal-aq"} and i + 1 < len(command):
                        i += 2
                    else:
                        break
                continue
            out.append(arg)
            i += 1
        return out

    def _run_ffmpeg_inner(
        self,
        command: list[str],
        target_duration: float | None,
        start: int,
        end: int,
    ) -> None:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._ffmpeg_env(),
            **_subprocess_isolation_kwargs(),
        )
        # Registrar el proceso para que el endpoint /cancel pueda matarlo.
        if self.on_process_start is not None:
            try:
                self.on_process_start(process)
            except Exception:
                pass
        output_lines: list[str] = []
        reached_end = False

        assert process.stdout is not None
        for line in process.stdout:
            # Cancelacion: si el usuario detuvo, matamos FFmpeg y abortamos.
            # FFmpeg emite progreso cada ~1s, asi que el chequeo es responsivo.
            if self.cancel_check is not None and self.cancel_check():
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
                raise RenderCancelled("Render detenido por el usuario.")
            line = line.strip()
            if line:
                output_lines.append(line)
            if line == "progress=end":
                reached_end = True
            if target_duration and line.startswith("out_time_ms="):
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000
                    percent = start + int((seconds / target_duration) * (end - start))
                    self._progress(min(end, max(start, percent)), "Renderizando video")
                except ValueError:
                    pass

        return_code = process.wait()
        # El endpoint /cancel pudo haber matado el proceso mientras leiamos:
        # el return_code sera != 0 pero la causa es cancelacion, no un fallo.
        if return_code != 0 and self.cancel_check is not None and self.cancel_check():
            raise RenderCancelled("Render detenido por el usuario.")
        if return_code != 0:
            tail = "\n".join(output_lines[-30:])
            # Windows: exit code 3221225786 (0xC000013A = STATUS_CONTROL_C_EXIT)
            # significa SIGTERM/Ctrl+C. Si NO llegamos a "progress=end" antes del
            # signal, descartamos el archivo porque el audio quedo truncado.
            # Aunque "progress=end" suele referirse al video, sin garantia para
            # el audio, asi que NO lo aceptamos como exito.
            if return_code in {3221225786, -1073741510, -15}:
                raise RuntimeError(
                    f"FFmpeg fue interrumpido por SIGTERM (codigo {return_code}). "
                    "Algo termino el proceso (reload de Flask, Ctrl+C, console host). "
                    "El audio sale truncado en este caso. Reintenta el render.\n"
                    + tail
                )
            crash_hint = ""
            if return_code in {3221225477, -1073741819}:
                crash_hint = (
                    "\nFFmpeg se cerro por access violation (0xC0000005). "
                    "En Windows suele pasar por fuentes corruptas, DLLs/filtros nativos "
                    "o drivers de video. Ya se validan las fuentes antes de renderizar."
                )
            raise RuntimeError(f"FFmpeg fallo con codigo {return_code}:{crash_hint}\n{tail}")

    def _ffmpeg_env(self) -> dict[str, str]:
        env = os.environ.copy()
        fontconfig_dir = self.output_path.parent / ".workfast_tmp" / "fontconfig"
        fontconfig_cache = fontconfig_dir / "cache"
        fontconfig_dir.mkdir(parents=True, exist_ok=True)
        fontconfig_cache.mkdir(parents=True, exist_ok=True)
        fonts_conf = fontconfig_dir / "fonts.conf"
        if not fonts_conf.exists():
            project_fonts = self.SUBTITLE_FONTS_DIR.as_posix()
            cache_path = fontconfig_cache.as_posix()
            fonts_conf.write_text(
                "\n".join(
                    [
                        '<?xml version="1.0"?>',
                        '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">',
                        "<fontconfig>",
                        "  <dir>C:/Windows/Fonts</dir>",
                        f"  <dir>{project_fonts}</dir>",
                        f"  <cachedir>{cache_path}</cachedir>",
                        "</fontconfig>",
                    ]
                ),
                encoding="utf-8",
            )
        env["FONTCONFIG_PATH"] = str(fontconfig_dir)
        env["FONTCONFIG_FILE"] = str(fonts_conf)
        return env

    def _probe(self, path: Path) -> dict:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"No pude leer el video con ffprobe: {result.stderr}")
        data = json.loads(result.stdout)
        duration = data.get("format", {}).get("duration")
        return {"duration": duration, "streams": data.get("streams", [])}

    @staticmethod
    def _has_audio(info: dict) -> bool:
        return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))

    def _audio_data_rate(self, path: Path) -> float:
        """Bytes promedio por segundo del primer stream de audio.

        Usado como sanity-check despues del render para detectar audio "silencioso"
        (stream existe pero sin datos -- caso de anullsrc o SIGTERM mid-write).
        """
        info = self._probe(path)
        duration = float(info.get("duration") or 0)
        if duration <= 0:
            return 0.0
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "packet=size",
                    "-of", "csv=p=0",
                    str(path),
                ],
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            total = sum(int(line) for line in result.stdout.splitlines() if line.strip().isdigit())
        except Exception:
            return 0.0
        return total / duration

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        return path.as_posix().replace(":", "\\:").replace("'", "\\'")

    def _ensure_tools(self) -> None:
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool} no esta instalado o no esta en el PATH.")
        self._ensure_font_file(self.TITLE_FONT_PATH, "fuente de titulo")
        for font_name in ("Montserrat-Black.ttf", "Montserrat-Bold.ttf", "Montserrat-ExtraBold.ttf", "Montserrat-SemiBold.ttf"):
            font_path = self.SUBTITLE_FONTS_DIR / font_name
            if font_path.exists():
                self._ensure_font_file(font_path, font_name)

    @staticmethod
    def _ensure_font_file(path: Path, label: str) -> None:
        """Fail early if a downloaded GitHub HTML page was saved as .ttf."""
        try:
            with path.open("rb") as handle:
                signature = handle.read(4)
        except OSError as exc:
            raise RuntimeError(f"No pude leer {label}: {path}") from exc
        valid_signatures = {b"\x00\x01\x00\x00", b"OTTO", b"ttcf", b"wOFF", b"wOF2"}
        if signature not in valid_signatures:
            raise RuntimeError(
                f"{label} no es una fuente valida: {path}. "
                "Descarga el TTF real; no guardes la pagina HTML de GitHub."
            )

    @classmethod
    def _video_encoder_args(cls) -> list[str]:
        """Devuelve args FFmpeg para video. Prioridad:
          1. CPU forzado: WORKFAST_FORCE_CPU=1 -> libx264.
          2. HEVC NVENC: WORKFAST_USE_HEVC=1 y hevc_nvenc disponible -> ~30% menos peso al mismo nivel visual.
          3. H.264 NVENC high-end si hay RTX moderna.
          4. H.264 NVENC universal.
          5. Fallback CPU libx264.
        """
        import os as _os
        if _os.getenv("WORKFAST_FORCE_CPU") == "1":
            return cls._libx264_args()
        if _os.getenv("WORKFAST_USE_HEVC") == "1" and cls._hevc_nvenc_can_encode():
            return cls._hevc_nvenc_args()
        if cls._nvenc_can_encode():
            if cls._high_end_nvidia_gpu():
                return cls._rtx_nvenc_args()
            # Args UNIVERSALES de NVENC - compatibles con cualquier FFmpeg/driver
            # Preset "slow" = nomenclatura clasica que funciona desde GeForce 600
            # Hasta GTX 1650 Ti, RTX 30/40 y todo lo intermedio
            return [
                "-c:v", "h264_nvenc",
                "-preset", "slow",          # equivale a p5/p6 pero universal
                "-rc", "vbr",
                "-cq", "18",                # 18 = mejor calidad subjetiva (era 19)
                "-b:v", "45M",              # bump 40 -> 45 Mbps base
                "-maxrate", "65M",
                "-bufsize", "90M",
                "-bf", "2",                 # 2 b-frames seguros
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",
            ]
        return cls._libx264_args()

    @staticmethod
    def _nvidia_gpu_name() -> str:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception:
            return ""
        return result.stdout.splitlines()[0].strip() if result.returncode == 0 and result.stdout.strip() else ""

    @classmethod
    def _high_end_nvidia_gpu(cls) -> bool:
        raw = cls._nvidia_gpu_name().lower()
        if not raw:
            return False
        return any(token in raw for token in ("rtx 4070", "rtx 4080", "rtx 4090", "rtx 5070", "rtx 5080", "rtx 5090"))

    @staticmethod
    def _rtx_nvenc_args() -> list[str]:
        """H.264 NVENC tuned for RTX 4070/5070+ — balanced quality vs speed.

        Antes usaba preset p7 + multipass qres (calidad máxima a 1x realtime).
        Ahora usa p5 sin multipass para 3x velocidad con calidad visible
        identica tras el re-encode de Facebook/YouTube. CRF aleatorio 17-18
        para que cada render tenga un hash distinto (ayuda con originalidad).
        """
        import random
        cq = random.choice(["17", "18"])
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",            # antes p7 (3x mas rapido, calidad casi identica)
            "-tune", "hq",
            "-rc", "vbr",
            # multipass removido: ahorra 30-40% sin diferencia visible
            "-cq", cq,                  # 17-18 random (cada render fingerprint distinto)
            "-b:v", "50M",
            "-maxrate", "75M",
            "-bufsize", "100M",
            "-bf", "2",                 # antes 3 (-10% tiempo, cero diferencia)
            "-spatial-aq", "1",
            "-temporal-aq", "1",
            "-aq-strength", "8",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ]

    @staticmethod
    def _build_audio_chain(speed: float, volume_db: float, pitch_factor: float = 1.0, already_denoised: bool = False) -> str:
        """Cadena de audio profesional por defecto.

        Default = voice_pro: anlmdn (denoise no-local-means, mas limpio que afftdn
        en ruido estacionario tipo aire acond/calle/ventiladores), notch 50/60Hz
        para hum electrico, EQ de presencia marcado, de-esser suave, compresor.

        Override con env var (rara vez necesario):
          WORKFAST_AUDIO_PROFILE=cinema  -> loudnorm I=-16 (streaming target)
          WORKFAST_AUDIO_PROFILE=legacy  -> cadena vieja con afftdn (fallback)
        """
        import os as _os
        import random as _rnd
        profile = (_os.getenv("WORKFAST_AUDIO_PROFILE") or "voice_pro").lower()

        # base: speed + volume.
        # IMPORTANTE: el pitch shift NO se hace aqui en FFmpeg porque sin
        # librubberband solo queda asetrate+atempo que produce efecto chipmunk
        # (Alvin y las ardillas). Lo hacemos antes en Python con pedalboard
        # (preserva formantes -> voz humana real). Ver audio_enhancer.pitch_shift_speech.
        base = f"aresample=48000,atempo={speed:.5f},volume={volume_db:.2f}dB,"

        if profile in ("voice_pro", "cinema", "default"):
            denoise = ""
            # Notch a 50 y 60Hz para hum electrico (Argentina/USA dual).
            # Usamos equalizer mono-band en lugar de anequalizer para evitar
            # parsing fragil con espacios en el filter graph.
            notch = (
                "equalizer=f=50:t=q:w=4:g=-18,"
                "equalizer=f=60:t=q:w=4:g=-18,"
            )
            shaping = (
                "highpass=f=80,lowpass=f=15500,"
                # Calidez sin barro
                "equalizer=f=140:t=q:w=0.9:g=1.2,"
                # Presencia (claridad de voz)
                "equalizer=f=3000:t=q:w=1.0:g=2.8,"
                "equalizer=f=5000:t=q:w=1.2:g=1.6,"
                # Aire (brillantez)
                "equalizer=f=9500:t=q:w=1.4:g=1.4,"
                # De-esser suave (atenua sibilantes 6-9kHz)
                "equalizer=f=7200:t=q:w=1.6:g=-2.5,"
            )
            # ── EVASIÓN PROFUNDA DE FINGERPRINT DE AUDIO ──────────────────────
            # Las plataformas (YouTube Content ID, FB AudioMatch, ACRCloud) usan:
            #   1) Chromaprint: hash perceptual de espectrograma, sensible a pitch
            #   2) Espectrograma de frecuencia: busca patrones espectrales
            #   3) Correlación de forma de onda temporal
            # Atacamos los tres frentes con cambios imperceptibles al oído:

            # a) Pitch micro-shift via asetrate (MÁS EFECTIVO contra Chromaprint):
            #    Desplaza TODAS las frecuencias ±0.8% uniformemente. Inaudible pero
            #    invalida el hash perceptual porque el espectrograma se corre en Y.
            #    aresample corrige la tasa de muestreo para que la duración no cambie.
            #    DEBE ir al inicio, antes de aresample y atempo.
            _pitch_delta = _rnd.uniform(-0.008, 0.008)   # ±0.8% ratio lineal
            _pitch_rate = int(48000 * (1.0 + _pitch_delta))

            # b) Notches en 3 zonas de frecuencia (grave, medio, agudo).
            #    Rangos amplios e impredecibles para que el patrón sea irreconocible.
            _n1_f = _rnd.randint(280, 750)       # graves: 280-750Hz (zona sensible de Chromaprint)
            _n1_g = _rnd.randint(5, 9)
            _n1_w = _rnd.randint(10, 20)
            _n2_f = _rnd.randint(1800, 4500)     # medios: 1.8-4.5kHz (zona de voz fundamental)
            _n2_g = _rnd.randint(4, 8)
            _n2_w = _rnd.randint(12, 22)
            _n3_f = _rnd.randint(10500, 16000)   # agudos: 10.5-16kHz
            _n3_g = _rnd.randint(6, 11)
            _n3_w = _rnd.randint(14, 26)

            # c) aphaser: variación de fase imperceptible.
            #    speed < 0.4 y decay < 0.3 → no crea chorus audible.
            _ph_delay = _rnd.randint(1, 3)        # 1-3ms, inaudible
            _ph_speed = _rnd.uniform(0.1, 0.35)
            _ph_decay = _rnd.uniform(0.1, 0.28)

            # d) adelay micro-estéreo (0-4ms por canal).
            #    El oído no percibe < 5ms; los fingerprinters sí.
            _dl_l = _rnd.randint(0, 4)
            _dl_r = _rnd.randint(0, 4)

            # e) Micro-jitter de tempo (< 0.3%).
            _tempo_jit = 1.0 + _rnd.uniform(-0.003, 0.003)

            # f) Loudnorm con target aleatorio: la curva de dinámica varía cada render.
            _ln_i = _rnd.randint(13, 16)
            _ln_tp = round(_rnd.uniform(1.1, 1.8), 1)
            _ln_lra = _rnd.randint(8, 13)
            target = f"I=-{_ln_i}:TP=-{_ln_tp}:LRA={_ln_lra}"

            # Cadena completa: pitch shift al inicio → base → shaping → originality → comp → tail
            # El asetrate va ANTES de aresample+atempo para que el speed sea correcto.
            pitch_prefix = f"asetrate={_pitch_rate},aresample=48000,"
            originality = (
                # Notches espectrales en 3 zonas
                f"equalizer=f={_n1_f}:t=q:w={_n1_w}:g=-{_n1_g},"
                f"equalizer=f={_n2_f}:t=q:w={_n2_w}:g=-{_n2_g},"
                f"equalizer=f={_n3_f}:t=q:w={_n3_w}:g=-{_n3_g},"
                # Variación de fase
                f"aphaser=in_gain=0.92:out_gain=0.92:delay={_ph_delay}:decay={_ph_decay:.2f}:speed={_ph_speed:.2f}:type=t,"
                # Adelay micro-estéreo
                f"adelay=delays={_dl_l}ms|{_dl_r}ms,"
                # Micro-jitter de tempo
                f"atempo={_tempo_jit:.5f},"
            )
            comp = (
                "acompressor=threshold=-20dB:ratio=2.5:attack=10:release=200:makeup=1.7,"
            )
            tail = f"loudnorm={target},alimiter=limit=0.96"
            return pitch_prefix + base + denoise + notch + shaping + originality + comp + tail

        # default original (no romper a usuarios actuales)
        return (
            base
            + "afftdn=nf=-28,highpass=f=75,lowpass=f=16000,"
            "equalizer=f=120:t=q:w=0.8:g=1.5,"
            "equalizer=f=3200:t=q:w=1.1:g=2.4,"
            "equalizer=f=8500:t=q:w=1.4:g=1.2,"
            "acompressor=threshold=-18dB:ratio=2.2:attack=12:release=180:makeup=1.5,"
            "loudnorm=I=-14:TP=-1.5:LRA=10,"
            "alimiter=limit=0.96"
        )

    @staticmethod
    def _hevc_nvenc_args() -> list[str]:
        """HEVC NVENC: ~30-40% menos peso al mismo nivel visual que h264_nvenc.
        Recomendado para GPUs Pascal+ (GTX 10xx, RTX 20/30/40)."""
        return [
            "-c:v", "hevc_nvenc",
            "-preset", "slow",
            "-rc", "vbr",
            "-cq", "20",
            "-b:v", "30M",
            "-maxrate", "50M",
            "-bufsize", "70M",
            "-bf", "3",
            "-profile:v", "main",
            "-tag:v", "hvc1",  # compatibilidad con QuickTime/iOS/redes sociales
            "-pix_fmt", "yuv420p",
        ]

    @classmethod
    def _hevc_nvenc_can_encode(cls) -> bool:
        """Test real de hevc_nvenc."""
        cache_key = "hevc_nvenc_real"
        if cache_key in cls._ENCODER_CACHE:
            return cls._ENCODER_CACHE[cache_key]
        if not cls._has_encoder("hevc_nvenc"):
            cls._ENCODER_CACHE[cache_key] = False
            return False
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=duration=0.1:size=320x240:rate=30",
                    "-c:v", "hevc_nvenc",
                    "-pix_fmt", "yuv420p", "-t", "0.1",
                    "-f", "null", "-",
                ],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=20,
            )
            ok = result.returncode == 0
            cls._ENCODER_CACHE[cache_key] = ok
            return ok
        except Exception:
            cls._ENCODER_CACHE[cache_key] = False
            return False

    @staticmethod
    def _libx264_args() -> list[str]:
        return [
            "-c:v", "libx264",
            "-preset", "slow",          # mas tiempo de encode pero mejor calidad
            "-crf", "16",               # 16 = visualmente sin perdida (era 17)
            "-tune", "film",            # tune para fuente real (no animacion)
            "-x264opts", "ref=4:bframes=3:me=umh:subme=8:trellis=2",
            "-maxrate", "65M",
            "-bufsize", "90M",
            "-profile:v", "high",
            "-level", "4.2",
            "-pix_fmt", "yuv420p",
        ]

    _NVENC_LAST_ERROR: str = ""

    @classmethod
    def _nvenc_can_encode(cls) -> bool:
        """Hace un test REAL de encode con NVENC. Si funciona, devuelve True.

        Permite override con WORKFAST_FORCE_NVENC=1 (salta el test).
        Hace 2 tests consecutivos: el minimo y luego con presets nuevos.
        Si el minimo pasa pero los presets fallan, igual devuelve True
        (los args completos solo se usaran si todo el set funciona)."""
        import os as _os
        cache_key = "nvenc_real"
        if cache_key in cls._ENCODER_CACHE:
            return cls._ENCODER_CACHE[cache_key]

        # Override manual: el usuario sabe que su GPU es compatible
        if _os.getenv("WORKFAST_FORCE_NVENC") == "1":
            cls._ENCODER_CACHE[cache_key] = True
            cls._NVENC_LAST_ERROR = "Forzado por WORKFAST_FORCE_NVENC=1"
            return True

        # Verifica primero que el encoder esta listado en FFmpeg
        if not cls._has_encoder("h264_nvenc"):
            cls._ENCODER_CACHE[cache_key] = False
            cls._NVENC_LAST_ERROR = "h264_nvenc no esta listado en FFmpeg"
            return False

        # Test MINIMO: solo -c:v h264_nvenc sin opciones extra
        # Esto es lo mas universal posible. Si esto falla, NVENC realmente no esta.
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=duration=0.1:size=320x240:rate=30",
                    "-c:v", "h264_nvenc",
                    "-pix_fmt", "yuv420p", "-t", "0.1",
                    "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            ok = result.returncode == 0
            if not ok:
                cls._NVENC_LAST_ERROR = (result.stderr or "").strip()[:500]
            else:
                cls._NVENC_LAST_ERROR = ""
            cls._ENCODER_CACHE[cache_key] = ok
            return ok
        except Exception as exc:
            cls._ENCODER_CACHE[cache_key] = False
            cls._NVENC_LAST_ERROR = f"Excepcion: {exc}"
            return False

    @classmethod
    def get_nvenc_status(cls) -> dict:
        """Devuelve estado y razon de NVENC para diagnostico."""
        return {
            "available": cls._nvenc_can_encode(),
            "last_error": cls._NVENC_LAST_ERROR or None,
        }

    @staticmethod
    def _has_encoder(name: str) -> bool:
        if name in VideoEditor._ENCODER_CACHE:
            return VideoEditor._ENCODER_CACHE[name]
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            available = result.returncode == 0 and name in result.stdout
            VideoEditor._ENCODER_CACHE[name] = available
            return available
        except Exception:
            VideoEditor._ENCODER_CACHE[name] = False
            return False

    @staticmethod
    def _has_filter(name: str) -> bool:
        """Check if a FFmpeg filter (e.g. 'rubberband', 'anlmdn') is available."""
        cache_key = f"_filter_{name}"
        if cache_key in VideoEditor._ENCODER_CACHE:
            return VideoEditor._ENCODER_CACHE[cache_key]
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10,
            )
            # ffmpeg -filters lista cada filtro como " T.. nombre  descripcion"
            available = result.returncode == 0 and any(
                line.split()[1] == name for line in result.stdout.splitlines()
                if len(line.split()) >= 2 and not line.startswith(("Filters:", " ---"))
            )
            VideoEditor._ENCODER_CACHE[cache_key] = available
            return available
        except Exception:
            VideoEditor._ENCODER_CACHE[cache_key] = False
            return False

    def _strip_silences(
        self,
        source: Path,
        threshold_db: float = -32.0,
        min_duration: float = 0.45,
        padding: float = 0.12,
    ) -> Path:
        """Detect and remove silent gaps so dialogue feels tight.

        Uses ffmpeg silencedetect to find silent ranges, builds the complementary
        speech segments and concatenates them via select/aselect into a new MP4.
        Returns the trimmed file path or the original if no silences were found.
        """
        detect_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(source),
            "-af",
            f"silencedetect=noise={threshold_db}dB:d={min_duration}",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(
            detect_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        silences: list[tuple[float, float]] = []
        current_start: float | None = None
        for line in result.stderr.splitlines():
            stripped = line.strip()
            if "silence_start" in stripped:
                try:
                    current_start = float(stripped.split("silence_start:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    current_start = None
            elif "silence_end" in stripped and current_start is not None:
                try:
                    end = float(stripped.split("silence_end:")[1].strip().split()[0])
                    if end > current_start:
                        silences.append((current_start, end))
                except (ValueError, IndexError):
                    pass
                current_start = None

        if not silences:
            return source

        # Build complement (speech) segments with safety padding so we never clip a syllable.
        speech: list[tuple[float, float]] = []
        cursor = 0.0
        total = float(self.duration)
        for s_start, s_end in silences:
            seg_end = max(cursor, min(total, s_start + padding))
            if seg_end > cursor + 0.08:
                speech.append((cursor, seg_end))
            cursor = max(cursor, min(total, s_end - padding))
        if cursor < total - 0.08:
            speech.append((cursor, total))

        if not speech:
            return source

        # Skip if we'd remove more than 70% of the video; too aggressive.
        kept = sum(end - start for start, end in speech)
        if kept / total < 0.30:
            return source

        # Build select expressions: between(t,s1,e1)+between(t,s2,e2)+...
        expr = "+".join(
            f"between(t,{start:.3f},{end:.3f})" for start, end in speech
        )
        filter_complex = (
            f"[0:v]select='{expr}',setpts=N/FRAME_RATE/TB[vtrim];"
            f"[0:a]aselect='{expr}',asetpts=N/SR/TB[atrim]"
        )

        out = self.temp_dir / "no_silence.mp4"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filter_complex,
            "-map",
            "[vtrim]",
            "-map",
            "[atrim]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(out),
        ]
        try:
            self._run_ffmpeg(cmd, target_duration=None, start=2, end=5)
        except RuntimeError:
            return source
        return out if out.exists() and out.stat().st_size > 0 else source

    def _progress(self, percent: int, step: str) -> None:
        if self.progress_callback:
            self.progress_callback(percent, step)

    @staticmethod
    def _even(value: int) -> int:
        return value if value % 2 == 0 else value + 1

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    # Margen lateral de seguridad y ancho usable para titulos (1080 - 2*72 = 936)
    TITLE_SAFE_MARGIN = 72
    # Factor empirico para Montserrat Black: ancho promedio de glifo ~0.60 * fontsize
    # (W,M llegan a 0.90; i,l a 0.30). Usamos 0.62 como estimado conservador.
    TITLE_AVG_GLYPH_RATIO = 0.62
    # Cache de PIL ImageFont por fontsize para no reinstanciar
    _TITLE_FONT_CACHE: dict[int, object] = {}

    @classmethod
    def _pil_font_for(cls, font_size: int):
        """Devuelve ImageFont.truetype cacheado o None si Pillow no esta."""
        if font_size in cls._TITLE_FONT_CACHE:
            return cls._TITLE_FONT_CACHE[font_size]
        try:
            from PIL import ImageFont  # type: ignore
        except ImportError:
            cls._TITLE_FONT_CACHE[font_size] = None
            return None
        # TITLE_FONT viene escapado para FFmpeg; necesitamos la ruta cruda.
        font_file = cls.TITLE_FONT_PATH
        try:
            font = ImageFont.truetype(str(font_file), font_size)
        except Exception:
            font = None
        cls._TITLE_FONT_CACHE[font_size] = font
        return font

    @classmethod
    def _estimate_title_pixel_width(cls, line: str, font_size: int) -> float:
        """Mide ancho real con Pillow; fallback a heuristica calibrada."""
        font = cls._pil_font_for(font_size)
        if font is not None:
            try:
                # getbbox devuelve (x0, y0, x1, y1); el ancho es x1 - x0
                bbox = font.getbbox(line)
                return float(bbox[2] - bbox[0])
            except Exception:
                pass
        # Fallback heuristico (cuando no hay Pillow)
        wide_chars = set("MWQGOmw@%&")
        narrow_chars = set("ilItrfj.,'!|()1 ")
        width = 0.0
        for ch in line:
            if ch in wide_chars:
                width += font_size * 0.88
            elif ch in narrow_chars:
                width += font_size * 0.34
            else:
                width += font_size * cls.TITLE_AVG_GLYPH_RATIO
        return width

    @classmethod
    def _wrap_title(cls, text: str) -> tuple[list[str], int]:
        """Envuelve y dimensiona el titulo midiendo ancho real en pixeles.

        Garantiza que ninguna linea exceda WIDTH - 2*safe_margin. Empieza con
        fontsize 92 (impacto Opus) y baja hasta 48 si no cabe en max 4 lineas.
        """
        clean_text = " ".join(text.split())
        if not clean_text:
            return [], 0

        max_width_px = cls.WIDTH - 2 * cls.TITLE_SAFE_MARGIN
        max_lines = 4
        # Tamanos candidatos en orden de preferencia (impacto -> caben mas chars)
        candidate_sizes = [60, 54, 48, 44, 40, 36, 32]

        for font_size in candidate_sizes:
            # Aproxima cuantos caracteres caben por linea con este fontsize
            avg_glyph_px = font_size * cls.TITLE_AVG_GLYPH_RATIO
            chars_per_line = max(8, int(max_width_px / avg_glyph_px))

            lines = textwrap.wrap(
                clean_text,
                width=chars_per_line,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [clean_text]

            # Verifica que cada linea quepa pixel-perfect; si no, prueba menor fontsize
            fits = all(
                cls._estimate_title_pixel_width(line, font_size) <= max_width_px
                for line in lines
            )
            if not fits:
                continue
            if len(lines) <= max_lines:
                return lines, font_size

        # Fallback: usa el fontsize mas chico y trunca a 4 lineas con elipsis
        font_size = candidate_sizes[-1]
        avg_glyph_px = font_size * cls.TITLE_AVG_GLYPH_RATIO
        chars_per_line = max(8, int(max_width_px / avg_glyph_px))
        lines = textwrap.wrap(
            clean_text,
            width=chars_per_line,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [clean_text]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            tail = lines[-1]
            # Recorta para dejar lugar a "..."
            while (
                cls._estimate_title_pixel_width(tail + "...", font_size) > max_width_px
                and len(tail) > 1
            ):
                tail = tail[:-1].rstrip()
            lines[-1] = f"{tail.rstrip()}..."
            
        # Ensure no individual line exceeds max pixel width (e.g. string of only "W"s or "M"s)
        for i in range(len(lines)):
            if cls._estimate_title_pixel_width(lines[i], font_size) > max_width_px:
                tail = lines[i]
                while cls._estimate_title_pixel_width(tail + "...", font_size) > max_width_px and len(tail) > 1:
                    tail = tail[:-1].rstrip()
                lines[i] = f"{tail}..."
                
        return lines, font_size

    @staticmethod
    def _clean_overlay_text(text: str) -> str:
        cleaned = []
        for character in text:
            category = unicodedata.category(character)
            if category in {"So", "Cs", "Co", "Cn"}:
                cleaned.append(" ")
                continue
            cleaned.append(character)
        return " ".join("".join(cleaned).split())

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "’")
            .replace("%", "\\%")
            .replace("\n", " ")
        )
