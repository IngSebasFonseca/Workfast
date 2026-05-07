# WorkFast — Mejoras aplicadas (producción lista)

Esta versión queda **lista para usar**. Todas las mejoras de calidad están activas por defecto, sin necesidad de env vars ni configuraciones extra.

## Lo que está corregido

### Títulos que se desbordaban a los lados
- `_wrap_title()` ahora mide ancho real en píxeles. Si Pillow está instalado, usa `ImageFont.getbbox()` (exacto). Si no, fallback heurístico calibrado para Montserrat Black.
- Empieza con fontsize **92** (impacto Opus) y baja automáticamente hasta **48** según haga falta para que ninguna línea pase del ancho seguro `1080 − 2×72 = 936 px`. Máximo 4 líneas; si no entra, recorta con `…`.
- Verificado con 8 casos (incluyendo títulos largos y palabras anchas tipo `MMMM…`): cero overflows.

### Subtítulos que se quedaban después del habla
- `_tighten_word_ends()` recorta el `end` de cada palabra al `start` de la siguiente menos 30 ms y al "ancho natural" (`start + 0.18 + len·0.085 + 0.18`). En la práctica reduce 150–270 ms de cola por palabra.
- Quitado el `+0.05 s` de pad en `_render_phrase_event`.
- `\fad(80,80)` → `\fad(60,40)`: salida casi instantánea.
- Cada frase se recorta para no solaparse con la siguiente.
- Whisper configurado con `vad_parameters` estrictos (silencio mín. 250 ms), `condition_on_previous_text=False`, `no_speech_threshold=0.55`.
- Cache de subtítulos invalidado con prefijo `v2|` para regenerar con la nueva precisión.

### Calidad de video por defecto
- **NVENC**: CQ 19 → **18**, bitrate base 40 → **45 Mbps** (max 65 Mbps).
- **CPU libx264**: CRF 17 → **16**, preset `medium` → `slow`, `tune=film`, `ref=4`, `bframes=3`, `me=umh`, `subme=8`, `trellis=2`.
- **Filtro HD** default: 42 % → **55 %**, con sharpen progresivo 0.50–1.10 (antes 0.45–1.00).
- **Lanczos + accurate_rnd + full_chroma** ya estaba (preserva detalle al upscalar 720→1080).

### Calidad de audio por defecto (perfil pro)
La cadena por defecto ahora es estilo "voz profesional":
1. `aformat=sample_fmts=fltp` (compatibilidad universal con `anlmdn`).
2. `anlmdn` — denoise non-local-means, mucho mejor que `afftdn` para ruido estacionario tipo aire acond, ventiladores, calle.
3. **Notch a 50 y 60 Hz** — mata el hum eléctrico (cualquier país).
4. Highpass 80 Hz, lowpass 15.500 Hz.
5. EQ de presencia: +1.2 dB @ 140 Hz (calidez), **+2.8 dB @ 3 kHz** (claridad de voz), +1.6 dB @ 5 kHz, +1.4 dB @ 9.5 kHz (aire).
6. **De-esser** suave: −2.5 dB @ 7.2 kHz (sibilantes).
7. Compressor 2.5:1 con make-up 1.7 dB.
8. Loudnorm I=−14, TP=−1.5, LRA=10 + alimiter 0.96.

## Cómo usarlo

Sin tocar nada, simplemente:

```text
1. Doble click en "Abrir WorkFast.bat" (o "Abrir Evse.bat").
2. Sube/importa video.
3. Selecciona perfil de marca (logo, sigueme, ending).
4. Pulsa "Procesar cola".
```

El render ya viene con:
- Título centrado sin overflow.
- Subtítulos sincronizados con la voz (anti-lag).
- Audio profesional limpiado y nivelado para redes.
- Video 1080×1920 a 60 fps, NVENC 45 Mbps, sharpen 55 %.

## Overrides opcionales (sólo si necesitas cambiar algo puntual)

| Variable de entorno | Para qué |
|---|---|
| `WORKFAST_USE_HEVC=1` | Salida HEVC (h265) en lugar de H.264. Archivos ~30 % más livianos al mismo nivel visual. Requiere GPU Pascal+ (GTX 10xx/20xx/30xx/40xx). |
| `WORKFAST_FORCE_CPU=1` | Forzar libx264 aunque haya NVENC. |
| `WORKFAST_FORCE_NVENC=1` | Saltar el self-test de NVENC. |
| `WORKFAST_AUDIO_PROFILE=cinema` | loudnorm I=−16 (target streaming, voz un poco más baja). |
| `WORKFAST_AUDIO_PROFILE=legacy` | Vuelve a la cadena vieja con `afftdn`. |

## Test de humo

`scripts/test_text_and_subs.py` corre sin FFmpeg ni Whisper y valida los tres fixes principales:

```powershell
python scripts\test_text_and_subs.py
```

Espera ver "Todos los tests OK".

## Archivos modificados

- `backend/video_processor/editor.py` — wrap título con medición pixel, audio chain `voice_pro` por defecto, args NVENC/libx264 con mejor calidad, `filter_intensity` default 0.55.
- `backend/video_processor/subtitles.py` — `_tighten_word_ends`, `\fad(60,40)`, sin pad +0.05.
- `backend/video_processor/transcriber.py` — VAD estricto, `condition_on_previous_text=False`, `no_speech_threshold=0.55`, cache `v2|`.
- `backend/main.py` — `filter_intensity` default 0.55.
- `frontend/index.html` — toggle de subtítulos con texto "Sincronia mejorada (anti-lag)", filtro HD default 55.
- `requirements.txt` — Pillow agregada (medición exacta de títulos).
- `scripts/test_text_and_subs.py` — nuevo, smoke tests de los fixes.

## Roadmap a futuro (opcional, no necesario hoy)

Si en algún momento querés más:

- **stable-ts** (`pip install stable-ts`) — el código ya lo detecta. Realinea timestamps por silencios reales, ~50 ms más preciso. Pulla torch (~700 MB), por eso no está como dep obligatoria.
- **Real-ESRGAN ncnn-vulkan** — upscale IA para fuentes <1080p.
- **DeepFilterNet 3** — denoise neuronal aún más limpio (CPU, modelo de 3 MB).
- **HEVC NVENC** ya disponible con `WORKFAST_USE_HEVC=1`.

Pero todo lo que pediste — títulos sin overflow, subtítulos sincronizados con la voz, audio limpio, video con calidad — ya está activado por defecto.
