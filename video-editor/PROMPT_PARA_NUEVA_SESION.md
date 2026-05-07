# Prompt para usar en nueva sesión

Cuando vuelvas con tokens, copiá y pegá ESTO entero al iniciar la nueva sesión. Tiene todo el contexto + plan + criterios para que el agente continúe sin que tengas que explicarle de nuevo.

---

## INICIO DEL PROMPT

Estoy continuando un proyecto: **WorkFast Video Editor**, una app web local en Windows para automatizar edición de videos verticales tipo TikTok / Reels / Shorts. Está en `C:\Users\fonck\Downloads\Workfast\video-editor`. Stack: backend Flask + Python + FFmpeg, frontend HTML/JS único (`frontend/index.html`). Tengo NVIDIA GTX 1650 Ti 4 GB.

### Lo que ya está hecho (NO retocar)
- Fix de overflow lateral de títulos (medición pixel-exacta, fontsize auto 92→48, max 4 líneas).
- Fix de subtítulos colgados (`_tighten_word_ends`, `\fad(60,40)`, sin pad +0.05 s, VAD estricto).
- Audio chain default = perfil "voice_pro": `anlmdn` + notch 50/60 Hz + EQ presencia +2.8 dB @ 3 kHz + de-esser + compressor pro + loudnorm I=-14.
- NVENC con CQ 18, 45 Mbps base; libx264 con CRF 16 preset slow tune=film.
- Filter intensity default 0.55, sharpen 0.50–1.10.
- Pillow agregada a `requirements.txt`.
- Esqueleto de `backend/video_processor/enhancer.py` con Real-ESRGAN + RIFE ncnn-vulkan (auto-download).
- Plan completo en `PLAN_IA_PRODUCCION.md`.
- Smoke tests en `scripts/test_text_and_subs.py`.

### Lo que HAY QUE HACER (en este orden, leyendo `PLAN_IA_PRODUCCION.md` para detalles)

**Fase 1 — Wiring del video enhancer (Real-ESRGAN + RIFE)**
1. Leé `backend/video_processor/enhancer.py` que ya está creado.
2. En `backend/main.py`, en el worker de `/api/process` (alrededor de línea 1047 en adelante), antes de instanciar `VideoEditor`, llamá a `enhance_video()` SI el toggle `enhance_with_ai` está activo en `data`. Pasá el progress_callback con un step "Mejorando calidad con IA".
3. En `frontend/index.html`, agregá un toggle "Mejorar calidad con IA (auto-detect)" en la sección de inspector, default ON. Al enviar, pasá `enhance_with_ai: $("enhanceWithAiInput").checked` en el payload.
4. Manejar errores con gracia: si falla la descarga del binario, log warning y seguir sin enhance (no romper el render).
5. Verificá con un video 720p de prueba.

**Fase 2 — DeepFilterNet 3 audio**
1. Agregá `deepfilternet>=0.5.6` a `requirements.txt`.
2. Creá `backend/video_processor/audio_enhancer.py` con `enhance_speech(input_wav: Path, output_wav: Path) -> Path` usando `from df.enhance import enhance, init_df`.
3. En `editor.py · _render_main_video`, antes del comando FFmpeg principal, extraé el audio a wav, pásalo por DeepFilterNet, y reemplazá el input de audio del FFmpeg principal con el wav limpiado (usar `-i wav_limpio.wav -map 1:a` o similar).
4. Cache: hashear input audio para no reprocesar.

**Fase 3 — WhisperX para subtítulos**
1. Agregá `whisperx>=3.1.0` a `requirements.txt`.
2. En `backend/video_processor/transcriber.py`, agregá `_transcribe_with_whisperx()` que use `whisperx.load_model()` + `whisperx.load_align_model(language)` + `whisperx.align(segments, alignment_model, ...)`. Auto-detect: si whisperx está instalado, usalo; sino faster-whisper.
4. WhisperX devuelve segmentos con palabras alineadas a wav2vec2 (~20-50 ms precisión). Mapear a `Word(text, start, end)` igual que el código existente.
5. La invalidación de cache `v2|` ya está; bumpear a `v3|` si querés forzar regeneración.

**Fase 4 — Face restoration con CodeFormer (opcional, hacelo solo si queda tiempo)**
1. `pip install gfpgan codeformer-pip facexlib` (verificar nombres correctos en PyPI).
2. Toggle UI "Restaurar caras IA" — solo activable cuando `enhance_with_ai` está ON.
3. En `enhancer.py`, agregá `restore_faces(video_in, video_out)` que extrae frames, detecta caras con `facexlib.detection`, las pasa por CodeFormer (severidad alta) o GFPGAN (suave), y recompone.
4. Posicionar este paso DESPUÉS del upscale.

**Fase 5 — Tests + polish**
1. Extender `scripts/test_text_and_subs.py` para validar import de los módulos nuevos sin ejecutarlos.
2. Crear `scripts/test_enhancer_e2e.py` que toma un video corto (5 s) y aplica el pipeline completo, valida que el output existe y tiene >0 bytes.
3. UI: que cada paso muestre un step distinto en la barra de progreso.
4. Doc final actualizado en `PLAN_IA_PRODUCCION.md`.

### Criterios de calidad (no negociables)
- Si una dep IA falla en runtime, el pipeline DEBE seguir con el fallback FFmpeg sin romper.
- Toda dep IA debe ser auto-detectable, no requerir setup manual del usuario más allá del `pip install -r requirements.txt`.
- Los binarios ncnn-vulkan se descargan en `assets/tools/` la primera vez. La descarga debe mostrar progreso al usuario.
- El render NO debe ser >5× más lento que sin IA. Si Real-ESRGAN tarda mucho en un video largo, mostrar advertencia en UI.
- Cero secrets en el código (no API keys, todo local).
- Mantener compatibilidad con el flujo "sin IA" (toggle off = pipeline FFmpeg puro como hoy).

### Entregables esperados al terminar
1. Toggle UI "Mejorar calidad con IA" funcional, default ON.
2. Pipeline e2e funcionando con video 720p → 1080p+ a 60fps con audio limpiado y subs precisos.
3. `requirements.txt` actualizado.
4. Tests pasando.
5. Doc final con instrucciones de "primera ejecución" (las descargas iniciales).
6. Confirmar al usuario qué tarda más, qué consume más VRAM, y cómo desactivar selectivamente.

### Estilo
- Spanish para mensajes al usuario, English para nombres de funciones internas (consistente con el código actual).
- No emojis en código.
- Código sintáctico al estilo del repo: type hints, dataclasses, sin frameworks pesados de patrones.
- Cada función con docstring corto que explique el "por qué" además del "qué".

Empezá por leer `PLAN_IA_PRODUCCION.md`, después `backend/video_processor/enhancer.py`, y arrancá con la Fase 1.

## FIN DEL PROMPT

---

## Anotaciones para vos (Joan)

- Antes de pegar el prompt, asegurate que la próxima sesión tiene acceso a la carpeta `Workfast` igual que esta.
- Si la sesión nueva no encuentra los archivos, simplemente decile "abrime la carpeta Workfast" y eso le da acceso.
- Cuando termine, pedile que corra `scripts/test_text_and_subs.py` para validar que nada se rompió.
- Probá con un video bajo (480p o 720p) para ver el upscale en acción. En 1080p ya no hace falta upscale.
- La primera ejecución va a tardar 5-10 min en descargar binarios + modelos wav2vec2. Después es instant cache.

## Lista de descargas que hace el sistema en primera ejecución

| Componente | Tamaño | Cuándo |
|---|---|---|
| `realesrgan-ncnn-vulkan` (binario + modelos) | ~50 MB | Primer video que se mejora |
| `rife-ncnn-vulkan` (binario + modelos) | ~30 MB | Primer video que se interpola |
| DeepFilterNet 3 modelo | ~3 MB | Primer audio limpiado |
| WhisperX modelo + wav2vec2 (es/en/pt) | ~700 MB-1.2 GB | Primer subtítulo generado |
| CodeFormer + GFPGAN modelos | ~600 MB | Primer face restore |

Total: ~2 GB la primera vez. Después se queda en cache local.
