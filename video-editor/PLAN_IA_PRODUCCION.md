# WorkFast — Plan de mejora IA (estado del arte 2026)

Plan completo basado en investigación del estado del arte actual. La idea es saltar de "edición con filtros FFmpeg" a "edición con IA real" en cada eslabón: video, audio, subtítulos y caras. Todo corre **local** en tu PC con NVIDIA y se descarga automáticamente la primera vez.

## Comparativa de las mejores librerías 2026 (investigado)

### Upscaling de video (mejorar calidad de la imagen)

| Librería | Calidad | Velocidad | VRAM | Recomendación |
|---|---|---|---|---|
| **SeedVR2** | Best overall (consistencia temporal real) | Lento | 12+ GB | Tope si tenés RTX 30/40 |
| **SUPIR** | Mejor realismo individual | Muy lento (10-50× más que ESRGAN) | 12+ GB | Imágenes hero, no batch |
| **FlashVSR** | Diffusion rápido | Medio | 8+ GB | Balance de calidad/velocidad |
| **Real-ESRGAN ncnn-vulkan** ⭐ | Excelente, GAN sólido | **Muy rápido** (Vulkan) | 2-4 GB | **El recomendado para 1650 Ti** |
| **AuraSR** | Bueno para texturas | Rápido | 4+ GB | Alternativa moderna |

**Decisión para WorkFast:** Real-ESRGAN ncnn-vulkan. Razones:
- Corre en cualquier GPU vía Vulkan, no necesita CUDA Toolkit
- Binario portable de ~30 MB, modelos de ~17 MB
- 2× / 4× upscale con buena calidad
- Modelo `realesr-animevideov3` es óptimo para video general
- Tu GTX 1650 Ti 4 GB lo corre sin drama

### Frame interpolation (suavizar 30→60 fps real)

| Librería | Calidad | Velocidad | Recomendación |
|---|---|---|---|
| **RIFE 4.6 ncnn-vulkan** ⭐ | Estado del arte | Tiempo real en GPU | **El recomendado** |
| **FILM** (Google) | Excelente | Lento | Para post-producción |
| **DAIN** | Buena | Lento | Legacy |

**Decisión:** RIFE 4.6 ncnn-vulkan. Mismo razonamiento que Real-ESRGAN: portable, rápido, sin torch.

### Speech enhancement (limpiar audio)

| Librería | Calidad | Tamaño | CPU/GPU | Recomendación |
|---|---|---|---|---|
| **DeepFilterNet 3** ⭐ | SOTA 2026 (PESQ/POLQA mejor) | ~3 MB modelo | CPU real-time | **El recomendado** |
| **Resemble Enhance** | Excelente, 44.1 kHz | ~500 MB | GPU recomendado | Pesado |
| **Demucs htdemucs_ft** | Mejor separación voz/música | ~80 MB | GPU | Para clips con música de fondo |
| **noisereduce** | Spectral gating básico | <1 MB | CPU | Sólo ruido estacionario |

**Decisión:** DeepFilterNet 3 como denoiser principal. Se ejecuta en CPU real-time, modelo de 3 MB, sin pre-instalación pesada. Para casos con música de fondo agregar **htdemucs** opcional.

### Subtítulos (alineación de timestamps)

| Librería | Precisión timestamps | Setup | Recomendación |
|---|---|---|---|
| **WhisperX** ⭐ | ~20-50 ms (wav2vec2 forced alignment) | Pulla torch + wav2vec2/idioma | **El recomendado** |
| **stable-ts** | ~50-100 ms (DTW) | Pulla torch | Más liviano que WhisperX |
| **faster-whisper + tweaks** | ~150-300 ms | Ya instalado | Lo que tenés ahora |
| **CrisperWhisper** | Verbatim, alta precisión | Reciente | Investigar |

**Decisión:** WhisperX para subtítulos. La inversión de torch (~700 MB) se justifica por la precisión absoluta de los timestamps, que es exactamente lo que pediste para "no más lag".

### Restauración de caras (talking heads)

| Librería | Calidad | Velocidad | Recomendación |
|---|---|---|---|
| **CodeFormer** ⭐ | Mejor en degradación severa | Medio | **El recomendado para vídeos viejos** |
| **GFPGAN** | Más rápido y estable en video | Rápido | Para videos ya decentes |
| **GPEN** | Alternativa | Medio | Backup |
| **RestoreFormer++** | Reciente | Medio | Investigar |

**Decisión:** CodeFormer + GFPGAN combinados (CodeFormer si la cara está muy degradada, GFPGAN si está OK pero blur). Detección automática con `face-detection-yolov8`.

## Arquitectura propuesta

```
[Video original]
       │
       ├─ enhancer.py
       │    ├─ Real-ESRGAN (si <1080p)
       │    ├─ RIFE (si <60fps)
       │    └─ CodeFormer (si detecta caras borrosas, opcional)
       │
[Video pre-mejorado] (1080p+ @ 60fps con caras restauradas)
       │
       ├─ audio_enhancer.py
       │    └─ DeepFilterNet 3 sobre la pista de audio
       │
[Audio limpio + Video HD] 
       │
       ├─ transcriber.py
       │    └─ WhisperX (forced alignment wav2vec2)
       │
[Word timestamps ±20ms]
       │
       └─ editor.py (FFmpeg pipeline existente)
              ├─ Compone bottom mirror + top
              ├─ Quema subtitulos .ass
              ├─ Drawtext titulo
              └─ Encode H.264 NVENC 45 Mbps
              
[VIDEO FINAL TikTok/Reels/Shorts profesional]
```

## Plan de implementación (próxima sesión)

### Fase 1: Enhancer de video (1-2h de trabajo de IA)
- ✅ Esqueleto en `backend/video_processor/enhancer.py` (creado)
- ✅ Wiring en `main.py`: llamar a `enhance_video()` antes de instanciar `VideoEditor`
- ✅ Toggle en UI "Mejorar calidad con IA" (default ON si <1080p)
- ✅ Detectar resolución y fps automáticamente
- ✅ Timeout y handling de errores (si falla la descarga del binario, fallback a procesar sin enhance)

### Fase 2: DeepFilterNet 3 audio (30 min)
- ✅ `pip install deepfilternet` en `requirements.txt`
- ✅ Crear `audio_enhancer.py` con `enhance_speech(input_wav, output_wav)`
- ✅ Pre-procesar audio en `editor.py` antes del FFmpeg principal
- ✅ Cache de DFN para no reprocesar el mismo video

### Fase 3: WhisperX subtítulos (1h)
- ✅ `pip install whisperx` en `requirements.txt`
- ✅ Reemplazar `_transcribe_with_faster_whisper` por `_transcribe_with_whisperx`
- ✅ Auto-download de modelo wav2vec2 por idioma (cacheado en `~/.cache/whisperx`)
- ✅ Mantener `_tighten_word_ends` como safety net (con WhisperX casi no recortará nada)

### Fase 4: Face restoration opcional (1-2h)
- 🔲 `facexlib` + `gfpgan` + `codeformer` en requirements
- 🔲 Detector de caras con yolov8-face
- 🔲 Toggle "Restaurar caras (talking head)" en UI
- 🔲 Procesar frame-by-frame con face mask, recomponer

### Fase 5: Tests y polish (30 min)
- ✅ Smoke test que valida descarga de binarios + aplicación end-to-end
- ✅ UI muestra progreso por etapa
- ✅ Si toggle "IA off" se desactiva todo y vuelve al pipeline FFmpeg puro
- ✅ Documentar `WORKFAST_DISABLE_AI=1` para entornos sin GPU

## Costos a considerar

| Aspecto | Costo |
|---|---|
| Espacio en disco | +700 MB (torch para WhisperX) + ~50 MB binarios ncnn + ~3 MB DFN model + modelos wav2vec2 ~300 MB/idioma |
| Primera ejecución | Descarga 1× (1-2 GB total) |
| Tiempo de render | +2-5× para video 1080p (upscale + interpolate). Audio +20%. Subs casi igual. |
| VRAM | 4 GB es suficiente. WhisperX small + Real-ESRGAN no concurrente. |

## Riesgos y mitigaciones

1. **GTX 1650 Ti 4 GB OOM con WhisperX large**
   - Mitigación: usar modelo `medium` o `small` con compute_type=int8.
2. **Real-ESRGAN tarda 5-10 min en video largo**
   - Mitigación: mostrar progreso por frame, paralelizar con `-j` flag de ncnn.
3. **Fallback si no hay GPU compatible con Vulkan**
   - Mitigación: detectar y desactivar features IA, dejar pipeline FFmpeg puro.
4. **Conflictos torch / faster-whisper / whisperx**
   - Mitigación: WhisperX ya internamente usa faster-whisper, así que no hay conflicto. Solo agregar la dep.

## Resultado esperado

Después de implementar las 5 fases, el flujo del usuario será exactamente igual:

1. Subir video (puede ser 480p o 720p de YouTube).
2. Click en "Procesar".
3. WorkFast detecta y aplica:
   - Real-ESRGAN → 1080p o más, nitidez IA.
   - RIFE → 60 fps suave real.
   - DeepFilterNet 3 → voz limpia studio-grade.
   - WhisperX → subtítulos con sincronía perfecta (±20 ms).
   - CodeFormer → caras restauradas si las hay.
   - FFmpeg pipeline existente → branding, títulos, encode final NVENC.
4. Output: video TikTok/Reels/Shorts a calidad profesional.

Esto pone WorkFast a la par de Submagic, Opus Clip, CapCut Pro y Captions.ai.
