# Estado actual de WorkFast / Evse Video Studio

Fecha ultima actualizacion: 2026-05-11 (sesion 3)
PC objetivo: Ryzen 7 8700F · 32 GB DDR5-4800 · NVIDIA RTX 5070 12 GB (Blackwell, compute 12.0, CUDA 13.2)
Ruta: `C:\Users\Usuario\Downloads\Workfast\video-editor`

Leer este archivo al inicio de cada sesion. Reemplaza todos los planes y notas viejos.

---

## Objetivo del programa

Aplicacion local para producir videos verticales 9:16 (TikTok, Reels, Shorts) con
un preset automatico: importacion desde YouTube/TikTok/Facebook, branding por perfil,
mejora IA de video, subtitulos karaoke, audio limpio, render GPU y descarga por lote.

Meta comercial: producir contenido a volumen con calidad suficiente para monetizar.
Flujo repetible, consistencia visual, subtitulos sincronizados.

---

## Como arrancar (estado actual)

1. Doble clic en `Abrir Evse.bat` (o `Abrir WorkFast.bat`, son alias equivalentes).
2. `start.bat` valida Python, FFmpeg y dependencias, luego abre el servidor.
3. El backend queda en `http://127.0.0.1:5000` y el navegador se abre solo.

Comando real del servidor:
```
venv\Scripts\python.exe -B backend\main.py
```

---

## Estado tecnico (2026-05-09)

Verificacion rapida:
```powershell
venv\Scripts\python.exe -m compileall -q backend scripts
```

Resultado: compilacion OK, sin errores, todos los modulos importan.

### Librerias y GPU — tabla de estado actual

| Componente | Version | GPU activa |
|---|---|---|
| PyTorch (venv principal) | 2.11.0+cu128 | Si — RTX 5070 |
| torchaudio | 2.11.0+cu128 | Si |
| onnxruntime-gpu | 1.26.0 | Si — CUDA + TensorRT |
| ctranslate2 (Whisper) | 4.7.1 | Si — CUDA |
| faster-whisper | 1.2.1 | Si |
| stable-ts | 2.19.1 | Si |
| deepfilternet | 0.5.6 | Si — via onnxruntime-gpu |
| PyTorch (venv Qwen3-TTS) | 2.11.0+cu128 | Si — RTX 5070 |
| transformers (Qwen3-TTS) | 4.57.3 | Si |
| Flask | 3.1.3 | N/A |
| yt-dlp | 2026.3.17 | N/A |

Nota: antes de 2026-05-09 el venv principal tenia torch 2.5.1+cpu (solo CPU).
Sesion 3 (2026-05-11): confirmado torch 2.11.0+cu128, DeepFilterNet en cuda:0.
deepfilternet 0.5.6 usa APIs removidas en torchaudio 2.x — se parcheo df/io.py
(ver seccion "Parche deepfilternet" mas abajo).

### ONNX providers disponibles
```
TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
```

---

## Capacidades actuales

### Video

- Render vertical 1080x1920 a 60 fps.
- Encoder principal: `h264_nvenc` (RTX 5070 detectada automaticamente).
- Fallback CPU `libx264` si NVENC falla.
- FFmpeg tambien tiene `av1_nvenc` disponible (no usado en output final por
  compatibilidad con plataformas, pero disponible para uso futuro).
- Fuentes Montserrat en `assets/fonts/` verificadas y funcionales.

### Mejora IA de video (enhancer)

- `backend/video_processor/enhancer.py`
- Real-ESRGAN ncnn-vulkan para upscale (solo en modo ultra).
- RIFE ncnn-vulkan para interpolacion a 60 fps (solo en modo ultra).
- Binarios en `assets/tools/`.
- **Modo actual: `balanced`** — no ejecuta redes neuronales de video.
  Aplica Lanczos + NVENC + nitidez. Cola rapida, GPU moderada.
- Modo ultra: usa Real-ESRGAN/RIFE, mas calidad, mas tiempo, RTX al 100%.
- Workers moderados: `NCNN_JOBS=1:2:1` para temperatura controlada.

### Subtitulos IA

- `backend/video_processor/transcriber.py`
- Motor: `faster-whisper` con `stable-ts` para timestamps de palabra.
- En esta PC: large-v3, CUDA, float16, batch 8.
- Modo: `WORKFAST_SUBTITLE_MODE=pro` (stable-ts + faster-whisper).
- Idiomas: original, espanol, ingles, portugues.
- **Paralelismo activado** (`WORKFAST_GPU_PARALLEL=1`): Whisper y mejora de
  video corren al mismo tiempo. Con 12 GB VRAM y ~10 GB libres es seguro.
- Cortar silencios: genera video temporal recortado primero para evitar
  desincronizacion en los subtitulos.
- Ajuste de timing: `WORKFAST_SUBTITLE_LEAD_MS=140` adelanta para habla rapida.
- UI: ajuste manual de offset en ms, boton "Preview 18s" para revisar sync.
- Estilo: karaoke por palabras, frases cortas (1-2 palabras, max 20 chars, 1.05s).

### Audio

- `backend/video_processor/audio_enhancer.py`
- Limpieza con DeepFilterNet (ahora en GPU via onnxruntime-gpu).
- Cadena FFmpeg `voice_pro`: denoise, notch 50/60 Hz, EQ de presencia,
  de-esser, compresion, loudnorm, limitador.
- Cache de audio en `assets/outputs/.workfast_tmp/audio_cache`.

### Voz (estado actual)

Motor actual: Qwen3-TTS (texto-a-voz, NO es voice conversion).

Flujo: audio original → Whisper transcribe → Qwen3-TTS sintetiza nueva voz
con ese texto → FFmpeg ajusta duracion para encajar en el video.

Limitacion: no es voice-to-voice exacto. El timing es aproximado por frase,
no palabra por palabra. Funciona, pero no es el nivel de ElevenLabs o CapCut.

Presets disponibles en UI:
- Original
- Qwen3-TTS: Mujer latina suave
- Qwen3-TTS: Hombre 30 anos
- Fallback local (pitch/formantes via Praat)

Siguiente paso para voz real: RVC via Applio (ver seccion de mejoras).
La infraestructura ya esta lista en el codigo (`WORKFAST_RVC_COMMAND`, carpeta
`assets/library/voice_models/rvc/`).

### Importacion de video

- YouTube, TikTok, Facebook via yt-dlp.
- Logica de cookies por plataforma: auto, archivo .txt, Edge, Chrome, Firefox, Brave.
- TikTok: descarga sin marca de agua via download_addr.
- Facebook: escala de cookies (facebook_cookies.txt > youtube_cookies.txt > browser).
- Brave no puede leerse directamente (DPAPI v127+). Usar siempre archivo facebook_cookies.txt.
- facebook_cookies.txt verificado 2026-05-09: cookies de auth validas hasta ago-nov 2026.
  ATENCION: cookie `datr` vence ~2026-05-22. Renovar con extension "Get cookies.txt LOCALLY".
- Perfil de importacion: marca videos ya importados para evitar duplicados.

### Frontend

- Archivo unico: `frontend/index.html` (2491 lineas, sin framework JS externo).
- Paleta de colores actualizada 2026-05-09:
  - Acento principal: `#8b5cf6` (violeta saturado tech, antes era #a78bfa lavanda)
  - Acento secundario: `#7c3aed` (violeta profundo)
  - Eliminado el rosa `#ec4899`, reemplazado por cyan `#06b6d4`
- Cola maxima: 50 videos (alineado con limite de subida de Facebook).
- Descarga ZIP: hasta 50 videos por lote (genera ZIP en RAM ~7 GB max, seguro con 32 GB).
- Descarga muestra mensaje "Preparando ZIP con N videos..." y nombre con fecha (evse_videos_YYYYMMDD.zip).
- Perfiles de marca: logo, sigueme, ending por perfil.
- Descarga en lote (ZIP).

---

## Flujo de render

`POST /api/process` → recibe video + assets + ajustes:

1. Si `remove_silences`: recorta silencios primero en archivo temporal.
2. Paralelo (si `GPU_PARALLEL=1`): subtitulos IA + mejora de video al mismo tiempo.
3. Render final con `VideoEditor` (editor.py).
4. Quema subtitulos si aplica.
5. Limpia temporales.
6. Devuelve MP4 + subtitulos `.ass` opcionales.

`POST /api/subtitles/preview` → preview corto de 18s con subtitulos para revisar
sincronizacion antes de renderizar toda la cola.

---

## Archivos principales

```
backend/
  main.py                    Flask API, jobs, importadores YT/TT/FB, render
  video_processor/
    editor.py                Pipeline FFmpeg final, NVENC, overlays, titulos, subs
    enhancer.py              Real-ESRGAN, RIFE, descarga de binarios
    transcriber.py           Whisper, stable-ts, traduccion, cache
    subtitles.py             Generador de ASS karaoke estilo short viral
    audio_enhancer.py        DeepFilterNet, Qwen3-TTS, hook RVC

frontend/
  index.html                 Toda la interfaz (UI sin framework)

scripts/
  smoke_test.py              Render sintetico end-to-end
  test_text_and_subs.py      Tests de titulos, subtitulos, audio
  test_enhancer_e2e.py       Imports de modulos IA
  qwen3_tts_render.py        Runner aislado Qwen3-TTS
  setup_qwen3_tts.ps1        Instala el venv aislado de Qwen3-TTS

assets/
  tools/                     Real-ESRGAN, RIFE (binarios ncnn-vulkan)
  library/
    voice_models/rvc/        Modelos RVC (cuando se configuren)
    youtube_chrome_profile/  Perfil Chrome para cookies
  fonts/                     Montserrat para titulos y subtitulos
  uploads/                   Videos del usuario (NO borrar)
  outputs/                   Videos procesados (se limpian solos, LRU 30)

.env                         Configuracion GPU, Whisper, voz, enhancer
requirements.txt             Dependencias Python del venv principal
```

---

## Config .env actual

```
WORKFAST_WHISPER_MODEL=auto
WORKFAST_WHISPER_DEVICE=auto
WORKFAST_WHISPER_COMPUTE=auto
WORKFAST_WHISPER_BATCHED=1
WORKFAST_WHISPER_BATCH_SIZE=8
WORKFAST_SUBTITLE_MODE=pro
WORKFAST_SUBTITLE_LEAD_MS=140
WORKFAST_ENHANCER_PROFILE=balanced
WORKFAST_ENHANCER_GPU=0
WORKFAST_NCNN_JOBS=1:2:1
WORKFAST_ENABLE_RIFE_BALANCED=0
WORKFAST_GPU_PARALLEL=1           <-- activado 2026-05-09 (antes era 0)
WORKFAST_QWEN3_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
WORKFAST_QWEN3_DEVICE=cuda:0
WORKFAST_QWEN3_ATTENTION=sdpa
WORKFAST_RVC_MODELS_DIR=assets/library/voice_models/rvc
WORKFAST_RVC_F0_METHOD=rmvpe
WORKFAST_RVC_DEVICE=cuda:0
WORKFAST_OLLAMA_MODEL=qwen3:14b
```

---

## Cambios sesion 3 (2026-05-11)

### DeepFilterNet en GPU — fix completo
El venv principal tenia torch 2.5.1+cpu. Cada vez que start.bat detectaba fallo
de importacion reinstalaba requirements.txt y volvía a bajar torch CPU.

Cambios aplicados:
- `torch` y `torchaudio`: 2.5.1+cpu → 2.11.0+cu128
- `requirements.txt`: linea 1 agregada `--extra-index-url https://download.pytorch.org/whl/cu128`
  y pins actualizados a `torch==2.11.0+cu128` / `torchaudio==2.11.0+cu128`
- `backend/video_processor/audio_enhancer.py`: modelo DeepFilterNet cacheado
  globalmente (`_df_cache`, `_get_df_model()`). No se recarga en cada video.
  El audio de entrada permanece en CPU; `df_features()` mueve los features a
  GPU internamente; `enhance()` ya devuelve tensor en CPU.

### Parche deepfilternet (venv/Lib/site-packages/df/io.py)
deepfilternet 0.5.6 usa APIs de torchaudio removidas en 2.x:
- `torchaudio.backend.common.AudioMetaData` → no existe en 2.11
- `torchaudio.info()` → eliminada en 2.11 (reemplazada por TorchCodec)
- `torchaudio.load()` / `torchaudio.save()` → requieren torchcodec en 2.11

Parche en `venv\Lib\site-packages\df\io.py`:
- `AudioMetaData`: try/except con dataclass de respaldo
- `ta.info()`: reemplazado por `soundfile.info()` en bloque try/except
- `ta.load()` / `ta.save()`: try/except con fallback a soundfile

ATENCION: si se reinstala deepfilternet, volver a aplicar el parche.
El parche ya esta documentado en requirements.txt como comentario.

### Descarga ZIP — fix RAM y velocidad
Antes: ZIP_DEFLATED + io.BytesIO → 23 GB en RAM, tarda varios minutos.
Ahora: ZIP_STORED (sin comprimir, los MP4 ya estan comprimidos) + archivo
temporal en disco (`assets/outputs/.workfast_tmp/bundle_UUID.zip`).
El archivo se borra automaticamente 2 horas despues (hilo daemon).
- `backend/main.py`: funcion `_delayed_unlink()` agregada; `download_bundle`
  reescrita; nombre del ZIP: `evse_videos_YYYYMMDD.zip`.

### Polling doble eliminado
`pollJob` en el frontend ya no hace requests HTTP propios.
Solo `pollAll` (setInterval cada 1.5s) hace los requests al backend.
`pollJob` ahora solo espera en un loop a que `pollAll` cambie `item.status`.

---

## Cambios sesion 2 (2026-05-09)

### Cola y descarga
- `MAX_QUEUE` subido de 20 → 50 (alineado con limite de subida de Facebook por tanda).
- `download_bundle` en backend: limite hardcodeado de 5 → 50. Ya no trunca el ZIP.
- Eliminada funcion `downloadBundle` duplicada en el frontend (habia dos; la segunda
  sobreescribia a la primera silenciosamente).
- `handleMissingResult`: ahora llama `syncQueueDom()` + `updateHero()` ademas de
  `syncGalleryDom()`. Antes el contador "Listo" en la cola no se actualizaba cuando
  el backend borraba archivos del disco tras el ZIP — causaba el desfase "12 vs 7".

### Cookies Facebook
- Verificadas y en orden. Ver seccion "Importacion de video" para estado de expiracion.

---

## Mejoras pendientes (en orden de prioridad)

### 1. Instalador .exe profesional

Objetivo: distribuir el programa como aplicacion instalable, con icono,
acceso directo, deteccion automatica de hardware en cualquier PC.

Plan tecnico:
- Launcher `.exe` con PyInstaller: convierte el `.bat` en ejecutable real con icono.
- Instalador con Inno Setup: genera un `EvsStudio-Setup.exe` que instala todo.
- Deteccion de hardware al instalar:
  - NVIDIA GPU → descarga wheels cu128, activa NVENC
  - AMD GPU → wheels CPU + vulkan para NCNN
  - Sin GPU → modo CPU completo, batch sizes reducidos
  - RAM < 16 GB → reduce batch_size de Whisper
- Incluye: FFmpeg, Python embebido o descarga automatica, dependencias.
- Crea: acceso directo en escritorio, entrada en Menu Inicio, desinstalador.
- El `.env` se genera automaticamente segun el hardware detectado.

Herramientas: PyInstaller (launcher), Inno Setup (installer .exe), Python
para el script de deteccion de hardware en tiempo de instalacion.

Estado: pendiente. El programa actual funciona solo con .bat.

### 2. Voz real tipo ElevenLabs / CapCut (RVC)

Objetivo: cambiar la voz a una persona completamente diferente manteniendo
timing exacto, no solo cambiar pitch o sintetizar texto.

Diferencia clave:
- Qwen3-TTS actual: texto → voz nueva (timing aproximado por frase)
- RVC: audio → voz nueva (timing exacto, misma cadencia, otra identidad)

Plan:
- Instalar Applio (interfaz RVC local, gratuita).
- Conseguir 2 modelos pre-entrenados con licencia permisiva:
  - mujer joven (dataset VCTK u otro libre)
  - hombre joven
- Configurar `WORKFAST_RVC_COMMAND` en .env con el comando de Applio.
- La RTX 5070 corre RVC en tiempo real o mas rapido.
- La infraestructura ya esta en el codigo, solo falta el modelo y el comando.

Comparativa:
- RVC local: 85-92% calidad, gratis, sin limite de minutos, privado
- ElevenLabs: 95-99% calidad, ~$22/mes, requiere internet y sube audio

Para produccion en volumen, RVC local es la mejor relacion calidad/costo.

### 3. Auto-clips desde video largo

Objetivo: de un video de 10-60 min sacar los mejores shorts automaticamente.

- Detectar silencios, picos de energia de voz, cambios de escena.
- Usar Ollama qwen3:14b (ya instalado) para: titulo, descripcion, hashtags,
  score de viralidad, identificar el gancho de los primeros 3 segundos.
- UI para aprobar/rechazar clips antes de renderizar.
- El hardware actual (Whisper large-v3 CUDA + Ollama local) puede hacer todo esto.

### 4. Vista previa de subtitulos antes de renderizar

- El endpoint `POST /api/subtitles/preview` ya existe.
- Falta: mostrar el video preview inline en la UI con controles de offset,
  tamano, posicion y estilo sin tener que renderizar toda la cola.

### 5. Reencuadre inteligente (crop dinamico)

- Detectar cara/cuerpo con YOLO o MediaPipe.
- Crop automatico 9:16 que sigue al sujeto.
- Util cuando el video original es 16:9 y el sujeto no esta centrado.

---

## Lo que NO se borra nunca

```
assets/uploads/          videos del usuario
assets/library/          perfiles, cookies, cache de subtitulos, assets de marca
assets/tools/            binarios Real-ESRGAN y RIFE
venv/                    entorno Python activo
assets/tools/qwen3_tts_venv/   venv aislado de Qwen3-TTS
```

## Reglas para la proxima sesion

1. Leer este archivo primero.
2. Verificar con: `venv\Scripts\python.exe -m compileall -q backend scripts`
3. No tocar `assets/uploads`, `assets/library`, `assets/tools` ni `venv`.
4. Si se cambian librerias, verificar imports:
   ```powershell
   $env:PYTHONPATH="backend"
   venv\Scripts\python.exe -B -c "from video_processor import VideoEditor; print('OK')"
   ```
5. Si se toca voz: RVC con modelos legales, no prometer ElevenLabs sin tenerlo probado.
6. No subir `.env` a Git. Contiene rutas locales y posibles claves.
