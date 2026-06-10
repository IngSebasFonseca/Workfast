# WorkFast / Evse Video Studio

App local para producir videos verticales 9:16 para TikTok, Reels y Shorts con
flujo automatico: importacion, branding, mejora IA, subtitulos, audio limpio y
render por GPU.

Para estado completo, diagnostico y roadmap tecnico, lee:

```text
ESTADO_ACTUAL_WORKFAST.md
```

## Inicio rapido

En Windows:

```text
Abrir WorkFast.bat
```

Tambien existe:

```text
Abrir Evse.bat
```

Ambos llaman a `start.bat`, preparan el entorno y abren:

```text
http://127.0.0.1:5000
```

## Capacidades actuales

- Exporta vertical 1080x1920 a 60 fps.
- Usa NVIDIA NVENC si esta disponible.
- Perfil RTX HQ para la RTX 5070.
- Mejora IA de video con Real-ESRGAN y RIFE ncnn-vulkan.
- Subtitulos IA locales con Whisper `large-v3` + `stable-ts` en modo pro.
- Traduccion de subtitulos a espanol, ingles y portugues.
- Subtitulos y mejora IA corren en paralelo cuando ambas opciones estan activas.
- Audio limpio con DeepFilterNet + cadena `voice_pro`.
- Presets de voz: original, Qwen3-TTS mujer/hombre y fallback local rapido.
- Qwen3-TTS local configurado como texto-a-voz para reemplazar la narracion.
- Hook listo para RVC/OpenVoice u otro conversor neuronal con `WORKFAST_VOICE_CONVERTER_CMD`.
- Cola de hasta 20 videos.
- Perfiles de marca con logo, imagen de sigueme y ending.
- Importacion desde YouTube/TikTok/Facebook mediante yt-dlp y cookies locales.

## Requisitos

- Windows 10/11.
- Python 3.10 o 3.11 de python.org.
- FFmpeg y ffprobe en PATH.
- Driver NVIDIA actualizado.
- Node.js 20+ ayuda con retos actuales de YouTube/yt-dlp.

## Comandos utiles

Instalar dependencias manualmente:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Arrancar backend manualmente:

```powershell
venv\Scripts\python.exe -B backend\main.py
```

Verificar:

```powershell
venv\Scripts\python.exe -m pip check
venv\Scripts\python.exe -m compileall -q backend scripts
venv\Scripts\python.exe scripts\test_text_and_subs.py
venv\Scripts\python.exe scripts\test_enhancer_e2e.py
venv\Scripts\python.exe scripts\smoke_test.py
```

## Estructura

```text
backend/
  main.py
  video_processor/
    editor.py
    enhancer.py
    transcriber.py
    subtitles.py
    audio_enhancer.py
frontend/
  index.html
assets/
  fonts/
  library/
  uploads/
  outputs/
  tools/
scripts/
  smoke_test.py
  test_text_and_subs.py
  test_enhancer_e2e.py
```

## Datos locales

Estas carpetas pueden contener archivos importantes del usuario:

- `assets/uploads`
- `assets/outputs`
- `assets/library`

No las borres sin revisar. Estan ignoradas por Git para no subir videos,
perfiles, cookies ni datos privados.

## Nota de voz IA

Qwen3-TTS transcribe el audio original, genera una pista nueva con una voz
mujer/hombre y ajusta la duracion al video. Es texto-a-voz, no voice-to-voice
labial exacto. Para conversion de voz neuronal pura tipo RVC/OpenVoice, configura
un motor externo con modelos o voces legales mediante:

```text
WORKFAST_VOICE_CONVERTER_CMD
WORKFAST_VOICE_REFERENCE
```
