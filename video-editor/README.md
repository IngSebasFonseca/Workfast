# WorkFast Video Editor

App local para automatizar un preset de edicion estilo CapCut/Filmora para videos verticales de redes sociales.

## Que hace

- Exporta un maestro vertical 9:16, 1080x1920 a 60 fps para TikTok, Facebook Reels y YouTube Shorts.
- Video de alta tasa: objetivo 40 Mbps, maximo 60 Mbps, buffer 80 Mbps.
- Usa NVIDIA NVENC automaticamente si FFmpeg detecta una GPU compatible.
- Duplica el video en dos capas.
- Capa inferior: espejo horizontal, zoom 196%, saturacion alta y desvanecimiento suave.
- Capa superior: zoom 96%, filtro HD ligero y enfoque.
- Acelera video y audio a 1.05x.
- Audio: +5.4 dB, reduccion de ruido, filtros de voz y limitador.
- Logo con movimiento de izquierda a derecha durante todo el video.
- Titulo centrado con fondo negro y texto verde cada 10 segundos.
- Imagen "sigueme" en intervalos.
- Ending opcional al final.
- Barra de progreso real por job.
- Cola de hasta 5 videos con descarga individual o ZIP.
- Los videos exportados usan el titulo como nombre de archivo.
- Importacion desde YouTube: pega un canal, playlist o video, elige uno e importalo como video principal.
- Sesion de YouTube desde Edge, Chrome, Firefox o Brave cuando YouTube pide confirmar que no eres bot.
- Audio HD: denoise, EQ de voz, compresion, loudness y AAC 320 kbps.
- Perfiles de marca: cada persona crea su perfil y guarda logo, imagen de sigueme y ending.
- Opcion para ocultar subtitulos ya pegados en la franja inferior; las pistas de subtitulos embebidas no se copian al exportar.
- Subtitulos IA opcionales con OpenAI: transcribe, traduce a espanol/ingles/portugues y puede quemarlos dentro del MP4 final.

## Requisitos

- Windows 10/11.
- Python 3.10 o superior.
- FFmpeg y ffprobe disponibles en PATH.
- Node.js 20 o superior para resolver los retos JavaScript actuales de YouTube.

Para comprobar FFmpeg:

```powershell
ffmpeg -version
ffprobe -version
```

## Inicio rapido para usar

Haz doble clic en `Abrir WorkFast.bat`.
El archivo inicia el servidor si hace falta y abre `http://127.0.0.1:5000` en el navegador.

## Uso

1. Sube el video principal.
2. O pega un link de YouTube, pulsa `Cargar` e importa el video que quieras.
3. Crea o selecciona un perfil.
4. Sube o elige logo, sigueme y ending para ese perfil.
5. El titulo se llena con el nombre del video importado/subido y puedes editarlo.
6. Opcional: configura `OPENAI_API_KEY` en `.env` y activa `Generar subtitulos nuevos`.
7. Pulsa `Procesar cola`.
8. Revisa la barra de progreso.
9. Descarga el resultado cuando termine.

Usa la importacion de YouTube solo con contenido propio, con permiso o con derechos de uso para tus redes.
El modo `Auto` intenta primero sin sesion, como el flujo inicial que ya funcionaba, y luego usa la sesion guardada si existe.
Si YouTube muestra `Sign in to confirm you're not a bot`, pulsa `Abrir login`, inicia sesion en la ventana de Chrome de WorkFast y luego pulsa `Guardar sesion`.
La sesion se guarda localmente como `assets/library/youtube_cookies.txt`; no se sube a GitHub.
Tambien puedes subir un archivo Netscape `cookies.txt` manualmente con `Subir cookies`.

Los videos subidos y generados se guardan localmente en:

- `assets/uploads`
- `assets/outputs`

Esas carpetas estan ignoradas por Git para no subir tus videos privados.

## Fase 2

El plan de ejecucion para subtitulos tipo Opus Clips, traducciones por idioma y clips automaticos desde videos largos esta en:

```text
ROADMAP_PHASE_2.md
```

## Desarrollo

Instalacion manual:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python backend\main.py
```

Prueba de humo:

```powershell
python scripts\smoke_test.py
```

## Estructura

```text
backend/
  main.py
  video_processor/editor.py
frontend/
  index.html
assets/
  uploads/
  outputs/
requirements.txt
start.bat
```
