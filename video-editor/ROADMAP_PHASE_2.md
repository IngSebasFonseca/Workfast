# Plan de ejecucion - Fase 2

## Objetivo

Convertir WorkFast Video Editor en una herramienta de produccion para redes: importar contenido propio, crear clips cortos, agregar subtitulos estilo Opus Clips, traducir versiones por idioma y exportar paquetes listos para publicar.

## Fase 2.1 - Base estable

Estado: en progreso.

- Importacion de YouTube con cookies del navegador para canales o videos propios.
- Layout responsive con columnas independientes y boton de procesamiento siempre visible.
- Cola visual de progreso para importacion y render.
- Validacion de errores claros cuando YouTube pide sesion o bloquea por verificacion anti-bot.

Criterio de aceptacion:

- Un canal con muchos videos no rompe el layout.
- Se puede elegir Edge, Chrome, Firefox o Brave como fuente de sesion de YouTube.
- Si YouTube bloquea, el usuario recibe una accion concreta en pantalla.

## Fase 2.2 - Subtitulos estilo Opus Clips

Prioridad: alta.

- Transcribir audio con timestamps por palabra.
- Generar subtitulos dinamicos en formato ASS con palabras resaltadas en blanco y verde.
- Agregar un toggle en la UI: "Subtitulos automaticos".
- Permitir tamaño, posicion, color principal y color de palabra activa.
- Guardar subtitulos como archivo lateral para poder re-renderizar sin retranscribir.

Criterio de aceptacion:

- Los subtitulos se sincronizan con la voz.
- No se salen del video 9:16.
- El usuario puede activar o desactivar esta capa antes de renderizar.

## Fase 2.3 - Quitar o cubrir subtitulos existentes

Prioridad: media.

- Si el video trae pistas de subtitulos separadas, eliminarlas del archivo antes del render.
- Si los subtitulos estan quemados en la imagen, detectar zonas tipicas y cubrirlas con blur, caja, recorte o inpainting.
- Crear vista previa de la zona que se va a cubrir para evitar tapar rostros o elementos importantes.

Criterio de aceptacion:

- Videos con subtitulos blandos exportan limpio.
- Videos con subtitulos quemados quedan suficientemente limpios para colocar nuevos subtitulos encima.

## Fase 2.4 - Traduccion por idioma

Prioridad: media.

- Traducir transcript a ingles, portugues, hindi, aleman y otros idiomas configurables.
- Renderizar subtitulos traducidos con el mismo estilo.
- Opcion posterior: doblaje con TTS o voz IA, manteniendo niveles de audio y musica.
- Exportar un MP4 por idioma con nombre claro.

Criterio de aceptacion:

- El usuario selecciona uno o varios idiomas y obtiene versiones separadas.
- Los titulos, subtitulos y metadatos quedan traducidos.

## Fase 2.5 - Modo clips largos tipo Opus

Prioridad: alta despues de subtitulos.

- Subir videos largos de hasta 1 hora o mas.
- Detectar cortes candidatos por silencios, cambios de escena, energia de audio y frases fuertes.
- Generar clips de 60 a 90 segundos.
- Crear titulo, descripcion, hashtags y puntaje de viralidad para cada clip.
- Permitir revisar, descartar y renderizar por lote con el preset actual.

Criterio de aceptacion:

- Un video largo genera una lista de clips candidatos.
- Cada candidato tiene score, titulo, descripcion y hashtags.
- El usuario puede renderizar todos o solo los seleccionados.

## Fase 2.6 - Produccion y escala

Prioridad: despues del primer flujo completo.

- Historial de renders e importaciones.
- Perfiles de marca por cuenta/red social.
- Procesamiento en segundo plano con cola persistente.
- Aceleracion GPU si el equipo tiene soporte.
- Exportacion de paquete: video, descripcion, hashtags, subtitulos y miniatura.

## Dependencias tecnicas sugeridas

- Faster Whisper o Whisper para transcripcion.
- FFmpeg ASS subtitles para estilos avanzados.
- OpenCV para deteccion/cobertura de subtitulos quemados.
- SQLite para historial, cola y perfiles.
- Un proveedor de traduccion/LLM para traduccion y metadatos.
- Opcional: TTS para doblaje por idioma.
