"""Ollama-based description + hashtag generator for Facebook Reels."""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:14b"

# /no_think al inicio desactiva el modo razonamiento de Qwen3 (más rápido, sin <think> blocks).
_PROMPT = """\
/no_think
Eres un experto en marketing digital para redes sociales hispanas.
Escribe una descripción llamativa para un Reel de Facebook.

Título del video: {title}
Extracto de transcripción: {transcript}

REGLAS:
- Descripción de 2-3 líneas: emotiva, directa, que invite a ver y compartir
- Máximo 2 emojis en toda la descripción
- 8 hashtags relevantes: mezcla de específicos del tema y de alcance masivo
- Responde SOLO con el formato indicado abajo, sin texto extra ni explicaciones

FORMATO EXACTO:
DESCRIPCION:
[tu descripción aquí]

HASHTAGS:
#tag1 #tag2 #tag3 #tag4 #tag5 #tag6 #tag7 #tag8\
"""


def generate_description(title: str, transcript: str = "", model: str = DEFAULT_MODEL) -> dict:
    """Call Ollama and return {"description": str, "hashtags": [str, ...]}.

    Raises RuntimeError if Ollama is not reachable.
    """
    prompt = _PROMPT.format(
        title=title[:300],
        transcript=transcript[:1500] if transcript else "No disponible",
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # API-level thinking disable (Ollama >= 0.6)
        "options": {"temperature": 0.72, "num_predict": 500},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama no responde en {OLLAMA_BASE}. "
            "Asegúrate de que Ollama esté corriendo (ollama serve)."
        ) from exc

    raw_response = data.get("response", "")
    # Strip any <think>...</think> blocks Qwen3 might still emit
    clean = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    return _parse(clean)


def _parse(text: str) -> dict:
    desc_lines: list[str] = []
    hashtags: list[str] = []
    mode: str | None = None

    for raw in text.strip().splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("DESCRIPCION:"):
            mode = "desc"
            rest = line[len("DESCRIPCION:"):].strip()
            if rest:
                desc_lines.append(rest)
        elif upper.startswith("HASHTAGS:"):
            mode = "tags"
            rest = line[len("HASHTAGS:"):].strip()
            hashtags.extend(w for w in rest.split() if w.startswith("#"))
        elif mode == "desc":
            desc_lines.append(line)
        elif mode == "tags":
            hashtags.extend(w for w in line.split() if w.startswith("#"))

    description = "\n".join(l for l in desc_lines if l).strip()
    hashtags = [h for h in hashtags if len(h) > 1][:10]
    return {"description": description, "hashtags": hashtags}


_HOOK_PROMPT = """\
/no_think
Sos experto en miniaturas virales de YouTube (estilo MrBeast).
Convertí este título en un GANCHO ultra corto para la miniatura.

Título del video: {title}
Extracto: {transcript}

REGLAS:
- Máximo 4 palabras, en MAYÚSCULAS
- Que genere curiosidad o impacto (no aburrido, no descriptivo)
- En español neutro
- Sin emojis, sin comillas, sin signos al inicio
- Marcá la palabra MÁS impactante entre asteriscos, ej: TODO ERA *FALSO*
- Responde SOLO con el gancho, nada más
"""

_BANNED_HOOK = {"", "GANCHO", "TITULO", "MINIATURA", "VIDEO"}


def generate_thumbnail_hook(title: str, transcript: str = "", model: str = DEFAULT_MODEL,
                            timeout: float = 12.0) -> str:
    """Genera un gancho corto (<=4 palabras, MAYUS) para la miniatura.

    Usa Ollama. Si no responde a tiempo, levanta RuntimeError (el caller cae
    al titulo original). Timeout corto a proposito: no debe colgar el render.
    """
    prompt = _HOOK_PROMPT.format(
        title=(title or "")[:200],
        transcript=(transcript[:600] if transcript else "No disponible"),
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.9, "num_predict": 24},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Ollama no disponible para el hook: {exc}") from exc

    raw = data.get("response", "")
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Tomar la primera linea no vacia, limpiar comillas/puntuacion
    line = next((l.strip() for l in clean.splitlines() if l.strip()), "")
    line = line.strip(' "\'.:!¡¿?').upper()
    # Cap a 4 palabras
    words = line.split()
    if len(words) > 4:
        line = " ".join(words[:4])
    if line in _BANNED_HOOK or len(line) < 2:
        raise RuntimeError("Hook vacio o invalido")
    return line


def check_ollama() -> dict:
    """Return {"running": bool, "models": [...], "has_model": bool, "active_model": str}."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        has = len(models) > 0
        active = models[0] if models else ""
        return {"running": True, "models": models, "has_model": has, "active_model": active}
    except Exception as exc:
        return {"running": False, "models": [], "has_model": False, "active_model": "", "error": str(exc)}
