from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


PRESETS = {
    "mujer_latina_suave": {
        "speaker": "Serena",
        "instruct": (
            "Soft warm Latin American woman in her late twenties. "
            "Gentle conversational delivery, natural Spanish rhythm, clear diction, "
            "friendly tone, not theatrical, not childish."
        ),
    },
    "hombre_30": {
        "speaker": "Aiden",
        "instruct": (
            "Natural man around 30 years old. Warm midrange voice, confident but relaxed, "
            "clear conversational delivery, friendly Latin American neutral style, not old."
        ),
    },
}


def _dtype_from_arg(value: str) -> torch.dtype | None:
    raw = (value or "auto").lower()
    if raw in {"none", "default"}:
        return None
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _effective_device(value: str) -> str:
    requested = (value or "auto").strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _model_kind(tts: Qwen3TTSModel, model_id: str) -> str:
    kind = getattr(tts.model, "tts_model_type", None)
    if kind:
        return str(kind)
    lowered = model_id.lower()
    if "voicedesign" in lowered or "voice-design" in lowered:
        return "voice_design"
    if "customvoice" in lowered or "custom-voice" in lowered:
        return "custom_voice"
    if "base" in lowered:
        return "base"
    return "voice_design"


def _split_text(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            words = part.split()
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            continue
        candidate = f"{current} {part}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio / peak * 0.98
    return np.clip(audio, -1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a WorkFast Qwen3-TTS voice track.")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--model", default=os.getenv("WORKFAST_QWEN3_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
    parser.add_argument("--device", default=os.getenv("WORKFAST_QWEN3_DEVICE", "auto"))
    parser.add_argument("--dtype", default=os.getenv("WORKFAST_QWEN3_DTYPE", "auto"))
    parser.add_argument("--attention", default=os.getenv("WORKFAST_QWEN3_ATTENTION", "sdpa"))
    parser.add_argument("--max-chars", type=int, default=int(os.getenv("WORKFAST_QWEN3_MAX_CHARS", "220")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("WORKFAST_QWEN3_MAX_NEW_TOKENS", "2048")))
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    chunks = _split_text(text, max(80, args.max_chars))
    if not chunks:
        raise SystemExit("No text to synthesize.")

    device = _effective_device(args.device)
    dtype = _dtype_from_arg(args.dtype)
    load_kwargs: dict = {"device_map": device}
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    attention = (args.attention or "").strip()
    if attention and attention.lower() not in {"none", "default"}:
        load_kwargs["attn_implementation"] = attention

    tts = Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)
    kind = _model_kind(tts, args.model)
    preset = PRESETS[args.preset]
    wavs: list[np.ndarray] = []
    sample_rate = 24000
    silence = None

    for chunk in chunks:
        if kind == "voice_design":
            out, sample_rate = tts.generate_voice_design(
                text=chunk,
                language=args.language or "Auto",
                instruct=preset["instruct"],
                max_new_tokens=args.max_new_tokens,
            )
        elif kind == "custom_voice":
            out, sample_rate = tts.generate_custom_voice(
                text=chunk,
                language=args.language or "Auto",
                speaker=preset["speaker"],
                instruct=preset["instruct"],
                max_new_tokens=args.max_new_tokens,
            )
        else:
            raise RuntimeError(
                "This WorkFast preset needs a Qwen3 VoiceDesign or CustomVoice model. "
                "Set WORKFAST_QWEN3_MODEL accordingly."
            )
        audio = _normalize_audio(out[0])
        wavs.append(audio)
        if silence is None:
            silence = np.zeros(int(sample_rate * 0.10), dtype=np.float32)
        wavs.append(silence)

    if wavs and silence is not None:
        wavs = wavs[:-1]
    final = np.concatenate(wavs) if wavs else np.zeros(sample_rate, dtype=np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), final, sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
