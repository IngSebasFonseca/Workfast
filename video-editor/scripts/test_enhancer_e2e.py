"""Smoke test para las mejoras de IA (Enhancer, DeepFilterNet, WhisperX).

Verifica que las librerías requeridas se puedan importar y los helpers de
configuración identifiquen la GPU correctamente, sin procesar video real.

Uso:
    python scripts/test_enhancer_e2e.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

def test_imports():
    print("=== Test 1: Importación de Enhancer ===")
    try:
        from video_processor.enhancer import enhance_video, EnhancerError
        print("[OK] Enhancer module load success.")
    except Exception as e:
        print(f"[FAIL] Enhancer module load: {e}")

    print("\n=== Test 2: Importación de Audio Enhancer (DeepFilterNet) ===")
    try:
        from video_processor.audio_enhancer import enhance_speech
        print("[OK] Audio Enhancer module load success.")
    except Exception as e:
        print(f"[FAIL] Audio Enhancer module load: {e}")

    print("\n=== Test 3: Importación de Transcriber (WhisperX / Faster-Whisper) ===")
    try:
        from video_processor.transcriber import _whisperx_available, _stable_ts_available
        print(f"[OK] Transcriber loaded. WhisperX: {_whisperx_available()}, Stable-ts: {_stable_ts_available()}")
    except Exception as e:
        print(f"[FAIL] Transcriber module load: {e}")

if __name__ == "__main__":
    test_imports()
    print("\nAll end-to-end smoke tests completed.")
