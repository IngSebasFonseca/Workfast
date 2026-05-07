from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from video_processor import VideoEditor  # noqa: E402


def main() -> None:
    source = ROOT / ".tmp" / "real5.mp4"
    output = ROOT / ".tmp" / "real5_rendered.mp4"
    logo = ROOT / "assets" / "uploads" / "logo_02fc53eaa5_1773290044550.png"
    follow = ROOT / "assets" / "uploads" / "follow_4faab96d9c_672686690_1502164695256195_5843398129386047960_n.jpg"
    ending = ROOT / "assets" / "uploads" / "ending_3be7dbc8a1_ending_spanol.mp4"

    if not source.exists():
        raise SystemExit(f"Missing benchmark source: {source}")

    progress = []
    started = time.perf_counter()
    editor = VideoEditor(
        input_video=source,
        output_path=output,
        logo_path=logo if logo.exists() else None,
        follow_image_path=follow if follow.exists() else None,
        ending_path=ending if ending.exists() else None,
        progress_callback=lambda percent, step: progress.append((percent, step)),
    )
    editor.process_complete(title_text="Benchmark")
    elapsed = time.perf_counter() - started

    print(f"Rendered: {output}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Progress updates: {len(progress)}")
    if progress:
        print(f"Last progress: {progress[-1]}")


if __name__ == "__main__":
    main()
