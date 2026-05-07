from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from video_processor import VideoEditor  # noqa: E402


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def main() -> None:
    tmpdir = ROOT / ".tmp" / "smoke"
    if tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        source = tmpdir / "source.mp4"
        output = tmpdir / "output.mp4"
        logo = tmpdir / "logo.png"
        follow = tmpdir / "follow.png"
        ending = tmpdir / "ending.mp4"

        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=720x1280:rate=30:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=880:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(source),
            ]
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x39ff88:s=320x120:d=1",
                "-frames:v",
                "1",
                str(logo),
            ]
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=white:s=360x160:d=1",
                "-frames:v",
                "1",
                str(follow),
            ]
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=720x1280:rate=30:duration=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(ending),
            ]
        )

        progress = []
        editor = VideoEditor(
            input_video=source,
            output_path=output,
            logo_path=logo,
            follow_image_path=follow,
            ending_path=ending,
            progress_callback=lambda percent, step: progress.append((percent, step)),
        )
        editor.process_complete(title_text="Smoke Test")

        if not output.exists() or output.stat().st_size <= 0:
            raise AssertionError("Output video was not created.")

        run(["ffprobe", "-v", "error", "-show_format", "-show_streams", str(output)])
        print(f"Smoke test OK: {output}")
        print(f"Progress updates: {len(progress)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
