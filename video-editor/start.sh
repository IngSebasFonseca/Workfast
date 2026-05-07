#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "WorkFast Video Editor"
echo "====================="

command -v python3 >/dev/null || {
  echo "ERROR: python3 is not installed."
  exit 1
}

command -v ffmpeg >/dev/null || {
  echo "ERROR: ffmpeg is not installed."
  exit 1
}

command -v ffprobe >/dev/null || {
  echo "ERROR: ffprobe is not installed."
  exit 1
}

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Server running at http://localhost:5000"
python backend/main.py
