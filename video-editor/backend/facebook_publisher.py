"""Facebook Graph API client for Reels publishing."""
from __future__ import annotations

import json
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Callable, Optional

GRAPH_BASE = "https://graph.facebook.com/v20.0"
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB per chunk


class FacebookAPIError(Exception):
    pass


# ── low-level helpers ──────────────────────────────────────────────────────────

def _raise_from_http_error(exc: urllib.error.HTTPError) -> None:
    body = exc.read()
    try:
        msg = json.loads(body).get("error", {}).get("message", str(exc))
    except Exception:
        msg = str(exc)
    raise FacebookAPIError(msg) from exc


def _graph_get(path: str, token: str, params: dict | None = None) -> dict:
    p = dict(params or {})
    p["access_token"] = token
    url = f"{GRAPH_BASE}/{path}?{urllib.parse.urlencode(p)}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        _raise_from_http_error(exc)


def _graph_post(path: str, token: str, data: dict, timeout: int = 60) -> dict:
    d = dict(data)
    d["access_token"] = token
    encoded = urllib.parse.urlencode(d).encode()
    req = urllib.request.Request(f"{GRAPH_BASE}/{path}", data=encoded, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        _raise_from_http_error(exc)


# ── public API ─────────────────────────────────────────────────────────────────

def get_pages(user_token: str) -> list[dict]:
    """Return all pages the user manages, with page-level access tokens."""
    data = _graph_get("me/accounts", user_token, {
        "fields": "id,name,access_token,fan_count,picture.width(60){url}"
    })
    return data.get("data", [])


def _upload_binary(upload_url: str, page_token: str, video_path: Path,
                   progress_cb: Optional[Callable[[float], None]] = None) -> None:
    """Upload video in 4 MB chunks using the Resumable Upload API."""
    file_size = video_path.stat().st_size
    offset = 0
    with open(video_path, "rb") as fh:
        while offset < file_size:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            req = urllib.request.Request(upload_url, data=chunk, method="POST")
            req.add_header("Authorization", f"OAuth {page_token}")
            req.add_header("Content-Type", "application/octet-stream")
            req.add_header("offset", str(offset))
            req.add_header("file_size", str(file_size))
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    resp_data = json.loads(resp.read())
                next_offset = resp_data.get("start_offset")
                offset = int(next_offset) if next_offset is not None else offset + len(chunk)
            except urllib.error.HTTPError as exc:
                _raise_from_http_error(exc)
            if progress_cb:
                progress_cb(min(offset / file_size, 1.0))


def upload_reel(
    page_id: str,
    page_token: str,
    video_path: Path,
    description: str,
    title: str = "",
    scheduled_ts: Optional[int] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> dict:
    """Full Reels upload: start → binary → finish. Returns FB API result dict."""
    file_size = video_path.stat().st_size

    # 1. Start session
    start = _graph_post(f"{page_id}/video_reels", page_token, {
        "upload_phase": "start",
        "file_size": str(file_size),
    })
    video_id = start.get("video_id")
    upload_url = start.get("upload_url")
    if not video_id or not upload_url:
        raise FacebookAPIError(f"upload start returned unexpected data: {start}")

    # 2. Upload binary
    _upload_binary(upload_url, page_token, video_path, progress_cb=progress_cb)

    # 3. Finish and publish/schedule
    finish_data: dict = {
        "upload_phase": "finish",
        "video_id": video_id,
        "description": description,
        "title": title or "",
    }
    if scheduled_ts:
        finish_data["video_state"] = "SCHEDULED"
        finish_data["scheduled_publish_time"] = str(scheduled_ts)
    else:
        finish_data["video_state"] = "PUBLISHED"

    result = _graph_post(f"{page_id}/video_reels", page_token, finish_data, timeout=120)
    return {"video_id": video_id, **result}


def check_video_status(video_id: str, page_token: str) -> dict:
    """Check processing + copyright status of an uploaded video."""
    try:
        return _graph_get(video_id, page_token, {
            "fields": "status,copyright_check_status"
        })
    except FacebookAPIError:
        return {}
