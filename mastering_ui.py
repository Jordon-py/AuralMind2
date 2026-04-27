"""
Flask dashboard for local AuralMind2 mastering control.

The UI keeps its own lightweight session model for waveform/spectrogram state
and delegates actual mastering work to the in-process server module through the
`MasteringUIBridge`.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_from_directory
from scipy import signal

from mastering_ui_bridge import MasteringUIBridge

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT_DIR / "data"
UPLOAD_DIR = DATA_ROOT / "ui_uploads"
EXPORT_DIR = ROOT_DIR / "Album_Ignorance_is_bliss" / "masters"
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"}

DATA_ROOT.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

bridge = MasteringUIBridge(export_root=EXPORT_DIR)
mastering_sessions: dict[str, "MasteringSession"] = {}
session_lock = threading.Lock()


def run_async(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _default_metrics() -> dict[str, float]:
    return {
        "lufs": -23.0,
        "true_peak": -1.0,
        "crest_db": 0.0,
        "stereo_corr": 0.0,
    }


def _sanitize_upload_name(filename: str) -> str:
    original = Path(filename or "upload.wav").name
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(original).stem)
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
    return f"{stem or 'audio'}_{uuid.uuid4().hex[:8]}{suffix}"


def _resolve_session_audio_path(audio_path: str) -> Path:
    candidate = Path(audio_path)
    if not candidate.is_absolute():
        candidate = DATA_ROOT / candidate
    resolved = candidate.resolve()
    if os.path.commonpath([str(resolved), str(DATA_ROOT.resolve())]) != str(DATA_ROOT.resolve()):
        raise ValueError("Audio path must stay inside the repo data directory")
    return resolved


def _relative_data_path(path: Path) -> str:
    return path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()


def _song_name_from_path(audio_path: str) -> str:
    return Path(audio_path).stem or "Untitled"


class MasteringSession:
    """UI-facing session state."""

    def __init__(self, session_id: str, audio_path: str, song_name: str):
        self.session_id = session_id
        self.audio_path = audio_path
        self.song_name = song_name
        self.created_at = datetime.now().isoformat()

        self.audio = None
        self.sr = None
        self.duration = 0.0
        self.spectrogram_data = None

        self.workflow = "standard"
        self.preset = None
        self.job_id = None
        self.status = "ready"
        self.progress = 0
        self.is_mastering = False
        self.error = None
        self.output_file = None
        self.metrics = _default_metrics()

        self.load_audio()

    def load_audio(self) -> None:
        audio, sr = sf.read(self.audio_path)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        self.audio = audio
        self.sr = sr
        self.duration = len(audio) / sr if sr else 0.0

    def compute_spectrogram(self) -> dict[str, Any] | None:
        if self.audio is None:
            return None

        frequencies, times, spectrum = signal.spectrogram(
            self.audio,
            self.sr,
            nperseg=2048,
            noverlap=1536,
        )
        spectrum_db = 10 * np.log10(spectrum + 1e-10)
        return {
            "frequencies": frequencies.tolist()[:256],
            "times": times.tolist(),
            "magnitude": (spectrum_db[:256, :].T).tolist(),
            "min_db": float(np.min(spectrum_db)),
            "max_db": float(np.max(spectrum_db)),
        }

    def apply_metrics(self, metrics: dict[str, Any] | None) -> None:
        if not metrics:
            return
        self.metrics = {
            "lufs": float(metrics.get("lufs", self.metrics["lufs"])),
            "true_peak": float(metrics.get("true_peak", self.metrics["true_peak"])),
            "crest_db": float(metrics.get("crest_db", self.metrics["crest_db"])),
            "stereo_corr": float(metrics.get("stereo_corr", self.metrics["stereo_corr"])),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "song_name": self.song_name,
            "audio_path": self.audio_path,
            "workflow": self.workflow,
            "preset": self.preset,
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "is_mastering": self.is_mastering,
            "metrics": self.metrics,
            "duration": self.duration,
            "error": self.error,
            "output_file": self.output_file,
        }


def _refresh_session(session: MasteringSession) -> MasteringSession:
    if not session.job_id:
        return session
    if session.status in {"done", "error"} and session.output_file:
        return session

    job_info = run_async(bridge.monitor_job(session.session_id))
    session.status = str(job_info.get("status", session.status))
    session.progress = int(job_info.get("progress", session.progress))
    session.is_mastering = session.status in {"queued", "running"}
    session.error = job_info.get("error")
    session.output_file = job_info.get("output_file")

    if job_info.get("final_metrics"):
        session.apply_metrics(job_info["final_metrics"])
    else:
        session.apply_metrics(job_info.get("analysis_metrics"))

    return session


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mastery")
def mastery():
    return render_template("mastery.html")


@app.route("/api/upload", methods=["POST"])
def upload_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No filename"}), 400

        filename = _sanitize_upload_name(file.filename)
        destination = UPLOAD_DIR / filename
        file.save(destination)

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "audio_path": _relative_data_path(destination),
                "song_name": destination.stem,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/session/new", methods=["POST"])
def create_session():
    try:
        data = request.get_json(silent=True) or {}
        audio_path = data.get("audio_path")
        if not audio_path:
            return jsonify({"error": "audio_path is required"}), 400

        resolved_path = _resolve_session_audio_path(audio_path)
        if not resolved_path.exists():
            return jsonify({"error": f"Audio file not found: {audio_path}"}), 404

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = MasteringSession(
            session_id=session_id,
            audio_path=str(resolved_path),
            song_name=data.get("song_name") or _song_name_from_path(audio_path),
        )

        with session_lock:
            mastering_sessions[session_id] = session

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "audio_path": session.audio_path,
                "song_name": session.song_name,
                "duration": session.duration,
                "metrics": session.metrics,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to create session: {exc}"}), 500


@app.route("/api/session/<session_id>/start", methods=["POST"])
def start_session(session_id: str):
    data = request.get_json(silent=True) or {}
    preset = data.get("preset")
    workflow = data.get("workflow", "standard")
    stem_mode = data.get("stem_mode", "auto")

    if not preset:
        return jsonify({"error": "preset is required"}), 400

    with session_lock:
        session = mastering_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        if session.job_id and session.status in {"queued", "running"}:
            return jsonify({"error": "Session is already mastering"}), 409

    try:
        job_info = run_async(
            bridge.start_mastering_session(
                session_id=session_id,
                audio_path=session.audio_path,
                preset=preset,
                song_name=session.song_name,
                workflow=workflow,
                stem_mode=stem_mode,
            )
        )
    except Exception as exc:
        with session_lock:
            session = mastering_sessions[session_id]
            session.status = "error"
            session.error = str(exc)
            session.is_mastering = False
        return jsonify({"error": str(exc)}), 500

    with session_lock:
        session = mastering_sessions[session_id]
        session.workflow = workflow
        session.preset = preset
        session.job_id = job_info["job_id"]
        session.status = job_info["status"]
        session.progress = 0
        session.is_mastering = True
        session.error = None
        session.apply_metrics(job_info.get("analysis_metrics"))

        return jsonify({"success": True, **session.to_json()})


@app.route("/api/session/<session_id>/spectrogram", methods=["GET"])
def get_spectrogram(session_id: str):
    with session_lock:
        session = mastering_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        if session.spectrogram_data is None:
            session.spectrogram_data = session.compute_spectrogram()
        return jsonify(session.spectrogram_data)


@app.route("/api/session/<session_id>/metrics", methods=["GET"])
def get_metrics(session_id: str):
    with session_lock:
        session = mastering_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        session = _refresh_session(session)
        return jsonify(session.metrics)


@app.route("/api/session/<session_id>/status", methods=["GET"])
def get_status(session_id: str):
    with session_lock:
        session = mastering_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        session = _refresh_session(session)
        return jsonify(session.to_json())


@app.route("/api/session/<session_id>/update", methods=["POST"])
def update_session(session_id: str):
    """
    Compatibility endpoint retained for older UI code.

    The connected UI no longer pushes simulated metrics into the backend, but
    keeping this route avoids breaking any local experiments that still call it.
    """

    data = request.get_json(silent=True) or {}
    with session_lock:
        session = mastering_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        session.is_mastering = bool(data.get("is_mastering", session.is_mastering))
        session.progress = int(data.get("progress", session.progress))
        if "lufs" in data:
            session.apply_metrics(data)
    return jsonify({"status": "updated"})


@app.route("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True, port=5000, host="127.0.0.1")
