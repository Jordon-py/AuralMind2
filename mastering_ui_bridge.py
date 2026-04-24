"""
Bridge between the Flask UI and the local AuralMind2 server module.

The important constraint is session stability: audio registration, analysis,
job launch, polling, and artifact export all have to use the same FastMCP
session id or the server will treat them as unrelated requests.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from server import (
    MasterRequest,
    analyze_audio,
    job_result,
    job_status,
    read_artifact,
    register_audio_from_path,
    run_master_job,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = ROOT_DIR / "Album_Ignorance_is_bliss" / "masters"


class MockContext:
    """Minimal FastMCP-compatible context for direct in-process tool calls."""

    def __init__(self, session_id: str):
        self.session_id = session_id


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _metrics_to_ui(metrics: Any) -> Dict[str, float]:
    if metrics is None:
        return {
            "lufs": -23.0,
            "true_peak": -1.0,
            "crest_db": 0.0,
            "stereo_corr": 0.0,
        }

    return {
        "lufs": float(metrics.integrated_lufs),
        "true_peak": float(metrics.true_peak_dbtp),
        "crest_db": float(metrics.crest_db),
        "stereo_corr": float(metrics.stereo_correlation),
    }


class MasteringUIBridge:
    """Owns real mastering jobs started from the Flask UI."""

    def __init__(self, export_root: Path | str = DEFAULT_EXPORT_DIR):
        self.export_root = Path(export_root)
        self.export_root.mkdir(parents=True, exist_ok=True)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

    def _ctx(self, session_id: str) -> MockContext:
        return MockContext(session_id=session_id)

    async def start_mastering_session(
        self,
        *,
        session_id: str,
        audio_path: str,
        preset: str,
        song_name: str = "Untitled",
        workflow: str = "standard",
        stem_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Register audio, analyze it, and launch a real mastering job."""

        ctx = self._ctx(session_id)
        register_result = await _maybe_await(register_audio_from_path(audio_path, ctx))
        audio_id = register_result.audio_id

        analysis = await _maybe_await(analyze_audio(audio_id, ctx))
        request = MasterRequest(
            audio_id=audio_id,
            preset_name=preset,
            stem_mode=stem_mode,
        )
        launch = await _maybe_await(run_master_job(request, ctx))

        job_info = {
            "session_id": session_id,
            "job_id": launch.job_id,
            "audio_id": audio_id,
            "audio_path": audio_path,
            "song_name": song_name,
            "preset": preset,
            "workflow": workflow,
            "stem_mode": stem_mode,
            "status": launch.status,
            "progress": 0,
            "analysis_metrics": _metrics_to_ui(analysis),
            "final_metrics": None,
            "output_file": None,
            "error": None,
            "started_at": datetime.now().isoformat(),
        }
        self.active_jobs[session_id] = job_info
        return job_info

    async def monitor_job(self, session_id: str) -> Dict[str, Any]:
        """Refresh job status and export the output once it completes."""

        if session_id not in self.active_jobs:
            raise ValueError(f"unknown_session: {session_id}")

        job_info = self.active_jobs[session_id]
        ctx = self._ctx(session_id)
        status_result = await _maybe_await(job_status(job_info["job_id"], ctx))

        job_info["status"] = status_result.status
        job_info["progress"] = int(status_result.progress)
        job_info["error"] = (
            status_result.error.message if getattr(status_result, "error", None) else None
        )

        if job_info["status"] == "done" and not job_info["output_file"]:
            await self.fetch_results(session_id)

        return job_info

    async def fetch_results(self, session_id: str) -> Optional[str]:
        """Export the mastered artifact for a completed job."""

        if session_id not in self.active_jobs:
            return None

        job_info = self.active_jobs[session_id]
        ctx = self._ctx(session_id)
        result = await _maybe_await(job_result(job_info["job_id"], ctx))

        if not result.artifacts:
            return None

        artifact_id = result.artifacts[0].artifact_id
        payload = await self._read_full_artifact(artifact_id, ctx)

        output_path = self._build_output_path(
            song_name=job_info["song_name"],
            preset=job_info["preset"],
            filename=result.artifacts[0].filename,
        )
        output_path.write_bytes(payload)

        metrics = await _maybe_await(analyze_audio(artifact_id, ctx))
        job_info["final_metrics"] = _metrics_to_ui(metrics)
        job_info["output_file"] = str(output_path)
        return str(output_path)

    async def _read_full_artifact(self, artifact_id: str, ctx: MockContext) -> bytes:
        chunks: list[bytes] = []
        offset = 0

        while True:
            chunk = await _maybe_await(read_artifact(artifact_id, offset=offset, ctx=ctx))
            chunks.append(base64.b64decode(chunk.data_b64))
            if chunk.is_last:
                break
            offset += chunk.length

        return b"".join(chunks)

    def _build_output_path(self, *, song_name: str, preset: str, filename: str) -> Path:
        ext = Path(filename).suffix or ".wav"
        safe_song = self._sanitize_filename(song_name or Path(filename).stem)
        safe_preset = self._sanitize_filename(preset)
        return self.export_root / f"{safe_song}_{safe_preset}_Master{ext}"

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        cleaned = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in value.strip()
        )
        return cleaned.strip("_") or "master"


async def export_job_to_album(
    job_id: str,
    preset: str,
    song_name: str,
    *,
    session_id: str = "ui_export",
    export_root: Path | str = DEFAULT_EXPORT_DIR,
) -> Optional[str]:
    """
    Compatibility helper for manual exports.

    This only works when the caller provides the same session id that launched
    the job; otherwise the server will correctly refuse cross-session access.
    """

    ctx = MockContext(session_id=session_id)
    result = await _maybe_await(job_result(job_id, ctx))
    if not result.artifacts:
        return None

    bridge = MasteringUIBridge(export_root=export_root)
    payload = await bridge._read_full_artifact(result.artifacts[0].artifact_id, ctx)
    output_path = bridge._build_output_path(
        song_name=song_name,
        preset=preset,
        filename=result.artifacts[0].filename,
    )
    output_path.write_bytes(payload)
    return str(output_path)
