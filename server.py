"""
AuralMind Maestro — FastMCP Server
===================================

Production-grade MCP server exposing the AuralMind mastering DSP pipeline
as LLM-callable tools.  Designed for non-blocking operation: heavy mastering
jobs run in a background thread-pool, while lightweight analysis and
validation tools respond immediately.

Architecture
------------

    LLM client ──→ FastMCP (stdio / streamable-HTTP)
                     ├── resources/  config://system-prompt
                     ├── prompts/    generate-mastering-strategy
                     └── tools/
                          ├── upload_audio_to_session   (sync,  ~instant)
                          ├── analyze_audio             (sync,  ~2 s)
                          ├── list_presets               (sync,  instant)
                          ├── propose_master_settings    (sync,  instant)
                          ├── run_master_job             (async, returns job_id)
                          ├── job_status                 (sync,  instant)
                          └── job_result                 (sync,  instant)

Mastering Control Loop
----------------------

    1.  upload_audio_to_session(file)   →  server_path
    2.  analyze_audio(server_path)      →  metrics + recommended preset
    3.  LLM reasons over metrics (optionally calls generate-mastering-strategy)
    4.  propose_master_settings(...)    →  validated settings JSON
    5.  run_master_job(server_path, settings)  →  job_id  (non-blocking)
    6.  job_status(job_id)              →  progress / stage description
    7.  job_result(job_id)              →  final metrics + WAV URI + report URI
"""

from __future__ import annotations

import os
import json
import uuid
import time
import logging
import threading
from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

from fastmcp import FastMCP

import tools.auralmind_maestro as maestro

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("auralmind.server")

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "AuralMind Maestro v7.3 Pro-Agent"
)
app = mcp.http_app(
    path='/mcp',
    middleware=[origins:=["*"], methods:=["*"], allow_headers:=["*"]],
    json_response=True,
    transport="streamable-http",
    event_store="memory",
    retry_interval=2
)



# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------
STORAGE_DIR = os.path.abspath("./maestro_sessions")
os.makedirs(STORAGE_DIR, exist_ok=True)

SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "system_prompt.md"
)

# ---------------------------------------------------------------------------
# Upload safety cap (200 MB hex = 100 MB audio)
# ---------------------------------------------------------------------------
MAX_UPLOAD_HEX_CHARS = 200 * 1024 * 1024 * 2  # 200 MB after decode

# ---------------------------------------------------------------------------
# Async Job Infrastructure
# ---------------------------------------------------------------------------
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="maestro-job")
_JOBS_LOCK = threading.Lock()


@dataclass
class JobState:
    """In-memory state for a background mastering job."""

    job_id: str
    status: str = "queued"  # queued → running → done | error
    progress: str = "Waiting in queue"
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_JOBS: Dict[str, JobState] = {}


def _get_job(job_id: str) -> Optional[JobState]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _set_job(job: JobState) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


# ============================================================================
# RESOURCE: System Prompt
# ============================================================================
@mcp.resource("config://system-prompt")
def get_system_prompt() -> str:
    """Returns the AuralMind Cognitive Mastering system prompt.

    This resource provides the full system prompt that instructs the LLM
    on how to act as a mastering engineer.  The LLM should read this once
    at conversation start to understand its role and output format.
    """
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# PROMPT: Mastering Strategy Generator
# ============================================================================
@mcp.prompt("generate-mastering-strategy")
def generate_strategy(lufs: float, crest: float, platform: str) -> str:
    """Generates a mastering strategy prompt seeded with audio metrics.

    The LLM calls this after ``analyze_audio`` to get a prompt that
    combines the system prompt with the track's measured metrics.
    The LLM should then respond with a JSON strategy object as
    specified in the system prompt.

    Args:
        lufs:     Integrated loudness of the input track (LUFS).
        crest:    Crest factor of the input track (dB).
        platform: Target platform (e.g. 'spotify', 'apple_music', 'club').
    """
    prompt_content = get_system_prompt()
    return f"""\
{prompt_content}

INPUT_METRICS:
{{
    "lufs": {lufs},
    "crest_db": {crest},
    "platform": "{platform}"
}}

Respond with the JSON strategy object.
"""


# ============================================================================
# TOOL: upload_audio_to_session
# ============================================================================
@mcp.tool()
def upload_audio_to_session(filename: str, hex_payload: str) -> Dict[str, str]:
    """Upload an audio file to the server before analysis or mastering.

    The LLM **must** call this first to transfer the user's audio file
    to the server.  The returned ``server_path`` is required by
    ``analyze_audio`` and ``run_master_job``.

    Args:
        filename:    Original filename (e.g. ``'my_beat.wav'``).  Used for
                     display only; the server generates a unique safe name.
        hex_payload: Entire audio file encoded as a hex string.  Maximum
                     decoded size: 200 MB.

    Returns:
        ``{"status": "success", "server_path": "<path>"}`` on success, or
        ``{"status": "error", "message": "<reason>"}`` on failure.
    """
    if len(hex_payload) > MAX_UPLOAD_HEX_CHARS:
        return {
            "status": "error",
            "message": "Payload exceeds 200 MB limit after decoding.",
        }

    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{os.path.basename(filename)}"
    full_path = os.path.join(STORAGE_DIR, safe_name)

    try:
        with open(full_path, "wb") as f:
            f.write(bytes.fromhex(hex_payload))
        log.info("Uploaded %s → %s", filename, full_path)
        return {"status": "success", "server_path": full_path}
    except Exception as exc:
        log.exception("Upload failed for %s", filename)
        return {"status": "error", "message": str(exc)}


# ============================================================================
# TOOL: analyze_audio
# ============================================================================
@mcp.tool()
def analyze_audio(server_path: str) -> Dict[str, Any]:
    """Comprehensive pre-mastering analysis — run BEFORE mastering.

    Returns perceptual metrics that the LLM should use to:
    1. Select the correct preset (``recommended_preset``).
    2. Decide ``target_lufs`` (``recommended_lufs``).
    3. Determine if the track needs transient sculpting (high ``crest_db``)
       or harshness control (high ``centroid_hz``).
    4. Check mono compatibility (``sub_mono_ok``).

    Args:
        server_path: Server-side audio path returned by ``upload_audio_to_session``.

    Returns:
        A dictionary with the following keys:

        - ``lufs_i``              — Integrated loudness (BS.1770-4) in LUFS.
        - ``tp_dbfs``             — True-peak level in dBFS (4× oversampled).
        - ``peak_dbfs``           — Sample peak in dBFS.
        - ``rms_dbfs``            — Broadband RMS in dBFS.
        - ``crest_db``            — Crest factor (peak − RMS) in dB.
        - ``corr_broadband``      — L/R correlation 2–12 kHz (≥ 0.85 = safe).
        - ``corr_low``            — L/R correlation 20–200 Hz (≥ 0.95 = mono-safe).
        - ``sub_mono_ok``         — True if low-band correlation ≥ 0.95.
        - ``centroid_hz``         — Spectral centroid in Hz (brightness proxy).
        - ``recommended_preset``  — Auto-selected preset name.
        - ``recommended_lufs``    — Suggested target LUFS for the preset.
        - ``sample_rate``         — Native sample rate of the input file.
        - ``duration_s``          — Duration in seconds.
    """
    if not os.path.exists(server_path):
        return {"error": f"File not found: {server_path}"}

    try:
        y, sr = maestro.load_audio(server_path)
        features = maestro.analyze_track_features(y, sr)

        corr_lo = float(features.get("corr_lo", 0.0))
        preset_name = maestro.auto_select_preset_name(features)
        presets = maestro.get_presets()
        recommended_lufs = (
            float(presets[preset_name].target_lufs)
            if preset_name in presets
            else -12.0
        )

        return {
            "lufs_i": features["lufs"],
            "tp_dbfs": features["tp_dbfs"],
            "peak_dbfs": features["peak_dbfs"],
            "rms_dbfs": features["rms_dbfs"],
            "crest_db": features["crest_db"],
            "corr_broadband": features["corr_hi"],
            "corr_low": corr_lo,
            "sub_mono_ok": corr_lo >= 0.95,
            "centroid_hz": features["centroid_hz"],
            "recommended_preset": preset_name,
            "recommended_lufs": recommended_lufs,
            "sample_rate": int(sr),
            "duration_s": round(len(y) / sr, 2),
        }
    except Exception as exc:
        log.exception("Analysis failed for %s", server_path)
        return {"error": f"Analysis failed: {exc}"}


# ============================================================================
# TOOL: list_presets
# ============================================================================
@mcp.tool()
def list_presets() -> Dict[str, Any]:
    """List all available mastering presets with key parameters.

    Use this to discover valid ``preset_name`` values and their defaults
    before calling ``propose_master_settings`` or ``run_master_job``.

    Returns:
        A dictionary mapping preset names to their key parameters::

            {
                "hi_fi_streaming": {
                    "target_lufs": -12.8,
                    "limiter_mode": "v2",
                    "governor_gr_limit_db": -3.0,
                    "description": "Transparent hi-fi for streaming platforms."
                },
                ...
            }
    """
    presets = maestro.get_presets()
    out: Dict[str, Any] = {}
    for name, p in presets.items():
        out[name] = {
            "target_lufs": p.target_lufs,
            "ceiling_dbfs": p.ceiling_dbfs,
            "limiter_mode": p.limiter_mode,
            "governor_gr_limit_db": p.governor_gr_limit_db,
            "match_strength": p.match_strength,
            "enable_harshness_limiter": p.enable_harshness_limiter,
        }
    return out


# ============================================================================
# TOOL: propose_master_settings
# ============================================================================
@mcp.tool()
def propose_master_settings(
    preset_name: str = "hi_fi_streaming",
    target_lufs: float = -12.0,
    warmth: float = 0.5,
    transient_boost_db: float = 1.0,
    enable_harshness_limiter: bool = True,
    enable_air_motion: bool = True,
    bit_depth: str = "float32",
) -> Dict[str, Any]:
    """Validate and preview mastering settings before submitting a job.

    The LLM should call this after ``analyze_audio`` to confirm the
    parameter set is valid.  The returned ``settings`` object can be
    passed directly to ``run_master_job``.

    Args:
        preset_name:             One of the available preset names (see ``list_presets``).
        target_lufs:             Target integrated loudness in LUFS
                                 (e.g. ``-14.0`` for Spotify, ``-11.0`` for competitive).
        warmth:                  Analog warmth tilt amount (``0.0`` to ``1.0``).
        transient_boost_db:      Pre-limiter transient sculpt boost in dB (``0.0`` to ``4.0``).
        enable_harshness_limiter: Enable the 6–10 kHz dynamic harshness limiter.
        enable_air_motion:       Enable correlation-guarded air-band 3D depth.
        bit_depth:               Export precision: ``'float32'`` or ``'float64'``.

    Returns:
        ``{"status": "validated", "settings": {...}}`` on success, or
        ``{"error": "<reason>"}`` on validation failure.
    """
    presets = maestro.get_presets()
    if preset_name not in presets:
        return {"error": f"Invalid preset. Available: {list(presets.keys())}"}

    warmth = max(0.0, min(1.0, float(warmth)))
    transient_boost_db = max(0.0, min(4.0, float(transient_boost_db)))
    if bit_depth not in ("float32", "float64"):
        bit_depth = "float32"

    return {
        "status": "validated",
        "settings": {
            "preset_name": preset_name,
            "target_lufs": float(target_lufs),
            "warmth": warmth,
            "transient_boost_db": transient_boost_db,
            "enable_harshness_limiter": enable_harshness_limiter,
            "enable_air_motion": enable_air_motion,
            "bit_depth": bit_depth,
            "subtype": "FLOAT" if bit_depth == "float32" else "DOUBLE",
        },
    }


# ============================================================================
# TOOL: run_master_job  (non-blocking — Enhancement B)
# ============================================================================
def _run_master_worker(job: JobState, target_path: str, settings: Dict[str, Any]) -> None:
    """Background worker — runs inside ThreadPoolExecutor."""
    try:
        job.status = "running"
        job.progress = "Loading preset and applying AI overrides"
        _set_job(job)

        presets = maestro.get_presets()
        preset_name = settings.get("preset_name", "hi_fi_streaming")
        preset = presets[preset_name]

        # AI overrides
        updates: Dict[str, Any] = {}
        if "target_lufs" in settings:
            updates["target_lufs"] = float(settings["target_lufs"])
        if "warmth" in settings:
            updates["warmth"] = max(0.0, min(1.0, float(settings["warmth"])))
        if "transient_boost_db" in settings:
            updates["transient_sculpt_boost_db"] = max(
                0.0, min(4.0, float(settings["transient_boost_db"]))
            )
        updates["enable_harshness_limiter"] = settings.get(
            "enable_harshness_limiter", True
        )
        updates["enable_air_motion"] = settings.get("enable_air_motion", True)
        updates["bit_depth"] = settings.get("bit_depth", "float32")

        preset = replace(preset, **updates)

        # Export paths
        bit_depth = settings.get("bit_depth", "float32")
        out_filename = f"mastered_{job.job_id}_{os.path.basename(target_path)}"
        out_path = os.path.join(STORAGE_DIR, out_filename)
        report_path = os.path.join(STORAGE_DIR, f"report_{job.job_id}.md")
        subtype = "FLOAT" if bit_depth == "float32" else "DOUBLE"

        job.progress = "Running DSP pipeline (governor + limiter + export)"
        _set_job(job)

        result = maestro.master(
            target_path=target_path,
            out_path=out_path,
            preset=preset,
            reference_path=None,
            report_path=report_path,
            out_subtype=subtype,
            dither=False,
        )

        result["file_uri"] = out_path
        result["report_uri"] = report_path
        result["precision"] = f"48000Hz / {bit_depth}"

        job.status = "done"
        job.progress = "Complete"
        job.result = result
        _set_job(job)
        log.info("Job %s completed successfully", job.job_id)

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.progress = f"Failed: {exc}"
        _set_job(job)
        log.exception("Job %s failed", job.job_id)


@mcp.tool()
def run_master_job(
    target_path: str,
    preset_name: str = "hi_fi_streaming",
    target_lufs: float = -12.0,
    warmth: float = 0.5,
    transient_boost_db: float = 1.0,
    enable_harshness_limiter: bool = True,
    enable_air_motion: bool = True,
    bit_depth: str = "float32",
) -> Dict[str, str]:
    """Submit a mastering job to the background thread pool (non-blocking).

    Returns a ``job_id`` immediately.  The LLM should then poll
    ``job_status(job_id)`` until the status is ``"done"`` or ``"error"``,
    then call ``job_result(job_id)`` to retrieve the output.

    This tool does NOT block the MCP request thread.  Mastering typically
    takes 30–120 seconds depending on track length and governor iterations.

    Args:
        target_path:             Server-side audio path from ``upload_audio_to_session``.
        preset_name:             Mastering preset name (default: ``'hi_fi_streaming'``).
        target_lufs:             Override target integrated loudness in LUFS.
        warmth:                  Analog warmth tilt (``0.0`` to ``1.0``).
        transient_boost_db:      Pre-limiter transient sculpt boost (``0.0`` to ``4.0`` dB).
        enable_harshness_limiter: Enable the 6–10 kHz dynamic harshness limiter.
        enable_air_motion:       Enable correlation-guarded air-band 3D depth.
        bit_depth:               Export precision: ``'float32'`` or ``'float64'``.

    Returns:
        ``{"status": "queued", "job_id": "<uuid>"}`` on success, or
        ``{"error": "<reason>"}`` on validation failure.
    """
    if not os.path.exists(target_path):
        return {"error": "target_path invalid. Use upload_audio_to_session first."}

    presets = maestro.get_presets()
    if preset_name not in presets:
        return {"error": f"Invalid preset. Available: {list(presets.keys())}"}

    job_id = str(uuid.uuid4())[:12]
    job = JobState(job_id=job_id)
    _set_job(job)

    settings = {
        "preset_name": preset_name,
        "target_lufs": float(target_lufs),
        "warmth": max(0.0, min(1.0, float(warmth))),
        "transient_boost_db": max(0.0, min(4.0, float(transient_boost_db))),
        "enable_harshness_limiter": bool(enable_harshness_limiter),
        "enable_air_motion": bool(enable_air_motion),
        "bit_depth": str(bit_depth),
    }

    _EXECUTOR.submit(_run_master_worker, job, target_path, settings)
    log.info("Submitted job %s for %s", job_id, target_path)

    return {"status": "queued", "job_id": job_id}


# ============================================================================
# TOOL: job_status
# ============================================================================
@mcp.tool()
def job_status(job_id: str) -> Dict[str, Any]:
    """Check the current status of a mastering job.

    Poll this after ``run_master_job`` until ``status`` is ``"done"``
    or ``"error"``.  When ``"done"``, call ``job_result(job_id)`` to
    retrieve the output files and metrics.

    Args:
        job_id: The job identifier returned by ``run_master_job``.

    Returns:
        ``{"job_id": "...", "status": "running", "progress": "...", "elapsed_s": 42.3}``
    """
    job = _get_job(job_id)
    if job is None:
        return {"error": f"Unknown job_id: {job_id}"}

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "elapsed_s": round(time.time() - job.created_at, 1),
    }


# ============================================================================
# TOOL: job_result
# ============================================================================
@mcp.tool()
def job_result(job_id: str) -> Dict[str, Any]:
    """Retrieve the final result of a completed mastering job.

    Only call this after ``job_status`` reports ``status == "done"``.
    If the job is still running, this returns an appropriate message.

    Args:
        job_id: The job identifier returned by ``run_master_job``.

    Returns:
        On success — the full mastering result dictionary including:

        - ``file_uri``    — Path to the mastered WAV file.
        - ``report_uri``  — Path to the mastering report (Markdown).
        - ``precision``   — Output format (e.g. ``"48000Hz / float32"``).
        - ``lufs``        — Final integrated loudness.
        - ``tp_dbfs``     — Final true-peak level.
        - Plus all other metrics from the mastering pipeline.

        On error — ``{"error": "<reason>"}``.
    """
    job = _get_job(job_id)
    if job is None:
        return {"error": f"Unknown job_id: {job_id}"}

    if job.status == "error":
        return {"error": job.error or "Unknown error", "job_id": job_id}

    if job.status != "done":
        return {
            "job_id": job_id,
            "status": job.status,
            "message": "Job is not yet complete. Call job_status to check progress.",
        }

    return job.result or {"error": "No result available"}


# ============================================================================
# Entrypoint
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    port = int(os.environ.get("PORT", "8000"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
