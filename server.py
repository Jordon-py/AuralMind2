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
                          ├── job_result                 (sync,  instant)
                          └── read_artifact              (sync,  chunked download)

Mastering Control Loop
----------------------

    1.  upload_audio_to_session(file)   →  audio_id
    2.  analyze_audio(audio_id)         →  metrics + recommended preset
    3.  LLM reasons over metrics (optionally calls generate-mastering-strategy)
    4.  propose_master_settings(...)    →  validated settings JSON
    5.  run_master_job(audio_id, settings)  →  job_id  (non-blocking)
    6.  job_status(job_id)              →  progress / stage description
    7.  job_result(job_id)              →  final metrics + artifact handles
    8.  read_artifact(artifact_id)      →  base64 chunks (audio/report)
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import json
import re
import uuid
import time
import logging
import threading
from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Literal, Tuple, List
import auralmind_match_maestro_v7_3
from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import BaseModel, Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


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

_http_middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "mcp-protocol-version",
            "mcp-session-id",
            "Authorization",
            "Content-Type",
        ],
        expose_headers=["mcp-session-id"],
    )
]

app = mcp.http_app(
    path='/mcp',
    middleware=_http_middleware,
    json_response=True,
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
# Upload safety cap (hex string length cap; ~200 MB decoded audio)
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 400 * 1024 * 1024  # 400 MB after decode
MAX_UPLOAD_HEX_CHARS = MAX_UPLOAD_BYTES * 2
MAX_READ_BYTES = 2 * (1024 * 1024)  # 2 MB chunks for artifact reads

HANDLE_RE = re.compile(r"^(aud|art|job)_[a-f0-9]{12}$")

_ARTIFACTS_LOCK = threading.Lock()
_ARTIFACTS: Dict[str, Dict[str, "ArtifactEntry"]] = {}

_MAESTRO_LOCK = threading.Lock()
_MAESTRO: Optional[Any] = None
_MAESTRO_ERROR: Optional[Dict[str, Any]] = None


@dataclass
class ArtifactEntry:
    artifact_id: str
    kind: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    data_filename: str
    created_at: float = field(default_factory=time.time)


class ErrorInfo(BaseModel):
    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured error details."
    )


class ArtifactRef(BaseModel):
    artifact_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str


class UploadResult(BaseModel):
    audio_id: str
    filename: str
    size_bytes: int
    sha256: str
    media_type: str


class UploadResponse(BaseModel):
    ok: bool
    result: Optional[UploadResult] = None
    error: Optional[ErrorInfo] = None


class AnalyzeResult(BaseModel):
    audio_id: str
    metrics: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    ok: bool
    result: Optional[AnalyzeResult] = None
    error: Optional[ErrorInfo] = None


class PresetsResult(BaseModel):
    presets: Dict[str, Any]


class PresetsResponse(BaseModel):
    ok: bool
    result: Optional[PresetsResult] = None
    error: Optional[ErrorInfo] = None


class SettingsResult(BaseModel):
    settings: Dict[str, Any]


class SettingsResponse(BaseModel):
    ok: bool
    result: Optional[SettingsResult] = None
    error: Optional[ErrorInfo] = None


class JobSubmitResult(BaseModel):
    job_id: str
    status: str
    audio_id: str


class JobSubmitResponse(BaseModel):
    ok: bool
    result: Optional[JobSubmitResult] = None
    error: Optional[ErrorInfo] = None


class JobStatusResult(BaseModel):
    job_id: str
    status: str
    progress: str
    elapsed_s: float


class JobStatusResponse(BaseModel):
    ok: bool
    result: Optional[JobStatusResult] = None
    error: Optional[ErrorInfo] = None


class JobResultResult(BaseModel):
    job_id: str
    status: str
    artifacts: List[ArtifactRef]
    metrics: Dict[str, Any]
    precision: str


class JobResultResponse(BaseModel):
    ok: bool
    result: Optional[JobResultResult] = None
    error: Optional[ErrorInfo] = None


class ArtifactReadResult(BaseModel):
    artifact_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    offset: int
    length: int
    is_last: bool
    data_b64: str


class ArtifactReadResponse(BaseModel):
    ok: bool
    result: Optional[ArtifactReadResult] = None
    error: Optional[ErrorInfo] = None

@mcp.tool()
def _ok(result: BaseModel | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(result, BaseModel):
        result = result.model_dump()
    return {"ok": True, "result": result, "error": None}

@mcp.tool()
def _err(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "result": None,
        "error": {"code": code, "message": message, "details": details},
    }

@mcp.tool()
def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

@mcp.tool()
def _valid_handle(handle: str, prefix: Optional[str] = None) -> bool:
    if not isinstance(handle, str) or not HANDLE_RE.match(handle):
        return False
    if prefix is None:
        return True
    return handle.startswith(f"{prefix}_")

@mcp.tool()
def _sanitize_filename(name: str, fallback: str = "audio") -> str:
    base = os.path.basename(name or fallback)
    cleaned = "".join(
        ch if 32 <= ord(ch) < 127 and ch not in "\\/:*?\"<>|" else "_"
        for ch in base
    ).strip()
    return cleaned or fallback

@mcp.tool()
def _guess_media_type(filename: str, fallback: str = "application/octet-stream") -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".ogg":
        return "audio/ogg"
    if ext in (".aif", ".aiff"):
        return "audio/aiff"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".md":
        return "text/markdown"
    return fallback

@mcp.tool()
def _get_session_info(ctx: Optional[Context]) -> Tuple[str, str]:
    if ctx is not None and ctx.session_id:
        sid = str(ctx.session_id)
        key = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16]
        session_key = f"s_{key}"
    else:
        session_key = "s_anon"
    session_dir = os.path.join(STORAGE_DIR, session_key)
    os.makedirs(session_dir, exist_ok=True)
    return session_key, session_dir

@mcp.tool()
def _artifact_meta_path(session_dir: str, artifact_id: str) -> str:
    return os.path.join(session_dir, f"{artifact_id}.json")

@mcp.tool()
def _artifact_data_path(session_dir: str, data_filename: str) -> str:
    return os.path.join(session_dir, data_filename)

@mcp.tool()
def _register_artifact(session_key: str, entry: ArtifactEntry, session_dir: str) -> None:
    with _ARTIFACTS_LOCK:
        _ARTIFACTS.setdefault(session_key, {})[entry.artifact_id] = entry
    meta_path = _artifact_meta_path(session_dir, entry.artifact_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_id": entry.artifact_id,
                "kind": entry.kind,
                "filename": entry.filename,
                "media_type": entry.media_type,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
                "data_filename": entry.data_filename,
                "created_at": entry.created_at,
            },
            f,
            indent=2,
        )

@mcp.tool()
def _load_artifact(session_key: str, session_dir: str, artifact_id: str) -> Optional[ArtifactEntry]:
    with _ARTIFACTS_LOCK:
        cached = _ARTIFACTS.get(session_key, {}).get(artifact_id)
    if cached is not None:
        return cached

    meta_path = _artifact_meta_path(session_dir, artifact_id)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = ArtifactEntry(
        artifact_id=data["artifact_id"],
        kind=data["kind"],
        filename=data["filename"],
        media_type=data["media_type"],
        size_bytes=int(data["size_bytes"]),
        sha256=data["sha256"],
        data_filename=data["data_filename"],
        created_at=float(data.get("created_at", time.time())),
    )
    _register_artifact(session_key, entry, session_dir)
    return entry

@mcp.tool()
def _store_bytes(
    session_key: str,
    session_dir: str,
    *,
    artifact_id: str,
    kind: str,
    filename: str,
    payload: bytes,
    media_type: str,
) -> ArtifactEntry:
    safe_name = _sanitize_filename(filename)
    ext = os.path.splitext(safe_name)[1].lower() or ".bin"
    data_filename = f"{artifact_id}{ext}"
    data_path = _artifact_data_path(session_dir, data_filename)
    with open(data_path, "wb") as f:
        f.write(payload)
    size_bytes = len(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    entry = ArtifactEntry(
        artifact_id=artifact_id,
        kind=kind,
        filename=safe_name,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=sha256,
        data_filename=data_filename,
    )
    _register_artifact(session_key, entry, session_dir)
    return entry

@mcp.tool()
def _register_existing_file(
    session_key: str,
    session_dir: str,
    *,
    artifact_id: str,
    kind: str,
    filename: str,
    data_filename: str,
    media_type: str,
) -> ArtifactEntry:
    data_path = _artifact_data_path(session_dir, data_filename)
    sha = hashlib.sha256()
    size = 0
    with open(data_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            sha.update(chunk)
    entry = ArtifactEntry(
        artifact_id=artifact_id,
        kind=kind,
        filename=_sanitize_filename(filename),
        media_type=media_type,
        size_bytes=size,
        sha256=sha.hexdigest(),
        data_filename=data_filename,
    )
    _register_artifact(session_key, entry, session_dir)
    return entry

@mcp.tool()
def analyze_track_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Lightweight analysis for auto-tuning and reporting.
    USE BEFORE MASTER!
    """
    return auralmind_match_maestro_v7.analyze_track_features(y, sr)

@mcp.tool()
def _get_maestro() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    global _MAESTRO
    global _MAESTRO_ERROR
    with _MAESTRO_LOCK:
        if _MAESTRO is not None:
            return _MAESTRO, None
        if _MAESTRO_ERROR is not None:
            return None, _MAESTRO_ERROR
        try:
            import tools.auralmind_maestro as maestro
        except Exception as exc:
            log.exception("Failed to import DSP engine")
            _MAESTRO_ERROR = {
                "code": "engine_unavailable",
                "message": "DSP engine unavailable. Check server dependencies.",
                "details": {"error": str(exc)},
            }
            return None, _MAESTRO_ERROR
        _MAESTRO = maestro
        return _MAESTRO, None

# ---------------------------------------------------------------------------
# Async Job Infrastructure
# ---------------------------------------------------------------------------
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="maestro-job")
_JOBS_LOCK = threading.Lock()


@dataclass
class JobState:
    """In-memory state for a background mastering job."""

    job_id: str
    session_key: str
    audio_id: str
    status: str = "queued"  # queued → running → done | error
    progress: str = "Waiting in queue"
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    artifacts: List[ArtifactEntry] = field(default_factory=list)
    precision: str = "48000Hz / float32"
    error: Optional[str] = None


_JOBS: Dict[str, JobState] = {}

@mcp.tool()
def _get_job(job_id: str, session_key: str) -> Optional[JobState]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return None
    if job.session_key != session_key:
        return None
    return job

@mcp.tool()
def _set_job(job: JobState) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


# ============================================================================
# RESOURCE: System Prompt
# ============================================================================
@mcp.tool()@mcp.resource("config://system-prompt")
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
@mcp.tool()@mcp.prompt("generate-mastering-strategy")
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
@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=UploadResponse.model_json_schema())
def upload_audio_to_session(
    filename: str,
    hex_payload: Optional[str] = None,
    payload_b64: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Upload an audio file to the server before analysis or mastering.

    The LLM **must** call this first to transfer the user's audio file
    to the server. The returned ``audio_id`` is required by
    ``analyze_audio`` and ``run_master_job``.

    Args:
        filename:    Original filename (e.g. ``'my_beat.wav'``). Used for
                     display only; the server generates a unique safe name.
        hex_payload: Entire audio file encoded as a hex string (legacy).
        payload_b64: Entire audio file encoded as base64 (preferred).

    Returns:
        Envelope with ``audio_id`` on success, or a machine-readable error.
    """
    if hex_payload and payload_b64:
        return _err(
            "ambiguous_payload",
            "Provide only one payload encoding.",
            {"allowed": ["hex_payload", "payload_b64"]},
        )
    if not hex_payload and not payload_b64:
        return _err(
            "missing_payload",
            "Provide hex_payload or payload_b64.",
            {"allowed": ["hex_payload", "payload_b64"]},
        )

    try:
        if hex_payload:
            if len(hex_payload) > MAX_UPLOAD_HEX_CHARS:
                return _err(
                    "payload_too_large",
                    "Payload exceeds 200 MB limit after decoding.",
                    {"limit_bytes": MAX_UPLOAD_BYTES},
                )
            payload = bytes.fromhex(hex_payload)
        else:
            payload = base64.b64decode(payload_b64 or "", validate=True)
    except (ValueError, binascii.Error) as exc:
        return _err("invalid_payload", "Payload decoding failed.", {"error": str(exc)})

    if len(payload) > MAX_UPLOAD_BYTES:
        return _err(
            "payload_too_large",
            "Payload exceeds 200 MB limit after decoding.",
            {"limit_bytes": MAX_UPLOAD_BYTES},
        )

    session_key, session_dir = _get_session_info(ctx)
    audio_id = _new_id("aud")
    safe_name = _sanitize_filename(filename)
    media_type = _guess_media_type(safe_name)

    try:
        entry = _store_bytes(
            session_key,
            session_dir,
            artifact_id=audio_id,
            kind="audio",
            filename=safe_name,
            payload=payload,
            media_type=media_type,
        )
    except Exception as exc:
        log.exception("Upload failed for %s", filename)
        return _err("upload_failed", "Upload failed.", {"error": str(exc)})

    log.info("Uploaded %s -> %s (%s bytes)", filename, audio_id, entry.size_bytes)
    return _ok(
        UploadResult(
            audio_id=entry.artifact_id,
            filename=entry.filename,
            size_bytes=entry.size_bytes,
            sha256=entry.sha256,
            media_type=entry.media_type,
        )
    )


# ============================================================================
# TOOL: analyze_audio
# ============================================================================
@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=AnalyzeResponse.model_json_schema())
def analyze_audio(audio_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Comprehensive pre-mastering analysis — run BEFORE mastering.

    Args:
        audio_id: Server-side audio handle returned by ``upload_audio_to_session``.

    Returns:
        Envelope with analysis metrics or a machine-readable error.
    """
    if not _valid_handle(audio_id, "aud"):
        return _err("invalid_audio_id", "audio_id format is invalid.", {"audio_id": audio_id})

    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind != "audio":
        return _err("not_found", "Audio not found.", {"audio_id": audio_id})

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    if not os.path.exists(data_path):
        return _err("not_found", "Audio file missing.", {"audio_id": audio_id})

    maestro, err = _get_maestro()
    if err is not None:
        return _err(err["code"], err["message"], err.get("details"))

    try:
        y, sr = maestro.load_audio(data_path)
        features = maestro.analyze_track_features(y, sr)

        corr_lo = float(features.get("corr_lo", 0.0))
        preset_name = maestro.auto_select_preset_name(features)
        presets = maestro.get_presets()
        recommended_lufs = (
            float(presets[preset_name].target_lufs)
            if preset_name in presets
            else -12.0
        )

        metrics = {
            "lufs_i": features["lufs"],
            "tp_dbfs": features["tp_dbfs"],
            "peak_dbfs": features["peak_dbfs"],
            "rms_dbfs": features["rms_dbfs"],
            "crest_db": features["crest_db"],
            "corr_broadband": float(features.get("corr_hi", 0.0)),
            "corr_low": corr_lo,
            "sub_mono_ok": corr_lo >= 0.95,
            "centroid_hz": features["centroid_hz"],
            "recommended_preset": preset_name,
            "recommended_lufs": recommended_lufs,
            "sample_rate": int(sr),
            "duration_s": round(len(y) / sr, 2),
        }

        return _ok(AnalyzeResult(audio_id=audio_id, metrics=metrics))
    except Exception as exc:
        log.exception("Analysis failed for %s", audio_id)
        return _err("analysis_failed", "Analysis failed.", {"error": str(exc)})


# ============================================================================
# TOOL: list_presets
# ============================================================================
@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=PresetsResponse.model_json_schema())
def list_presets(ctx: Optional[Context] = None) -> Dict[str, Any]:
    """List all available mastering presets with key parameters."""
    maestro, err = _get_maestro()
    if err is not None:
        return _err(err["code"], err["message"], err.get("details"))

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
    return _ok(PresetsResult(presets=out))

@mcp.tool()
def _build_settings(
    maestro: Any,
    preset_name: str,
    target_lufs: float,
    warmth: float,
    transient_boost_db: float,
    enable_harshness_limiter: bool,
    enable_air_motion: bool,
    bit_depth: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    presets = maestro.get_presets()
    if preset_name not in presets:
        return None, _err(
            "invalid_preset",
            "Invalid preset.",
            {"available": list(presets.keys())},
        )

    warmth_val = max(0.0, min(1.0, float(warmth)))
    transient_val = max(0.0, min(4.0, float(transient_boost_db)))
    bit_depth_val = str(bit_depth)
    if bit_depth_val not in ("float32", "float64"):
        bit_depth_val = "float32"

    settings = {
        "preset_name": preset_name,
        "target_lufs": float(target_lufs),
        "warmth": warmth_val,
        "transient_boost_db": transient_val,
        "enable_harshness_limiter": bool(enable_harshness_limiter),
        "enable_air_motion": bool(enable_air_motion),
        "bit_depth": bit_depth_val,
        "subtype": "FLOAT" if bit_depth_val == "float32" else "DOUBLE",
    }
    return settings, None


# ============================================================================
# TOOL: propose_master_settings
# ============================================================================
@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=SettingsResponse.model_json_schema())
def propose_master_settings(
    preset_name: str = "hi_fi_streaming",
    target_lufs: float = -12.0,
    warmth: float = 0.5,
    transient_boost_db: float = 1.0,
    enable_harshness_limiter: bool = True,
    enable_air_motion: bool = True,
    bit_depth: str = "float32",
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Validate and preview mastering settings before submitting a job."""
    maestro, err = _get_maestro()
    if err is not None:
        return _err(err["code"], err["message"], err.get("details"))

    settings, err_payload = _build_settings(
        maestro,
        preset_name,
        target_lufs,
        warmth,
        transient_boost_db,
        enable_harshness_limiter,
        enable_air_motion,
        bit_depth,
    )
    if err_payload is not None:
        return err_payload
    return _ok(SettingsResult(settings=settings))


# ============================================================================
# TOOL: run_master_job  (non-blocking — Enhancement B)
@mcp.tool()# ============================================================================
def _run_master_worker(
    job: JobState,
    session_dir: str,
    audio_entry: ArtifactEntry,
    settings: Dict[str, Any],
) -> None:
    """Background worker — runs inside ThreadPoolExecutor."""
    maestro, err = _get_maestro()
    if err is not None:
        job.status = "error"
        job.error = err["message"]
        job.progress = f"Failed: {err['message']}"
        _set_job(job)
        return

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
        subtype = "FLOAT" if bit_depth == "float32" else "DOUBLE"
        audio_artifact_id = _new_id("art")
        report_artifact_id = _new_id("art")
        out_filename = f"{audio_artifact_id}.wav"
        report_filename = f"{report_artifact_id}.md"
        out_path = os.path.join(session_dir, out_filename)
        report_path = os.path.join(session_dir, report_filename)

        job.progress = "Running DSP pipeline (governor + limiter + export)"
        _set_job(job)

        data_path = _artifact_data_path(session_dir, audio_entry.data_filename)
        result = maestro.master(
            target_path=data_path,
            out_path=out_path,
            preset=preset,
            reference_path=None,
            report_path=report_path,
            out_subtype=subtype,
            dither=False,
        )

        result.pop("out_path", None)
        precision = f"48000Hz / {bit_depth}"
        job.precision = precision

        audio_ref = _register_existing_file(
            job.session_key,
            session_dir,
            artifact_id=audio_artifact_id,
            kind="mastered_audio",
            filename=f"mastered_{audio_entry.filename}",
            data_filename=out_filename,
            media_type="audio/wav",
        )
        report_ref = _register_existing_file(
            job.session_key,
            session_dir,
            artifact_id=report_artifact_id,
            kind="report",
            filename=f"report_{job.job_id}.md",
            data_filename=report_filename,
            media_type="text/markdown",
        )

        job.status = "done"
        job.progress = "Complete"
        job.result = result
        job.artifacts = [audio_ref, report_ref]
        _set_job(job)
        log.info("Job %s completed successfully", job.job_id)

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.progress = f"Failed: {exc}"
        _set_job(job)
        log.exception("Job %s failed", job.job_id)


@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=JobSubmitResponse.model_json_schema())
def run_master_job(
    audio_id: str,
    preset_name: str = "hi_fi_streaming",
    target_lufs: float = -12.0,
    warmth: float = 0.5,
    transient_boost_db: float = 1.0,
    enable_harshness_limiter: bool = True,
    enable_air_motion: bool = True,
    bit_depth: str = "float32",
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Submit a mastering job to the background thread pool (non-blocking)."""
    if not _valid_handle(audio_id, "aud"):
        return _err("invalid_audio_id", "audio_id format is invalid.", {"audio_id": audio_id})

    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind != "audio":
        return _err("not_found", "Audio not found.", {"audio_id": audio_id})

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    if not os.path.exists(data_path):
        return _err("not_found", "Audio file missing.", {"audio_id": audio_id})

    maestro, err = _get_maestro()
    if err is not None:
        return _err(err["code"], err["message"], err.get("details"))

    settings, err_payload = _build_settings(
        maestro,
        preset_name,
        target_lufs,
        warmth,
        transient_boost_db,
        enable_harshness_limiter,
        enable_air_motion,
        bit_depth,
    )
    if err_payload is not None:
        return err_payload

    job_id = _new_id("job")
    job = JobState(job_id=job_id, session_key=session_key, audio_id=audio_id)
    _set_job(job)

    _EXECUTOR.submit(_run_master_worker, job, session_dir, entry, settings)
    log.info("Submitted job %s for %s", job_id, audio_id)

    return _ok(JobSubmitResult(job_id=job_id, status="queued", audio_id=audio_id))


# ============================================================================
# TOOL: job_status
# ============================================================================
@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=JobStatusResponse.model_json_schema())
def job_status(job_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Check the current status of a mastering job."""
    if not _valid_handle(job_id, "job"):
        return _err("invalid_job_id", "job_id format is invalid.", {"job_id": job_id})

    session_key, _ = _get_session_info(ctx)
    job = _get_job(job_id, session_key)
    if job is None:
        return _err("not_found", "Unknown job_id.", {"job_id": job_id})

    return _ok(
        JobStatusResult(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            elapsed_s=round(time.time() - job.created_at, 1),
        )
    )


# ============================================================================
# TOOL: job_result
# ============================================================================
@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=JobResultResponse.model_json_schema())
def job_result(job_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Retrieve the final result of a completed mastering job."""
    if not _valid_handle(job_id, "job"):
        return _err("invalid_job_id", "job_id format is invalid.", {"job_id": job_id})

    session_key, _ = _get_session_info(ctx)
    job = _get_job(job_id, session_key)
    if job is None:
        return _err("not_found", "Unknown job_id.", {"job_id": job_id})

    if job.status == "error":
        return _err(
            "job_failed",
            "Job failed.",
            {"job_id": job_id, "error": job.error or "Unknown error"},
        )

    if job.status != "done":
        return _err(
            "job_not_complete",
            "Job is not yet complete. Call job_status to check progress.",
            {"job_id": job_id, "status": job.status},
        )

    artifacts = [
        ArtifactRef(
            artifact_id=a.artifact_id,
            filename=a.filename,
            media_type=a.media_type,
            size_bytes=a.size_bytes,
            sha256=a.sha256,
        )
        for a in job.artifacts
    ]
    metrics = job.result or {}

    return _ok(
        JobResultResult(
            job_id=job.job_id,
            status=job.status,
            artifacts=artifacts,
            metrics=metrics,
            precision=job.precision,
        )
    )


# ============================================================================
# TOOL: read_artifact
# ============================================================================
@mcp.tool()@mcp.tool(exclude_args=["ctx"], output_schema=ArtifactReadResponse.model_json_schema())
def read_artifact(
    artifact_id: str,
    offset: int = 0,
    length: int = MAX_READ_BYTES,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Read artifact bytes as base64 in bounded chunks."""
    if not _valid_handle(artifact_id):
        return _err(
            "invalid_artifact_id",
            "artifact_id format is invalid.",
            {"artifact_id": artifact_id},
        )
    if offset < 0 or length < 0:
        return _err(
            "invalid_range",
            "offset and length must be non-negative.",
            {"offset": offset, "length": length},
        )
    if length > MAX_READ_BYTES:
        return _err(
            "chunk_too_large",
            "Requested length exceeds server chunk limit.",
            {"max_bytes": MAX_READ_BYTES},
        )

    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, artifact_id)
    if entry is None:
        return _err("not_found", "Artifact not found.", {"artifact_id": artifact_id})

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    if not os.path.exists(data_path):
        return _err("not_found", "Artifact file missing.", {"artifact_id": artifact_id})

    if offset > entry.size_bytes:
        return _err(
            "offset_out_of_range",
            "Offset exceeds artifact size.",
            {"offset": offset, "size_bytes": entry.size_bytes},
        )

    with open(data_path, "rb") as f:
        f.seek(offset)
        chunk = f.read(length)

    b64 = base64.b64encode(chunk).decode("ascii")
    is_last = (offset + len(chunk)) >= entry.size_bytes

    return _ok(
        ArtifactReadResult(
            artifact_id=entry.artifact_id,
            filename=entry.filename,
            media_type=entry.media_type,
            size_bytes=entry.size_bytes,
            sha256=entry.sha256,
            offset=offset,
            length=len(chunk),
            is_last=is_last,
            data_b64=b64,
        )
    )


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
