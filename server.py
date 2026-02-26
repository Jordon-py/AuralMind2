
"""
AuralMind Maestro - FastMCP Server
=================================

Production-grade MCP server exposing the AuralMind mastering DSP pipeline
as LLM-callable tools. Designed for non-blocking operation: heavy mastering
jobs run in a background thread pool, while lightweight analysis and
validation tools respond immediately.

Architecture
------------

    LLM client -> FastMCP (stdio / streamable-HTTP)
                     -> resources/  config://system-prompt, config://mcp-docs
                     -> prompts/    generate-mastering-strategy
                     -> tools/
                          -> upload_audio_to_session   (sync,  ~instant)
                          -> analyze_audio             (sync,  ~2 s)
                          -> list_presets              (sync,  instant)
                          -> propose_master_settings   (sync,  instant)
                          -> run_master_job            (async, returns job_id)
                          -> job_status                (sync,  instant)
                          -> job_result                (sync,  instant)
                          -> read_artifact             (sync,  chunked download)
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Literal, Annotated, TypeVar

from fastmcp import FastMCP, Context
from fastmcp.prompts import Message, PromptResult
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


log = logging.getLogger("auralmind.server")

SERVER_NAME = "AuralMind Maestro v7.3 Pro-Agent"
Platform = Literal["spotify", "apple_music", "youtube", "soundcloud", "club"]
BitDepth = Literal["float32", "float64"]

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    SERVER_NAME,
    on_duplicate="error",
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
    path="/mcp",
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
MCP_DOCS_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "mcp_docs.md"
)

# ---------------------------------------------------------------------------
# Upload safety caps
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 400 * 1024 * 1024  # 400 MB after decode
MAX_UPLOAD_HEX_CHARS = MAX_UPLOAD_BYTES * 2
MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MB chunks for artifact reads
HANDLE_RE = re.compile(r"^(aud|art|job)_[a-f0-9]{12}$")

_ARTIFACTS_LOCK = threading.Lock()
_ARTIFACTS: Dict[str, Dict[str, "ArtifactEntry"]] = {}
_MAESTRO_LOCK = threading.Lock()
_MAESTRO: Optional[Any] = None
_MAESTRO_ERROR: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactRef(StrictBaseModel):
    artifact_id: str = Field(..., description="Artifact handle.")
    filename: str = Field(..., description="Stored filename.")
    media_type: str = Field(..., description="MIME type.")
    size_bytes: int = Field(..., description="Artifact size in bytes.")
    sha256: str = Field(..., description="SHA-256 hash of the artifact.")


class UploadResult(StrictBaseModel):
    audio_id: str = Field(..., description="Server-side handle for the uploaded audio.")
    filename: str = Field(..., description="Sanitized filename stored on the server.")
    size_bytes: int = Field(..., description="Payload size in bytes.")
    sha256: str = Field(..., description="SHA-256 hash of the payload.")
    media_type: str = Field(..., description="Detected media type.")


class AudioMetrics(StrictBaseModel):
    lufs_i: float = Field(..., description="Integrated loudness (LUFS).")
    tp_dbfs: float = Field(..., description="True peak in dBFS.")
    peak_dbfs: float = Field(..., description="Peak level in dBFS.")
    rms_dbfs: float = Field(..., description="RMS level in dBFS.")
    crest_db: float = Field(..., description="Crest factor in dB.")
    corr_broadband: float = Field(..., description="Broadband correlation.")
    corr_low: float = Field(..., description="Low-band correlation.")
    sub_mono_ok: bool = Field(..., description="True if sub region is mono-safe.")
    centroid_hz: float = Field(..., description="Spectral centroid in Hz.")
    recommended_preset: str = Field(..., description="Recommended preset name.")
    recommended_lufs: float = Field(..., description="Recommended LUFS target.")
    sample_rate: int = Field(..., description="Sample rate in Hz.")
    duration_s: float = Field(..., description="Duration in seconds.")


class AnalyzeResult(StrictBaseModel):
    audio_id: str = Field(..., description="Audio handle that was analyzed.")
    metrics: AudioMetrics = Field(..., description="Analysis metrics.")


class PresetSummary(StrictBaseModel):
    target_lufs: float = Field(..., description="Target LUFS for the preset.")
    ceiling_dbfs: float = Field(..., description="Limiter ceiling in dBFS.")
    limiter_mode: str = Field(..., description="Limiter engine name.")
    governor_gr_limit_db: float = Field(..., description="Governor GR limit in dB.")
    match_strength: float = Field(..., description="Match EQ strength.")
    enable_harshness_limiter: bool = Field(..., description="Harshness limiter toggle.")


class PresetsResult(StrictBaseModel):
    presets: Dict[str, PresetSummary] = Field(
        ...,
        description="Preset map keyed by name.",
    )


class MasterSettings(StrictBaseModel):
    preset_name: str = Field(..., description="Preset name to start from.")
    target_lufs: float = Field(..., description="Target integrated loudness (LUFS).")
    warmth: float = Field(..., ge=0.0, le=1.0, description="Analog warmth amount.")
    transient_boost_db: float = Field(
        ...,
        ge=0.0,
        le=4.0,
        description="Transient sculpt boost.",
    )
    enable_harshness_limiter: bool = Field(
        ...,
        description="Enable harshness limiter.",
    )
    enable_air_motion: bool = Field(
        ...,
        description="Enable air motion stage.",
    )
    bit_depth: BitDepth = Field(..., description="Export bit depth.")
    subtype: str = Field(..., description="libsndfile subtype used for export.")


class SettingsResult(StrictBaseModel):
    settings: MasterSettings = Field(
        ...,
        description="Validated mastering settings.",
    )


class JobSubmitResult(StrictBaseModel):
    job_id: str = Field(..., description="Background job handle.")
    status: str = Field(..., description="Initial job status.")
    audio_id: str = Field(..., description="Audio handle submitted.")


class JobStatusResult(StrictBaseModel):
    job_id: str = Field(..., description="Job handle.")
    status: str = Field(..., description="Job status.")
    progress: str = Field(..., description="Human-readable progress message.")
    elapsed_s: float = Field(..., description="Seconds since job submission.")


class JobResultResult(StrictBaseModel):
    job_id: str = Field(..., description="Job handle.")
    status: str = Field(..., description="Final job status.")
    artifacts: List[ArtifactRef] = Field(
        ...,
        description="Output artifact references.",
    )
    metrics: Dict[str, Any] = Field(
        ...,
        description="Final mastering metrics.",
    )
    precision: str = Field(..., description="Export precision string.")


class ArtifactReadResult(StrictBaseModel):
    artifact_id: str = Field(..., description="Artifact handle.")
    filename: str = Field(..., description="Stored filename.")
    media_type: str = Field(..., description="MIME type.")
    size_bytes: int = Field(..., description="Artifact size in bytes.")
    sha256: str = Field(..., description="SHA-256 hash of the artifact.")
    offset: int = Field(..., description="Byte offset for this chunk.")
    length: int = Field(..., description="Length of this chunk in bytes.")
    is_last: bool = Field(..., description="True if this is the final chunk.")
    data_b64: str = Field(..., description="Base64-encoded chunk bytes.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _valid_handle(handle: str, prefix: Optional[str] = None) -> bool:
    if not isinstance(handle, str) or not HANDLE_RE.match(handle):
        return False
    if prefix is None:
        return True
    return handle.startswith(f"{prefix}_")


def _sanitize_filename(name: str, fallback: str = "audio") -> str:
    base = os.path.basename(name or fallback)
    cleaned = "".join(
        ch if 32 <= ord(ch) < 127 and ch not in "\\/:*?\"<>|" else "_"
        for ch in base
    ).strip()
    return cleaned or fallback


def _guess_media_type(
    filename: str,
    fallback: str = "application/octet-stream",
) -> str:
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


def _artifact_meta_path(session_dir: str, artifact_id: str) -> str:
    return os.path.join(session_dir, f"{artifact_id}.json")


def _artifact_data_path(session_dir: str, data_filename: str) -> str:
    return os.path.join(session_dir, data_filename)


def _register_artifact(session_key: str, entry: "ArtifactEntry", session_dir: str) -> None:
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


def _load_artifact(
    session_key: str,
    session_dir: str,
    artifact_id: str,
) -> Optional["ArtifactEntry"]:
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


def _store_bytes(
    session_key: str,
    session_dir: str,
    *,
    artifact_id: str,
    kind: str,
    filename: str,
    payload: bytes,
    media_type: str,
) -> "ArtifactEntry":
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


def _register_existing_file(
    session_key: str,
    session_dir: str,
    *,
    artifact_id: str,
    kind: str,
    filename: str,
    data_filename: str,
    media_type: str,
) -> "ArtifactEntry":
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


def _build_settings(
    maestro: Any,
    preset_name: str,
    target_lufs: float,
    warmth: float,
    transient_boost_db: float,
    enable_harshness_limiter: bool,
    enable_air_motion: bool,
    bit_depth: str,
) -> MasterSettings:
    presets = maestro.get_presets()
    if preset_name not in presets:
        raise ValueError(f"Invalid preset. Available: {list(presets.keys())}")

    warmth_val = max(0.0, min(1.0, float(warmth)))
    transient_val = max(0.0, min(4.0, float(transient_boost_db)))
    bit_depth_val = str(bit_depth)
    if bit_depth_val not in ("float32", "float64"):
        bit_depth_val = "float32"

    settings = MasterSettings(
        preset_name=preset_name,
        target_lufs=float(target_lufs),
        warmth=warmth_val,
        transient_boost_db=transient_val,
        enable_harshness_limiter=bool(enable_harshness_limiter),
        enable_air_motion=bool(enable_air_motion),
        bit_depth=bit_depth_val,  # type: ignore[arg-type]
        subtype="FLOAT" if bit_depth_val == "float32" else "DOUBLE",
    )
    return settings


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
    status: str = "queued"  # queued -> running -> done | error
    progress: str = "Waiting in queue"
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    artifacts: List["ArtifactEntry"] = field(default_factory=list)
    precision: str = "48000Hz / float32"
    error: Optional[str] = None


_JOBS: Dict[str, JobState] = {}


def _get_job(job_id: str, session_key: str) -> Optional[JobState]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return None
    if job.session_key != session_key:
        return None
    return job


def _set_job(job: JobState) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


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


def _run_master_worker(
    job: JobState,
    session_dir: str,
    audio_entry: ArtifactEntry,
    settings: MasterSettings,
) -> None:
    """Background worker - runs inside ThreadPoolExecutor."""
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
        preset_name = settings.preset_name or "hi_fi_streaming"
        preset = presets[preset_name]

        updates: Dict[str, Any] = {
            "target_lufs": float(settings.target_lufs),
            "warmth": float(settings.warmth),
            "transient_sculpt_boost_db": float(settings.transient_boost_db),
            "enable_harshness_limiter": bool(settings.enable_harshness_limiter),
            "enable_air_motion": bool(settings.enable_air_motion),
            "bit_depth": settings.bit_depth,
        }

        preset = replace(preset, **updates)

        bit_depth = settings.bit_depth
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


# ===========================================================================
# RESOURCES
# ===========================================================================
@mcp.resource(
    uri="config://system-prompt",
    name="SystemPrompt",
    description="Cognitive mastering system prompt.",
    mime_type="text/markdown",
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"config"},
)
def get_system_prompt() -> str:
    """Returns the AuralMind Cognitive Mastering system prompt."""
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource(
    uri="config://mcp-docs",
    name="McpDocs",
    description="LLM-facing MCP usage guide for AuralMind Maestro.",
    mime_type="text/markdown",
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"config", "docs"},
)
def get_mcp_docs() -> str:
    """Returns the MCP usage guide bundled with the server."""
    with open(MCP_DOCS_PATH, "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource(
    uri="config://server-info",
    name="ServerInfo",
    description="Server configuration and limits.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"config"},
)
def get_server_info() -> str:
    """Provides server metadata and limits as JSON."""
    payload = {
        "name": SERVER_NAME,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_read_bytes": MAX_READ_BYTES,
        "supported_bit_depths": ["float32", "float64"],
    }
    return json.dumps(payload, indent=2)


# ===========================================================================
# PROMPTS
# ===========================================================================
@mcp.prompt(
    name="generate-mastering-strategy",
    description="Builds a mastering-strategy prompt seeded with analysis metrics.",
    tags={"mastering", "prompt"},
)
def generate_strategy(
    lufs: Annotated[float, Field(description="Integrated loudness (LUFS).")],
    crest: Annotated[float, Field(description="Crest factor (dB).")],
    platform: Annotated[Platform, Field(description="Target platform.")],
) -> PromptResult:
    """Generates a prompt with the system instructions and measured metrics."""
    prompt_content = get_system_prompt()
    metrics = {
        "lufs": float(lufs),
        "crest_db": float(crest),
        "platform": platform,
    }
    prompt = (
        f"{prompt_content}\n\n"
        f"INPUT_METRICS:\n{json.dumps(metrics, indent=2)}\n\n"
        "Respond with the JSON strategy object."
    )
    return PromptResult(
        messages=[Message(prompt)],
        description="Mastering strategy prompt with embedded metrics.",
        meta={"platform": platform},
    )


# ===========================================================================
# TOOLS
# ===========================================================================
@mcp.tool()
def upload_audio_to_session(
    filename: Annotated[
        str,
        Field(description="Original filename (used for display).", min_length=1),
    ],
    payload_b64: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Base64-encoded audio payload (preferred).",
        ),
    ] = None,
    hex_payload: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Hex-encoded audio payload (legacy).",
        ),
    ] = None,
    ctx: Context = None,
) -> UploadResult:
    """Upload an audio file to the server before analysis or mastering."""
    if hex_payload and payload_b64:
        raise ValueError("ambiguous_payload: Provide only one payload encoding (hex_payload or payload_b64).")
    if not hex_payload and not payload_b64:
        raise ValueError("missing_payload: Provide hex_payload or payload_b64.")

    try:
        if hex_payload:
            if len(hex_payload) > MAX_UPLOAD_HEX_CHARS:
                raise ValueError(f"payload_too_large: Payload exceeds upload limit {MAX_UPLOAD_BYTES} bytes after decoding.")
            payload = bytes.fromhex(hex_payload)
        else:
            payload = base64.b64decode(payload_b64 or "", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"invalid_payload: Payload decoding failed. Error: {exc}")

    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError(f"payload_too_large: Payload exceeds upload limit {MAX_UPLOAD_BYTES} bytes after decoding.")

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
        raise RuntimeError(f"upload_failed: Upload failed. Error: {exc}")

    log.info("Uploaded %s -> %s (%s bytes)", filename, audio_id, entry.size_bytes)
    return UploadResult(
        audio_id=entry.artifact_id,
        filename=entry.filename,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
        media_type=entry.media_type,
    )


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False})
def analyze_audio(
    audio_id: Annotated[
        str,
        Field(
            description="Audio handle returned by upload_audio_to_session.",
            pattern=r"^aud_[a-f0-9]{12}$",
        ),
    ],
    ctx: Context = None,
) -> AnalyzeResult:
    """Comprehensive pre-mastering analysis - run BEFORE mastering."""
    if not _valid_handle(audio_id, "aud"):
        raise ValueError(f"invalid_audio_id: audio_id format is invalid: {audio_id}")

    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind != "audio":
        raise ValueError(f"not_found: Audio not found: {audio_id}")

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    if not os.path.exists(data_path):
        raise ValueError(f"not_found: Audio file missing: {audio_id}")

    maestro, err = _get_maestro()
    if err is not None:
        raise RuntimeError(f"{err['code']}: {err['message']} {err.get('details', '')}")

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

        metrics = AudioMetrics(
            lufs_i=float(features["lufs"]),
            tp_dbfs=float(features["tp_dbfs"]),
            peak_dbfs=float(features["peak_dbfs"]),
            rms_dbfs=float(features["rms_dbfs"]),
            crest_db=float(features["crest_db"]),
            corr_broadband=float(features.get("corr_hi", 0.0)),
            corr_low=corr_lo,
            sub_mono_ok=corr_lo >= 0.95,
            centroid_hz=float(features["centroid_hz"]),
            recommended_preset=preset_name,
            recommended_lufs=recommended_lufs,
            sample_rate=int(sr),
            duration_s=round(len(y) / sr, 2),
        )

        return AnalyzeResult(audio_id=audio_id, metrics=metrics)
    except Exception as exc:
        log.exception("Analysis failed for %s", audio_id)
        raise RuntimeError(f"analysis_failed: Analysis failed. Error: {exc}")


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False})
def list_presets() -> PresetsResult:
    """List all available mastering presets with key parameters."""
    maestro, err = _get_maestro()
    if err is not None:
        raise RuntimeError(f"{err['code']}: {err['message']} {err.get('details', '')}")


    presets = maestro.get_presets()
    out: Dict[str, PresetSummary] = {}
    for name, p in presets.items():
        out[name] = PresetSummary(
            target_lufs=float(p.target_lufs),
            ceiling_dbfs=float(p.ceiling_dbfs),
            limiter_mode=str(getattr(p, "limiter_mode", "v2")),
            governor_gr_limit_db=float(p.governor_gr_limit_db),
            match_strength=float(p.match_strength),
            enable_harshness_limiter=bool(p.enable_harshness_limiter),
        )
    return PresetsResult(presets=out)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False})
def propose_master_settings(
    preset_name: Annotated[
        str,
        Field(description="Preset name to use as a baseline."),
    ] = "hi_fi_streaming",
    target_lufs: Annotated[
        float,
        Field(description="Target integrated loudness (LUFS)."),
    ] = -12.0,
    warmth: Annotated[
        float,
        Field(description="Analog warmth amount (0.0-1.0).", ge=0.0, le=1.0),
    ] = 0.5,
    transient_boost_db: Annotated[
        float,
        Field(description="Transient sculpt boost in dB (0.0-4.0).", ge=0.0, le=4.0),
    ] = 1.0,
    enable_harshness_limiter: Annotated[
        bool,
        Field(description="Enable harshness limiter stage."),
    ] = True,
    enable_air_motion: Annotated[
        bool,
        Field(description="Enable air motion stage."),
    ] = True,
    bit_depth: Annotated[
        BitDepth,
        Field(description="Export bit depth."),
    ] = "float32",
) -> SettingsResult:
    """Validate and preview mastering settings before submitting a job."""
    maestro, err = _get_maestro()
    if err is not None:
        raise RuntimeError(f"{err['code']}: {err['message']} {err.get('details', '')}")

    settings = _build_settings(
        maestro,
        preset_name,
        target_lufs,
        warmth,
        transient_boost_db,
        enable_harshness_limiter,
        enable_air_motion,
        bit_depth,
    )
    return SettingsResult(settings=settings)


@mcp.tool()
def run_master_job(
    audio_id: Annotated[
        str,
        Field(
            description="Audio handle returned by upload_audio_to_session.",
            pattern=r"^aud_[a-f0-9]{12}$",
        ),
    ],
    preset_name: Annotated[
        str,
        Field(description="Preset name to use as a baseline."),
    ] = "hi_fi_streaming",
    target_lufs: Annotated[
        float,
        Field(description="Target integrated loudness (LUFS)."),
    ] = -12.0,
    warmth: Annotated[
        float,
        Field(description="Analog warmth amount (0.0-1.0).", ge=0.0, le=1.0),
    ] = 0.5,
    transient_boost_db: Annotated[
        float,
        Field(description="Transient sculpt boost in dB (0.0-4.0).", ge=0.0, le=4.0),
    ] = 1.0,
    enable_harshness_limiter: Annotated[
        bool,
        Field(description="Enable harshness limiter stage."),
    ] = True,
    enable_air_motion: Annotated[
        bool,
        Field(description="Enable air motion stage."),
    ] = True,
    bit_depth: Annotated[
        BitDepth,
        Field(description="Export bit depth."),
    ] = "float32",
    ctx: Context = None,
) -> JobSubmitResult:
    """Submit a mastering job to the background thread pool (non-blocking)."""
    if not _valid_handle(audio_id, "aud"):
        raise ValueError(f"invalid_audio_id: audio_id format is invalid: {audio_id}")

    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind != "audio":
        raise ValueError(f"not_found: Audio not found: {audio_id}")

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    if not os.path.exists(data_path):
        raise ValueError(f"not_found: Audio file missing: {audio_id}")

    maestro, err = _get_maestro()
    if err is not None:
        raise RuntimeError(f"{err['code']}: {err['message']} {err.get('details', '')}")


    settings = _build_settings(
        maestro,
        preset_name,
        target_lufs,
        warmth,
        transient_boost_db,
        enable_harshness_limiter,
        enable_air_motion,
        bit_depth,
    )

    job_id = _new_id("job")
    job = JobState(job_id=job_id, session_key=session_key, audio_id=audio_id)
    _set_job(job)

    _EXECUTOR.submit(_run_master_worker, job, session_dir, entry, settings)
    log.info("Submitted job %s for %s", job_id, audio_id)

    return JobSubmitResult(job_id=job_id, status="queued", audio_id=audio_id)



@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False})
def job_status(
    job_id: Annotated[
        str,
        Field(
            description="Job handle returned by run_master_job.",
            pattern=r"^job_[a-f0-9]{12}$",
        ),
    ],
    ctx: Context = None,
) -> JobStatusResult:
    """Check the current status of a mastering job."""
    if not _valid_handle(job_id, "job"):
        raise ValueError(f"invalid_job_id: job_id format is invalid: {job_id}")

    session_key, _ = _get_session_info(ctx)
    job = _get_job(job_id, session_key)
    if job is None:
        raise ValueError(f"not_found: Unknown job_id: {job_id}")

    return JobStatusResult(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        elapsed_s=round(time.time() - job.created_at, 1),
    )



@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False})
def job_result(
    job_id: Annotated[
        str,
        Field(
            description="Job handle returned by run_master_job.",
            pattern=r"^job_[a-f0-9]{12}$",
        ),
    ],
    ctx: Context = None,
) -> JobResultResult:
    """Retrieve the final result of a completed mastering job."""
    if not _valid_handle(job_id, "job"):
        raise ValueError(f"invalid_job_id: job_id format is invalid: {job_id}")

    session_key, _ = _get_session_info(ctx)
    job = _get_job(job_id, session_key)
    if job is None:
        raise ValueError(f"not_found: Unknown job_id: {job_id}")

    if job.status == "error":
        raise RuntimeError(f"job_failed: Job failed with error: {job.error or 'Unknown error'}")

    if job.status != "done":
        raise RuntimeError(f"job_not_complete: Job is not yet complete. Call job_status to check progress. Current status: {job.status}")


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

    return JobResultResult(
        job_id=job.job_id,
        status=job.status,
        artifacts=artifacts,
        metrics=metrics,
        precision=job.precision,
    )



@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False})
def read_artifact(
    artifact_id: Annotated[
        str,
        Field(
            description="Artifact handle from upload_audio_to_session or job_result.",
            pattern=r"^(aud|art)_[a-f0-9]{12}$",
        ),
    ],
    offset: Annotated[
        int,
        Field(description="Byte offset for the chunk.", ge=0),
    ] = 0,
    length: Annotated[
        int,
        Field(
            description="Bytes to read (max 2 MB).",
            ge=1,
            le=MAX_READ_BYTES,
        ),
    ] = MAX_READ_BYTES,
    ctx: Context = None,
) -> ArtifactReadResult:
    """Read artifact bytes as base64 in bounded chunks."""
    if not _valid_handle(artifact_id):
        raise ValueError(f"invalid_artifact_id: artifact_id format is invalid: {artifact_id}")
    if offset < 0 or length <= 0:
        raise ValueError(f"invalid_range: offset and length must be non-negative. offset={offset}, length={length}")
    if length > MAX_READ_BYTES:
        raise ValueError(f"chunk_too_large: Requested length exceeds server chunk limit {MAX_READ_BYTES}")

    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, artifact_id)
    if entry is None:
        raise ValueError(f"not_found: Artifact not found: {artifact_id}")

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    if not os.path.exists(data_path):
        raise ValueError(f"not_found: Artifact file missing: {artifact_id}")

    if offset > entry.size_bytes:
        raise ValueError(f"offset_out_of_range: Offset exceeds artifact size. offset={offset}, size={entry.size_bytes}")


    with open(data_path, "rb") as f:
        f.seek(offset)
        chunk = f.read(length)

    b64 = base64.b64encode(chunk).decode("ascii")
    is_last = (offset + len(chunk)) >= entry.size_bytes

    return ArtifactReadResult(
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



# ===========================================================================
# Entrypoint
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    port = int(os.environ.get("PORT", "8000"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
