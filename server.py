from __future__ import annotations

"""
AuralMind2 - FastMCP mastering server
=====================================

This module exposes the AuralMind mastering pipeline over FastMCP with
streamable HTTP as the deployment default and stdio as the local-client
override. Heavy mastering runs are queued in background workers while
analysis, discovery, resources, and prompts stay synchronous and predictable.

Runtime surface
---------------

    LLM client -> FastMCP
                     -> 34 tools
                     -> 10 resources
                     -> 4 prompts
                     -> ASGI app export for streamable HTTP hosts

The exported `app` is intended for hosts such as Render. The `main()` entry
point preserves direct `python server.py` execution and selects transport from
environment variables.
"""

import os
import re
import json
import time
import uuid
import asyncio
import base64
import binascii
import hashlib
import logging
import tempfile
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Literal, Annotated, Callable

from fastmcp import FastMCP, Context
from fastmcp.prompts import Message
from pydantic import BaseModel, Field, ConfigDict, RootModel, model_validator
import numpy as np
import soundfile as sf
from starlette.requests import Request
from starlette.responses import JSONResponse

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

log = logging.getLogger("auralmind.server")

SERVER_NAME = "AuralMind2"
VERSION = "0.1.0"
ACTIVE_TRANSPORT_ENV = "ACTIVE_TRANSPORT"
DEFAULT_ACTIVE_TRANSPORT = "streamable-http"
SUPPORTED_TRANSPORTS = ("stdio", "streamable-http")
_TRANSPORT_ALIASES = {
    "http": "streamable-http",
    "streamable_http": "streamable-http",
    "streamablehttp": "streamable-http",
}
HTTP_HOST_ENV = "MCP_HOST"
HTTP_PORT_ENV = "PORT"
HTTP_PATH_ENV = "MCP_PATH"
DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8080
DEFAULT_HTTP_PATH = "/mcp"
HTTP_APP_TRANSPORT = "streamable-http"

Platform = Literal["spotify", "apple_music", "youtube", "soundcloud", "club"]
# prefer float64 for audio processing and float32 for audio output
BitDepth = Literal["float32", "float64"]
JobStatus = Literal["queued", "running", "done", "error"]

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name=SERVER_NAME,
)

# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_STORAGE_DIR = os.path.join(tempfile.gettempdir(), "maestro_sessions")
STORAGE_DIR = os.path.abspath(os.environ.get("MAESTRO_SESSION_DIR", DEFAULT_STORAGE_DIR))
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
DATA_DIR_REAL = os.path.realpath(DATA_DIR)

SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "system_prompt.md"
)
MCP_DOCS_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "mcp_docs.md"
)
MAINTAINER_GUIDE_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "maintainer_guide.md"
)

# ---------------------------------------------------------------------------
# Upload safety caps
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 400 * 1024 * 1024  # 400 MB after decode
MAX_UPLOAD_B64_CHARS = int(MAX_UPLOAD_BYTES * 4 / 3) + 4

BOOTSTRAP_WORKFLOW_STEPS = [
    "Upload audio (upload_audio_to_session or register_audio_from_path)",
    "Analyze (analyze_audio)",
    "Select/Optimize preset (list_presets / analyze_and_optimize_governor)",
    "Run master job (run_master_job)",
    "Poll status (job_status)",
    "Download result (job_result -> read_artifact)"
]

BOOTSTRAP_EXAMPLE_CALLS = {
    "bootstrap": {"method": "tools/call", "params": {"name": "bootstrap", "arguments": {}}},
    "list_presets": {"method": "tools/call", "params": {"name": "list_presets", "arguments": {}}},
    "master_once": {"method": "tools/call", "params": {"name": "master_audio", "arguments": {"audio_id": "aud_...", "preset_name": "hi_fi_streaming"}}}
}
MAX_UPLOAD_HEX_CHARS = MAX_UPLOAD_BYTES * 2
UPLOAD_CHUNK_MAX_BYTES = int(os.environ.get("UPLOAD_CHUNK_MAX_BYTES", str(1024 * 1024)))  # 1 MiB
MAX_UPLOAD_CHUNK_B64_CHARS = int(UPLOAD_CHUNK_MAX_BYTES * 4 / 3) + 4
MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MB chunks for artifact reads
CONNECT_PREVIEW_LIMIT = 10
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}
HANDLE_RE = re.compile(r"^(aud|art|job)_[a-f0-9]{12}$")
UPLOAD_ID_RE = re.compile(r"^upl_[a-f0-9]{12}$")
VISIBLE_MASTER_FIELDS = {
    "preset_name",
    "target_lufs",
    "warmth",
    "transient_boost_db",
    "enable_harshness_limiter",
    "enable_air_motion",
    "bit_depth",
}
SAFE_OVERRIDE_FIELDS = {"governor_search_steps", "governor_gr_limit_db", "stem_gains_db"}
CONTROL_PROFILE_FIELDS = {
    "spatial_width",
    "brightness_tilt",
    "harshness_control",
    "movement_amount",
    "low_end_focus",
}

PLATFORM_TARGET_RANGES: Dict[str, Tuple[float, float]] = {
    "spotify": (-13.8, -11.4),
    "apple_music": (-14.2, -11.8),
    "youtube": (-14.4, -12.0),
    "soundcloud": (-13.2, -10.8),
    "club": (-11.2, -9.4),
}
PLATFORM_TARGET_DEFAULTS: Dict[str, float] = {
    "spotify": -12.6,
    "apple_music": -13.0,
    "youtube": -13.4,
    "soundcloud": -12.0,
    "club": -10.2,
}
PLATFORM_POLICY_NOTES: Dict[str, str] = {
    "spotify": "Keep competitive loudness while leaving headroom for streaming normalization.",
    "apple_music": "Bias a touch more open and dynamic than Spotify-oriented masters.",
    "youtube": "Avoid over-driving loudness because playback normalization is conservative.",
    "soundcloud": "Allow a slightly hotter result for mixed playback environments.",
    "club": "Optimize for impact and density while still guarding true peak and low-end stability.",
}

_ARTIFACTS_LOCK = threading.Lock()
_ARTIFACTS: Dict[str, Dict[str, "ArtifactEntry"]] = {}
_UPLOAD_LOCK = threading.Lock()
_MAESTRO_LOCK = threading.Lock()
_MAESTRO: Optional[Any] = None
_MAESTRO_ERROR: Optional[Dict[str, Any]] = None

_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, "JobState"] = {}
_JOB_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("MAX_MASTER_JOBS", "2"))
)


def _active_transport() -> str:
    raw_transport = str(os.environ.get(ACTIVE_TRANSPORT_ENV, DEFAULT_ACTIVE_TRANSPORT)).strip().lower()
    normalized_transport = _TRANSPORT_ALIASES.get(raw_transport, raw_transport)
    if normalized_transport not in SUPPORTED_TRANSPORTS:
        log.warning(
            "Unsupported %s=%r. Falling back to %s.",
            ACTIVE_TRANSPORT_ENV,
            raw_transport,
            DEFAULT_ACTIVE_TRANSPORT,
        )
        return DEFAULT_ACTIVE_TRANSPORT
    return normalized_transport


def _http_host() -> str:
    host = str(os.environ.get(HTTP_HOST_ENV, DEFAULT_HTTP_HOST)).strip()
    return host or DEFAULT_HTTP_HOST


def _http_port() -> int:
    raw_port = str(os.environ.get(HTTP_PORT_ENV, DEFAULT_HTTP_PORT)).strip()
    try:
        parsed = int(raw_port)
    except ValueError:
        log.warning("Invalid %s=%r. Falling back to %s.", HTTP_PORT_ENV, raw_port, DEFAULT_HTTP_PORT)
        return DEFAULT_HTTP_PORT

    if parsed < 1 or parsed > 65535:
        log.warning("Out-of-range %s=%r. Falling back to %s.", HTTP_PORT_ENV, raw_port, DEFAULT_HTTP_PORT)
        return DEFAULT_HTTP_PORT
    return parsed


def _http_path() -> str:
    path = str(os.environ.get(HTTP_PATH_ENV, DEFAULT_HTTP_PATH)).strip()
    if not path:
        return DEFAULT_HTTP_PATH
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _run_kwargs_for_active_transport() -> Dict[str, Any]:
    transport = _active_transport()
    run_kwargs: Dict[str, Any] = {"transport": transport}
    if transport != "stdio":
        run_kwargs["host"] = _http_host()
        run_kwargs["port"] = _http_port()
        run_kwargs["path"] = _http_path()
        run_kwargs["json_response"] = True
    return run_kwargs


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorEnvelope(StrictBaseModel):
    code: str = Field(..., description="Stable error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional diagnostic details.")


class CapabilitiesOut(StrictBaseModel):
    server_name: str = Field(SERVER_NAME, description="Name of the MCP server.")
    version: str = Field(..., description="Server version.")
    transport: str = Field(..., description="Active transport (stdio, sse, or streamable-http).")
    features: List[str] = Field(..., description="List of enabled features.")


class ToolCatalogEntry(StrictBaseModel):
    name: str = Field(..., description="Tool name.")
    description: str = Field(..., description="Purpose of the tool.")
    input_model: str = Field(..., description="Name/schema of the input Pydantic model.")
    output_model: str = Field(..., description="Name/schema of the output Pydantic model.")


class ResourceCatalogEntry(StrictBaseModel):
    uri: str = Field(..., description="Resource URI.")
    description: str = Field(..., description="Resource description.")
    mime_type: str = Field(..., description="Resource MIME type.")
    annotations: Dict[str, Any] = Field(..., description="Resource annotations/hints.")


class PromptCatalogEntry(StrictBaseModel):
    name: str = Field(..., description="Prompt name.")
    description: str = Field(..., description="Prompt description.")
    args_schema: Dict[str, Any] = Field(..., description="JSON Schema for prompt arguments.")


class BootstrapOut(StrictBaseModel):
    capabilities: CapabilitiesOut
    tools: List[ToolCatalogEntry]
    resources: List[ResourceCatalogEntry]
    prompts: List[PromptCatalogEntry]
    workflow_steps: List[str] = Field(..., description="Strict ordered list of task steps.")
    example_calls: Dict[str, Any] = Field(..., description="Copy/paste payloads for common tasks.")


class AudioMetrics(StrictBaseModel):
    integrated_lufs: float = Field(..., description="Integrated loudness (LUFS).")
    true_peak_dbtp: float = Field(..., description="True peak in dBTP.")
    crest_db: float = Field(..., description="Crest factor in dB.")
    stereo_correlation: float = Field(..., description="Stereo correlation coefficient.")
    duration_s: float = Field(..., description="Duration in seconds.")
    # Optional metadata
    peak_dbfs: Optional[float] = None
    rms_dbfs: Optional[float] = None
    centroid_hz: Optional[float] = None


class AnalyzeIn(StrictBaseModel):
    audio_id: str = Field(..., description="Handle of the audio to analyze.")


class AnalyzeResult(StrictBaseModel):
    audio_id: str = Field(..., description="Audio handle analyzed.")
    metrics: AudioMetrics = Field(..., description="Analysis metrics.")


class JobIdIn(StrictBaseModel):
    job_id: str = Field(..., description="Job ID.")


class PresetSummary(StrictBaseModel):
    target_lufs: float = Field(..., description="Target LUFS.")
    ceiling_dbfs: float = Field(..., description="Limiter ceiling.")
    limiter_mode: str = Field(..., description="Limiter engine.")
    governor_gr_limit_db: float = Field(..., description="Governor limit.")
    match_strength: float = Field(..., description="Match EQ strength.")
    enable_harshness_limiter: bool = Field(..., description="Harshness limiter flag.")
    enable_air_motion: bool = Field(..., description="Air motion flag.")
    bit_depth: BitDepth = Field(..., description="Default bit depth.")


class PresetsOut(StrictBaseModel):
    presets: Dict[str, PresetSummary] = Field(..., description="Map of presets.")


class MasteringControlProfile(StrictBaseModel):
    spatial_width: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Stereo image intent. -1 is tighter and more mono-safe, +1 is widest.",
    )
    brightness_tilt: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Spectral tilt. -1 is darker/smoother, +1 is brighter/more forward.",
    )
    harshness_control: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Upper-mid fatigue control. -1 is relaxed, +1 is aggressively protective.",
    )
    movement_amount: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Macro motion and hook-lift intensity. -1 is static, +1 is animated.",
    )
    low_end_focus: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Low-end intent. -1 is lighter/leaner, +1 is heavier and tighter.",
    )


class MasterSettings(StrictBaseModel):
    preset_name: str = Field("hi_fi_streaming", description="Base preset.")
    target_lufs: float = Field(-12.0, description="Target LUFS.")
    warmth: float = Field(0.5, ge=0.0, le=1.0, description="Warmth (0-1).")
    transient_boost_db: float = Field(1.0, ge=0.0, le=4.0, description="Transient boost.")
    enable_harshness_limiter: bool = Field(True, description="Enable harshness filter.")
    enable_air_motion: bool = Field(True, description="Enable spatial air.")
    bit_depth: BitDepth = Field("float32", description="Output precision.")
    control_profile: Optional[MasteringControlProfile] = Field(
        None,
        description="Optional high-level control profile compiled into bounded DSP decisions.",
    )
    governor_search_steps: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description="Override governor binary search steps.",
    )
    governor_gr_limit_db: Optional[float] = Field(
        None,
        ge=-6.0,
        le=-0.4,
        description="Override governor GR limit.",
    )
    stem_gains_db: Optional[Dict[str, float]] = Field(None, description="Demucs stem gain adjustments (dB).")


class MasterRequest(MasterSettings):
    audio_id: str = Field(..., description="Source audio handle.")


class MasterResult(StrictBaseModel):
    run_id: str = Field(..., description="Unique ID for this mastering run.")
    master_wav_id: str = Field(..., description="Handle for the output WAV.")
    metrics_before: AudioMetrics
    metrics_after: AudioMetrics
    tuning_trace_id: str = Field(..., description="Handle for the tuning trace JSON.")
    artifacts: List[str] = Field(default_factory=list, description="Artifact handles created by this run.")


class ProposedSettingsOut(StrictBaseModel):
    settings: MasterSettings = Field(..., description="Validated mastering settings.")


class StrategyPlanIn(StrictBaseModel):
    audio_id: str = Field(..., description="Source audio handle.")
    goal: str = Field(..., min_length=3, description="Plain-language mastering intent.")
    platform: Platform = Field("spotify", description="Target platform.")
    control_profile: Optional[MasteringControlProfile] = Field(
        None,
        description="Optional high-level LLM control profile to apply after semantic planning.",
    )
    governor_search_steps: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description="Optional governor search override.",
    )
    governor_gr_limit_db: Optional[float] = Field(
        None,
        ge=-6.0,
        le=-0.4,
        description="Optional governor GR ceiling override.",
    )
    stem_gains_db: Optional[Dict[str, float]] = Field(
        None,
        description="Optional Demucs stem gain adjustments in dB.",
    )


class StrategyPlanOut(StrictBaseModel):
    audio_id: str = Field(..., description="Source audio handle.")
    goal: str = Field(..., description="Original mastering goal.")
    platform: Platform = Field(..., description="Target platform.")
    metrics: AudioMetrics = Field(..., description="Measured source metrics used for planning.")
    chosen_preset: str = Field(..., description="Base preset chosen before final overrides.")
    settings: MasterSettings = Field(..., description="Resolved mastering settings to execute.")
    reasoning: List[str] = Field(..., description="Decision trail explaining how the plan was resolved.")
    warnings: List[str] = Field(default_factory=list, description="Potential risks or caveats for the chosen plan.")


class JobLaunchOut(StrictBaseModel):
    job_id: str = Field(..., description="Queued mastering job ID.")
    status: JobStatus = Field(..., description="Initial job status.")
    audio_id: str = Field(..., description="Source audio handle.")


class JobStatusOut(StrictBaseModel):
    job_id: str = Field(..., description="Job ID.")
    status: JobStatus = Field(..., description="Current job status.")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage (0-100).")
    elapsed_s: float = Field(..., description="Elapsed time in seconds.")
    error: Optional[ErrorEnvelope] = Field(None, description="Failure details, if any.")


class ArtifactSummary(StrictBaseModel):
    artifact_id: str = Field(..., description="Artifact handle.")
    filename: str = Field(..., description="Stored filename.")
    media_type: str = Field(..., description="MIME type.")
    size_bytes: int = Field(..., description="Size in bytes.")
    sha256: str = Field(..., description="SHA-256 hash.")


class JobResultOut(StrictBaseModel):
    job_id: str = Field(..., description="Job ID.")
    status: JobStatus = Field(..., description="Final job status.")
    artifacts: List[ArtifactSummary] = Field(..., description="Generated artifacts.")
    metrics: AudioMetrics = Field(..., description="Final mastering metrics.")
    precision: BitDepth = Field(..., description="Output precision.")


class ClosedLoopRequest(StrictBaseModel):
    audio_id: str = Field(..., description="Source audio handle.")
    goal: str = Field(..., description="Mastering goal (e.g. 'Club-ready', 'Intimate Acoustic').")
    platform: Platform = Field("spotify", description="Target platform.")
    control_profile: Optional[MasteringControlProfile] = Field(
        None,
        description="Optional high-level control profile layered on top of the semantic plan.",
    )
    governor_search_steps: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description="Override governor binary search steps.",
    )
    governor_gr_limit_db: Optional[float] = Field(
        None,
        ge=-6.0,
        le=-0.4,
        description="Override governor GR limit.",
    )
    stem_gains_db: Optional[Dict[str, float]] = Field(None, description="Demucs stem gain adjustments (dB).")


class TuneDelta(StrictBaseModel):
    param: str = Field(..., description="Parameter changed.")
    old_value: Any
    new_value: Any
    reason_code: str = Field(..., description="Stable reason code.")
    reason_detail: str = Field(..., description="Detailed explanation.")


class ClosedLoopResult(StrictBaseModel):
    best_run_id: str = Field(..., description="ID of the best mastering run.")
    artifacts: List[str] = Field(..., description="List of generated artifact handles.")
    runner_summary_id: str = Field(..., description="Handle for the runner summary JSON.")
    metrics_final: AudioMetrics


class FileReadIn(StrictBaseModel):
    path: str = Field(..., description="File path to read (within allowlist).")


class FileReadOut(StrictBaseModel):
    content: str = Field(..., description="Text content of the file.")


class FileWriteIn(StrictBaseModel):
    path: str = Field(..., description="File path to write (within allowlist).")
    content: str = Field(..., description="Text content to write.")


class FileWriteOut(StrictBaseModel):
    success: bool
    path: str


class UploadIn(StrictBaseModel):
    filename: str = Field(..., description="Original filename.")
    payload_b64: Optional[str] = Field(None, description="Base64 payload.")
    hex_payload: Optional[str] = Field(None, description="Hex payload (legacy).")

    @model_validator(mode="after")
    def _validate_payload(self) -> "UploadIn":
        if not self.payload_b64 and not self.hex_payload:
            raise ValueError("payload_required")
        if self.payload_b64 and self.hex_payload:
            raise ValueError("payload_conflict")
        return self


class UploadResult(StrictBaseModel):
    audio_id: str = Field(..., description="Server-side handle for the uploaded audio.")
    filename: str = Field(..., description="Sanitized filename stored on the server.")
    size_bytes: int = Field(..., description="Payload size in bytes.")
    sha256: str = Field(..., description="SHA-256 hash of the payload.")
    media_type: str = Field(..., description="Detected media type.")


class UploadInitIn(StrictBaseModel):
    filename: str = Field(..., description="Original filename.")
    total_bytes: int = Field(..., ge=1, le=MAX_UPLOAD_BYTES, description="Total decoded byte length.")
    sha256: Optional[str] = Field(None, description="Expected lowercase SHA-256 hex digest.")


class UploadInitOut(StrictBaseModel):
    upload_id: str = Field(..., description="Upload handle.")
    filename: str = Field(..., description="Sanitized filename.")
    total_bytes: int = Field(..., description="Expected total byte count.")
    received_bytes: int = Field(..., description="Bytes received so far.")
    next_index: int = Field(..., description="Next chunk index expected.")
    chunk_max_bytes: int = Field(..., description="Maximum bytes per chunk.")
    done: bool = Field(..., description="True when upload bytes are complete.")


class UploadChunkIn(StrictBaseModel):
    upload_id: str = Field(..., description="Upload handle from upload_init.")
    index: int = Field(..., ge=0, description="Sequential chunk index starting at 0.")
    chunk_b64: str = Field(..., description="Base64 chunk payload.")


class UploadFinalizeIn(StrictBaseModel):
    upload_id: str = Field(..., description="Upload handle from upload_init.")


class UploadStatusOut(StrictBaseModel):
    upload_id: str = Field(..., description="Upload handle.")
    filename: str = Field(..., description="Sanitized filename.")
    total_bytes: int = Field(..., description="Expected total byte count.")
    received_bytes: int = Field(..., description="Bytes received so far.")
    next_index: int = Field(..., description="Next chunk index expected.")
    done: bool = Field(..., description="True when upload bytes are complete.")
    expected_sha256: Optional[str] = Field(None, description="Optional expected digest.")


class AudioAssetInfo(StrictBaseModel):
    filename: str = Field(..., description="Base filename within the data directory.")
    size_bytes: int = Field(..., description="File size in bytes.")
    format: str = Field(..., description="Audio format (wav, flac, etc).")
    duration_seconds: Optional[float] = Field(None, description="Optional duration in seconds.")


class AudioAssetList(RootModel[List[AudioAssetInfo]]):
    pass


class ConnectSongPreview(StrictBaseModel):
    filename: str = Field(..., description="Base filename inside the data directory.")
    size_bytes: int = Field(..., description="File size in bytes.")
    format: str = Field(..., description="Audio format extension without dot.")
    duration_seconds: Optional[float] = Field(None, description="Optional duration in seconds.")
    modified_at: str = Field(..., description="UTC ISO-8601 last-modified timestamp.")


class ConnectPacketOut(StrictBaseModel):
    generated_at: str = Field(..., description="UTC ISO-8601 packet generation timestamp.")
    preview_limit: int = Field(..., description="Maximum songs included in preview.")
    total_songs: int = Field(..., description="Total matching songs in data directory.")
    songs_preview: List[ConnectSongPreview] = Field(..., description="Most recent songs available to master.")
    recommended_first_path: str = Field(..., description="Suggested first path based on song availability.")
    workflow_steps: List[str] = Field(..., description="Ordered first-contact workflow guidance.")
    example_calls: Dict[str, Any] = Field(..., description="Copy/paste tool call templates.")


class RegisterAudioPathIn(StrictBaseModel):
    path: str = Field(..., description="Path to an audio file within the data directory.")


class RegisterAudioResult(StrictBaseModel):
    audio_id: str = Field(..., description="Server-side handle for the registered audio.")
    format: str = Field(..., description="Audio format (wav, flac, etc).")
    size_bytes: int = Field(..., description="File size in bytes.")
    checksum: str = Field(..., description="SHA-256 checksum of the file.")
    registered_at: str = Field(..., description="UTC ISO-8601 timestamp of registration.")


class ArtifactReadIn(StrictBaseModel):
    artifact_id: str = Field(..., description="Artifact handle.")
    offset: int = Field(0, ge=0, description="Byte offset (default 0).")
    length: int = Field(MAX_READ_BYTES, ge=1, le=MAX_READ_BYTES, description="Bytes to read.")


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


class CancelJobIn(StrictBaseModel):
    job_id: str = Field(..., description="Job ID to cancel.")


class CancelJobOut(StrictBaseModel):
    job_id: str
    success: bool
    message: str


class DeleteArtifactIn(StrictBaseModel):
    artifact_id: str = Field(..., description="Artifact handle to delete.")


class DeleteArtifactOut(StrictBaseModel):
    artifact_id: str
    success: bool


class AudioMetricsDelta(StrictBaseModel):
    lufs_delta: float = Field(..., description="After - Before integrated LUFS.")
    true_peak_delta: float = Field(..., description="After - Before true peak.")
    crest_delta: float = Field(..., description="After - Before crest factor.")
    correlation_delta: float = Field(..., description="After - Before stereo correlation.")


class CompareMetricsIn(StrictBaseModel):
    audio_id_a: str = Field(..., description="Baseline audio or metrics artifact ID.")
    audio_id_b: str = Field(..., description="Target audio or metrics artifact ID to compare against baseline.")


class CompareMetricsOut(StrictBaseModel):
    delta: AudioMetricsDelta

class MusicalEqIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact to process.")
    key: str = Field(..., description="The musical key (e.g., 'C', 'G#').")
    scale: str = Field(..., description="The scale type ('major' or 'minor').")

class MusicalEqOut(StrictBaseModel):
    artifact_id: str
    message: str
    ascii_graph: str

class TempoDynamicsIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact to process.")
    bpm: float = Field(..., description="Detected or desired strict BPM.")
    note_division: str = Field("1/4", description="Release time note division (e.g., '1/4', '1/8').")

class TempoDynamicsOut(StrictBaseModel):
    artifact_id: str
    message: str
    pulse_grid: str

class HarmonicExcitationIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact to process.")
    drive_amount: float = Field(50.0, description="Saturation drive amount (0.0 to 100.0).")
    harmonics: str = Field("even", description="Harmonic profile: 'even' (tube), 'odd' (tape), or 'both'.")

class HarmonicExcitationOut(StrictBaseModel):
    artifact_id: str
    message: str
    meter: str

class StartInteractiveMasteringIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact to process.")
    preset_name: str = Field(..., description="Preset to use for Stage 1 (e.g., 'punchy_pop').")

class StartInteractiveMasteringOut(StrictBaseModel):
    session_token: str
    message: str
    metrics: AudioMetrics
    stage1_settings: MasterRequest

class CommitInteractiveMasteringIn(StrictBaseModel):
    session_token: str = Field(..., description="The session token returned by start_interactive_mastering.")
    warmth: float = Field(..., description="The new warmth value determined by the AI.")
    transient_boost_db: float = Field(..., description="The new transient boost dB determined by the AI.")

class CommitInteractiveMasteringOut(StrictBaseModel):
    artifact_id: str
    message: str
    ascii_console: str
    final_metrics: AudioMetrics

class SemanticABMasteringIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact to process.")
    preset_a: str = Field(..., description="Preset name for Option A.")
    preset_b: str = Field(..., description="Preset name for Option B.")

class SemanticABMasteringOut(StrictBaseModel):
    artifact_id_a: str
    artifact_id_b: str
    message: str
    comparison_matrix: str
    heatmap_a: str
    heatmap_b: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
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


@dataclass
class JobState:
    job_id: str
    audio_id: str
    status: JobStatus
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[ErrorEnvelope] = None
    result: Optional[MasterResult] = None
    settings: Optional[MasterSettings] = None
    session_key: str = ""
    session_dir: str = ""
    future: Optional[Future] = None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def _valid_handle(handle: str, prefix: Optional[str] = None) -> bool:
    if not isinstance(handle, str) or not HANDLE_RE.match(handle):
        return False
    if prefix is None:
        return True
    return handle.startswith(f"{prefix}_")


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_allowed_path(path: str) -> bool:
    norm = _normalize_path(path)
    for allowed_root in (_normalize_path(STORAGE_DIR), _normalize_path(DATA_DIR_REAL)):
        try:
            if os.path.commonpath([norm, allowed_root]) == allowed_root:
                return True
        except ValueError:
            continue
    return False


def _resolve_data_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("invalid_path")
    candidate = path.strip()
    candidate_norm = candidate.replace("\\", "/")
    if candidate_norm.lower().startswith("./"):
        candidate_norm = candidate_norm[2:]
    if candidate_norm.lower().startswith("data/"):
        candidate_norm = candidate_norm[5:]
    candidate = candidate_norm
    if not candidate:
        raise ValueError("invalid_path")
    if any(part == ".." for part in re.split(r"[\\/]+", candidate)):
        raise ValueError("path_traversal")
    if not os.path.isabs(candidate):
        candidate = os.path.join(DATA_DIR, candidate)
    abs_path = os.path.abspath(candidate)
    real_path = os.path.realpath(abs_path)
    try:
        common = os.path.commonpath([os.path.normcase(real_path), os.path.normcase(DATA_DIR_REAL)])
    except ValueError as exc:
        raise ValueError("access_denied: Path outside allowlist.") from exc
    if common != os.path.normcase(DATA_DIR_REAL):
        raise ValueError("access_denied: Path outside allowlist.")
    return real_path


def _audio_format_from_path(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError("unsupported_format")
    return ext, ext[1:]


def _safe_audio_duration(path: str) -> Optional[float]:
    try:
        info = sf.info(path)
    except Exception:
        return None
    duration = getattr(info, "duration", None)
    if duration is None:
        return None
    return round(float(duration), 3)


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def _scan_connect_previews() -> List[ConnectSongPreview]:
    indexed: List[Tuple[float, ConnectSongPreview]] = []
    with os.scandir(DATA_DIR) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in ALLOWED_AUDIO_EXTENSIONS:
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            modified = float(st.st_mtime)
            indexed.append(
                (
                    modified,
                    ConnectSongPreview(
                        filename=entry.name,
                        size_bytes=int(st.st_size),
                        format=ext[1:],
                        duration_seconds=_safe_audio_duration(entry.path),
                        modified_at=_iso_utc(modified),
                    ),
                )
            )
    indexed.sort(key=lambda item: (-item[0], item[1].filename.lower()))
    return [item[1] for item in indexed]


def _build_connect_packet(preview_limit: int = CONNECT_PREVIEW_LIMIT) -> ConnectPacketOut:
    limit = max(1, int(preview_limit))
    catalog = _scan_connect_previews()
    preview = catalog[:limit]
    sample_path = preview[0].filename if preview else "song.wav"
    recommended = "register_from_data" if catalog else "upload_then_master"

    workflow_steps = [
        "1. get_connect_packet or read auralmind://connect-kit",
        "2. list_data_audio",
        "3. register_audio_from_path",
        "4. analyze_audio",
        "5. plan_mastering_strategy or propose_master_settings",
        "6. run_master_job (or master_closed_loop)",
        "7. job_status + job_result + read_artifact",
    ]
    if not catalog:
        workflow_steps.insert(2, "3. upload_init -> upload_chunk -> upload_finalize")

    example_calls: Dict[str, Any] = {
        "list_data_audio": {},
        "register_audio_from_path": {"path": sample_path},
        "analyze_audio": {"audio_id": "aud_1234567890ab"},
        "plan_mastering_strategy": {
            "audio_id": "aud_1234567890ab",
            "goal": "Wide, punchy streaming master with tight low end",
            "platform": "spotify",
            "control_profile": {
                "spatial_width": 0.45,
                "low_end_focus": 0.5,
            },
        },
        "run_master_job": {
            "audio_id": "aud_1234567890ab",
            "preset_name": "hi_fi_streaming",
            "target_lufs": -12.4,
            "warmth": 0.2,
            "transient_boost_db": 2.1,
            "enable_harshness_limiter": True,
            "enable_air_motion": True,
            "bit_depth": "float32",
            "control_profile": {
                "spatial_width": 0.45,
                "brightness_tilt": 0.25,
                "movement_amount": 0.35,
                "low_end_focus": 0.5,
            },
        },
        "job_status": {"job_id": "job_1234567890ab"},
        "job_result": {"job_id": "job_1234567890ab"},
        "master_closed_loop": {
            "audio_id": "aud_1234567890ab",
            "goal": "Streaming-ready, clear and punchy",
            "platform": "spotify",
            "control_profile": {
                "spatial_width": 0.3,
                "movement_amount": 0.2,
            },
        },
    }
    if not catalog:
        example_calls["upload_init"] = {"filename": "song.wav", "total_bytes": 12345678, "sha256": "<sha256>"}
        example_calls["upload_chunk"] = {"upload_id": "upl_1234567890ab", "index": 0, "chunk_b64": "<base64-chunk>"}
        example_calls["upload_finalize"] = {"upload_id": "upl_1234567890ab"}

    return ConnectPacketOut(
        generated_at=_iso_utc(time.time()),
        preview_limit=limit,
        total_songs=len(catalog),
        songs_preview=preview,
        recommended_first_path=recommended,
        workflow_steps=workflow_steps,
        example_calls=example_calls,
    )


def _tool_catalog_entries() -> List[ToolCatalogEntry]:
    return [
        ToolCatalogEntry(name="bootstrap", description="Discovery", input_model="Empty", output_model="BootstrapOut"),
        ToolCatalogEntry(name="capabilities", description="Server capability summary", input_model="Empty", output_model="CapabilitiesOut"),
        ToolCatalogEntry(name="get_connect_packet", description="First-contact packet with song preview and next calls", input_model="Empty", output_model="ConnectPacketOut"),
        ToolCatalogEntry(name="list_audio_assets", description="List audio files in the data directory", input_model="Empty", output_model="AudioAssetList"),
        ToolCatalogEntry(name="list_data_audio", description="Compatibility alias for list_audio_assets", input_model="Empty", output_model="AudioAssetList"),
        ToolCatalogEntry(name="register_audio_from_path", description="Register a server-side audio file from data/", input_model="RegisterAudioPathIn", output_model="RegisterAudioResult"),
        ToolCatalogEntry(name="upload_init", description="Start a resumable upload session", input_model="UploadInitIn", output_model="UploadInitOut"),
        ToolCatalogEntry(name="upload_chunk", description="Append a chunk to a resumable upload", input_model="UploadChunkIn", output_model="UploadStatusOut"),
        ToolCatalogEntry(name="upload_status", description="Inspect resumable upload progress", input_model="upload_id:string", output_model="UploadStatusOut"),
        ToolCatalogEntry(name="upload_finalize", description="Finalize a resumable upload into an audio handle", input_model="UploadFinalizeIn", output_model="UploadResult"),
        ToolCatalogEntry(name="upload_audio_to_session", description="Legacy one-shot upload", input_model="UploadIn", output_model="UploadResult"),
        ToolCatalogEntry(name="analyze_audio", description="Analyze a source audio handle", input_model="AnalyzeIn", output_model="AudioMetrics"),
        ToolCatalogEntry(name="list_presets", description="List mastering presets", input_model="Empty", output_model="PresetsOut"),
        ToolCatalogEntry(name="plan_mastering_strategy", description="Resolve semantic mastering intent into executable settings", input_model="StrategyPlanIn", output_model="StrategyPlanOut"),
        ToolCatalogEntry(name="propose_master_settings", description="Validate and normalize mastering settings", input_model="MasterSettings", output_model="ProposedSettingsOut"),
        ToolCatalogEntry(name="run_master_job", description="Queue an async mastering job", input_model="MasterRequest", output_model="JobLaunchOut"),
        ToolCatalogEntry(name="job_status", description="Poll a mastering job", input_model="JobIdIn", output_model="JobStatusOut"),
        ToolCatalogEntry(name="job_result", description="Fetch completed mastering job results", input_model="JobIdIn", output_model="JobResultOut"),
        ToolCatalogEntry(name="master_audio", description="Run a single-pass master immediately", input_model="MasterRequest", output_model="MasterResult"),
        ToolCatalogEntry(name="master_closed_loop", description="Run expert multi-pass mastering", input_model="ClosedLoopRequest", output_model="ClosedLoopResult"),
        ToolCatalogEntry(name="read_artifact", description="Read artifact bytes in chunks", input_model="ArtifactReadIn", output_model="ArtifactReadResult"),
        ToolCatalogEntry(name="safe_read_text", description="Read a text file inside the allowlist", input_model="FileReadIn", output_model="FileReadOut"),
        ToolCatalogEntry(name="safe_write_text", description="Write a text file inside the allowlist", input_model="FileWriteIn", output_model="FileWriteOut"),
        ToolCatalogEntry(name="cancel_job", description="Cancel a queued or running job", input_model="CancelJobIn", output_model="CancelJobOut"),
        ToolCatalogEntry(name="delete_artifact", description="Delete a stored artifact", input_model="DeleteArtifactIn", output_model="DeleteArtifactOut"),
        ToolCatalogEntry(name="compare_audio_metrics", description="Compare metrics from two analysis or artifact handles", input_model="CompareMetricsIn", output_model="CompareMetricsOut"),
        ToolCatalogEntry(name="apply_musical_eq", description="Apply key-aware resonant EQ", input_model="MusicalEqIn", output_model="MusicalEqOut"),
        ToolCatalogEntry(name="apply_tempo_dynamics", description="Apply tempo-synced groove compression", input_model="TempoDynamicsIn", output_model="TempoDynamicsOut"),
        ToolCatalogEntry(name="apply_harmonic_excitation", description="Apply harmonic saturation", input_model="HarmonicExcitationIn", output_model="HarmonicExcitationOut"),
        ToolCatalogEntry(name="start_interactive_mastering", description="Start an interactive stage-1 master", input_model="StartInteractiveMasteringIn", output_model="StartInteractiveMasteringOut"),
        ToolCatalogEntry(name="commit_interactive_mastering", description="Commit interactive mastering tweaks", input_model="CommitInteractiveMasteringIn", output_model="CommitInteractiveMasteringOut"),
        ToolCatalogEntry(name="semantic_a_b_mastering", description="Run semantic A/B mastering variants", input_model="SemanticABMasteringIn", output_model="SemanticABMasteringOut"),
        ToolCatalogEntry(name="analyze_and_optimize_governor", description="Optimize governor loops via crest analysis", input_model="AnalyzeAndOptimizeGovernorIn", output_model="AnalyzeAndOptimizeGovernorOut"),
        ToolCatalogEntry(name="ai_stem_remix", description="Analyze Demucs stems for mix intervention", input_model="AiStemRemixIn", output_model="AiStemRemixOut"),
    ]


def _resource_catalog_entries() -> List[ResourceCatalogEntry]:
    return [
        ResourceCatalogEntry(uri="auralmind://connect-kit", description="Connect-time discovery payload", mime_type="application/json", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="config://system-prompt", description="System prompt", mime_type="text/markdown", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="config://mcp-docs", description="Usage docs", mime_type="text/markdown", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="config://maintainer-guide", description="Maintainer architecture guide", mime_type="text/markdown", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="config://server-info", description="Server limits and transport metadata", mime_type="application/json", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="auralmind://workflow", description="Workflow steps", mime_type="application/json", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="auralmind://metrics", description="Metrics and scoring thresholds", mime_type="application/json", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="auralmind://presets", description="Preset guide", mime_type="application/json", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="auralmind://control-surface", description="Bounded LLM control profile and precedence", mime_type="application/json", annotations={"readOnlyHint": True, "idempotentHint": True}),
        ResourceCatalogEntry(uri="auralmind://contracts", description="Tool contracts", mime_type="application/json", annotations={"readOnlyHint": True, "idempotentHint": True}),
    ]


def _prompt_catalog_entries() -> List[PromptCatalogEntry]:
    return [
        PromptCatalogEntry(name="on_connect", description="Client onboarding", args_schema={}),
        PromptCatalogEntry(name="master_once", description="Single-pass plan", args_schema={"file_uri": "string", "goal": "string", "platform": "string"}),
        PromptCatalogEntry(name="master_closed_loop_prompt", description="Closure plan", args_schema={"file_uri": "string", "goal": "string", "platform": "string"}),
        PromptCatalogEntry(name="generate-mastering-strategy", description="Strategy generator", args_schema={"integrated_lufs": "float", "crest_db": "float", "platform": "string"}),
    ]


def _tool_contract_map() -> Dict[str, Dict[str, str]]:
    return {
        entry.name: {"input": entry.input_model, "output": entry.output_model}
        for entry in _tool_catalog_entries()
    }


def _bootstrap_example_calls(packet: ConnectPacketOut) -> Dict[str, Any]:
    examples = {
        "bootstrap": {"method": "tools/call", "params": {"name": "bootstrap", "arguments": {}}},
        "capabilities": {"method": "tools/call", "params": {"name": "capabilities", "arguments": {}}},
        "get_connect_packet": {"method": "tools/call", "params": {"name": "get_connect_packet", "arguments": {}}},
        "list_presets": {"method": "tools/call", "params": {"name": "list_presets", "arguments": {}}},
        "plan_mastering_strategy": {
            "method": "tools/call",
            "params": {
                "name": "plan_mastering_strategy",
                "arguments": {
                    "audio_id": "aud_1234567890ab",
                    "goal": "Wide, controlled, streaming-ready master with punchy lows",
                    "platform": "spotify",
                },
            },
        },
        "propose_master_settings": {
            "method": "tools/call",
            "params": {
                "name": "propose_master_settings",
                "arguments": {
                    "preset_name": "hi_fi_streaming",
                    "target_lufs": -12.4,
                    "warmth": 0.2,
                    "transient_boost_db": 2.1,
                    "enable_harshness_limiter": True,
                    "enable_air_motion": True,
                    "bit_depth": "float32",
                    "control_profile": {
                        "spatial_width": 0.45,
                        "brightness_tilt": 0.25,
                    },
                },
            },
        },
        "master_once": {
            "method": "tools/call",
            "params": {
                "name": "master_audio",
                "arguments": {
                    "audio_id": "aud_1234567890ab",
                    "preset_name": "hi_fi_streaming",
                },
            },
        },
    }
    for name, arguments in packet.example_calls.items():
        examples[name] = {"method": "tools/call", "params": {"name": name, "arguments": arguments}}
    return examples


def _decode_base64_payload(payload_b64: str) -> bytes:
    compact = re.sub(r"\s+", "", payload_b64 or "")
    if not compact:
        raise ValueError("missing_payload")
    if len(compact) > MAX_UPLOAD_B64_CHARS:
        raise ValueError("payload_too_large")
    try:
        return base64.b64decode(compact, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid_base64") from exc


def _decode_hex_payload(payload_hex: str) -> bytes:
    compact = re.sub(r"\s+", "", (payload_hex or "").strip())
    if compact.startswith("0x"):
        compact = compact[2:]
    if not compact:
        raise ValueError("missing_payload")
    if len(compact) > MAX_UPLOAD_HEX_CHARS:
        raise ValueError("payload_too_large")
    if len(compact) % 2 != 0:
        raise ValueError("invalid_hex_length")
    try:
        return binascii.unhexlify(compact)
    except binascii.Error as exc:
        raise ValueError("invalid_hex") from exc


def _uploads_root(session_dir: str) -> str:
    root = os.path.join(session_dir, ".uploads")
    os.makedirs(root, exist_ok=True)
    return root


def _upload_meta_path(session_dir: str, upload_id: str) -> str:
    return os.path.join(_uploads_root(session_dir), f"{upload_id}.json")


def _upload_part_path(session_dir: str, upload_id: str) -> str:
    return os.path.join(_uploads_root(session_dir), f"{upload_id}.part")


def _save_upload_meta(session_dir: str, upload_id: str, meta: Dict[str, Any]) -> None:
    meta_path = _upload_meta_path(session_dir, upload_id)
    tmp_path = f"{meta_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, meta_path)


def _load_upload_meta(session_dir: str, upload_id: str) -> Dict[str, Any]:
    meta_path = _upload_meta_path(session_dir, upload_id)
    if not os.path.exists(meta_path):
        raise ValueError("not_found: Upload not found.")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _delete_upload_meta(session_dir: str, upload_id: str) -> None:
    meta_path = _upload_meta_path(session_dir, upload_id)
    if os.path.exists(meta_path):
        os.remove(meta_path)


def _upload_status_from_meta(meta: Dict[str, Any]) -> UploadStatusOut:
    total = int(meta["total_bytes"])
    received = int(meta["received_bytes"])
    return UploadStatusOut(
        upload_id=str(meta["upload_id"]),
        filename=str(meta["filename"]),
        total_bytes=total,
        received_bytes=received,
        next_index=int(meta["next_index"]),
        done=received >= total,
        expected_sha256=meta.get("sha256"),
    )


def _decode_base64_chunk(chunk_b64: str) -> bytes:
    compact = re.sub(r"\s+", "", chunk_b64 or "")
    if not compact:
        raise ValueError("missing_chunk")
    if len(compact) > MAX_UPLOAD_CHUNK_B64_CHARS:
        raise ValueError("chunk_too_large")
    try:
        return base64.b64decode(compact, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid_base64_chunk") from exc


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
    sid = getattr(ctx, "session_id", None) if ctx is not None else None
    if sid:
        sid = str(sid)
        # Store by a short hash instead of raw session_id to avoid leaking client IDs to disk paths.
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
    # Keep a memory cache for fast lookups, but also persist metadata so handles survive process restarts.
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
    # Fast path from in-memory cache.
    with _ARTIFACTS_LOCK:
        cached = _ARTIFACTS.get(session_key, {}).get(artifact_id)
    if cached is not None:
        return cached

    # Fallback to persisted JSON metadata written during registration.
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


def _store_file_from_path(
    session_key: str,
    session_dir: str,
    *,
    artifact_id: str,
    kind: str,
    filename: str,
    source_path: str,
    media_type: str,
) -> "ArtifactEntry":
    safe_name = _sanitize_filename(filename)
    ext = os.path.splitext(filename)[1].lower() or ".bin"
    data_filename = f"{artifact_id}{ext}"
    data_path = _artifact_data_path(session_dir, data_filename)
    sha = hashlib.sha256()
    size_bytes = 0
    with open(source_path, "rb") as src, open(data_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            size_bytes += len(chunk)
            sha.update(chunk)
    entry = ArtifactEntry(
        artifact_id=artifact_id,
        kind=kind,
        filename=safe_name,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=sha.hexdigest(),
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


def _make_error(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorEnvelope:
    return ErrorEnvelope(code=code, message=message, details=details)


def _get_job(job_id: str) -> Optional[JobState]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return replace(job) if job else None


def _update_job(job_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        for key, value in updates.items():
            if key == "progress":
                value = max(0, min(100, int(value)))
            setattr(job, key, value)


def _job_elapsed(job: JobState) -> float:
    start = job.started_at or job.created_at
    end = job.finished_at or time.time()
    return max(0.0, end - start)


def _artifact_summary(entry: ArtifactEntry) -> ArtifactSummary:
    return ArtifactSummary(
        artifact_id=entry.artifact_id,
        filename=entry.filename,
        media_type=entry.media_type,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
    )


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


def _serialize_preset(preset: Any, *, include_extended: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "target_lufs": float(getattr(preset, "target_lufs", -12.0)),
        "ceiling_dbfs": float(getattr(preset, "ceiling_dbfs", -1.0)),
        "limiter_mode": str(getattr(preset, "limiter_mode", "v2")),
        "governor_gr_limit_db": float(getattr(preset, "governor_gr_limit_db", -3.0)),
        "match_strength": float(getattr(preset, "match_strength", 0.5)),
        "enable_harshness_limiter": bool(getattr(preset, "enable_harshness_limiter", True)),
        "enable_air_motion": bool(getattr(preset, "enable_air_motion", True)),
        "bit_depth": str(getattr(preset, "bit_depth", "float32")),
    }
    if include_extended:
        payload.update(
            {
                "warmth": float(getattr(preset, "warmth", 0.0)),
                "transient_boost_db": float(getattr(preset, "transient_sculpt_boost_db", 0.0)),
                "width_mid": float(getattr(preset, "width_mid", 1.0)),
                "width_hi": float(getattr(preset, "width_hi", 1.0)),
                "air_motion_mix": float(getattr(preset, "air_motion_mix", 0.0)),
                "harshness_max_cut_db": float(getattr(preset, "harshness_max_cut_db", 0.0)),
                "movement_amount": float(getattr(preset, "movement_amount", 0.0)),
                "hooklift_mix": float(getattr(preset, "hooklift_mix", 0.0)),
                "mono_sub_base_mix": float(getattr(preset, "mono_sub_base_mix", 0.55)),
                "enable_stem_separation": bool(getattr(preset, "enable_stem_separation", False)),
            }
        )
    return payload


def _fields_set(model: BaseModel) -> set[str]:
    return set(getattr(model, "model_fields_set", set()))


def _normalize_control_profile(
    profile: Optional[MasteringControlProfile],
) -> Optional[MasteringControlProfile]:
    if profile is None:
        return None
    values: Dict[str, float] = {}
    for field_name in CONTROL_PROFILE_FIELDS:
        value = getattr(profile, field_name, None)
        if value is None:
            continue
        values[field_name] = round(max(-1.0, min(1.0, float(value))), 3)
    if not values:
        return None
    return MasteringControlProfile(**values)


def _merge_control_profiles(
    *profiles: Optional[MasteringControlProfile],
) -> Optional[MasteringControlProfile]:
    merged: Dict[str, float] = {}
    for profile in profiles:
        normalized = _normalize_control_profile(profile)
        if normalized is None:
            continue
        for field_name in CONTROL_PROFILE_FIELDS:
            value = getattr(normalized, field_name, None)
            if value is not None:
                merged[field_name] = value
    if not merged:
        return None
    return MasteringControlProfile(**merged)


def _master_settings_from_preset(preset: Any) -> MasterSettings:
    return MasterSettings(
        preset_name=str(getattr(preset, "name", "hi_fi_streaming")),
        target_lufs=float(getattr(preset, "target_lufs", -12.0)),
        warmth=float(getattr(preset, "warmth", 0.0)),
        transient_boost_db=float(getattr(preset, "transient_sculpt_boost_db", 1.0)),
        enable_harshness_limiter=bool(getattr(preset, "enable_harshness_limiter", True)),
        enable_air_motion=bool(getattr(preset, "enable_air_motion", True)),
        bit_depth=str(getattr(preset, "bit_depth", "float32")),
        control_profile=None,
        governor_search_steps=None,
        governor_gr_limit_db=None,
        stem_gains_db=None,
    )


def _apply_control_profile_to_settings(
    settings: MasterSettings,
    profile: MasteringControlProfile,
) -> MasterSettings:
    updated = settings.model_copy(deep=True)
    spatial = float(profile.spatial_width or 0.0)
    brightness = float(profile.brightness_tilt or 0.0)
    harshness = float(profile.harshness_control or 0.0)
    movement = float(profile.movement_amount or 0.0)
    low_end = float(profile.low_end_focus or 0.0)

    updated.warmth = max(0.0, min(1.0, updated.warmth + (low_end * 0.18) - (brightness * 0.08)))
    updated.transient_boost_db = max(
        0.0,
        min(4.0, updated.transient_boost_db + (movement * 0.65) + (brightness * 0.15)),
    )
    if spatial >= 0.1:
        updated.enable_air_motion = True
    elif spatial <= -0.75:
        updated.enable_air_motion = False

    if harshness >= -0.2 or brightness >= 0.6:
        updated.enable_harshness_limiter = True
    elif harshness <= -0.75:
        updated.enable_harshness_limiter = False

    updated.control_profile = profile
    return updated


def _normalize_master_settings(settings: MasterSettings) -> MasterSettings:
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    if settings.preset_name not in presets:
        raise ValueError(f"unknown_preset: {settings.preset_name}")

    bit_depth = str(settings.bit_depth)
    if bit_depth not in ("float32", "float64"):
        raise ValueError("invalid_bit_depth")

    governor_steps = settings.governor_search_steps
    if governor_steps is not None:
        governor_steps = max(1, min(16, int(governor_steps)))

    governor_gr_limit = settings.governor_gr_limit_db
    if governor_gr_limit is not None:
        governor_gr_limit = round(max(-6.0, min(-0.4, float(governor_gr_limit))), 2)

    stem_gains: Optional[Dict[str, float]] = None
    if settings.stem_gains_db:
        stem_gains = {
            str(name): round(max(-12.0, min(12.0, float(gain))), 2)
            for name, gain in settings.stem_gains_db.items()
        }

    return MasterSettings(
        preset_name=str(settings.preset_name),
        target_lufs=round(max(-20.0, min(-6.0, float(settings.target_lufs))), 2),
        warmth=round(max(0.0, min(1.0, float(settings.warmth))), 3),
        transient_boost_db=round(max(0.0, min(4.0, float(settings.transient_boost_db))), 3),
        enable_harshness_limiter=bool(settings.enable_harshness_limiter),
        enable_air_motion=bool(settings.enable_air_motion),
        bit_depth=bit_depth,  # type: ignore[arg-type]
        control_profile=_normalize_control_profile(settings.control_profile),
        governor_search_steps=governor_steps,
        governor_gr_limit_db=governor_gr_limit,
        stem_gains_db=stem_gains,
    )


def _finalize_master_settings(
    preset_name: str,
    *,
    semantic_overrides: Optional[Dict[str, Any]] = None,
    control_profile: Optional[MasteringControlProfile] = None,
    explicit_overrides: Optional[Dict[str, Any]] = None,
    safe_overrides: Optional[Dict[str, Any]] = None,
) -> MasterSettings:
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    if preset_name not in presets:
        raise ValueError(f"unknown_preset: {preset_name}")

    working = _master_settings_from_preset(presets[preset_name])
    if semantic_overrides:
        for field_name, value in semantic_overrides.items():
            setattr(working, field_name, value)

    normalized_profile = _normalize_control_profile(control_profile)
    if normalized_profile is not None:
        working = _apply_control_profile_to_settings(working, normalized_profile)

    if explicit_overrides:
        for field_name, value in explicit_overrides.items():
            setattr(working, field_name, value)

    if safe_overrides:
        for field_name, value in safe_overrides.items():
            setattr(working, field_name, value)

    working.control_profile = normalized_profile
    return _normalize_master_settings(working)


def _build_master_settings(
    *,
    preset_name: str,
    target_lufs: float,
    warmth: float,
    transient_boost_db: float,
    enable_harshness_limiter: bool,
    enable_air_motion: bool,
    bit_depth: BitDepth,
    control_profile: Optional[MasteringControlProfile] = None,
    governor_search_steps: Optional[int] = None,
    governor_gr_limit_db: Optional[float] = None,
    stem_gains_db: Optional[Dict[str, float]] = None,
) -> MasterSettings:
    return _finalize_master_settings(
        str(preset_name),
        explicit_overrides={
            "target_lufs": target_lufs,
            "warmth": warmth,
            "transient_boost_db": transient_boost_db,
            "enable_harshness_limiter": enable_harshness_limiter,
            "enable_air_motion": enable_air_motion,
            "bit_depth": bit_depth,
        },
        control_profile=control_profile,
        safe_overrides={
            "governor_search_steps": governor_search_steps,
            "governor_gr_limit_db": governor_gr_limit_db,
            "stem_gains_db": stem_gains_db,
        },
    )


def _master_request_from_settings(audio_id: str, settings: MasterSettings) -> MasterRequest:
    return MasterRequest(audio_id=audio_id, **settings.model_dump())


def _resolve_settings_from_request(settings: MasterSettings) -> MasterSettings:
    explicit_fields = _fields_set(settings)
    explicit_overrides = {
        field_name: getattr(settings, field_name)
        for field_name in VISIBLE_MASTER_FIELDS
        if field_name in explicit_fields and field_name != "preset_name"
    }
    safe_overrides = {
        field_name: getattr(settings, field_name)
        for field_name in SAFE_OVERRIDE_FIELDS
        if field_name in explicit_fields
    }
    return _finalize_master_settings(
        preset_name=str(settings.preset_name),
        control_profile=settings.control_profile if "control_profile" in explicit_fields else None,
        explicit_overrides=explicit_overrides,
        safe_overrides=safe_overrides,
    )


def _goal_text(goal: str) -> str:
    return re.sub(r"\s+", " ", str(goal or "").strip().lower())


def _goal_has(goal: str, *terms: str) -> bool:
    goal_text = _goal_text(goal)
    return any(term in goal_text for term in terms)


def _platform_target_for_preset(platform: Platform, base_target_lufs: float) -> float:
    lo, hi = PLATFORM_TARGET_RANGES[str(platform)]
    default_target = PLATFORM_TARGET_DEFAULTS[str(platform)]
    blended = (float(base_target_lufs) + default_target) / 2.0
    return round(max(lo, min(hi, blended)), 2)


def _build_semantic_control_profile(
    goal: str,
    platform: Platform,
    metrics: AudioMetrics,
) -> Tuple[Optional[MasteringControlProfile], List[str], List[str]]:
    values = {field_name: 0.0 for field_name in CONTROL_PROFILE_FIELDS}
    reasoning: List[str] = []
    warnings: List[str] = []
    goal_text = _goal_text(goal)

    if _goal_has(goal_text, "wide", "3d", "depth", "spacious", "atmospheric", "cinematic", "immersive", "air"):
        values["spatial_width"] += 0.7
        reasoning.append("Goal language requests a wider or more dimensional stereo field.")
    if _goal_has(goal_text, "mono", "focused", "tight", "centered"):
        values["spatial_width"] -= 0.45
        reasoning.append("Goal language asks for a tighter or more center-focused image.")

    if _goal_has(goal_text, "bright", "crisp", "air", "sheen", "open", "forward"):
        values["brightness_tilt"] += 0.6
        reasoning.append("Goal language asks for extra brightness, sheen, or forward presence.")
    if _goal_has(goal_text, "dark", "warm", "smooth", "round", "vintage"):
        values["brightness_tilt"] -= 0.55
        reasoning.append("Goal language asks for a darker, warmer, or smoother top end.")

    if _goal_has(goal_text, "smooth", "de-ess", "less harsh", "soft", "gentle", "fatigue"):
        values["harshness_control"] += 0.65
        reasoning.append("Goal language prioritizes fatigue control and smoother upper mids.")
    if _goal_has(goal_text, "aggressive", "edgy", "bite", "cut", "attack"):
        values["harshness_control"] -= 0.25
        reasoning.append("Goal language tolerates more upper-mid edge for aggression and cut.")

    if _goal_has(goal_text, "movement", "lift", "hook", "bounce", "energetic", "club", "anthem", "pump"):
        values["movement_amount"] += 0.7
        reasoning.append("Goal language asks for more movement, lift, or club energy.")
    if _goal_has(goal_text, "static", "natural", "transparent", "restrained"):
        values["movement_amount"] -= 0.35
        reasoning.append("Goal language prefers restrained automation and less macro motion.")

    if _goal_has(goal_text, "808", "sub", "bass", "low end", "knock", "weight", "trap"):
        values["low_end_focus"] += 0.7
        reasoning.append("Goal language emphasizes bass weight, 808 control, or low-end punch.")
    if _goal_has(goal_text, "lean", "light", "tight low end", "clean low end"):
        values["low_end_focus"] -= 0.4
        reasoning.append("Goal language asks for a leaner or cleaner low-end presentation.")

    if metrics.centroid_hz is not None and metrics.centroid_hz > 4200:
        values["harshness_control"] += 0.3
        warnings.append("Measured centroid is bright; the plan adds extra harshness protection.")
    if metrics.stereo_correlation < 0.08:
        values["spatial_width"] = min(values["spatial_width"], 0.2)
        warnings.append("Stereo correlation is already fragile; width expansion is intentionally capped.")
    if metrics.crest_db > 12.0:
        values["movement_amount"] -= 0.2
        reasoning.append("High crest factor suggests preserving transient openness instead of adding extra movement.")
    if metrics.crest_db < 8.0:
        values["movement_amount"] += 0.2
        reasoning.append("Low crest factor suggests adding a touch more punch and motion.")
    if str(platform) == "club":
        values["low_end_focus"] += 0.2
        values["movement_amount"] += 0.15
        reasoning.append("Club targeting increases low-end focus and movement by default.")

    normalized_values = {
        field_name: round(max(-1.0, min(1.0, value)), 3)
        for field_name, value in values.items()
    }
    if all(abs(value) < 0.01 for value in normalized_values.values()):
        return None, reasoning, warnings
    return MasteringControlProfile(**normalized_values), reasoning, warnings


def _choose_semantic_preset(
    goal: str,
    platform: Platform,
    metrics: AudioMetrics,
    maestro: Any,
) -> Tuple[str, str]:
    features = {
        "lufs": metrics.integrated_lufs,
        "tp_dbfs": metrics.true_peak_dbtp,
        "crest_db": metrics.crest_db,
        "centroid_hz": metrics.centroid_hz or 2800.0,
    }

    if str(platform) == "club" or _goal_has(goal, "club", "festival", "anthem", "dancefloor", "banger"):
        if _goal_has(goal, "trap", "808", "sub", "bass"):
            return "competitive_trap", "Trap or sub-heavy goal language points to the competitive_trap preset."
        return "club_clean", "Club-oriented goal language points to the club_clean preset."
    if _goal_has(goal, "cinematic", "film", "epic", "spacious", "wide", "3d", "atmospheric"):
        return "cinematic", "Cinematic or spacious goal language points to the cinematic preset."
    if _goal_has(goal, "radio", "commercial", "pop", "vocal", "hook"):
        return "radio_loud", "Commercial or vocal-forward goal language points to the radio_loud preset."
    if _goal_has(goal, "intimate", "acoustic", "organic", "dynamic", "natural", "open"):
        return "hi_fi_streaming", "Dynamic or organic goal language points to the hi_fi_streaming preset."

    auto_name = str(maestro.auto_select_preset_name(features))
    return auto_name, f"No explicit semantic preset match was found, so auto_select_preset_name chose {auto_name}."


def _build_semantic_overrides(
    goal: str,
    platform: Platform,
    metrics: AudioMetrics,
    preset: Any,
    control_profile: Optional[MasteringControlProfile],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    reasoning = [PLATFORM_POLICY_NOTES[str(platform)]]
    warnings: List[str] = []
    goal_text = _goal_text(goal)

    target_lufs = _platform_target_for_preset(platform, float(getattr(preset, "target_lufs", -12.0)))
    if _goal_has(goal_text, "loud", "competitive", "aggressive", "club", "anthem"):
        target_lufs += 0.35
        reasoning.append("Goal language leans competitive, so the target LUFS is nudged hotter.")
    if _goal_has(goal_text, "dynamic", "open", "organic", "acoustic", "natural"):
        target_lufs -= 0.45
        reasoning.append("Goal language asks for openness, so the target LUFS is nudged lower.")
    if metrics.true_peak_dbtp > -0.4:
        target_lufs -= 0.25
        warnings.append("Input true peak is already hot; the plan trims loudness slightly for safer headroom.")
    if metrics.centroid_hz is not None and metrics.centroid_hz > 4500:
        target_lufs -= 0.15
        warnings.append("Bright source material receives a small loudness backoff to protect harshness.")
    target_lufs = round(
        max(PLATFORM_TARGET_RANGES[str(platform)][0], min(PLATFORM_TARGET_RANGES[str(platform)][1], target_lufs)),
        2,
    )

    warmth = float(getattr(preset, "warmth", 0.0))
    if _goal_has(goal_text, "warm", "analog", "round", "dark", "intimate"):
        warmth += 0.18
        reasoning.append("Goal language requests more warmth or intimacy.")
    if _goal_has(goal_text, "clean", "bright", "crisp", "modern"):
        warmth -= 0.08
        reasoning.append("Goal language favors a cleaner or more modern tonal balance.")

    transient_boost = float(getattr(preset, "transient_sculpt_boost_db", 1.8))
    if _goal_has(goal_text, "punch", "impact", "attack", "slam"):
        transient_boost += 0.35
        reasoning.append("Goal language asks for more punch or attack.")
    if _goal_has(goal_text, "soft", "glue", "smooth"):
        transient_boost -= 0.2
        reasoning.append("Goal language prefers a smoother or more glued envelope.")
    if metrics.crest_db > 12.5:
        transient_boost -= 0.15
        reasoning.append("High crest factor suggests a lighter transient push to preserve openness.")
    elif metrics.crest_db < 8.0:
        transient_boost += 0.2
        reasoning.append("Low crest factor suggests a small transient lift to recover punch.")

    enable_harshness_limiter = bool(getattr(preset, "enable_harshness_limiter", True))
    if control_profile is not None and (control_profile.harshness_control or 0.0) >= -0.2:
        enable_harshness_limiter = True
    if _goal_has(goal_text, "raw", "edgy") and not _goal_has(goal_text, "smooth", "soft"):
        reasoning.append("The plan keeps some upper-mid edge because the goal language allows aggression.")

    enable_air_motion = bool(getattr(preset, "enable_air_motion", True))
    if control_profile is not None and (control_profile.spatial_width or 0.0) <= -0.75:
        enable_air_motion = False

    return (
        {
            "target_lufs": target_lufs,
            "warmth": warmth,
            "transient_boost_db": transient_boost,
            "enable_harshness_limiter": enable_harshness_limiter,
            "enable_air_motion": enable_air_motion,
            "bit_depth": str(getattr(preset, "bit_depth", "float32")),
        },
        reasoning,
        warnings,
    )


def _plan_mastering_strategy_internal(
    req: StrategyPlanIn,
    ctx: Context = None,
    *,
    session_key: Optional[str] = None,
    session_dir: Optional[str] = None,
) -> StrategyPlanOut:
    if session_key is None or session_dir is None:
        session_key, session_dir = _get_session_info(ctx)
    metrics = _analyze_internal(req.audio_id, ctx, session_key=session_key, session_dir=session_dir)
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])

    chosen_preset, preset_reason = _choose_semantic_preset(req.goal, req.platform, metrics, maestro)
    presets = maestro.get_presets()
    semantic_profile, profile_reasoning, profile_warnings = _build_semantic_control_profile(
        req.goal,
        req.platform,
        metrics,
    )
    merged_profile = _merge_control_profiles(semantic_profile, req.control_profile)
    semantic_overrides, semantic_reasoning, semantic_warnings = _build_semantic_overrides(
        req.goal,
        req.platform,
        metrics,
        presets[chosen_preset],
        merged_profile,
    )
    if req.control_profile is not None:
        profile_reasoning.append("Caller-supplied control_profile was layered on top of the semantic defaults.")

    safe_overrides = {
        field_name: getattr(req, field_name)
        for field_name in SAFE_OVERRIDE_FIELDS
        if getattr(req, field_name) is not None
    }
    settings = _finalize_master_settings(
        chosen_preset,
        semantic_overrides=semantic_overrides,
        control_profile=merged_profile,
        safe_overrides=safe_overrides,
    )

    warnings = profile_warnings + semantic_warnings
    if req.stem_gains_db:
        warnings.append("Stem gain overrides assume Demucs is available; otherwise the master falls back without stem remixing.")
    if metrics.stereo_correlation < 0.0:
        warnings.append("Negative stereo correlation indicates potential phase issues in the source.")
    if metrics.true_peak_dbtp > -0.1:
        warnings.append("Source true peak is very hot; conservative limiting behavior is recommended.")

    return StrategyPlanOut(
        audio_id=req.audio_id,
        goal=req.goal,
        platform=req.platform,
        metrics=metrics,
        chosen_preset=chosen_preset,
        settings=settings,
        reasoning=[preset_reason] + semantic_reasoning + profile_reasoning,
        warnings=warnings,
    )


def _preset_overrides_from_settings(base_preset: Any, settings: MasterSettings) -> Dict[str, Any]:
    p_args: Dict[str, Any] = {
        "target_lufs": settings.target_lufs,
        "warmth": settings.warmth,
        "transient_sculpt_boost_db": settings.transient_boost_db,
        "enable_harshness_limiter": settings.enable_harshness_limiter,
        "enable_air_motion": settings.enable_air_motion,
        "bit_depth": settings.bit_depth,
    }
    if settings.governor_search_steps is not None:
        p_args["governor_search_steps"] = settings.governor_search_steps
    if settings.governor_gr_limit_db is not None:
        p_args["governor_gr_limit_db"] = settings.governor_gr_limit_db
    if settings.stem_gains_db:
        p_args["stem_gains_db"] = settings.stem_gains_db
        p_args["enable_stem_separation"] = True

    profile = settings.control_profile
    if profile is None:
        return p_args

    spatial = float(profile.spatial_width or 0.0)
    brightness = float(profile.brightness_tilt or 0.0)
    harshness = float(profile.harshness_control or 0.0)
    movement = float(profile.movement_amount or 0.0)
    low_end = float(profile.low_end_focus or 0.0)

    p_args["width_mid"] = round(max(0.95, min(1.16, float(getattr(base_preset, "width_mid", 1.04)) + (spatial * 0.05))), 3)
    p_args["width_hi"] = round(max(1.0, min(1.45, float(getattr(base_preset, "width_hi", 1.24)) + (spatial * 0.12))), 3)
    p_args["microshift_mix"] = round(
        max(0.0, min(0.28, float(getattr(base_preset, "microshift_mix", 0.14)) + (spatial * 0.06))),
        3,
    )
    p_args["air_motion_mix"] = round(
        max(
            0.0,
            min(
                0.24,
                float(getattr(base_preset, "air_motion_mix", 0.12)) + (spatial * 0.05) + (brightness * 0.02),
            ),
        ),
        3,
    )
    p_args["hi_factor"] = round(
        max(0.55, min(0.9, float(getattr(base_preset, "hi_factor", 0.72)) + (brightness * 0.08) - (harshness * 0.03))),
        3,
    )
    p_args["microdetail_amount"] = round(
        max(
            0.0,
            min(
                0.4,
                float(getattr(base_preset, "microdetail_amount", 0.2)) + (brightness * 0.08) - (harshness * 0.03),
            ),
        ),
        3,
    )
    p_args["glow_mix"] = round(
        max(0.2, min(0.75, float(getattr(base_preset, "glow_mix", 0.5)) + (brightness * 0.08))),
        3,
    )
    p_args["harshness_max_cut_db"] = round(
        max(
            0.5,
            min(
                4.0,
                float(getattr(base_preset, "harshness_max_cut_db", 2.0)) + (harshness * 1.2) - (brightness * 0.25),
            ),
        ),
        3,
    )
    p_args["harshness_mix"] = round(
        max(0.2, min(0.9, float(getattr(base_preset, "harshness_mix", 0.6)) + (harshness * 0.18))),
        3,
    )
    p_args["deess_mix"] = round(
        max(0.2, min(0.85, float(getattr(base_preset, "deess_mix", 0.5)) + (harshness * 0.14))),
        3,
    )
    p_args["movement_amount"] = round(
        max(0.0, min(0.35, float(getattr(base_preset, "movement_amount", 0.15)) + (movement * 0.12))),
        3,
    )
    p_args["hooklift_mix"] = round(
        max(0.0, min(0.4, float(getattr(base_preset, "hooklift_mix", 0.2)) + (movement * 0.1))),
        3,
    )
    p_args["transient_sculpt_mix"] = round(
        max(0.1, min(0.55, float(getattr(base_preset, "transient_sculpt_mix", 0.34)) + (movement * 0.08))),
        3,
    )
    p_args["enable_movement"] = p_args["movement_amount"] >= 0.03
    p_args["enable_hooklift"] = p_args["hooklift_mix"] >= 0.03
    p_args["mono_sub_base_mix"] = round(
        max(0.45, min(0.65, float(getattr(base_preset, "mono_sub_base_mix", 0.55)) - (low_end * 0.05))),
        3,
    )
    return p_args


def _calculate_score(metrics: AudioMetrics, target_lufs: float, ceiling: float) -> float:
    """Deterministic scoring of mastering quality. Lower is better."""
    lufs_delta = abs(metrics.integrated_lufs - target_lufs)
    tp_violation = max(0.0, metrics.true_peak_dbtp - (ceiling + 0.1))

    penalty_crest = 0.0
    if metrics.crest_db < 8.0:
        penalty_crest = 8.0 - metrics.crest_db
    elif metrics.crest_db > 12.0:
        penalty_crest = metrics.crest_db - 12.0

    penalty_corr = max(0.0, 0.05 - metrics.stereo_correlation)

    score = (2.0 * lufs_delta) + (5.0 * tp_violation) + (1.5 * penalty_crest) + (2.0 * penalty_corr)
    return round(score, 3)


def _calculate_retune(metrics: AudioMetrics, current: MasterRequest) -> Tuple[MasterRequest, List[TuneDelta]]:
    """Generates a retune plan if Run1 fails thresholds."""
    deltas = []
    # Clone current settings
    next_req = MasterRequest(**current.model_dump())

    # LUFS correction
    lufs_error = next_req.target_lufs - metrics.integrated_lufs
    if abs(lufs_error) > 0.3:
        old = next_req.target_lufs
        # Adjust target to compensate
        next_req.target_lufs = round(next_req.target_lufs + (lufs_error * 0.8), 1)
        deltas.append(TuneDelta(
            param="target_lufs", old_value=old, new_value=next_req.target_lufs,
            reason_code="lufs_drift", reason_detail=f"Correcting {lufs_error:.1f}dB drift"
        ))

    # Peak correction
    if metrics.true_peak_dbtp > -0.1:
        old_warmth = next_req.warmth
        next_req.warmth = max(0.0, next_req.warmth - 0.1)
        deltas.append(TuneDelta(
            param="warmth", old_value=old_warmth, new_value=next_req.warmth,
            reason_code="peak_violation", reason_detail="Reducing warmth to lower crest/peaks"
        ))

    # Crest correction
    if metrics.crest_db < 7.5:
        old_tb = next_req.transient_boost_db
        next_req.transient_boost_db = min(4.0, next_req.transient_boost_db + 0.5)
        deltas.append(TuneDelta(
            param="transient_boost_db", old_value=old_tb, new_value=next_req.transient_boost_db,
            reason_code="low_crest", reason_detail="Increasing transients to recover dynamics"
        ))

    return next_req, deltas


# ---------------------------------------------------------------------------
# Mastering Execution
# ---------------------------------------------------------------------------
def _master_internal(
    audio_id: str,
    req: MasterRequest,
    run_id: str,
    ctx: Context = None, # pyright: ignore[reportArgumentType]
    *,
    session_key: Optional[str] = None,
    session_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> MasterResult:
    """Synchronous mastering execution for a single run."""
    if session_key is None or session_dir is None:
        session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind != "audio":
        raise ValueError(f"not_found: {audio_id}")

    # Metrics before
    if progress_cb:
        progress_cb(10)
    metrics_before = _analyze_internal(audio_id, ctx, session_key=session_key, session_dir=session_dir)

    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])

    # Prepare paths
    if progress_cb:
        progress_cb(30)
    master_wav_id = _new_id("art")
    out_wav_path = os.path.join(session_dir, f"{master_wav_id}.wav")

    # Run maestro
    presets = maestro.get_presets()
    if req.preset_name not in presets:
        raise ValueError(f"unknown_preset: {req.preset_name}")

    # Base preset
    base_p = presets[req.preset_name]
    preset = replace(base_p, **_preset_overrides_from_settings(base_p, req))

    maestro.master(
        target_path=_artifact_data_path(session_dir, entry.data_filename),
        out_path=out_wav_path,
        preset=preset
    )

    # Register output WAV
    if progress_cb:
        progress_cb(70)
    _register_existing_file(
        session_key, session_dir,
        artifact_id=master_wav_id,
        kind="mastered_audio",
        filename=f"{master_wav_id}.wav",
        data_filename=f"{master_wav_id}.wav",
        media_type="audio/wav"
    )

    # Metrics after
    metrics_after = _analyze_internal(master_wav_id, ctx, session_key=session_key, session_dir=session_dir)
    if progress_cb:
        progress_cb(85)

    # Save metrics JSONs as required by spec
    metrics_before_id = _new_id("art")
    metrics_after_id = _new_id("art")
    metrics_payloads = [
        (metrics_before, metrics_before_id, "metrics_before"),
        (metrics_after, metrics_after_id, "metrics_after"),
    ]
    for m, mid, label in metrics_payloads:
        filename = f"{mid}.json"
        with open(os.path.join(session_dir, filename), "w", encoding="utf-8") as f:
            json.dump({"label": label, "metrics": m.model_dump()}, f, indent=2)
        _register_existing_file(
            session_key,
            session_dir,
            artifact_id=mid,
            kind="metrics",
            filename=filename,
            data_filename=filename,
            media_type="application/json",
        )

    # Tuning trace
    trace_id = _new_id("art")
    trace_data = {
        "run_id": run_id,
        "settings": req.model_dump(),
        "metrics_before": metrics_before.model_dump(),
        "metrics_after": metrics_after.model_dump()
    }
    trace_filename = f"{trace_id}.json"
    with open(os.path.join(session_dir, trace_filename), "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
    _register_existing_file(
        session_key,
        session_dir,
        artifact_id=trace_id,
        kind="trace",
        filename=trace_filename,
        data_filename=trace_filename,
        media_type="application/json",
    )
    if progress_cb:
        progress_cb(95)

    return MasterResult(
        run_id=run_id,
        master_wav_id=master_wav_id,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        tuning_trace_id=trace_id,
        artifacts=[master_wav_id, metrics_before_id, metrics_after_id, trace_id],
    )


def _run_master_job_worker(job_id: str) -> None:
    job = _get_job(job_id)
    if job is None:
        return
    if job.settings is None:
        _update_job(
            job_id,
            status="error",
            finished_at=time.time(),
            error=_make_error("job_settings_missing", "Job settings missing."),
        )
        return

    # Explicit lifecycle transitions keep polling deterministic for clients:
    # queued -> running -> done|error.
    _update_job(job_id, status="running", started_at=time.time(), progress=5)
    try:
        req = MasterRequest(audio_id=job.audio_id, **job.settings.model_dump())

        def _progress(pct: int) -> None:
            _update_job(job_id, progress=pct)

        result = _master_internal(
            job.audio_id,
            req,
            run_id=job_id,
            session_key=job.session_key,
            session_dir=job.session_dir,
            progress_cb=_progress,
        )
        _update_job(
            job_id,
            status="done",
            finished_at=time.time(),
            progress=100,
            result=result,
        )
    except Exception as exc:
        log.exception("Master job failed: %s", job_id)
        _update_job(
            job_id,
            status="error",
            finished_at=time.time(),
            progress=100,
            error=_make_error("job_failed", str(exc), {"job_id": job_id}),
        )


# ===========================================================================
# RESOURCES
# ===========================================================================
@mcp.resource(
    uri="auralmind://workflow",
    name="WorkflowSteps",
    description="Ordered steps for mastering.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def get_workflow_resource() -> str:
    packet = _build_connect_packet()
    return json.dumps(
        {
            "workflow": packet.workflow_steps,
            "recommended_first_path": packet.recommended_first_path,
        },
        indent=2,
    )


@mcp.resource(
    uri="auralmind://metrics",
    name="MetricsThresholds",
    description="Scoring thresholds and target metrics.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_metrics_resource() -> str:
    thresholds = {
        "lufs_target_tolerance": 0.7,
        "true_peak_ceiling_tolerance": 0.1,
        "crest_factor_range": [8.0, 12.0],
        "stereo_correlation_min": 0.05,
        "scoring_weights": {
            "lufs_delta": 2.0,
            "true_peak_violation": 5.0,
            "crest_penalty": 1.5,
            "correlation_penalty": 2.0
        }
    }
    return json.dumps(thresholds, indent=2)


@mcp.resource(
    uri="auralmind://control-surface",
    name="ControlSurface",
    description="Bounded LLM control profile and precedence rules.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_control_surface_resource() -> str:
    payload = {
        "control_profile": {
            "range": [-1.0, 1.0],
            "fields": {
                "spatial_width": "Stereo image intent. Negative tightens the image, positive widens it.",
                "brightness_tilt": "Spectral tilt. Negative darkens, positive brightens.",
                "harshness_control": "Upper-mid fatigue control. Negative relaxes protection, positive increases it.",
                "movement_amount": "Macro motion and hook-lift intensity. Negative restrains movement, positive adds lift.",
                "low_end_focus": "Low-end weight and tightness. Negative lightens the low end, positive tightens and emphasizes it.",
            },
        },
        "safe_overrides": {
            "governor_search_steps": {"min": 1, "max": 16},
            "governor_gr_limit_db": {"min": -6.0, "max": -0.4},
            "stem_gains_db": {"min_db": -12.0, "max_db": 12.0},
        },
        "precedence": [
            "base preset",
            "semantic planner",
            "control_profile",
            "explicit master setting fields",
            "safe overrides",
        ],
        "notes": [
            "Use plan_mastering_strategy when starting from natural-language goals.",
            "Use propose_master_settings to validate a fully-specified request before execution.",
            "control_profile is intentionally bounded so LLMs can steer the master without exposing the full raw DSP surface.",
        ],
    }
    return json.dumps(payload, indent=2)


@mcp.resource(
    uri="auralmind://presets",
    name="PresetsAtlas",
    description="Detailed preset guide.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_presets_resource() -> str:
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    payload: Dict[str, Any] = {}
    for name, p in presets.items():
        payload[name] = _serialize_preset(p, include_extended=True)
    return json.dumps({"presets": payload}, indent=2)


@mcp.resource(
    uri="auralmind://contracts",
    name="ToolContracts",
    description="Simplified tool I/O contracts.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_contracts_resource() -> str:
    model_schemas = {
        "CapabilitiesOut": CapabilitiesOut.model_json_schema(),
        "BootstrapOut": BootstrapOut.model_json_schema(),
        "ConnectSongPreview": ConnectSongPreview.model_json_schema(),
        "ConnectPacketOut": ConnectPacketOut.model_json_schema(),
        "AnalyzeIn": AnalyzeIn.model_json_schema(),
        "MasteringControlProfile": MasteringControlProfile.model_json_schema(),
        "UploadIn": UploadIn.model_json_schema(),
        "UploadResult": UploadResult.model_json_schema(),
        "UploadInitIn": UploadInitIn.model_json_schema(),
        "UploadInitOut": UploadInitOut.model_json_schema(),
        "UploadChunkIn": UploadChunkIn.model_json_schema(),
        "UploadFinalizeIn": UploadFinalizeIn.model_json_schema(),
        "UploadStatusOut": UploadStatusOut.model_json_schema(),
        "AudioAssetInfo": AudioAssetInfo.model_json_schema(),
        "AudioAssetList": AudioAssetList.model_json_schema(),
        "RegisterAudioPathIn": RegisterAudioPathIn.model_json_schema(),
        "RegisterAudioResult": RegisterAudioResult.model_json_schema(),
        "AudioMetrics": AudioMetrics.model_json_schema(),
        "PresetsOut": PresetsOut.model_json_schema(),
        "MasterSettings": MasterSettings.model_json_schema(),
        "StrategyPlanIn": StrategyPlanIn.model_json_schema(),
        "StrategyPlanOut": StrategyPlanOut.model_json_schema(),
        "ProposedSettingsOut": ProposedSettingsOut.model_json_schema(),
        "MasterRequest": MasterRequest.model_json_schema(),
        "MasterResult": MasterResult.model_json_schema(),
        "ClosedLoopRequest": ClosedLoopRequest.model_json_schema(),
        "ClosedLoopResult": ClosedLoopResult.model_json_schema(),
        "JobLaunchOut": JobLaunchOut.model_json_schema(),
        "JobIdIn": JobIdIn.model_json_schema(),
        "JobStatusOut": JobStatusOut.model_json_schema(),
        "JobResultOut": JobResultOut.model_json_schema(),
        "ArtifactReadIn": ArtifactReadIn.model_json_schema(),
        "ArtifactReadResult": ArtifactReadResult.model_json_schema(),
        "FileReadIn": FileReadIn.model_json_schema(),
        "FileReadOut": FileReadOut.model_json_schema(),
        "FileWriteIn": FileWriteIn.model_json_schema(),
        "FileWriteOut": FileWriteOut.model_json_schema(),
        "CancelJobIn": CancelJobIn.model_json_schema(),
        "CancelJobOut": CancelJobOut.model_json_schema(),
        "DeleteArtifactIn": DeleteArtifactIn.model_json_schema(),
        "DeleteArtifactOut": DeleteArtifactOut.model_json_schema(),
        "CompareMetricsIn": CompareMetricsIn.model_json_schema(),
        "CompareMetricsOut": CompareMetricsOut.model_json_schema(),
        "AudioMetricsDelta": AudioMetricsDelta.model_json_schema(),
        "MusicalEqIn": MusicalEqIn.model_json_schema(),
        "MusicalEqOut": MusicalEqOut.model_json_schema(),
        "TempoDynamicsIn": TempoDynamicsIn.model_json_schema(),
        "TempoDynamicsOut": TempoDynamicsOut.model_json_schema(),
        "HarmonicExcitationIn": HarmonicExcitationIn.model_json_schema(),
        "HarmonicExcitationOut": HarmonicExcitationOut.model_json_schema(),
        "StartInteractiveMasteringIn": StartInteractiveMasteringIn.model_json_schema(),
        "StartInteractiveMasteringOut": StartInteractiveMasteringOut.model_json_schema(),
        "CommitInteractiveMasteringIn": CommitInteractiveMasteringIn.model_json_schema(),
        "CommitInteractiveMasteringOut": CommitInteractiveMasteringOut.model_json_schema(),
        "SemanticABMasteringIn": SemanticABMasteringIn.model_json_schema(),
        "SemanticABMasteringOut": SemanticABMasteringOut.model_json_schema(),
        "AnalyzeAndOptimizeGovernorIn": AnalyzeAndOptimizeGovernorIn.model_json_schema(),
        "AnalyzeAndOptimizeGovernorOut": AnalyzeAndOptimizeGovernorOut.model_json_schema(),
        "AiStemRemixIn": AiStemRemixIn.model_json_schema(),
        "StemLufsReport": StemLufsReport.model_json_schema(),
        "AiStemRemixOut": AiStemRemixOut.model_json_schema(),
    }
    tool_map = _tool_contract_map()
    return json.dumps({"models": model_schemas, "tools": tool_map}, indent=2)


@mcp.resource(
    uri="auralmind://connect-kit",
    name="ConnectKit",
    description="Connect-time discovery payload with song preview and next-call templates.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_connect_kit_resource() -> str:
    packet = _build_connect_packet()
    payload = {
        "notes": [
            "Read this resource immediately after connect.",
            "Use `register_audio_from_path` for server-side files.",
            "Use `upload_init/upload_chunk/upload_finalize` if no songs are present.",
            "Use `plan_mastering_strategy` when the mastering goal starts as natural language.",
        ],
        "packet": packet.model_dump(),
    }
    return json.dumps(payload, indent=2)


@mcp.resource(
    uri="config://system-prompt",
    name="SystemPrompt",
    description="Cognitive mastering system prompt.",
    mime_type="text/markdown",
    annotations={"readOnlyHint": True, "idempotentHint": True},
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
)
def get_mcp_docs() -> str:
    """Returns the MCP usage guide bundled with the server."""
    with open(MCP_DOCS_PATH, "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource(
    uri="config://maintainer-guide",
    name="MaintainerGuide",
    description="Maintainer architecture and extension guide.",
    mime_type="text/markdown",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_maintainer_guide() -> str:
    with open(MAINTAINER_GUIDE_PATH, "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource(
    uri="config://server-info",
    name="ServerInfo",
    description="Server configuration and limits.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_server_info() -> str:
    """Provides server metadata and limits as JSON."""
    payload = {
        "name": SERVER_NAME,
        "version": VERSION,
        "transport": _active_transport(),
        "http_host": _http_host(),
        "http_port": _http_port(),
        "http_path": _http_path(),
        "supported_transports": list(SUPPORTED_TRANSPORTS),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_upload_b64_chars": MAX_UPLOAD_B64_CHARS,
        "max_upload_hex_chars": MAX_UPLOAD_HEX_CHARS,
        "upload_chunk_max_bytes": UPLOAD_CHUNK_MAX_BYTES,
        "max_upload_chunk_b64_chars": MAX_UPLOAD_CHUNK_B64_CHARS,
        "connect_preview_limit": CONNECT_PREVIEW_LIMIT,
        "max_read_bytes": MAX_READ_BYTES,
        "supported_bit_depths": ["float32", "float64"],
        "data_dir": DATA_DIR,
        "allowed_audio_extensions": sorted(ALLOWED_AUDIO_EXTENSIONS),
    }
    return json.dumps(payload, indent=2)


# ===========================================================================
# PROMPTS
# ===========================================================================
@mcp.prompt(name="on_connect")
async def on_connect_prompt() -> list[Message]:
    """Directed onboarding for new clients."""
    packet = _build_connect_packet()
    preview_names = ", ".join(song.filename for song in packet.songs_preview[:5]) if packet.songs_preview else "None"
    if packet.total_songs > 0:
        flow_hint = (
            "1) call `get_connect_packet` or read `auralmind://connect-kit` "
            "2) call `list_data_audio` "
            "3) call `register_audio_from_path` using one preview filename "
            "4) call `analyze_audio` "
            "5) call `plan_mastering_strategy` or `propose_master_settings` "
            "6) call `run_master_job` (or `master_closed_loop`)."
        )
    else:
        flow_hint = (
            "No songs found in `data/`. "
            "Use upload flow: `upload_init` -> `upload_chunk` -> `upload_finalize`, then `analyze_audio`, `plan_mastering_strategy`, and `run_master_job`."
        )
    return [
        Message(
            role="assistant",
            content=(
                f"Welcome to AuralMind Maestro. Songs detected: {packet.total_songs}. "
                f"Recent songs: {preview_names}. "
                f"{flow_hint} "
                "Use `bootstrap` for complete catalogs and `config://mcp-docs` for full usage guidance."
            ),
        )
    ]


@mcp.prompt(name="master_once")
async def master_once_prompt(
    file_uri: str,
    goal: str,
    platform: Platform = "spotify"
) -> str:
    """Single-pass mastering guide."""
    return (
        f"Master {file_uri} for {platform} with goal '{goal}'. "
        "Steps: 1) register_audio_from_path or upload_init/upload_chunk/upload_finalize "
        "(or upload_audio_to_session) 2) analyze_audio 3) plan_mastering_strategy 4) master_audio."
    )


@mcp.prompt(name="master_closed_loop_prompt")
def master_closed_loop_prompt(
    file_uri: str,
    goal: str,
    platform: Platform = "spotify"
) -> str:
    """Deterministic 2nd-run planning prompt."""
    return (
        f"Master {file_uri} for {platform} with goal '{goal}'. "
        "Use `master_closed_loop` to automate semantic planning, run1, scoring, and optional retune."
    )


@mcp.prompt(
    name="generate-mastering-strategy",
    description="Legacy strategy generator.",
)
def generate_strategy(
    integrated_lufs: Annotated[float, Field(description="Integrated loudness (LUFS).")],
    crest_db: Annotated[float, Field(description="Crest factor (dB).")],
    platform: Annotated[Platform, Field(description="Target platform.")],
) -> str:
    """Generates a prompt with the system instructions and measured metrics."""
    prompt_content = get_system_prompt()
    metrics = {
        "integrated_lufs": float(integrated_lufs),
        "crest_db": float(crest_db),
        "platform": platform,
    }
    prompt = (
        f"{prompt_content}\n\n"
        f"INPUT_METRICS:\n{json.dumps(metrics, indent=2)}\n\n"
        "Respond with the JSON strategy object."
    )
    return prompt


# ===========================================================================
# TOOLS
# ===========================================================================
@mcp.tool()
def bootstrap() -> BootstrapOut:
    """First-contact discovery: returns capabilities, catalogs, and example calls."""
    packet = _build_connect_packet()

    return BootstrapOut(
        capabilities=capabilities(),
        tools=_tool_catalog_entries(),
        resources=_resource_catalog_entries(),
        prompts=_prompt_catalog_entries(),
        workflow_steps=list(packet.workflow_steps),
        example_calls=_bootstrap_example_calls(packet),
    )


@mcp.tool()
def capabilities() -> CapabilitiesOut:
    """Returns server capabilities and features."""
    return CapabilitiesOut(
        server_name=SERVER_NAME,
        version=VERSION,
        transport=_active_transport(),
        features=[
            "server_name",
            "async_jobs",
            "bootstrap_discovery",
            "closed_loop_mastering",
            "semantic_strategy_planning",
            "control_profile",
            "resources",
            "prompts",
            "safe_filesystem",
            "server_side_ingest",
            "chunked_upload",
            "connect_discovery",
        ]
    )


@mcp.tool()
def get_connect_packet() -> ConnectPacketOut:
    """Returns a first-contact packet with song preview and call templates."""
    return _build_connect_packet()


@mcp.tool()
def list_audio_assets() -> AudioAssetList:
    """List audio files available inside the data directory."""
    assets: List[AudioAssetInfo] = []
    with os.scandir(DATA_DIR) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in ALLOWED_AUDIO_EXTENSIONS:
                continue
            size_bytes = entry.stat(follow_symlinks=False).st_size
            duration = _safe_audio_duration(entry.path)
            assets.append(AudioAssetInfo(
                filename=entry.name,
                size_bytes=size_bytes,
                format=ext[1:],
                duration_seconds=duration,
            ))
    assets.sort(key=lambda item: item.filename.lower())
    return AudioAssetList(assets)


@mcp.tool()
def list_data_audio() -> AudioAssetList:
    """Alias for list_audio_assets for client compatibility."""
    return list_audio_assets()


@mcp.tool()
def register_audio_from_path(
    path: Annotated[str, Field(description="Path to an audio file within the data directory.")],
    ctx: Context = None,
) -> RegisterAudioResult:
    """Register a server-side audio file without upload."""
    resolved = _resolve_data_path(path)
    if not os.path.isfile(resolved):
        raise ValueError("not_found")
    if not os.access(resolved, os.R_OK):
        raise ValueError("unreadable")

    _, fmt = _audio_format_from_path(resolved)
    size_hint = os.path.getsize(resolved)
    if size_hint <= 0:
        raise ValueError("empty_file")

    session_key, session_dir = _get_session_info(ctx)
    audio_id = _new_id("aud")
    filename = os.path.basename(resolved)
    media_type = _guess_media_type(filename, fallback="audio/wav")
    entry = _store_file_from_path(
        session_key,
        session_dir,
        artifact_id=audio_id,
        kind="audio",
        filename=filename,
        source_path=resolved,
        media_type=media_type,
    )
    registered_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    log.info(
        "registered_audio_from_path audio_id=%s filename=%s size_bytes=%s format=%s",
        audio_id,
        entry.filename,
        entry.size_bytes,
        fmt,
    )
    return RegisterAudioResult(
        audio_id=audio_id,
        format=fmt,
        size_bytes=entry.size_bytes,
        checksum=entry.sha256,
        registered_at=registered_at,
    )


@mcp.tool()
def analyze_audio(
    audio_id: Annotated[str, Field(description="Audio ID to analyze.")],
    ctx: Context = None,
) -> AudioMetrics:
    """Comprehensive pre-mastering analysis."""
    try:
        return _analyze_internal(audio_id, ctx)
    except Exception as exc:
        raise RuntimeError(f"analysis_failed: {exc}")


def _analyze_internal(
    audio_id: str,
    ctx: Context = None,
    *,
    session_key: Optional[str] = None,
    session_dir: Optional[str] = None,
) -> AudioMetrics:
    # Existing analysis logic refactored
    if session_key is None or session_dir is None:
        session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind not in ("audio", "mastered_audio"):
        raise ValueError(f"not_found: Audio not found: {audio_id}")

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    maestro, err = _get_maestro()
    if err: raise RuntimeError(err["message"])

    y, sr = maestro.load_audio(data_path)
    features = maestro.analyze_track_features(y, sr)

    return AudioMetrics(
        integrated_lufs=float(features["lufs"]),
        true_peak_dbtp=float(features["tp_dbfs"]),
        crest_db=float(features["crest_db"]),
        stereo_correlation=float(features.get("corr_hi", 0.0)),
        duration_s=round(len(y) / sr, 2),
        peak_dbfs=float(features["peak_dbfs"]),
        rms_dbfs=float(features["rms_dbfs"]),
        centroid_hz=float(features["centroid_hz"])
    )


@mcp.tool()
def list_presets() -> PresetsOut:
    """List all available mastering presets."""
    maestro, err = _get_maestro()
    if err: raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    out = {}
    for name, p in presets.items():
        out[name] = PresetSummary(**_serialize_preset(p))
    return PresetsOut(presets=out)


@mcp.tool()
def plan_mastering_strategy(req: StrategyPlanIn, ctx: Context = None) -> StrategyPlanOut:
    """Resolve natural-language mastering intent into executable settings."""
    return _plan_mastering_strategy_internal(req, ctx)


@mcp.tool()
def propose_master_settings(req: MasterSettings) -> ProposedSettingsOut:
    """Validate and normalize mastering settings against preset defaults and control-profile rules."""
    return ProposedSettingsOut(settings=_resolve_settings_from_request(req))


@mcp.tool()
def run_master_job(req: MasterRequest, ctx: Context = None) -> JobLaunchOut:
    """Start mastering asynchronously. Returns job_id immediately."""
    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, req.audio_id)
    if entry is None or entry.kind != "audio":
        raise ValueError("not_found: Audio not found.")

    settings = _resolve_settings_from_request(req)

    job_id = _new_id("job")
    job = JobState(
        job_id=job_id,
        audio_id=req.audio_id,
        status="queued",
        progress=0,
        session_key=session_key,
        session_dir=session_dir,
        settings=settings,
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    future = _JOB_EXECUTOR.submit(_run_master_job_worker, job_id)
    _update_job(job_id, future=future)

    return JobLaunchOut(job_id=job_id, status="queued", audio_id=req.audio_id)


@mcp.tool()
def job_status(
    job_id: Annotated[str, Field(description="Job ID.")],
    ctx: Context = None,
) -> JobStatusOut:
    """Poll for job progress."""
    job = _get_job(job_id)
    if job is None:
        raise ValueError("not_found: Job not found.")
    session_key, _ = _get_session_info(ctx)
    if job.session_key != session_key:
        raise ValueError("not_found: Job not found.")

    return JobStatusOut(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        elapsed_s=round(_job_elapsed(job), 2),
        error=job.error,
    )


@mcp.tool()
def job_result(
    job_id: Annotated[str, Field(description="Job ID.")],
    ctx: Context = None,
) -> JobResultOut:
    """Fetch results once a job is complete."""
    job = _get_job(job_id)
    if job is None:
        raise ValueError("not_found: Job not found.")
    session_key, session_dir = _get_session_info(ctx)
    if job.session_key != session_key:
        raise ValueError("not_found: Job not found.")
    if job.status == "error":
        raise RuntimeError(job.error.message if job.error else "job_failed")
    if job.status != "done":
        raise ValueError("not_ready: Job still running.")
    if job.result is None:
        raise RuntimeError("job_missing_result")

    artifacts: List[ArtifactSummary] = []
    for artifact_id in job.result.artifacts:
        entry = _load_artifact(session_key, session_dir, artifact_id)
        if entry is not None:
            artifacts.append(_artifact_summary(entry))

    precision = job.settings.bit_depth if job.settings else "float32"
    return JobResultOut(
        job_id=job.job_id,
        status=job.status,
        artifacts=artifacts,
        metrics=job.result.metrics_after,
        precision=precision,
    )


@mcp.tool()
def safe_read_text(req: FileReadIn) -> FileReadOut:
    """Safely read a text file within session or data directories."""
    path = os.path.abspath(req.path)
    # Basic jail check
    if not _is_allowed_path(path):
        raise ValueError("access_denied: Path outside allowlist.")

    with open(path, "r", encoding="utf-8") as f:
        return FileReadOut(content=f.read())


@mcp.tool()
def safe_write_text(req: FileWriteIn) -> FileWriteOut:
    """Safely write a text file within session or data directories."""
    path = os.path.abspath(req.path)
    if not _is_allowed_path(path):
        raise ValueError("access_denied: Path outside allowlist.")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(req.content)
    return FileWriteOut(success=True, path=path)


@mcp.tool()
def cancel_job(req: CancelJobIn, ctx: Context = None) -> CancelJobOut:
    """Cancel a queued or running job."""
    session_key, _ = _get_session_info(ctx)
    with _JOBS_LOCK:
        job = _JOBS.get(req.job_id)
        if job is None or job.session_key != session_key:
            raise ValueError("not_found: Job not found.")

        if job.status in ("done", "error"):
            return CancelJobOut(job_id=req.job_id, success=False, message=f"Job already finished with status '{job.status}'.")

        if job.future and not job.future.done():
            job.future.cancel()

        job.status = "error"
        job.error = _make_error("cancelled", "Job was cancelled by user.", {"job_id": req.job_id})
        job.finished_at = time.time()

    return CancelJobOut(job_id=req.job_id, success=True, message="Job cancelled.")


@mcp.tool()
def delete_artifact(req: DeleteArtifactIn, ctx: Context = None) -> DeleteArtifactOut:
    """Delete an artifact to free session storage space."""
    session_key, session_dir = _get_session_info(ctx)
    with _ARTIFACTS_LOCK:
        cache = _ARTIFACTS.get(session_key, {"init": None})
        if cache and req.artifact_id in cache:
            entry = cache.pop(req.artifact_id)
        else:
            entry = None

    if entry is None:
        entry = _load_artifact(session_key, session_dir, req.artifact_id)
        if entry is None:
            raise ValueError("not_found: Artifact not found.")
        with _ARTIFACTS_LOCK:
            _ARTIFACTS.get(session_key, {}).pop(req.artifact_id, None)

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    meta_path = _artifact_meta_path(session_dir, entry.artifact_id)

    for p in (data_path, meta_path):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError as e:
            log.warning("Failed to delete %s: %s", p, e)

    return DeleteArtifactOut(artifact_id=req.artifact_id, success=True)


def _get_metrics_for_id(session_key: str, session_dir: str, ref_id: str, ctx: Context) -> AudioMetrics:
    entry = _load_artifact(session_key, session_dir, ref_id)
    if entry is None:
        raise ValueError(f"not_found: Artifact {ref_id} not found.")

    if entry.kind == "metrics":
        data_path = _artifact_data_path(session_dir, entry.data_filename)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return AudioMetrics(**data["metrics"])
    elif entry.kind in ("audio", "mastered_audio"):
        return _analyze_internal(ref_id, ctx, session_key=session_key, session_dir=session_dir)
    else:
        raise ValueError(f"invalid_artifact: {ref_id} is neither audio nor metrics JSON.")


@mcp.tool()
def compare_audio_metrics(req: CompareMetricsIn, ctx: Context = None) -> CompareMetricsOut:
    """Compare two sets of audio metrics or audio files."""
    session_key, session_dir = _get_session_info(ctx)

    metrics_a = _get_metrics_for_id(session_key, session_dir, req.audio_id_a, ctx)
    metrics_b = _get_metrics_for_id(session_key, session_dir, req.audio_id_b, ctx)

    delta = AudioMetricsDelta(
        lufs_delta=round(metrics_b.integrated_lufs - metrics_a.integrated_lufs, 2),
        true_peak_delta=round(metrics_b.true_peak_dbtp - metrics_a.true_peak_dbtp, 2),
        crest_delta=round(metrics_b.crest_db - metrics_a.crest_db, 2),
        correlation_delta=round(metrics_b.stereo_correlation - metrics_a.stereo_correlation, 3),
    )
    return CompareMetricsOut(delta=delta)


@mcp.tool()
def master_audio(req: MasterRequest, ctx: Context = None) -> MasterResult:
    """Run a single mastering pass on the provided audio."""
    run_id = f"once_{uuid.uuid4().hex[:6]}"
    settings = _resolve_settings_from_request(req)
    resolved_req = _master_request_from_settings(req.audio_id, settings)
    return _master_internal(req.audio_id, resolved_req, run_id, ctx)


@mcp.tool()
def master_closed_loop(req: ClosedLoopRequest, ctx: Context = None) -> ClosedLoopResult:
    """Deterministic closed-loop mastering orchestrator (max 2 runs)."""
    session_key, session_dir = _get_session_info(ctx)
    plan = _plan_mastering_strategy_internal(
        StrategyPlanIn(
            audio_id=req.audio_id,
            goal=req.goal,
            platform=req.platform,
            control_profile=req.control_profile,
            governor_search_steps=req.governor_search_steps,
            governor_gr_limit_db=req.governor_gr_limit_db,
            stem_gains_db=req.stem_gains_db,
        ),
        ctx,
        session_key=session_key,
        session_dir=session_dir,
    )
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    run1_req = _master_request_from_settings(req.audio_id, plan.settings)
    res1 = _master_internal(
        req.audio_id,
        run1_req,
        "run1",
        ctx,
        session_key=session_key,
        session_dir=session_dir,
    )

    # Score Run 1
    target = run1_req.target_lufs
    # Extract ceiling from preset
    ceiling = float(getattr(presets[plan.chosen_preset], "ceiling_dbfs", -1.0))

    score1 = _calculate_score(res1.metrics_after, target, ceiling)

    # Check violations
    violations = (abs(res1.metrics_after.integrated_lufs - target) > 0.7 or
                  res1.metrics_after.true_peak_dbtp > (ceiling + 0.1))

    best_res = res1
    best_run_id = "run1"
    runner_summary = {
        "plan": plan.model_dump(),
        "run1": {"score": score1, "metrics": res1.metrics_after.model_dump(), "settings": run1_req.model_dump()},
    }

    if violations:
        # 2. Retune and Run 2 using ORIGINAL input
        run2_req, deltas = _calculate_retune(res1.metrics_after, run1_req)
        res2 = _master_internal(
            req.audio_id,
            run2_req,
            "run2",
            ctx,
            session_key=session_key,
            session_dir=session_dir,
        )
        score2 = _calculate_score(res2.metrics_after, run2_req.target_lufs, ceiling)

        runner_summary["run2"] = {"score": score2, "metrics": res2.metrics_after.model_dump(), "settings": run2_req.model_dump(), "deltas": [d.model_dump() for d in deltas]}

        if score2 < score1:
            best_res = res2
            best_run_id = "run2"

    # Save summary
    summary_id = _new_id("art")
    summary_filename = f"{summary_id}.json"
    with open(os.path.join(session_dir, summary_filename), "w", encoding="utf-8") as f:
        json.dump(runner_summary, f, indent=2)
    _register_existing_file(
        session_key,
        session_dir,
        artifact_id=summary_id,
        kind="summary",
        filename=summary_filename,
        data_filename=summary_filename,
        media_type="application/json",
    )

    artifacts = list(best_res.artifacts) + [summary_id]

    return ClosedLoopResult(
        best_run_id=best_run_id,
        artifacts=artifacts,
        runner_summary_id=summary_id,
        metrics_final=best_res.metrics_after
    )


# ===========================================================================
# TOOLS - SYSTEM
# ===========================================================================
@mcp.tool()
def upload_init(req: UploadInitIn, ctx: Context = None) -> UploadInitOut:
    """Initialize a resumable chunked upload."""
    ext = os.path.splitext(req.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError("unsupported_format")

    expected_sha = req.sha256.lower() if req.sha256 else None
    if expected_sha and not re.fullmatch(r"[a-f0-9]{64}", expected_sha):
        raise ValueError("invalid_sha256")

    _, session_dir = _get_session_info(ctx)
    upload_id = f"upl_{uuid.uuid4().hex[:12]}"
    meta: Dict[str, Any] = {
        "upload_id": upload_id,
        "filename": _sanitize_filename(req.filename),
        "total_bytes": int(req.total_bytes),
        "received_bytes": 0,
        "next_index": 0,
        "sha256": expected_sha,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with _UPLOAD_LOCK:
        with open(_upload_part_path(session_dir, upload_id), "wb"):
            pass
        _save_upload_meta(session_dir, upload_id, meta)

    return UploadInitOut(
        upload_id=upload_id,
        filename=meta["filename"],
        total_bytes=meta["total_bytes"],
        received_bytes=0,
        next_index=0,
        chunk_max_bytes=UPLOAD_CHUNK_MAX_BYTES,
        done=False,
    )


@mcp.tool()
def upload_status(
    upload_id: Annotated[str, Field(description="Upload handle from upload_init.")],
    ctx: Context = None,
) -> UploadStatusOut:
    """Read resumable upload status."""
    if not UPLOAD_ID_RE.match(upload_id):
        raise ValueError("invalid_upload_id")
    _, session_dir = _get_session_info(ctx)
    with _UPLOAD_LOCK:
        meta = _load_upload_meta(session_dir, upload_id)
        return _upload_status_from_meta(meta)


@mcp.tool()
def upload_chunk(req: UploadChunkIn, ctx: Context = None) -> UploadStatusOut:
    """Append one ordered chunk to an active upload."""
    if not UPLOAD_ID_RE.match(req.upload_id):
        raise ValueError("invalid_upload_id")
    chunk = _decode_base64_chunk(req.chunk_b64)
    if not chunk:
        raise ValueError("empty_chunk")
    if len(chunk) > UPLOAD_CHUNK_MAX_BYTES:
        raise ValueError("chunk_too_large")

    _, session_dir = _get_session_info(ctx)
    with _UPLOAD_LOCK:
        meta = _load_upload_meta(session_dir, req.upload_id)
        next_index = int(meta["next_index"])
        total = int(meta["total_bytes"])
        received = int(meta["received_bytes"])

        # Idempotency: if the caller retries an already-accepted chunk index, return current status.
        if req.index < next_index:
            return _upload_status_from_meta(meta)
        # Strict sequencing protects against missing/duplicated chunk writes.
        if req.index != next_index:
            raise ValueError(f"out_of_order_chunk: expected index {next_index}")
        if received >= total:
            return _upload_status_from_meta(meta)
        if received + len(chunk) > total:
            raise ValueError("chunk_overflow")

        part_path = _upload_part_path(session_dir, req.upload_id)
        current_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if current_size != received:
            raise ValueError("upload_state_mismatch")
        with open(part_path, "ab") as f:
            f.write(chunk)

        meta["received_bytes"] = received + len(chunk)
        meta["next_index"] = next_index + 1
        meta["updated_at"] = time.time()
        _save_upload_meta(session_dir, req.upload_id, meta)
        return _upload_status_from_meta(meta)


@mcp.tool()
def upload_finalize(req: UploadFinalizeIn, ctx: Context = None) -> UploadResult:
    """Finalize upload, verify checksum, and register audio artifact."""
    if not UPLOAD_ID_RE.match(req.upload_id):
        raise ValueError("invalid_upload_id")
    session_key, session_dir = _get_session_info(ctx)

    with _UPLOAD_LOCK:
        meta = _load_upload_meta(session_dir, req.upload_id)
        total = int(meta["total_bytes"])
        received = int(meta["received_bytes"])
        if received != total:
            raise ValueError("upload_incomplete")

        part_path = _upload_part_path(session_dir, req.upload_id)
        if not os.path.exists(part_path):
            raise ValueError("upload_missing_part")
        part_size = os.path.getsize(part_path)
        if part_size != total:
            raise ValueError("upload_state_mismatch")

        sha = hashlib.sha256()
        with open(part_path, "rb") as f:
            while True:
                buf = f.read(1024 * 1024)
                if not buf:
                    break
                sha.update(buf)
        digest = sha.hexdigest()
        expected = meta.get("sha256")
        if expected and digest != expected:
            raise ValueError("sha256_mismatch")

        filename = str(meta["filename"])
        ext = os.path.splitext(filename)[1].lower() or ".bin"
        audio_id = _new_id("aud")
        data_filename = f"{audio_id}{ext}"
        os.replace(part_path, _artifact_data_path(session_dir, data_filename))
        entry = _register_existing_file(
            session_key,
            session_dir,
            artifact_id=audio_id,
            kind="audio",
            filename=filename,
            data_filename=data_filename,
            media_type=_guess_media_type(filename, fallback="audio/wav"),
        )
        _delete_upload_meta(session_dir, req.upload_id)

    return UploadResult(
        audio_id=audio_id,
        filename=entry.filename,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
        media_type=entry.media_type,
    )


@mcp.tool()
def upload_audio_to_session(
    filename: Annotated[str, Field(description="Original filename.")],
    payload_b64: Annotated[Optional[str], Field(default=None, description="Base64 payload.")] = None,
    hex_payload: Annotated[Optional[str], Field(default=None, description="Hex payload (legacy).")] = None,
    ctx: Context = None,
) -> UploadResult:
    """Upload audio for processing."""
    if payload_b64 and hex_payload:
        raise ValueError("payload_conflict")
    if not payload_b64 and not hex_payload:
        raise ValueError("missing_payload")

    payload = _decode_base64_payload(payload_b64) if payload_b64 else _decode_hex_payload(hex_payload)
    if not payload:
        raise ValueError("empty_payload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("payload_too_large")
    session_key, session_dir = _get_session_info(ctx)
    audio_id = _new_id("aud")
    media_type = _guess_media_type(filename, fallback="audio/wav")
    entry = _store_bytes(session_key, session_dir, artifact_id=audio_id, kind="audio",
                         filename=filename, payload=payload, media_type=media_type)

    return UploadResult(audio_id=audio_id, filename=entry.filename, size_bytes=entry.size_bytes,
                        sha256=entry.sha256, media_type=entry.media_type)


@mcp.tool()
def read_artifact(
    artifact_id: Annotated[str, Field(description="Artifact ID.")],
    offset: int = 0,
    length: int = MAX_READ_BYTES,
    ctx: Context = None
) -> ArtifactReadResult:
    """Read artifact bytes."""
    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, artifact_id)
    if entry is None: raise ValueError("not_found")
    if offset < 0:
        raise ValueError("invalid_offset")
    if length <= 0 or length > MAX_READ_BYTES:
        raise ValueError("invalid_length")
    if offset >= entry.size_bytes:
        raise ValueError("offset_out_of_range")

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    with open(data_path, "rb") as f:
        f.seek(offset)
        chunk = f.read(min(length, entry.size_bytes - offset))

    return ArtifactReadResult(
        artifact_id=entry.artifact_id, filename=entry.filename, media_type=entry.media_type,
        size_bytes=entry.size_bytes, sha256=entry.sha256, offset=offset, length=len(chunk),
        is_last=(offset + len(chunk)) >= entry.size_bytes,
        data_b64=base64.b64encode(chunk).decode("ascii")
    )


# ===========================================================================
# Advanced AI / LLM Tools
# ===========================================================================
def _load_audio_artifact(
    audio_id: str,
    ctx: Context = None,
    *,
    session_key: Optional[str] = None,
    session_dir: Optional[str] = None,
    allowed_kinds: Tuple[str, ...] = ("audio", "mastered_audio"),
) -> Tuple[str, str, ArtifactEntry, Any, np.ndarray, int, str]:
    if session_key is None or session_dir is None:
        session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind not in allowed_kinds:
        raise ValueError(f"not_found: Audio not found: {audio_id}")
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])
    data_path = _artifact_data_path(session_dir, entry.data_filename)
    audio, sr = maestro.load_audio(data_path)
    return session_key, session_dir, entry, maestro, audio, sr, data_path


def _store_audio_artifact(
    session_key: str,
    session_dir: str,
    maestro: Any,
    audio: np.ndarray,
    sr: int,
    *,
    filename: str,
    kind: str = "mastered_audio",
) -> ArtifactEntry:
    artifact_id = _new_id("art")
    ext = os.path.splitext(filename)[1].lower() or ".wav"
    data_filename = f"{artifact_id}{ext}"
    out_path = _artifact_data_path(session_dir, data_filename)
    maestro.write_audio(out_path, audio, sr, subtype="FLOAT", dither=False)
    return _register_existing_file(
        session_key,
        session_dir,
        artifact_id=artifact_id,
        kind=kind,
        filename=filename,
        data_filename=data_filename,
        media_type=_guess_media_type(filename, fallback="audio/wav"),
    )


def _gaussian_tone_curve(length: int, sr: int, peaks: List[Tuple[float, float, float]]) -> np.ndarray:
    freqs = np.fft.rfftfreq(length, d=1.0 / float(sr))
    curve = np.ones_like(freqs, dtype=np.float32)
    for hz, gain_db, width_hz in peaks:
        bell = np.exp(-0.5 * np.square((freqs - hz) / max(25.0, width_hz))).astype(np.float32)
        curve *= np.power(10.0, (gain_db * bell) / 20.0).astype(np.float32)
    return curve.astype(np.complex64)


def _render_band_heatmap(maestro: Any, audio_path: str, label: str) -> str:
    audio, sr = maestro.load_audio(audio_path)
    mono = maestro.to_mono(audio).astype(np.float32)
    excerpt = mono[: min(len(mono), sr * 3)]
    if excerpt.size < 1024:
        return f"{label}: [---]  (insufficient duration)"
    freqs = np.fft.rfftfreq(excerpt.size, d=1.0 / float(sr))
    mags = np.abs(np.fft.rfft(excerpt))
    lows = float(np.sum(mags[(freqs >= 20.0) & (freqs < 250.0)]))
    mids = float(np.sum(mags[(freqs >= 250.0) & (freqs < 2000.0)]))
    highs = float(np.sum(mags[(freqs >= 2000.0) & (freqs < 10000.0)]))
    total = max(lows + mids + highs, 1e-9)
    blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    indices = [
        min(7, int((lows / total) * 7)),
        min(7, int((mids / total) * 7)),
        min(7, int((highs / total) * 7)),
    ]
    return f"{label}: [{blocks[indices[0]]}{blocks[indices[1]]}{blocks[indices[2]]}]  (Low/Mid/High)"


@mcp.tool()
async def apply_musical_eq(req: MusicalEqIn, ctx: Context) -> MusicalEqOut:
    """Apply a simple key-aware spectral emphasis and return a session-scoped artifact."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    base_note = req.key.upper()
    if base_note not in notes:
        raise ValueError(f"unknown_key: {req.key}")

    session_key, session_dir, _, maestro, audio, sr, _ = _load_audio_artifact(req.audio_id, ctx)

    root_midi = 36 + notes.index(base_note)

    def midi_to_freq(midi_note: int) -> float:
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    root_freq = midi_to_freq(root_midi)
    third_freq = midi_to_freq(root_midi + (3 if req.scale.lower() == "minor" else 4))
    fifth_freq = midi_to_freq(root_midi + 7)

    await ctx.report_progress(20, 100, "Building harmonic emphasis curve...")
    curve = _gaussian_tone_curve(
        len(audio),
        sr,
        [
            (root_freq, 1.2, max(35.0, root_freq * 0.18)),
            (third_freq, 0.7, max(45.0, third_freq * 0.16)),
            (fifth_freq, 0.9, max(50.0, fifth_freq * 0.15)),
        ],
    )

    await ctx.report_progress(65, 100, "Applying musical EQ emphasis...")
    shaped = np.empty_like(audio, dtype=np.float32)
    for channel_index in range(audio.shape[1]):
        spectrum = np.fft.rfft(audio[:, channel_index])
        shaped[:, channel_index] = np.fft.irfft(spectrum * curve, n=len(audio)).astype(np.float32)
    peak = float(np.max(np.abs(shaped)))
    if peak > 1.0:
        shaped = (shaped / peak).astype(np.float32)

    await ctx.report_progress(90, 100, "Writing processed artifact...")
    entry = _store_audio_artifact(
        session_key,
        session_dir,
        maestro,
        shaped,
        sr,
        filename=f"musical_eq_{base_note}_{req.scale.lower()}.wav",
    )
    ascii_graph = (
        f"Musical EQ ({base_note} {req.scale.lower()}):\n"
        f"Root  {root_freq:7.1f} Hz  +1.2 dB\n"
        f"Third {third_freq:7.1f} Hz  +0.7 dB\n"
        f"Fifth {fifth_freq:7.1f} Hz  +0.9 dB"
    )
    return MusicalEqOut(
        artifact_id=entry.artifact_id,
        message=f"Applied harmonic emphasis around {base_note} {req.scale.lower()}.",
        ascii_graph=ascii_graph,
    )


@mcp.tool()
async def apply_tempo_dynamics(req: TempoDynamicsIn, ctx: Context) -> TempoDynamicsOut:
    """Apply a tempo-synced limiter release and store the result in the current session."""
    if req.bpm <= 0:
        raise ValueError("invalid_bpm")
    session_key, session_dir, _, maestro, audio, sr, _ = _load_audio_artifact(req.audio_id, ctx)
    ms_per_beat = 60000.0 / float(req.bpm)
    div_map = {"1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25}
    multiplier = div_map.get(req.note_division, 1.0)
    release_ms = ms_per_beat * multiplier

    await ctx.report_progress(30, 100, "Applying tempo-synced limiter...")
    limited_audio, stats = await asyncio.to_thread(
        maestro.true_peak_limiter_v2,
        audio,
        sr,
        ceiling_dbfs=-1.0,
        oversample=4,
        lookahead_ms=2.0,
        attack_ms=5.0,
        release_ms=release_ms,
    )
    await ctx.report_progress(90, 100, "Writing processed artifact...")
    entry = _store_audio_artifact(
        session_key,
        session_dir,
        maestro,
        limited_audio,
        sr,
        filename=f"tempo_lock_{int(req.bpm)}_{req.note_division.replace('/', '-')}.wav",
    )
    pulse_grid = (
        f"Tempo Groove Grid\n"
        f"BPM: {req.bpm:.2f}\n"
        f"Division: {req.note_division}\n"
        f"Release: {release_ms:.1f} ms\n"
        f"Limiter Min Gain: {stats['min_gain_db']:.2f} dB"
    )
    return TempoDynamicsOut(
        artifact_id=entry.artifact_id,
        message=f"Applied tempo-synced limiting at {req.bpm:.2f} BPM.",
        pulse_grid=pulse_grid,
    )


@mcp.tool()
async def apply_harmonic_excitation(req: HarmonicExcitationIn, ctx: Context) -> HarmonicExcitationOut:
    """Apply a bounded harmonic saturation pass and return the rendered artifact."""
    session_key, session_dir, _, maestro, audio, sr, _ = _load_audio_artifact(req.audio_id, ctx)
    drive_mult = max(0.0, min(5.0, (float(req.drive_amount) / 100.0) * 5.0))
    saturated = np.copy(audio).astype(np.float32)
    total_samples = saturated.shape[0]
    chunk_size = max(sr, int(sr * 1.5))
    num_chunks = max(1, int(np.ceil(total_samples / float(chunk_size))))

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size
        end = min(total_samples, start + chunk_size)
        chunk = saturated[start:end]
        if req.harmonics == "even":
            chunk = np.where(chunk > 0, np.tanh(chunk * (1.0 + drive_mult)), chunk)
        elif req.harmonics == "odd":
            chunk = np.tanh(chunk * (1.0 + drive_mult))
        else:
            chunk = np.where(
                chunk > 0,
                np.tanh(chunk * (1.0 + (drive_mult * 1.5))),
                np.tanh(chunk * (1.0 + drive_mult)),
            )
        saturated[start:end] = chunk.astype(np.float32)
        progress = 20 + int(((chunk_index + 1) / num_chunks) * 65)
        await ctx.report_progress(progress, 100, f"Saturating chunk {chunk_index + 1}/{num_chunks}...")

    peak = float(np.max(np.abs(saturated)))
    if peak > 1.0:
        saturated = (saturated / peak).astype(np.float32)

    entry = _store_audio_artifact(
        session_key,
        session_dir,
        maestro,
        saturated,
        sr,
        filename=f"harmonic_{req.harmonics}.wav",
    )
    bars = int(max(0.0, min(10.0, float(req.drive_amount) / 10.0)))
    return HarmonicExcitationOut(
        artifact_id=entry.artifact_id,
        message=f"Applied {req.harmonics} harmonic excitation at {req.drive_amount:.1f}% drive.",
        meter=f"[{'|' * bars}{'-' * (10 - bars)}] {req.drive_amount:.1f}% {req.harmonics}",
    )


class AnalyzeAndOptimizeGovernorIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact to analyze.")
    preset_name: str = Field(..., description="Preset to base the optimization on.")


class AnalyzeAndOptimizeGovernorOut(StrictBaseModel):
    crest_factor_db: float = Field(..., description="Measured crest factor of the input audio.")
    recommended_governor_steps: int = Field(..., description="Ideal search steps for the loudness governor.")
    recommended_governor_gr_limit_db: float = Field(..., description="Recommended GR ceiling.")
    music_theory_reasoning: str = Field(..., description="Explanation of the recommendation.")


class AiStemRemixIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact.")


class StemLufsReport(StrictBaseModel):
    vocals: float
    drums: float
    bass: float
    other: float


class AiStemRemixOut(StrictBaseModel):
    message: str = Field(..., description="AI info.")
    stem_lufs: StemLufsReport = Field(..., description="Calculated LUFS for each separated stem.")
    mix_theory_advice: str = Field(..., description="Suggested gain tweaks based on standard mix theory.")


INTERACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}


@mcp.tool()
async def start_interactive_mastering(req: StartInteractiveMasteringIn, ctx: Context) -> StartInteractiveMasteringOut:
    """Render a first-pass master and persist a session token for a second-stage commit."""
    session_key, session_dir = _get_session_info(ctx)
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    if req.preset_name not in presets:
        raise ValueError(f"unknown_preset: {req.preset_name}")

    settings = _finalize_master_settings(req.preset_name)
    run1_req = _master_request_from_settings(req.audio_id, settings)
    await ctx.report_progress(25, 100, "Rendering interactive stage 1 master...")
    res1 = await asyncio.to_thread(
        _master_internal,
        req.audio_id,
        run1_req,
        "interactive_stage1",
        None,
        session_key=session_key,
        session_dir=session_dir,
    )
    await ctx.report_progress(100, 100, "Stage 1 complete.")

    session_token = _new_id("art")
    INTERACTIVE_SESSIONS[session_token] = {
        "session_key": session_key,
        "session_dir": session_dir,
        "audio_id": req.audio_id,
        "stage1_settings": settings.model_dump(),
        "stage1_metrics": res1.metrics_after.model_dump(),
    }
    return StartInteractiveMasteringOut(
        session_token=session_token,
        message="Stage 1 complete. Review metrics and commit final warmth/transient changes.",
        metrics=res1.metrics_after,
        stage1_settings=run1_req,
    )


@mcp.tool()
async def commit_interactive_mastering(req: CommitInteractiveMasteringIn, ctx: Context) -> CommitInteractiveMasteringOut:
    """Apply final interactive tweaks to the saved stage-1 mastering session."""
    session_key, _ = _get_session_info(ctx)
    session = INTERACTIVE_SESSIONS.pop(req.session_token, None)
    if session is None:
        raise ValueError("invalid_or_expired_session")
    if session["session_key"] != session_key:
        raise ValueError("not_found: Interactive session not found.")

    base_settings = MasterSettings(**session["stage1_settings"])
    resolved_settings = _finalize_master_settings(
        base_settings.preset_name,
        control_profile=base_settings.control_profile,
        explicit_overrides={
            "target_lufs": base_settings.target_lufs,
            "warmth": req.warmth,
            "transient_boost_db": req.transient_boost_db,
            "enable_harshness_limiter": base_settings.enable_harshness_limiter,
            "enable_air_motion": base_settings.enable_air_motion,
            "bit_depth": base_settings.bit_depth,
        },
        safe_overrides={
            "governor_search_steps": base_settings.governor_search_steps,
            "governor_gr_limit_db": base_settings.governor_gr_limit_db,
            "stem_gains_db": base_settings.stem_gains_db,
        },
    )
    run2_req = _master_request_from_settings(session["audio_id"], resolved_settings)
    await ctx.report_progress(35, 100, "Rendering interactive final pass...")
    res2 = await asyncio.to_thread(
        _master_internal,
        session["audio_id"],
        run2_req,
        "interactive_stage2",
        None,
        session_key=session["session_key"],
        session_dir=session["session_dir"],
    )
    ascii_console = (
        "AURALMIND INTERACTIVE MASTER\n"
        f"Warmth:        {req.warmth:.2f}\n"
        f"Transient dB:  {req.transient_boost_db:.2f}\n"
        f"LUFS:          {res2.metrics_after.integrated_lufs:.2f}\n"
        f"True Peak:     {res2.metrics_after.true_peak_dbtp:.2f} dBTP"
    )
    return CommitInteractiveMasteringOut(
        artifact_id=res2.master_wav_id,
        message="Interactive mastering final pass rendered.",
        ascii_console=ascii_console,
        final_metrics=res2.metrics_after,
    )


@mcp.tool()
async def semantic_a_b_mastering(req: SemanticABMasteringIn, ctx: Context) -> SemanticABMasteringOut:
    """Render two preset variants in parallel and summarize the output differences."""
    session_key, session_dir, _, maestro, _, _, _ = _load_audio_artifact(req.audio_id, ctx)
    presets = maestro.get_presets()
    if req.preset_a not in presets or req.preset_b not in presets:
        raise ValueError("unknown_preset")

    req_a = _master_request_from_settings(req.audio_id, _finalize_master_settings(req.preset_a))
    req_b = _master_request_from_settings(req.audio_id, _finalize_master_settings(req.preset_b))

    await ctx.report_progress(20, 100, "Rendering A/B mastering variants...")
    res_a, res_b = await asyncio.gather(
        asyncio.to_thread(
            _master_internal,
            req.audio_id,
            req_a,
            "semantic_a",
            None,
            session_key=session_key,
            session_dir=session_dir,
        ),
        asyncio.to_thread(
            _master_internal,
            req.audio_id,
            req_b,
            "semantic_b",
            None,
            session_key=session_key,
            session_dir=session_dir,
        ),
    )
    await ctx.report_progress(90, 100, "Summarizing rendered variants...")

    entry_a = _load_artifact(session_key, session_dir, res_a.master_wav_id)
    entry_b = _load_artifact(session_key, session_dir, res_b.master_wav_id)
    if entry_a is None or entry_b is None:
        raise RuntimeError("rendered_artifact_missing")

    heatmap_a = _render_band_heatmap(
        maestro,
        _artifact_data_path(session_dir, entry_a.data_filename),
        f"Option A ({req.preset_a})",
    )
    heatmap_b = _render_band_heatmap(
        maestro,
        _artifact_data_path(session_dir, entry_b.data_filename),
        f"Option B ({req.preset_b})",
        )
    matrix = (
        f"| Metric | A: {req.preset_a} | B: {req.preset_b} |\n"
        f"|---|---|---|\n"
        f"| LUFS | {res_a.metrics_after.integrated_lufs:.2f} | {res_b.metrics_after.integrated_lufs:.2f} |\n"
        f"| True Peak | {res_a.metrics_after.true_peak_dbtp:.2f} | {res_b.metrics_after.true_peak_dbtp:.2f} |\n"
        f"| Crest | {res_a.metrics_after.crest_db:.2f} | {res_b.metrics_after.crest_db:.2f} |"
    )
    return SemanticABMasteringOut(
        artifact_id_a=res_a.master_wav_id,
        artifact_id_b=res_b.master_wav_id,
        message="A/B semantic mastering complete.",
        comparison_matrix=matrix,
        heatmap_a=heatmap_a,
        heatmap_b=heatmap_b,
    )


@mcp.tool()
async def analyze_and_optimize_governor(req: AnalyzeAndOptimizeGovernorIn, ctx: Context) -> AnalyzeAndOptimizeGovernorOut:
    """Recommend governor settings from source crest factor and preset intent."""
    _, _, _, maestro, audio, _, _ = _load_audio_artifact(req.audio_id, ctx)
    presets = maestro.get_presets()
    if req.preset_name not in presets:
        raise ValueError(f"unknown_preset: {req.preset_name}")

    await ctx.report_progress(40, 100, "Analyzing crest factor...")
    mono = maestro.to_mono(audio)
    peak_db = float(maestro.lin_to_db(maestro.peak(mono) + 1e-12))
    rms_db = float(maestro.lin_to_db(maestro.rms(mono) + 1e-12))
    crest = peak_db - rms_db
    base_gr = float(getattr(presets[req.preset_name], "governor_gr_limit_db", -3.0))

    if crest >= 14.0:
        steps = 3
        gr_limit = max(-1.4, base_gr)
        reason = (
            f"High crest factor ({crest:.2f} dB) indicates very dynamic material. "
            f"A short {steps}-step search with a conservative GR ceiling around {gr_limit:.2f} dB "
            "preserves transient openness."
        )
    elif crest >= 9.5:
        steps = 5
        gr_limit = round(min(-1.8, base_gr - 0.2), 2)
        reason = (
            f"Balanced crest factor ({crest:.2f} dB) can tolerate a moderate search depth. "
            f"A {steps}-step search and GR ceiling near {gr_limit:.2f} dB should stay musical."
        )
    else:
        steps = 7
        gr_limit = round(min(-3.6, base_gr - 1.0), 2)
        reason = (
            f"Low crest factor ({crest:.2f} dB) suggests dense material. "
            f"A deeper {steps}-step search with a GR ceiling near {gr_limit:.2f} dB can chase loudness safely."
        )
    await ctx.report_progress(100, 100, "Governor recommendation ready.")
    return AnalyzeAndOptimizeGovernorOut(
        crest_factor_db=round(crest, 2),
        recommended_governor_steps=steps,
        recommended_governor_gr_limit_db=gr_limit,
        music_theory_reasoning=reason,
    )


@mcp.tool()
async def ai_stem_remix(req: AiStemRemixIn, ctx: Context) -> AiStemRemixOut:
    """Analyze Demucs stems and return mix-balancing advice without leaving the current session model."""
    session_key, session_dir, _, maestro, audio, sr, _ = _load_audio_artifact(req.audio_id, ctx)
    if not getattr(maestro, "HAS_DEMUCS", False):
        raise ValueError("demucs_unavailable")

    await ctx.report_progress(20, 100, "Separating stems with Demucs...")
    stems, _ = await asyncio.to_thread(
        maestro.demucs_separate_stems,
        audio,
        sr,
        model_name="htdemucs",
        device="cpu",
        split=True,
        overlap=0.23,
        shifts=1,
    )
    await ctx.report_progress(80, 100, "Calculating stem loudness...")

    def stem_lufs(name: str) -> float:
        stem_audio = stems.get(name)
        if stem_audio is None:
            return -100.0
        return float(maestro.integrated_loudness_lufs(stem_audio, sr))

    report = StemLufsReport(
        vocals=round(stem_lufs("vocals"), 2),
        drums=round(stem_lufs("drums"), 2),
        bass=round(stem_lufs("bass"), 2),
        other=round(stem_lufs("other"), 2),
    )
    await ctx.report_progress(100, 100, "Stem analysis complete.")

    vocal_gap = report.vocals - report.other
    advice_parts = [
        f"Vocals vs other bed gap: {vocal_gap:.2f} dB.",
        "Modern vocal-forward masters usually keep vocals roughly 1 to 2 dB above the instrumental bed.",
    ]
    if vocal_gap < 1.0:
        advice_parts.append("Consider lifting vocals or trimming competing harmonic content.")
    elif vocal_gap > 2.5:
        advice_parts.append("Vocals already dominate; avoid extra vocal lift unless intelligibility is still poor.")
    if report.bass > report.drums + 2.0:
        advice_parts.append("Bass is leading the drum anchor; consider reducing bass slightly or nudging drums up.")
    advice_parts.append("Example override: stem_gains_db={'vocals': 1.0, 'bass': -0.5}.")

    return AiStemRemixOut(
        message=f"Demucs stems analyzed for session {session_key}.",
        stem_lufs=report,
        mix_theory_advice=" ".join(advice_parts),
    )


@mcp.custom_route("/", methods=["GET"])
async def root_info(_request: Request) -> JSONResponse:
    """Expose a lightweight root document so HTTP deployments are self-describing."""
    return JSONResponse(
        {
            "name": SERVER_NAME,
            "version": VERSION,
            "transport": HTTP_APP_TRANSPORT,
            "mcp_path": _http_path(),
            "health_path": "/health",
            "message": "AuralMind2 is running. Use the MCP endpoint at the configured mcp_path.",
        }
    )


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request: Request) -> JSONResponse:
    """Expose a simple health endpoint for hosts and smoke tests."""
    return JSONResponse(
        {
            "ok": True,
            "name": SERVER_NAME,
            "version": VERSION,
            "transport": HTTP_APP_TRANSPORT,
            "mcp_path": _http_path(),
        }
    )

# --- INITIALIZE MCP INSTRUCTIONS (LOG STARTUP) ---
try:
    _bs = bootstrap()
    log.info(
        "Server initialized with %s tools, %s resources, transport=%s.",
        len(_bs.tools),
        len(_bs.resources),
        _active_transport(),
    )
except Exception as e:
    log.warning(f"Failed to initialize server metadata: {e}")


def create_http_app() -> Any:
    """Expose an ASGI app for streamable HTTP hosts such as Render."""
    return mcp.http_app(
        path=_http_path(),
        transport=HTTP_APP_TRANSPORT,
        json_response=True,
    )


app = create_http_app()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    mcp.run(**_run_kwargs_for_active_transport())


if __name__ == "__main__":
    main()
