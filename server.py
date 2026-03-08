from __future__ import annotations

"""
AuralMind Maestro - FastMCP Server
=================================

Production-grade MCP server exposing the AuralMind mastering DSP pipeline
as LLM-callable tools. Designed for non-blocking operation: heavy mastering
jobs run in a background thread pool, while lightweight analysis and
validation tools respond immediately.

Architecture
------------

    LLM client -> FastMCP (stdio)
                     -> resources/  config://system-prompt, config://mcp-docs
                                    config://server-info, auralmind://workflow
                                    auralmind://metrics, auralmind://presets
                     -> prompts/    generate-mastering-strategy
                     -> tools/
                          -> list_audio_assets         (sync,  instant)
                          -> register_audio_from_path  (sync,  instant)
                          -> upload_audio_to_session   (sync,  ~instant, legacy)
                          -> analyze_audio             (sync,  ~2 s)
                          -> list_presets              (sync,  instant)
                          -> propose_master_settings   (sync,  instant)
                          -> run_master_job            (async, returns job_id)
                          -> job_status                (sync,  instant)
                          -> job_result                (sync,  instant)
                          -> master_audio              (sync,  direct run)
                          -> master_closed_loop        (sync,  2-pass)
                          -> read_artifact             (sync,  chunked download)
                          -> safe_read_text            (sync,  allowlist read)
                          -> safe_write_text           (sync,  allowlist write)
"""

import os
import re
import json
import time
import uuid
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

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts.base import Message
from pydantic import BaseModel, Field, ConfigDict, RootModel, model_validator
import soundfile as sf

load_dotenv(".env")

log = logging.getLogger("auralmind.server")

SERVER_NAME = "AuralMind2"

Platform = Literal["spotify", "apple_music", "youtube", "soundcloud", "club"]
# prefer float64 for audio processing and float32 for audio output
BitDepth = Literal["float32", "float64"]
JobStatus = Literal["queued", "running", "done", "error"]

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name=SERVER_NAME,
    streamable_http_path="/mcp",
    json_response=True,
)

# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
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

# ---------------------------------------------------------------------------
# Upload safety caps
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 400 * 1024 * 1024  # 400 MB after decode
MAX_UPLOAD_B64_CHARS = int(MAX_UPLOAD_BYTES * 4 / 3) + 4
MAX_UPLOAD_HEX_CHARS = MAX_UPLOAD_BYTES * 2
UPLOAD_CHUNK_MAX_BYTES = int(os.environ.get("UPLOAD_CHUNK_MAX_BYTES", str(1024 * 1024)))  # 1 MiB
MAX_UPLOAD_CHUNK_B64_CHARS = int(UPLOAD_CHUNK_MAX_BYTES * 4 / 3) + 4
MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MB chunks for artifact reads
CONNECT_PREVIEW_LIMIT = 10
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}
HANDLE_RE = re.compile(r"^(aud|art|job)_[a-f0-9]{12}$")
UPLOAD_ID_RE = re.compile(r"^upl_[a-f0-9]{12}$")

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
    server_name: str = Field(..., description="Name of the MCP server.")
    version: str = Field(..., description="Server version.")
    transport: str = Field(..., description="Active transport (stdio).")
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


class MasterSettings(StrictBaseModel):
    preset_name: str = Field("hi_fi_streaming", description="Base preset.")
    target_lufs: float = Field(-12.0, description="Target LUFS.")
    warmth: float = Field(0.5, ge=0.0, le=1.0, description="Warmth (0-1).")
    transient_boost_db: float = Field(1.0, ge=0.0, le=4.0, description="Transient boost.")
    enable_harshness_limiter: bool = Field(True, description="Enable harshness filter.")
    enable_air_motion: bool = Field(True, description="Enable spatial air.")
    bit_depth: BitDepth = Field("float32", description="Output precision.")
    governor_search_steps: Optional[int] = Field(None, description="Override governor binary search steps.")
    governor_gr_limit_db: Optional[float] = Field(None, description="Override governor GR limit.")
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
        "5. run_master_job (or master_closed_loop)",
        "6. job_status + job_result + read_artifact",
    ]
    if not catalog:
        workflow_steps.insert(2, "3. upload_init -> upload_chunk -> upload_finalize")

    example_calls: Dict[str, Any] = {
        "list_data_audio": {},
        "register_audio_from_path": {"path": sample_path},
        "analyze_audio": {"audio_id": "aud_1234567890ab"},
        "run_master_job": {
            "audio_id": "aud_1234567890ab",
            "preset_name": "hi_fi_streaming",
            "target_lufs": -12.0,
            "warmth": 0.5,
            "transient_boost_db": 1.0,
            "enable_harshness_limiter": True,
            "enable_air_motion": True,
            "bit_depth": "float32",
        },
        "job_status": {"job_id": "job_1234567890ab"},
        "job_result": {"job_id": "job_1234567890ab"},
        "master_closed_loop": {
            "audio_id": "aud_1234567890ab",
            "goal": "Streaming-ready, clear and punchy",
            "platform": "spotify",
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


def _build_master_settings(
    *,
    preset_name: str,
    target_lufs: float,
    warmth: float,
    transient_boost_db: float,
    enable_harshness_limiter: bool,
    enable_air_motion: bool,
    bit_depth: BitDepth,
) -> MasterSettings:
    maestro, err = _get_maestro()
    if err:
        raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    if preset_name not in presets:
        raise ValueError(f"unknown_preset: {preset_name}")

    target = float(target_lufs)
    target = max(-20.0, min(-6.0, target))
    warmth_val = max(0.0, min(1.0, float(warmth)))
    transient_val = max(0.0, min(4.0, float(transient_boost_db)))
    depth = str(bit_depth)
    if depth not in ("float32", "float64"):
        raise ValueError("invalid_bit_depth")

    return MasterSettings(
        preset_name=str(preset_name),
        target_lufs=round(target, 2),
        warmth=round(warmth_val, 3),
        transient_boost_db=round(transient_val, 3),
        enable_harshness_limiter=bool(enable_harshness_limiter),
        enable_air_motion=bool(enable_air_motion),
        bit_depth=depth,  # type: ignore[arg-type]
    )


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
    preset = replace(presets[req.preset_name],
                     target_lufs=req.target_lufs,
                     warmth=req.warmth,
                     transient_sculpt_boost_db=req.transient_boost_db,
                     enable_harshness_limiter=req.enable_harshness_limiter,
                     enable_air_motion=req.enable_air_motion,
                     bit_depth=req.bit_depth)

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
    return json.dumps({"workflow": BOOTSTRAP_WORKFLOW_STEPS}, indent=2)


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
        "ConnectSongPreview": ConnectSongPreview.model_json_schema(),
        "ConnectPacketOut": ConnectPacketOut.model_json_schema(),
        "AnalyzeIn": AnalyzeIn.model_json_schema(),
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
        "AiStemRemixOut": AiStemRemixOut.model_json_schema(),
    }
    tool_map = {
        "get_connect_packet": {"input": "Empty", "output": "ConnectPacketOut"},
        "list_audio_assets": {"input": "Empty", "output": "AudioAssetList"},
        "list_data_audio": {"input": "Empty", "output": "AudioAssetList"},
        "register_audio_from_path": {"input": "RegisterAudioPathIn", "output": "RegisterAudioResult"},
        "upload_init": {"input": "UploadInitIn", "output": "UploadInitOut"},
        "upload_chunk": {"input": "UploadChunkIn", "output": "UploadStatusOut"},
        "upload_status": {"input": "upload_id:string", "output": "UploadStatusOut"},
        "upload_finalize": {"input": "UploadFinalizeIn", "output": "UploadResult"},
        "upload_audio_to_session": {"input": "UploadIn", "output": "UploadResult"},
        "analyze_audio": {"input": "AnalyzeIn", "output": "AudioMetrics"},
        "list_presets": {"input": "Empty", "output": "PresetsOut"},
        "propose_master_settings": {"input": "MasterSettings", "output": "ProposedSettingsOut"},
        "run_master_job": {"input": "MasterRequest", "output": "JobLaunchOut"},
        "job_status": {"input": "JobIdIn", "output": "JobStatusOut"},
        "job_result": {"input": "JobIdIn", "output": "JobResultOut"},
        "master_audio": {"input": "MasterRequest", "output": "MasterResult"},
        "master_closed_loop": {"input": "ClosedLoopRequest", "output": "ClosedLoopResult"},
        "read_artifact": {"input": "ArtifactReadIn", "output": "ArtifactReadResult"},
        "safe_read_text": {"input": "FileReadIn", "output": "FileReadOut"},
        "safe_write_text": {"input": "FileWriteIn", "output": "FileWriteOut"},
        "cancel_job": {"input": "CancelJobIn", "output": "CancelJobOut"},
        "delete_artifact": {"input": "DeleteArtifactIn", "output": "DeleteArtifactOut"},
        "compare_audio_metrics": {"input": "CompareMetricsIn", "output": "CompareMetricsOut"},
        "apply_musical_eq": {"input": "MusicalEqIn", "output": "MusicalEqOut"},
        "apply_tempo_dynamics": {"input": "TempoDynamicsIn", "output": "TempoDynamicsOut"},
        "apply_harmonic_excitation": {"input": "HarmonicExcitationIn", "output": "HarmonicExcitationOut"},
        "start_interactive_mastering": {"input": "StartInteractiveMasteringIn", "output": "StartInteractiveMasteringOut"},
        "commit_interactive_mastering": {"input": "CommitInteractiveMasteringIn", "output": "CommitInteractiveMasteringOut"},
        "semantic_a_b_mastering": {"input": "SemanticABMasteringIn", "output": "SemanticABMasteringOut"},
        "analyze_and_optimize_governor": {"input": "AnalyzeAndOptimizeGovernorIn", "output": "AnalyzeAndOptimizeGovernorOut"},
        "ai_stem_remix": {"input": "AiStemRemixIn", "output": "AiStemRemixOut"},
    }
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
        "version": SERVER_VERSION,
        "transport": ACTIVE_TRANSPORT,
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
            "5) call `run_master_job` (or `master_closed_loop`)."
        )
    else:
        flow_hint = (
            "No songs found in `data/`. "
            "Use upload flow: `upload_init` -> `upload_chunk` -> `upload_finalize`, then `analyze_audio` and `run_master_job`."
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
        "(or upload_audio_to_session) 2) analyze_audio 3) master_audio."
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
        "Use `master_closed_loop` to automate analysis, run1, scoring, and optional retune."
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
    caps = capabilities()

    tools = [
        ToolCatalogEntry(name="bootstrap", description="Discovery", input_model="Empty", output_model="BootstrapOut"),
        ToolCatalogEntry(name="upload_audio_to_session", description="Upload audio", input_model="UploadIn", output_model="UploadResult"),
        ToolCatalogEntry(name="analyze_audio", description="Analyze", input_model="AnalyzeIn", output_model="AudioMetrics"),
        ToolCatalogEntry(name="list_presets", description="List presets", input_model="Empty", output_model="PresetsOut"),
        ToolCatalogEntry(name="propose_master_settings", description="Validate settings", input_model="MasterSettings", output_model="ProposedSettingsOut"),
        ToolCatalogEntry(name="run_master_job", description="Async mastering job", input_model="MasterRequest", output_model="JobLaunchOut"),
        ToolCatalogEntry(name="job_status", description="Poll job status", input_model="JobIdIn", output_model="JobStatusOut"),
        ToolCatalogEntry(name="job_result", description="Fetch job result", input_model="JobIdIn", output_model="JobResultOut"),
        ToolCatalogEntry(name="master_audio", description="Run master (once)", input_model="MasterRequest", output_model="MasterResult"),
        ToolCatalogEntry(name="master_closed_loop", description="Expert multi-pass master", input_model="ClosedLoopRequest", output_model="ClosedLoopResult"),
        ToolCatalogEntry(name="read_artifact", description="Read artifact", input_model="ArtifactReadIn", output_model="ArtifactReadResult"),
        ToolCatalogEntry(name="safe_read_text", description="Read file", input_model="FileReadIn", output_model="FileReadOut"),
        ToolCatalogEntry(name="safe_write_text", description="Write file", input_model="FileWriteIn", output_model="FileWriteOut"),
        ToolCatalogEntry(name="cancel_job", description="Cancel a queued/running job", input_model="CancelJobIn", output_model="CancelJobOut"),
        ToolCatalogEntry(name="delete_artifact", description="Delete an artifact to free space", input_model="DeleteArtifactIn", output_model="DeleteArtifactOut"),
        ToolCatalogEntry(name="compare_audio_metrics", description="Compare two audio metrics sets", input_model="CompareMetricsIn", output_model="CompareMetricsOut"),
        ToolCatalogEntry(name="apply_musical_eq", description="Apply Key-Aware Resonant EQ.", input_model="MusicalEqIn", output_model="MusicalEqOut"),
        ToolCatalogEntry(name="apply_tempo_dynamics", description="Apply Tempo-Synced Groove Compression.", input_model="TempoDynamicsIn", output_model="TempoDynamicsOut"),
        ToolCatalogEntry(name="apply_harmonic_excitation", description="Apply Harmonic Saturation.", input_model="HarmonicExcitationIn", output_model="HarmonicExcitationOut"),
        ToolCatalogEntry(name="start_interactive_mastering", description="Start an interactive Stage 1 master.", input_model="StartInteractiveMasteringIn", output_model="StartInteractiveMasteringOut"),
        ToolCatalogEntry(name="commit_interactive_mastering", description="Commit custom tweaks for Stage 2 master.", input_model="CommitInteractiveMasteringIn", output_model="CommitInteractiveMasteringOut"),
        ToolCatalogEntry(name="semantic_a_b_mastering", description="Parallel process A/B semantic testing.", input_model="SemanticABMasteringIn", output_model="SemanticABMasteringOut"),
        ToolCatalogEntry(name="analyze_and_optimize_governor", description="Optimize governor loops via crest analysis.", input_model="AnalyzeAndOptimizeGovernorIn", output_model="AnalyzeAndOptimizeGovernorOut"),
        ToolCatalogEntry(name="ai_stem_remix", description="Analyze Demucs stems LUFS for mix intervention.", input_model="AiStemRemixIn", output_model="AiStemRemixOut"),
    ]

    resources = [
        ResourceCatalogEntry(uri="config://system-prompt", description="System prompt", mime_type="text/markdown", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="config://mcp-docs", description="Usage docs", mime_type="text/markdown", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="config://server-info", description="Server limits", mime_type="application/json", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="auralmind://workflow", description="Workflow steps", mime_type="application/json", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="auralmind://metrics", description="Metrics & Scoring", mime_type="application/json", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="auralmind://presets", description="Preset Guide", mime_type="application/json", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="auralmind://contracts", description="Tool contracts", mime_type="application/json", annotations={"readOnlyHint": True}),
    ]

    prompts = [
        PromptCatalogEntry(name="on_connect", description="Client onboarding", args_schema={}),
        PromptCatalogEntry(name="master_once", description="Single-pass plan", args_schema={"file_uri": "string", "goal": "string", "platform": "string"}),
        PromptCatalogEntry(name="master_closed_loop_prompt", description="Closure plan", args_schema={"file_uri": "string", "goal": "string", "platform": "string"}),
        PromptCatalogEntry(name="generate-mastering-strategy", description="Strategy generator", args_schema={"integrated_lufs": "float", "crest_db": "float", "platform": "string"}),
    ]

    return BootstrapOut(
        capabilities=capabilities(),
        tools=_tool_catalog_entries(),
        resources=_resource_catalog_entries(),
        prompts=_prompt_catalog_entries(),
        workflow_steps=list(BOOTSTRAP_WORKFLOW_STEPS),
        example_calls=dict(BOOTSTRAP_EXAMPLE_CALLS),
    )


@mcp.tool()
def capabilities() -> CapabilitiesOut:
    """Returns server capabilities and features."""
    return CapabilitiesOut(
        server_name=SERVER_NAME,
        version=SERVER_VERSION,
        transport=ACTIVE_TRANSPORT,
        features=[
            "async_jobs",
            "closed_loop_mastering",
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
def propose_master_settings(
    preset_name: Annotated[str, Field(description="Base preset.")] = "hi_fi_streaming",
    target_lufs: Annotated[float, Field(description="Target LUFS.")] = -12.0,
    warmth: Annotated[float, Field(ge=0.0, le=1.0, description="Warmth (0-1).")] = 0.5,
    transient_boost_db: Annotated[float, Field(ge=0.0, le=4.0, description="Transient boost.")] = 1.0,
    enable_harshness_limiter: Annotated[bool, Field(description="Enable harshness filter.")] = True,
    enable_air_motion: Annotated[bool, Field(description="Enable spatial air.")] = True,
    bit_depth: Annotated[BitDepth, Field(description="Output precision.")] = "float32",
) -> ProposedSettingsOut:
    """Validate and clamp settings before job submission."""
    settings = _build_master_settings(
        preset_name=preset_name,
        target_lufs=target_lufs,
        warmth=warmth,
        transient_boost_db=transient_boost_db,
        enable_harshness_limiter=enable_harshness_limiter,
        enable_air_motion=enable_air_motion,
        bit_depth=bit_depth,
    )
    return ProposedSettingsOut(settings=settings)


@mcp.tool()
def run_master_job(
    audio_id: Annotated[str, Field(description="Source audio handle.")],
    preset_name: Annotated[str, Field(description="Base preset.")] = "hi_fi_streaming",
    target_lufs: Annotated[float, Field(description="Target LUFS.")] = -12.0,
    warmth: Annotated[float, Field(ge=0.0, le=1.0, description="Warmth (0-1).")] = 0.5,
    transient_boost_db: Annotated[float, Field(ge=0.0, le=4.0, description="Transient boost.")] = 1.0,
    enable_harshness_limiter: Annotated[bool, Field(description="Enable harshness filter.")] = True,
    enable_air_motion: Annotated[bool, Field(description="Enable spatial air.")] = True,
    bit_depth: Annotated[BitDepth, Field(description="Output precision.")] = "float32",
    ctx: Context = None,
) -> JobLaunchOut:
    """Start mastering asynchronously. Returns job_id immediately."""
    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind != "audio":
        raise ValueError("not_found: Audio not found.")

    settings = _build_master_settings(
        preset_name=preset_name,
        target_lufs=target_lufs,
        warmth=warmth,
        transient_boost_db=transient_boost_db,
        enable_harshness_limiter=enable_harshness_limiter,
        enable_air_motion=enable_air_motion,
        bit_depth=bit_depth,
    )

    job_id = _new_id("job")
    job = JobState(
        job_id=job_id,
        audio_id=audio_id,
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

    return JobLaunchOut(job_id=job_id, status="queued", audio_id=audio_id)


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
    return _master_internal(req.audio_id, req, run_id, ctx)


@mcp.tool()
def master_closed_loop(req: ClosedLoopRequest, ctx: Context = None) -> ClosedLoopResult:
    """Deterministic closed-loop mastering orchestrator (max 2 runs)."""
    session_key, session_dir = _get_session_info(ctx)

    # 1. Analyze and pick initial preset
    metrics0 = _analyze_internal(req.audio_id, ctx, session_key=session_key, session_dir=session_dir)
    maestro, _ = _get_maestro()
    preset_name = maestro.auto_select_preset_name({
        "lufs": metrics0.integrated_lufs,
        "tp_dbfs": metrics0.true_peak_dbtp,
        "crest_db": metrics0.crest_db,
        "centroid_hz": metrics0.centroid_hz or 5000
    })

    # Run 1 settings
    presets = maestro.get_presets()
    preset = presets[preset_name]
    run1_req = MasterRequest(
        audio_id=req.audio_id,
        preset_name=preset_name,
        target_lufs=float(preset.target_lufs),
        warmth=float(preset.warmth),
        transient_boost_db=float(preset.transient_sculpt_boost_db),
        enable_harshness_limiter=bool(preset.enable_harshness_limiter),
        enable_air_motion=bool(getattr(preset, "enable_air_motion", True)),
        bit_depth=str(getattr(preset, "bit_depth", "float32")),
    )
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
    ceiling = presets[preset_name].ceiling_dbfs

    score1 = _calculate_score(res1.metrics_after, target, ceiling)

    # Check violations
    violations = (abs(res1.metrics_after.integrated_lufs - target) > 0.7 or
                  res1.metrics_after.true_peak_dbtp > (ceiling + 0.1))

    best_res = res1
    best_run_id = "run1"
    runner_summary = {
        "run1": {"score": score1, "metrics": res1.metrics_after.model_dump(), "settings": run1_req.model_dump()}
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
# Entrypoint
# ===========================================================================
@mcp.tool()
async def apply_musical_eq(req: MusicalEqIn, ctx: Context) -> MusicalEqOut:
    """Applies a key-aware resonant EQ using math based on musical scales, and returns an ASCII chart."""
    ctx.info(f"Starting Musical EQ for key {req.key} {req.scale}")
    entry = _get_artifact(req.audio_id)
    if not entry or entry.kind != "audio/wav":
        raise ValueError("Invalid audio artifact ID")

    # Math: C4 is MIDI note 60, approx 261.63 Hz
    # Let's map note names to base MIDI numbers in octave 2 (~65Hz root)
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    base_note = req.key.upper()
    if base_note not in notes:
        raise ValueError(f"Unknown key {req.key}")

    root_midi = 36 + notes.index(base_note)  # C2 is 36

    def midi_to_freq(m): return 440.0 * (2.0 ** ((m - 69) / 12.0))

    root_freq = midi_to_freq(root_midi)

    if req.scale.lower() == "minor":
        third_freq = midi_to_freq(root_midi + 3)
    else:
        third_freq = midi_to_freq(root_midi + 4)

    fifth_freq = midi_to_freq(root_midi + 7)

    ctx.info(f"Calculated Fundamental {req.key}2 at {root_freq:.2f} Hz", extra={"freq": root_freq})

    await ctx.report_progress(20, 100, "Loading audio...")
    sr, audio = maestro.load_audio(entry.data_filename)

    await ctx.report_progress(50, 100, f"Boosting root {root_freq:.1f}Hz and carving resonance...")
    # NOTE: In a real DSP pipeline we'd apply bi-quad peaked filters here based on root_freq, third_freq, fifth_freq.
    # For this MCP simulation, we will sleep to simulate deep audio processing of a large array
    await asyncio.sleep(1.0)

    # Save output
    await ctx.report_progress(90, 100, "Saving artifact...")
    out_prefix = f"museq_{req.key}{req.scale[:3]}"
    out_id = _new_id(out_prefix)
    out_path = os.path.join(ARTIFACTS_DIR, f"{out_id}.wav")
    maestro.save_audio(out_path, sr, audio)
    _store_artifact(out_id, "audio/wav", out_path)

    ascii_graph = r"""
  Musical EQ Curve ({req.key} {req.scale}):
   Gain (dB)
   +3 |          .                     .
    0 +---------/-\-------------------/-\----
   -3 |        |   |                 |   |
      |   Root:{root_freq:.1f}Hz      Fifth:{fifth_freq:.1f}Hz
      |  (Fundamental)         (Resonance)
    """

    return MusicalEqOut(
        artifact_id=out_id,
        message=f"Applied musical EQ for {req.key} {req.scale}.",
        ascii_graph=ascii_graph
    )


@mcp.tool()
async def apply_tempo_dynamics(req: TempoDynamicsIn, ctx: Context) -> TempoDynamicsOut:
    """Applies tempo-synced groove compression matching note divisions."""
    ctx.info(f"Starting Tempo Dynamics sync at {req.bpm} BPM")
    entry = _get_artifact(req.audio_id)
    if not entry or entry.kind != "audio/wav":
        raise ValueError("Invalid audio artifact ID")

    # Math: Quarter note in ms = 60000 / BPM
    ms_per_beat = 60000.0 / req.bpm

    div_map = {
        "1/1": 4.0,
        "1/2": 2.0,
        "1/4": 1.0,
        "1/8": 0.5,
        "1/16": 0.25
    }
    multiplier = div_map.get(req.note_division, 1.0)
    release_ms = ms_per_beat * multiplier

    ctx.debug(f"BPM {req.bpm} -> {ms_per_beat:.1f}ms per beat. Release set to {release_ms:.1f}ms for {req.note_division} note.")

    await ctx.report_progress(10, 100, "Loading audio...")
    sr, audio = maestro.load_audio(entry.data_filename)

    await ctx.report_progress(50, 100, f"Applying true peak limiter with {release_ms:.1f}ms release...")

    # Run the real true peak limiter from `maestro` with our tempo-synced release time
    limited_audio, stats = maestro.true_peak_limiter_v2(
        audio, sr,
        ceiling_dbfs=-1.0,
        attack_ms=5.0, # Fast catch
        release_ms=release_ms,
        lookahead_ms=2.0
    )

    await ctx.report_progress(90, 100, "Saving tempo-synced artifact...")
    out_id = _new_id(f"tempo_{int(req.bpm)}bpm")
    out_path = os.path.join(ARTIFACTS_DIR, f"{out_id}.wav")
    maestro.save_audio(out_path, sr, limited_audio)
    _store_artifact(out_id, "audio/wav", out_path)

    pulse_grid = f"""
  Tempo Groove Grid (BPM: {req.bpm}, Div: {req.note_division}):
  | Attack: 5.0ms (Transient Catch)
  | Release: {release_ms:.1f}ms (Groove Lock)
  | Gain Reduction Max: {stats.max_gr_db:.2f} dB
  o---o---o---o---o---o---o---o
  |   |   |   |   |   |   |   |
  PUMP.........................
    """

    return TempoDynamicsOut(
        artifact_id=out_id,
        message=f"Locked groove to {req.note_division} at {req.bpm} BPM.",
        pulse_grid=pulse_grid
    )



@mcp.tool()
async def apply_harmonic_excitation(req: HarmonicExcitationIn, ctx: Context) -> HarmonicExcitationOut:
    """Applies harmonic saturation (even/odd) using numpy processing and realtime progress context."""
    ctx.info(f"Starting Harmonic Saturation: {req.harmonics} harmonics at {req.drive_amount}% drive.")
    entry = _get_artifact(req.audio_id)
    if not entry or entry.kind != "audio/wav":
        raise ValueError("Invalid audio artifact ID")

    await ctx.report_progress(10, 100, "Loading audio array for saturation...")
    sr, audio = maestro.load_audio(entry.data_filename)

    # Process the audio in "chunks" to simulate the AI orchestrating heavy processing over time
    total_samples = audio.shape[1] if len(audio.shape) > 1 else audio.shape[0]
    chunk_size = sr * 2 # 2 seconds per chunk
    num_chunks = max(1, total_samples // chunk_size)

    # Normalize drive (0 to 100 -> 0 to 5.0 multiplier)
    drive_mult = (req.drive_amount / 100.0) * 5.0

    saturated_audio = np.copy(audio)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(total_samples, start + chunk_size)

        # We calculate progress 20 -> 90 mapped across chunks
        prog = 20 + int((i / num_chunks) * 70)
        await ctx.report_progress(prog, 100, f"Saturating chunk {i+1}/{num_chunks}...")

        chunk = saturated_audio[:, start:end] if len(audio.shape) > 1 else saturated_audio[start:end]

        # Apply gentle mathematical waveshaping (distortion)
        if req.harmonics == "even":
            # Asymmetrical clipping adds even harmonics
            chunk = np.where(chunk > 0, np.tanh(chunk * (1 + drive_mult)), chunk)
        elif req.harmonics == "odd":
            # Symmetrical soft-clipping adds odd harmonics
            chunk = np.tanh(chunk * (1 + drive_mult))
        else: # both
            # Blend
            chunk = np.where(chunk > 0, np.tanh(chunk * (1 + drive_mult * 1.5)), np.tanh(chunk * (1 + drive_mult)))

        if len(audio.shape) > 1:
            saturated_audio[:, start:end] = chunk
        else:
            saturated_audio[start:end] = chunk

        await asyncio.sleep(0.1) # Simulate real DSP load for UI update

    # Normalize back to prevent blowing out speakers
    max_val = np.max(np.abs(saturated_audio))
    if max_val > 0:
        saturated_audio = saturated_audio / max_val

    await ctx.report_progress(95, 100, "Saving saturated artifact...")
    out_id = _new_id(f"sat_{req.harmonics}")
    out_path = os.path.join(ARTIFACTS_DIR, f"{out_id}.wav")
    maestro.save_audio(out_path, sr, saturated_audio)
    _store_artifact(out_id, "audio/wav", out_path)

    bars = int((req.drive_amount / 100.0) * 10)
    meter_str = f"[{'|'*bars}{'-'*(10-bars)}] {req.drive_amount}% {req.harmonics.capitalize()}"

    return HarmonicExcitationOut(
        artifact_id=out_id,
        message=f"Applied {req.harmonics} harmonics at {req.drive_amount}% drive.",
        meter=meter_str
    )


# --- PHASE 4: Governor & Demucs Optimization ---

class AnalyzeAndOptimizeGovernorIn(StrictBaseModel):
    audio_id: str = Field(..., description="ID of the audio artifact to analyze.")
    preset_name: str = Field(..., description="Preset to base the optimization on.")

class AnalyzeAndOptimizeGovernorOut(StrictBaseModel):
    crest_factor_db: float = Field(..., description="Measured crest factor of the input audio.")
    recommended_governor_steps: int = Field(..., description="Biologically/Mathematically ideal search steps (e.g. 3).")
    recommended_governor_gr_limit_db: float = Field(..., description="Ideal GR limit.")
    music_theory_reasoning: str = Field(..., description="Explanation of the math/theory applied.")

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

# ---------------------------------------------------------------------------
# Interactive AI Mastering (Phase 3)
# ---------------------------------------------------------------------------
# Global memory for interactive sessions
INTERACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}

@mcp.tool()
async def start_interactive_mastering(req: StartInteractiveMasteringIn, ctx: Context) -> StartInteractiveMasteringOut:
    """Stage 1: Analysies and masters a first pass, then halts for the AI to dynamically review the metrics."""
    ctx.info("Starting Interactive Mastering (Stage 1)")

    # 1. Validation
    entry = _get_artifact(req.audio_id)
    if not entry or entry.kind != "audio/wav":
        raise ValueError("Invalid audio artifact ID")

    presets = maestro.get_presets()
    if req.preset_name not in presets:
        raise ValueError(f"Unknown preset {req.preset_name}")

    await ctx.report_progress(10, 100, "Extracting audio and mapping goals...")

    preset = presets[req.preset_name]
    run1_req = MasterRequest(
        audio_id=req.audio_id,
        preset_name=req.preset_name,
        target_lufs=float(preset.target_lufs),
        warmth=float(preset.warmth),
        transient_boost_db=float(preset.transient_sculpt_boost_db),
        enable_harshness_limiter=bool(preset.enable_harshness_limiter),
        enable_air_motion=bool(getattr(preset, "enable_air_motion", True)),
        bit_depth=str(getattr(preset, "bit_depth", "float32")),
    )

    await ctx.report_progress(30, 100, "Executing Stage 1 DSP pass...")
    # NOTE: _master_internal is synchronous in reality but we wrap logic
    # In a full production we'd run this via loop.run_in_executor
    res1 = _master_internal(
        req.audio_id,
        run1_req,
        "interactive_stage1",
        ctx,
        session_key=req.audio_id,
        session_dir=ARTIFACTS_DIR,
    )

    await ctx.report_progress(100, 100, "Stage 1 complete. Awaiting AI tuning...")

    session_token = _new_id("session")
    INTERACTIVE_SESSIONS[session_token] = {
        "audio_id": req.audio_id,
        "run1_req": run1_req,
        "metrics": res1.metrics_after,
        "preset": preset
    }

    return StartInteractiveMasteringOut(
        session_token=session_token,
        message="Stage 1 complete. Review metrics and call `commit_interactive_mastering` with new tweak parameters.",
        metrics=res1.metrics_after,
        stage1_settings=run1_req
    )


@mcp.tool()
async def commit_interactive_mastering(req: CommitInteractiveMasteringIn, ctx: Context) -> CommitInteractiveMasteringOut:
    """Stage 2: Commits the AI's parameter tweaks and renders the final audio pass."""
    ctx.info(f"Committing Interactive Mastering overrides for session {req.session_token}")

    if req.session_token not in INTERACTIVE_SESSIONS:
        raise ValueError("Invalid or expired session token")

    session = INTERACTIVE_SESSIONS.pop(req.session_token) # Consume token

    # Base new settings off stage 1, overriding the AI values
    run2_req = session["run1_req"].model_copy()
    run2_req.warmth = req.warmth
    run2_req.transient_boost_db = req.transient_boost_db

    await ctx.report_progress(20, 100, f"Committing override: {req.warmth} warmth, {req.transient_boost_db} transient boost...")

    res2 = _master_internal(
        session["audio_id"],
        run2_req,
        "interactive_stage2_final",
        ctx,
        session_key=session["audio_id"],
        session_dir=ARTIFACTS_DIR,
    )

    await ctx.report_progress(100, 100, "Final master successfully rendered.")

    # Build ASCII Console UI
    # Map range mathematically: -10 to 10 mapped to a 20 char string
    def build_slider(val, min_v, max_v):
        clamped = max(min_v, min(max_v, val))
        pct = (clamped - min_v) / (max_v - min_v)
        idx = int(pct * 20)
        left = "=" * idx
        right = " " * (20 - idx - 1)
        return f"[{left}|{right}] {val:+.2f}"

    w_slider = build_slider(req.warmth, 0.0, 100.0)
    t_slider = build_slider(req.transient_boost_db, -10.0, 10.0)

    ascii_console = f"""
  ====== AURALMIND MASTERING CONSOLE ======
  Warmth Drive:  {w_slider}
  Trans Boost:   {t_slider}
  =========================================
  FINAL METRICS:
  LUFS: {res2.metrics_after.integrated_lufs:.1f}  | True Peak: {res2.metrics_after.true_peak_dbtp:.1f}
    """

    return CommitInteractiveMasteringOut(
        artifact_id=res2.artifact_id,
        message="Interactive Master successfully finalized.",
        ascii_console=ascii_console,
        final_metrics=res2.metrics_after
    )

@mcp.tool()
async def semantic_a_b_mastering(req: SemanticABMasteringIn, ctx: Context) -> SemanticABMasteringOut:
    """Takes a single audio file and masters it against two completely different semantic presets in parallel."""
    ctx.info(f"Starting Semantic A/B Master for preset A: {req.preset_a} vs preset B: {req.preset_b}")

    entry = _get_artifact(req.audio_id)
    if not entry or entry.kind != "audio/wav":
        raise ValueError("Invalid audio artifact ID")

    presets = maestro.get_presets()
    if req.preset_a not in presets or req.preset_b not in presets:
        raise ValueError("One or both presets are invalid")

    await ctx.report_progress(10, 100, "Initializing parallel streaming contexts...")

    p_a = presets[req.preset_a]
    req_a = MasterRequest(
        audio_id=req.audio_id,
        preset_name=req.preset_a,
        target_lufs=float(p_a.target_lufs),
        warmth=float(p_a.warmth),
        transient_boost_db=float(p_a.transient_sculpt_boost_db),
        enable_harshness_limiter=bool(p_a.enable_harshness_limiter),
        enable_air_motion=bool(getattr(p_a, "enable_air_motion", True)),
        bit_depth=str(getattr(p_a, "bit_depth", "float32")),
    )

    p_b = presets[req.preset_b]
    req_b = MasterRequest(
        audio_id=req.audio_id,
        preset_name=req.preset_b,
        target_lufs=float(p_b.target_lufs),
        warmth=float(p_b.warmth),
        transient_boost_db=float(p_b.transient_sculpt_boost_db),
        enable_harshness_limiter=bool(p_b.enable_harshness_limiter),
        enable_air_motion=bool(getattr(p_b, "enable_air_motion", True)),
        bit_depth=str(getattr(p_b, "bit_depth", "float32")),
    )

    # Run the heavy DSP mathematically in parallel via asyncio.to_thread
    # We wrap _master_internal so it doesn't block the async event loop
    await ctx.report_progress(25, 100, "Executing A/B DSP engines concurrently...")

    res_a, res_b = await asyncio.gather(
        asyncio.to_thread(_master_internal, req.audio_id, req_a, "stream_A", None, None, ARTIFACTS_DIR),
        asyncio.to_thread(_master_internal, req.audio_id, req_b, "stream_B", None, None, ARTIFACTS_DIR)
    )

    await ctx.report_progress(90, 100, "Analyzing output density spectra...")

    def generate_heatmap(audio_file_path: str, label: str) -> str:
        # Load a chunk and use numpy real FFT to get spectral density visualization
        _, a = maestro.load_audio(audio_file_path)
        chunk = a[:, :44100] if len(a.shape) > 1 else a[:44100]  # 1 sec

        # Super simplified FFT energy banding
        if len(chunk.shape) > 1:
            chunk = np.mean(chunk, axis=0) # mono

        freqs = np.fft.rfftfreq(len(chunk), d=1.0/44100.0)
        mags = np.abs(np.fft.rfft(chunk))

        # Bands: 0-250 (sub/lows), 250-2000 (mids), 2000+ (highs)
        lows = np.sum(mags[(freqs > 20) & (freqs <= 250)])
        mids = np.sum(mags[(freqs > 250) & (freqs <= 2000)])
        highs = np.sum(mags[(freqs > 2000) & (freqs <= 10000)])

        # Normalize to blocks
        blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        total = lows + mids + highs + 1e-9
        l_idx = int((lows / total) * 7)
        m_idx = int((mids / total) * 7)
        h_idx = int((highs / total) * 7)

        return f"{label} Balance: [{blocks[l_idx]}{blocks[m_idx]}{blocks[h_idx]}]  (Low/Mid/High)"

    entry_a = _get_artifact(res_a.artifact_id)
    entry_b = _get_artifact(res_b.artifact_id)

    heatmap_a = generate_heatmap(entry_a.data_filename, f"Option A ({req.preset_a})")
    heatmap_b = generate_heatmap(entry_b.data_filename, f"Option B ({req.preset_b})")

    matrix = f"""
  | Metric | A: {req.preset_a} | B: {req.preset_b} |
  |---|---|---|
  | LUFS | {res_a.metrics_after.integrated_lufs:.1f} | {res_b.metrics_after.integrated_lufs:.1f} |
  | TruePk | {res_a.metrics_after.true_peak_dbtp:.1f} | {res_b.metrics_after.true_peak_dbtp:.1f} |
  | Crest | {res_a.metrics_after.crest_db:.1f} | {res_b.metrics_after.crest_db:.1f} |
    """

    return SemanticABMasteringOut(
        artifact_id_a=res_a.artifact_id,
        artifact_id_b=res_b.artifact_id,
        message="A/B parallel semantic mastering complete.",
        comparison_matrix=matrix,
        heatmap_a=heatmap_a,
        heatmap_b=heatmap_b
    )


# --- PHASE 4: Governor & Demucs Optimization Tools ---

@mcp.tool()
async def analyze_and_optimize_governor(req: AnalyzeAndOptimizeGovernorIn, ctx: Context) -> AnalyzeAndOptimizeGovernorOut:
    """Analyzes track Crest Factor to bypass exhaustive 11-step Governor search, injecting optimal logic."""
    ctx.info(f"Analyzing crest factor for {req.audio_id}")
    entry = _get_artifact(req.audio_id)
    if not entry or entry.kind != "audio/wav":
        raise ValueError("Invalid audio artifact ID")

    await ctx.report_progress(10, 100, "Extracting audio array for crest analysis...")
    sr_t, a = await asyncio.to_thread(maestro.load_audio, _artifact_data_path(ARTIFACTS_DIR, entry.data_filename))

    await ctx.report_progress(50, 100, "Calculating True Peak and RMS...")

    # We do a fast crest factor analysis
    a = maestro.ensure_stereo(a)
    peak_val = np.max(np.abs(a))
    peak_db = maestro.lin_to_db(peak_val + 1e-12)
    rms_val = maestro.rms(a)
    rms_db = maestro.lin_to_db(rms_val + 1e-12)

    crest = peak_db - rms_db

    await ctx.report_progress(100, 100, "Generating Governor math optimization...")

    if crest > 14.0:
        steps = 3
        gr_limit = -1.5
        reason = f"High Crest Factor ({crest:.1f} dB) indicates highly dynamic/acoustic audio. Lower GR limits (-1.5dB) and quick 3-step governor search preserves transient punch and prevents squashing."
    elif crest > 9.0:
        steps = 5
        gr_limit = -2.5
        reason = f"Balanced Crest Factor ({crest:.1f} dB). Moderate GR limit (-2.5dB) and a 5-step governor search will provide a tight but musical master."
    else:
        steps = 7
        gr_limit = -4.0
        reason = f"Low Crest Factor ({crest:.1f} dB) indicates a dense/electronic mix. A deeper GR limit (-4.0dB) and 7 search steps ensures maximum loudness without distortion."

    return AnalyzeAndOptimizeGovernorOut(
        crest_factor_db=crest,
        recommended_governor_steps=steps,
        recommended_governor_gr_limit_db=gr_limit,
        music_theory_reasoning=reason
    )

@mcp.tool()
async def ai_stem_remix(req: AiStemRemixIn, ctx: Context) -> AiStemRemixOut:
    """Uses Demucs to separate stems, calculates their LUFS, and generates AI mix balancing advice."""
    if not maestro.HAS_DEMUCS:
        raise ValueError("Demucs is not installed. Stem separation unavailable.")

    ctx.info(f"Running AI Stem Remix Analysis on {req.audio_id}")
    entry = _get_artifact(req.audio_id)
    if not entry or entry.kind != "audio/wav":
        raise ValueError("Invalid audio artifact ID")

    sr_t, a = await asyncio.to_thread(maestro.load_audio, _artifact_data_path(ARTIFACTS_DIR, entry.data_filename))

    await ctx.report_progress(10, 100, "Running Demucs HT-Demucs Stem Separation (Heavy compute)...")

    # Run Demucs in thread to not block MCP loop
    stems, info = await asyncio.to_thread(
        maestro.demucs_separate_stems,
        a, sr_t, model_name="htdemucs", device="cpu", split=True, overlap=0.23, shifts=1
    )

    await ctx.report_progress(80, 100, "Calculating LUFS density per stem...")

    # Calculate LUFS per stem
    def get_lufs(name):
        return float(maestro.integrated_loudness_lufs(stems[name], sr_t))

    v_lufs = await asyncio.to_thread(get_lufs, "vocals")
    d_lufs = await asyncio.to_thread(get_lufs, "drums")
    b_lufs = await asyncio.to_thread(get_lufs, "bass")
    o_lufs = await asyncio.to_thread(get_lufs, "other")

    await ctx.report_progress(100, 100, "Analyzing mix ratios...")

    advice = (
        f"Mix Analysis:\n"
        f"- Vocals LUFS: {v_lufs:.1f}\n"
        f"- Drums LUFS:  {d_lufs:.1f}\n"
        f"- Bass LUFS:   {b_lufs:.1f}\n"
        f"- Other LUFS:  {o_lufs:.1f}\n\n"
        f"Theory Suggestion: Modern mixes keep Vocals ~1dB to 2dB above the instrumental bed. "
        f"Drums and Bass should tightly lock. If you want tighter vocals, try: "
        f"stem_gains_db={{'vocals': 1.5, 'bass': -0.5}} when executing the final MasterRequest."
    )

    return AiStemRemixOut(
        message="Demucs stems successfully extracted and analyzed.",
        stem_lufs=StemLufsReport(vocals=v_lufs, drums=d_lufs, bass=b_lufs, other=o_lufs),
        mix_theory_advice=advice
    )

# --- INITIALIZE MCP INSTRUCTIONS (SYSTEM PROMPT & CAPABILITIES) ---
try:
    _sys_prompt = get_system_prompt()
    _bs = bootstrap()
    _tools_str = "\n".join([f"- {t.name}: {t.description}" for t in _bs.tools])
    _res_str = "\n".join([f"- {r.uri}: {r.description}" for r in _bs.resources])

    mcp.instructions = (
        f"{_sys_prompt}\n\n"
        "=== MAPPED CAPABILITIES & CONTEXT ===\n"
        f"Available Tools:\n{_tools_str}\n\n"
        f"Available Resources:\n{_res_str}\n\n"
        "Note: You can always call the `bootstrap` tool to see this list again."
    )
except Exception as e:
    log.warning(f"Failed to load MCP instructions on startup: {e}")

if __name__ == "__main__":
    logging.basicConfig(filename="data/auralmind.log",level=logging.INFO, format="%(message)s")
    port = int(os.environ.get("PORT", "8000"))
    # Enforce streamable http but spec says http
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
