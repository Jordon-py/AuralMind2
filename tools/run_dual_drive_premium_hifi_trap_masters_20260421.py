from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
C_SOURCE_ROOT = Path(r"C:\Users\goku\Documents\AuralMind2\data")
D_SOURCE_ROOT = Path(r"D:\music")
SOURCE_ROOTS = {"C": C_SOURCE_ROOT, "D": D_SOURCE_ROOT}
TARGET_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a"}
SERVER_ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}
SKIP_NAME_MARKERS = ("auralmind", "master", "trapgod")
TARGET_SAMPLE_RATE = 48000
TARGET_BIT_DEPTH = "float32"
MOVEMENT_AMOUNT = 0.26
POLL_SECONDS = 3.0
MAX_JOB_WAIT_SECONDS = 60 * 60 * 4


@dataclass
class CandidateFile:
    normalized_song_name: str
    source_drive: str
    path: str
    modified_time: str
    modified_timestamp: float
    size_bytes: int
    sample_rate: Optional[int]
    channels: Optional[int]
    duration_seconds: Optional[float]


@dataclass
class SkippedFile:
    source_drive: str
    path: str
    reason: str
    detail: str


@dataclass
class AmbiguityNote:
    source_drive: str
    path: str
    original_name: str
    normalized_song_name: str
    note: str


@dataclass
class PlannedOutput:
    normalized_song_name: str
    source_drive: str
    source_file_path: str
    source_file_modified_time: str
    selection_reason: str
    output_file_path: str
    mastering_variant: str
    mastering_mode: str
    tools_or_script_used: str
    key_settings: Dict[str, Any]
    sample_rate_out: Optional[int]
    bit_depth_out: Optional[str]
    status: str
    error_message: str
    prepared_input_path: Optional[str] = None
    engine_artifact_id: Optional[str] = None


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def modified_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def safe_output_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "untitled_song"


def normalize_song_identity(path: Path, drive: str) -> Tuple[str, List[AmbiguityNote]]:
    original = path.stem
    normalized = original.lower()
    notes: List[AmbiguityNote] = []

    normalized = re.sub(r"[_\-.]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    before = normalized
    normalized = re.sub(r"\s+\((\d{1,3})\)$", "", normalized).strip()
    if normalized != before:
        notes.append(
            AmbiguityNote(
                source_drive=drive,
                path=str(path),
                original_name=original,
                normalized_song_name=safe_output_name(normalized),
                note="Removed trailing numeric parenthetical as duplicate/export clutter.",
            )
        )

    clutter_patterns = [
        r"\s+copy(?:\s+\d+)?$",
        r"\s+bounce(?:\s+\d+)?$",
        r"\s+export(?:\s+\d+)?$",
        r"\s+mixdown(?:\s+\d+)?$",
        r"\s+mix\s+\d+$",
        r"\s+take\s+\d+$",
        r"\s+version\s+\d+$",
        r"\s+v\d+$",
    ]
    for pattern in clutter_patterns:
        before = normalized
        normalized = re.sub(pattern, "", normalized).strip()
        if normalized != before:
            notes.append(
                AmbiguityNote(
                    source_drive=drive,
                    path=str(path),
                    original_name=original,
                    normalized_song_name=safe_output_name(normalized),
                    note=f"Removed safe trailing clutter marker matching {pattern}.",
                )
            )

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return safe_output_name(normalized), notes


def ffprobe_audio(path: Path, timeout_seconds: int = 20) -> Tuple[bool, Dict[str, Any], str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True, {}, "ffprobe unavailable; decode validation limited to extension and size."

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return False, {}, f"ffprobe timed out after {timeout_seconds}s"
    except OSError as exc:
        return False, {}, f"ffprobe launch failed: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, {}, detail[:1000] or f"ffprobe failed with exit code {result.returncode}"

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return False, {}, f"ffprobe returned invalid JSON: {exc}"

    streams = payload.get("streams") or []
    if not streams:
        return False, payload, "no audio stream found"

    stream = streams[0]
    fmt = payload.get("format") or {}
    duration_raw = stream.get("duration") or fmt.get("duration")
    parsed = {
        "codec_name": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if str(stream.get("sample_rate", "")).isdigit() else None,
        "channels": int(stream["channels"]) if str(stream.get("channels", "")).isdigit() else None,
        "duration_seconds": float(duration_raw) if duration_raw not in (None, "N/A", "") else None,
    }
    return True, parsed, "ok"


def iter_audio_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TARGET_EXTENSIONS),
        key=lambda item: str(item).lower(),
    )


def discover_candidates(validate_decode: bool = True) -> Tuple[List[CandidateFile], List[SkippedFile], List[AmbiguityNote]]:
    candidates: List[CandidateFile] = []
    skipped: List[SkippedFile] = []
    ambiguity: List[AmbiguityNote] = []

    for drive, root in SOURCE_ROOTS.items():
        if not root.exists():
            skipped.append(SkippedFile(drive, str(root), "root_missing", "Source root does not exist."))
            continue

        for path in iter_audio_files(root):
            lower_name = path.name.lower()
            marker = next((token for token in SKIP_NAME_MARKERS if token in lower_name), None)
            if marker:
                skipped.append(SkippedFile(drive, str(path), "filename_marker", f"Filename contains '{marker}'."))
                continue

            try:
                stat = path.stat()
            except OSError as exc:
                skipped.append(SkippedFile(drive, str(path), "stat_failed", str(exc)))
                continue

            if stat.st_size == 0:
                skipped.append(SkippedFile(drive, str(path), "zero_byte", "File size is zero bytes."))
                continue

            probe_ok = True
            probe: Dict[str, Any] = {}
            probe_detail = "decode validation disabled"
            if validate_decode:
                probe_ok, probe, probe_detail = ffprobe_audio(path)
            if not probe_ok:
                skipped.append(SkippedFile(drive, str(path), "decode_validation_failed", probe_detail))
                continue

            normalized, notes = normalize_song_identity(path, drive)
            ambiguity.extend(notes)
            candidates.append(
                CandidateFile(
                    normalized_song_name=normalized,
                    source_drive=drive,
                    path=str(path),
                    modified_time=modified_iso(path),
                    modified_timestamp=stat.st_mtime,
                    size_bytes=stat.st_size,
                    sample_rate=probe.get("sample_rate"),
                    channels=probe.get("channels"),
                    duration_seconds=probe.get("duration_seconds"),
                )
            )

    return candidates, skipped, ambiguity


def variant_settings(variant: str) -> Dict[str, Any]:
    if variant == "A":
        return {
            "preset_name": "competitive_trap",
            "target_lufs": -11.2,
            "warmth": 0.24,
            "transient_boost_db": 2.2,
            "enable_air_motion": True,
            "enable_harshness_limiter": True,
            "enable_hooklift": True,
            "enable_masking_eq": True,
            "governor_gr_limit_db": "-1.15",
            "governor_search_steps": "6",
            "stem_mode": "off",
            "stem_gains_db": "",
            "bit_depth": TARGET_BIT_DEPTH,
            "control_profile": {
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.78,
                "spatial_width": 0.16,
                "brightness_tilt": 0.14,
                "harshness_control": 0.42,
                "hooklift": True,
                "mono_sub": True,
                "no_stems": True,
            },
        }
    return {
        "preset_name": "competitive_trap",
        "target_lufs": -10.8,
        "warmth": 0.20,
        "transient_boost_db": 2.55,
        "enable_air_motion": True,
        "enable_harshness_limiter": True,
        "enable_hooklift": True,
        "enable_masking_eq": True,
        "governor_gr_limit_db": "-1.05",
        "governor_search_steps": "6",
        "stem_mode": "off",
        "stem_gains_db": "",
        "bit_depth": TARGET_BIT_DEPTH,
        "control_profile": {
            "movement_amount": MOVEMENT_AMOUNT,
            "low_end_focus": 0.82,
            "spatial_width": 0.24,
            "brightness_tilt": 0.18,
            "harshness_control": 0.46,
            "hooklift": True,
            "mono_sub": True,
            "no_stems": True,
        },
    }


def build_output_path(output_root: Path, candidate: CandidateFile, selection_reason: str, variant: str) -> Path:
    song_dir = output_root / "outputs" / candidate.normalized_song_name / f"from_{candidate.source_drive}"
    reason_label = "single" if selection_reason == "single-file-variant" else selection_reason
    file_name = (
        f"{candidate.normalized_song_name}__from_{candidate.source_drive}"
        f"__{reason_label}__variant{variant}.wav"
    )
    return song_dir / file_name


def build_plan(candidates: List[CandidateFile], output_root: Path) -> Tuple[List[PlannedOutput], Dict[str, List[CandidateFile]]]:
    groups: Dict[str, List[CandidateFile]] = {}
    for candidate in candidates:
        key = f"{candidate.source_drive}:{candidate.normalized_song_name}"
        groups.setdefault(key, []).append(candidate)

    planned: List[PlannedOutput] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: (item.modified_timestamp, item.path.lower()))
        if len(group) == 1:
            source = group[0]
            for variant in ("A", "B"):
                planned.append(
                    PlannedOutput(
                        normalized_song_name=source.normalized_song_name,
                        source_drive=source.source_drive,
                        source_file_path=source.path,
                        source_file_modified_time=source.modified_time,
                        selection_reason="single-file-variant",
                        output_file_path=str(build_output_path(output_root, source, "single-file-variant", variant)),
                        mastering_variant=variant,
                        mastering_mode="pending",
                        tools_or_script_used="pending",
                        key_settings=variant_settings(variant),
                        sample_rate_out=None,
                        bit_depth_out=None,
                        status="planned",
                        error_message="",
                    )
                )
            continue

        for selection_reason, source, variant in (("oldest", group[0], "A"), ("newest", group[-1], "B")):
            planned.append(
                PlannedOutput(
                    normalized_song_name=source.normalized_song_name,
                    source_drive=source.source_drive,
                    source_file_path=source.path,
                    source_file_modified_time=source.modified_time,
                    selection_reason=selection_reason,
                    output_file_path=str(build_output_path(output_root, source, selection_reason, variant)),
                    mastering_variant=variant,
                    mastering_mode="pending",
                    tools_or_script_used="pending",
                    key_settings=variant_settings(variant),
                    sample_rate_out=None,
                    bit_depth_out=None,
                    status="planned",
                    error_message="",
                )
            )

    return planned, groups


def summarize_discovery(
    candidates: List[CandidateFile],
    skipped: List[SkippedFile],
    groups: Dict[str, List[CandidateFile]],
    planned: List[PlannedOutput],
    output_root: Path,
    validate_decode: bool,
) -> Dict[str, Any]:
    c_groups = sorted(key.split(":", 1)[1] for key in groups if key.startswith("C:"))
    d_groups = sorted(key.split(":", 1)[1] for key in groups if key.startswith("D:"))
    duplicate_names = sorted(set(c_groups).intersection(d_groups))
    return {
        "output_folder_path": str(output_root),
        "candidate_files": len(candidates),
        "skipped_files": len(skipped),
        "song_groups_on_C": len(c_groups),
        "song_groups_on_D": len(d_groups),
        "duplicate_normalized_identities_across_drives": len(duplicate_names),
        "duplicate_names": duplicate_names,
        "total_expected_mastered_outputs": len(planned),
        "tool_pipeline_chosen": (
            "AuralMind2 server async run_master_job with explicit premium trap settings; "
            "fallback to tools/auralmind_maestro.py if server route fails."
        ),
        "realtime_ai_enabled": "yes when server async job route succeeds; fallback_local otherwise",
        "exact_32bit_48000_supported": True,
        "decode_validation": "ffprobe" if validate_decode and shutil.which("ffprobe") else "limited",
    }


def default_output_root() -> Path:
    base = Path.home() / "Desktop" / f"AuralMind_Premium_HiFi_Trap_Masters_{datetime.now():%Y-%m-%d_%H%M}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    return candidate


def ensure_output_tree(output_root: Path) -> None:
    for child in ("manifests", "reports", "logs", "outputs", ".prepared_inputs"):
        (output_root / child).mkdir(parents=True, exist_ok=True)


def setup_logger(output_root: Path) -> logging.Logger:
    logger = logging.getLogger("dual_drive_premium_hifi_trap")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(output_root / "logs" / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def convert_to_engine_wav(source: Path, prepared_root: Path, logger: logging.Logger) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to prepare this source for engine decoding but was not found.")

    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared = prepared_root / f"{safe_output_name(source.stem)}_{uuid.uuid4().hex[:10]}_prepared_48k.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        "2",
        "-sample_fmt",
        "flt",
        str(prepared),
    ]
    logger.info("Preparing engine WAV input: %s", source)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg source preparation failed: {detail[:1200]}")
    return prepared


def source_for_engine(source: Path, output_root: Path, logger: logging.Logger) -> Path:
    if source.suffix.lower() in SERVER_ALLOWED_EXTENSIONS:
        return source
    return convert_to_engine_wav(source, output_root / ".prepared_inputs", logger)


def probe_rendered_audio(path: Path) -> Tuple[Optional[int], Optional[str], Dict[str, Any]]:
    ok, payload, detail = ffprobe_audio(path)
    if not ok:
        return None, None, {"probe_error": detail}

    sample_rate = payload.get("sample_rate")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return sample_rate, None, payload

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_fmt,bits_per_sample,bits_per_raw_sample,sample_rate,codec_name",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return sample_rate, None, payload
    try:
        stream = (json.loads(result.stdout or "{}").get("streams") or [{}])[0]
    except json.JSONDecodeError:
        return sample_rate, None, payload

    fmt = stream.get("sample_fmt")
    bits = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample")
    bit_depth = None
    if fmt in {"flt", "fltp"}:
        bit_depth = "32-bit float"
    elif fmt in {"dbl", "dblp"}:
        bit_depth = "64-bit float"
    elif bits:
        bit_depth = f"{bits}-bit"
    return int(stream.get("sample_rate") or sample_rate), bit_depth, stream


class MinimalContext:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.request_id = f"dual_drive_{uuid.uuid4().hex[:12]}"


def copy_server_artifact(server_module: Any, ctx: MinimalContext, artifact_id: str, output_path: Path) -> Path:
    session_key, session_dir = server_module._get_session_info(ctx)
    entry = server_module._load_artifact(session_key, session_dir, artifact_id)
    source = Path(session_dir) / entry.data_filename
    if not source.exists():
        raise FileNotFoundError(f"Server artifact payload missing: {source}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return output_path


def master_with_server(item: PlannedOutput, output_root: Path, logger: logging.Logger) -> PlannedOutput:
    os.environ["AURALMIND_AUDIO_ROOTS"] = ",".join(
        str(path) for path in [C_SOURCE_ROOT, D_SOURCE_ROOT, output_root / ".prepared_inputs"]
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import server  # type: ignore

    ctx = MinimalContext(session_id=f"dual_drive_premium_{uuid.uuid4().hex[:12]}")
    source = Path(item.source_file_path)
    engine_source = source_for_engine(source, output_root, logger)
    item.prepared_input_path = str(engine_source) if engine_source != source else None

    register = server.register_audio_from_path(str(engine_source), ctx=ctx)
    audio_id = register["audio_id"]
    settings_payload = dict(item.key_settings)
    control_payload = settings_payload.pop("control_profile")
    control = server.MasteringControlProfile(**control_payload)
    settings = server.MasterSettings(**settings_payload, control_profile=control)
    normalized = server.propose_master_settings(settings).settings

    request = server.MasterRequest(audio_id=audio_id, **normalized.model_dump())
    launch = server.run_master_job(request, ctx=ctx)
    job_id = launch["job_id"]
    logger.info("Started realtime AI async mastering job %s for %s variant %s", job_id, item.normalized_song_name, item.mastering_variant)

    deadline = time.time() + MAX_JOB_WAIT_SECONDS
    last_progress: Optional[Tuple[str, float]] = None
    while time.time() < deadline:
        status = server.job_status(job_id)
        state = status.get("status")
        progress = float(status.get("progress", 0.0) or 0.0)
        marker = (str(state), round(progress, 2))
        if marker != last_progress:
            logger.info("Job %s status=%s progress=%.2f", job_id, state, progress)
            last_progress = marker
        if state == "completed":
            result = server.job_result(job_id)
            artifact_id = result.get("output_audio_id")
            if not artifact_id:
                raise RuntimeError(f"Server job completed without output_audio_id: {result}")
            item.engine_artifact_id = artifact_id
            output_path = Path(item.output_file_path)
            copy_server_artifact(server, ctx, artifact_id, output_path)
            sample_rate, bit_depth, _ = probe_rendered_audio(output_path)
            item.sample_rate_out = sample_rate
            item.bit_depth_out = bit_depth
            item.mastering_mode = "realtime_ai"
            item.tools_or_script_used = "server.py run_master_job -> tools/auralmind_maestro.py"
            item.status = "completed"
            item.error_message = ""
            return item
        if state == "failed":
            error = status.get("error") or "unknown server job failure"
            raise RuntimeError(str(error))
        time.sleep(POLL_SECONDS)

    raise TimeoutError(f"Server job did not complete within {MAX_JOB_WAIT_SECONDS}s")


def master_with_local_maestro(item: PlannedOutput, output_root: Path, logger: logging.Logger) -> PlannedOutput:
    tools_root = REPO_ROOT / "tools"
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    import auralmind_maestro as maestro  # type: ignore

    source = Path(item.source_file_path)
    engine_source = source
    if source.suffix.lower() not in {".wav", ".flac", ".aif", ".aiff"}:
        engine_source = convert_to_engine_wav(source, output_root / ".prepared_inputs", logger)
        item.prepared_input_path = str(engine_source)

    base = maestro.get_presets().get("competitive_trap") or next(iter(maestro.get_presets().values()))
    settings = item.key_settings
    cp = settings["control_profile"]
    variant = item.mastering_variant
    local_preset = replace(
        base,
        sr=TARGET_SAMPLE_RATE,
        bit_depth=TARGET_BIT_DEPTH,
        target_lufs=float(settings["target_lufs"]),
        warmth=float(settings["warmth"]),
        transient_boost_db=float(settings["transient_boost_db"]),
        stem_mode="off",
        movement_amount=MOVEMENT_AMOUNT,
        enable_hooklift=True,
        enable_air_motion=True,
        enable_harshness_limiter=True,
        enable_masking_eq=True,
        width_hi=0.19 if variant == "A" else 0.26,
        width_mid=0.08 if variant == "A" else 0.12,
        air_motion_mix=0.16 if variant == "A" else 0.20,
        hooklift_mix=0.18 if variant == "A" else 0.22,
        mono_sub_base_mix=0.90 if cp["low_end_focus"] >= 0.80 else 0.86,
        governor_search_steps=6,
        governor_gr_limit_db=float(settings["governor_gr_limit_db"]),
    )

    output_path = Path(item.output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running fallback local Maestro master: %s", output_path)
    maestro.master(
        str(engine_source),
        str(output_path),
        local_preset,
        out_subtype="FLOAT",
        dither=False,
        report_path=str(output_root / "reports" / f"{output_path.stem}_maestro_report.json"),
    )

    sample_rate, bit_depth, _ = probe_rendered_audio(output_path)
    item.sample_rate_out = sample_rate
    item.bit_depth_out = bit_depth
    item.mastering_mode = "fallback_local"
    item.tools_or_script_used = "tools/auralmind_maestro.py direct master()"
    item.status = "completed"
    item.error_message = ""
    return item


def master_item(item: PlannedOutput, output_root: Path, logger: logging.Logger) -> PlannedOutput:
    logger.info(
        "Mastering %s from drive %s selection=%s variant=%s",
        item.normalized_song_name,
        item.source_drive,
        item.selection_reason,
        item.mastering_variant,
    )
    try:
        return master_with_server(item, output_root, logger)
    except Exception as server_exc:
        logger.exception("Server realtime route failed; falling back to local Maestro: %s", server_exc)
        try:
            completed = master_with_local_maestro(item, output_root, logger)
            completed.error_message = f"Server route failed; local fallback succeeded. Server error: {server_exc}"
            return completed
        except Exception as fallback_exc:
            logger.exception("Local fallback failed: %s", fallback_exc)
            item.mastering_mode = "fallback_local"
            item.tools_or_script_used = "server.py failed; tools/auralmind_maestro.py fallback failed"
            item.status = "failed"
            item.error_message = f"Server error: {server_exc}; fallback error: {fallback_exc}"
            return item


def selected_breakdown(groups: Dict[str, List[CandidateFile]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: (item.modified_timestamp, item.path.lower()))
        drive, normalized = key.split(":", 1)
        if len(group) == 1:
            rows.append(
                {
                    "normalized_song_name": normalized,
                    "source_drive": drive,
                    "eligible_files": 1,
                    "single_source": group[0].path,
                    "single_modified_time": group[0].modified_time,
                }
            )
        else:
            rows.append(
                {
                    "normalized_song_name": normalized,
                    "source_drive": drive,
                    "eligible_files": len(group),
                    "oldest_source": group[0].path,
                    "oldest_modified_time": group[0].modified_time,
                    "newest_source": group[-1].path,
                    "newest_modified_time": group[-1].modified_time,
                }
            )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_report(
    output_root: Path,
    summary: Dict[str, Any],
    candidates: List[CandidateFile],
    skipped: List[SkippedFile],
    ambiguity: List[AmbiguityNote],
    groups: Dict[str, List[CandidateFile]],
    planned: List[PlannedOutput],
    started_at: str,
    finished_at: str,
) -> None:
    completed = [item for item in planned if item.status == "completed"]
    failed = [item for item in planned if item.status == "failed"]
    realtime = [item for item in completed if item.mastering_mode == "realtime_ai"]
    fallback = [item for item in completed if item.mastering_mode == "fallback_local"]
    export_mismatches = [
        item
        for item in completed
        if item.sample_rate_out != TARGET_SAMPLE_RATE or item.bit_depth_out != "32-bit float"
    ]
    c_groups = [key for key in groups if key.startswith("C:")]
    d_groups = [key for key in groups if key.startswith("D:")]

    per_song_lines = [
        f"- {row['normalized_song_name']} from {row['source_drive']}: {row['eligible_files']} eligible file(s)"
        for row in selected_breakdown(groups)
    ]
    failures = (
        "\n".join(
            f"- {item.normalized_song_name} from {item.source_drive} variant {item.mastering_variant}: {item.error_message}"
            for item in failed
        )
        or "- None"
    )
    ambiguity_lines = (
        "\n".join(
            f"- {note.normalized_song_name} from {note.source_drive}: {note.note} ({note.path})"
            for note in ambiguity
        )
        or "- None"
    )
    export_lines = (
        "\n".join(
            f"- {item.output_file_path}: {item.sample_rate_out} Hz, {item.bit_depth_out}"
            for item in completed
        )
        or "- No completed outputs"
    )
    mismatch_lines = (
        "\n".join(
            f"- {item.output_file_path}: expected 48000 Hz / 32-bit float, got {item.sample_rate_out} Hz / {item.bit_depth_out}"
            for item in export_mismatches
        )
        or "- None"
    )

    report = f"""# AuralMind Premium HiFi Trap Mastering Report

## Overview
- Started: {started_at}
- Finished: {finished_at}
- Output folder: `{output_root}`
- Planned masters: {len(planned)}
- Completed masters: {len(completed)}
- Failed masters: {len(failed)}

## Scanned Roots
- `C:\\Users\\goku\\Documents\\AuralMind2\\data\\`
- `D:\\music\\`

## Tooling Discovered
- AuralMind2 server route: `server.py` async `run_master_job`, `MasterSettings`, `MasteringControlProfile`
- Local fallback route: `tools/auralmind_maestro.py`
- `ffprobe`: {'available' if shutil.which('ffprobe') else 'not available'}
- `ffmpeg`: {'available' if shutil.which('ffmpeg') else 'not available'}

## Whether Real-Time AI Mastering Was Used
- Server async real-time route completions: {len(realtime)}
- Local fallback completions: {len(fallback)}
- The run used `realtime_ai` for items completed through `server.py run_master_job`. Any fallback is documented per output in `manifest.json`.

## Skip Rules Applied
- Candidate extensions: `.wav`, `.mp3`, `.flac`, `.aiff`, `.aif`, `.m4a`
- Filename exclusions: `auralmind`, `master`, `trapgod`
- Zero-byte files skipped
- Basic decode validation: {summary['decode_validation']}
- Skipped files: {len(skipped)}

## Grouping Logic Summary
- Lowercased names, stripped extensions, normalized underscores/dashes/dots to spaces, collapsed spaces, removed safe trailing duplicate/export counters, then converted to path-safe normalized identifiers.
- C and D drive groups were kept separate for source selection and output routing.

## Ambiguity Handling Notes
{ambiguity_lines}

## Per-Song Breakdown
{chr(10).join(per_song_lines) if per_song_lines else "- No eligible songs discovered"}

## Per-Drive Breakdown
- C drive groups: {len(c_groups)}
- D drive groups: {len(d_groups)}
- Duplicate normalized identities across drives: {summary['duplicate_normalized_identities_across_drives']}

## Output Summary
- Total expected outputs: {summary['total_expected_mastered_outputs']}
- Completed outputs: {len(completed)}
- Failed outputs: {len(failed)}

## Failures and Recoveries
{failures}

## Chosen Mastering Chain and Why
- Primary route: AuralMind2 `server.py run_master_job` because it is the environment-native async mastering route and exposes bounded premium mastering settings.
- Fallback route: `tools/auralmind_maestro.py` direct render because it preserves the same core DSP path and supports explicit 48 kHz / 32-bit float WAV output when server orchestration fails.
- Stems were disabled, movement was fixed at {MOVEMENT_AMOUNT}, hooklift was enabled, mono-sub anchoring was enabled, and top-end width/air were applied conservatively per variant.

## Export Format Results
{export_lines}

## Export Format Mismatches
{mismatch_lines}

## Final Totals
- Candidate files: {len(candidates)}
- Skipped files: {len(skipped)}
- Song groups processed: {len(groups)}
- Masters created: {len(completed)}
- Failures: {len(failed)}

## Manual Verification Checklist
- Level-match each new master against its source by ear before judging loudness.
- Check the first chorus or hook for vocal/drum transient smear.
- Check 30-80 Hz in mono to confirm the sub is centered and disciplined.
- Check upper hats, air, and vocal edge at low volume to confirm brightness is polished, not brittle.
- Collapse to mono for one hook and one verse to confirm width does not phase-cancel.
- Compare Variant A vs Variant B per song: A should feel cleaner and more polished; B should feel slightly more forward and wider without muddy subs or crushed impact.
"""
    (output_root / "reports" / "processing_report.md").write_text(report, encoding="utf-8")


def manifest_payload(
    output_root: Path,
    summary: Dict[str, Any],
    candidates: List[CandidateFile],
    skipped: List[SkippedFile],
    ambiguity: List[AmbiguityNote],
    groups: Dict[str, List[CandidateFile]],
    planned: List[PlannedOutput],
    started_at: str,
    finished_at: str,
) -> Dict[str, Any]:
    return {
        "run": {
            "started_at": started_at,
            "finished_at": finished_at,
            "output_folder": str(output_root),
            "repo_root": str(REPO_ROOT),
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "target_bit_depth": TARGET_BIT_DEPTH,
            "movement_amount": MOVEMENT_AMOUNT,
            "stems": "off",
        },
        "summary": summary,
        "scanned_roots": {drive: str(root) for drive, root in SOURCE_ROOTS.items()},
        "candidate_files": [asdict(item) for item in candidates],
        "skipped_files": [asdict(item) for item in skipped],
        "ambiguity_notes": [asdict(item) for item in ambiguity],
        "selected_breakdown": selected_breakdown(groups),
        "planned_outputs": [asdict(item) for item in planned],
    }


def print_dry_run(summary: Dict[str, Any], groups: Dict[str, List[CandidateFile]], planned: List[PlannedOutput]) -> None:
    print("\nDRY-RUN SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSELECTED OLDEST/NEWEST BY GROUP")
    print(json.dumps(selected_breakdown(groups), indent=2, ensure_ascii=False))
    print("\nPLANNED ITEMS")
    compact = [
        {
            "normalized_song_name": item.normalized_song_name,
            "source_drive": item.source_drive,
            "source_file_path": item.source_file_path,
            "source_file_modified_time": item.source_file_modified_time,
            "selection_reason": item.selection_reason,
            "mastering_variant": item.mastering_variant,
            "output_file_path": item.output_file_path,
        }
        for item in planned
    ]
    print(json.dumps(compact, indent=2, ensure_ascii=False))


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir).resolve() if args.output_dir else default_output_root()
    candidates, skipped, ambiguity = discover_candidates(validate_decode=not args.skip_decode_validation)
    planned, groups = build_plan(candidates, output_root)
    summary = summarize_discovery(
        candidates=candidates,
        skipped=skipped,
        groups=groups,
        planned=planned,
        output_root=output_root,
        validate_decode=not args.skip_decode_validation,
    )
    print_dry_run(summary, groups, planned)

    if args.dry_run:
        return 0

    ensure_output_tree(output_root)
    logger = setup_logger(output_root)
    started_at = local_now_iso()
    logger.info("Starting dual-drive premium hi-fi trap mastering run.")
    logger.info("Output root: %s", output_root)
    logger.info("Candidates=%s skipped=%s planned=%s", len(candidates), len(skipped), len(planned))

    completed_plan: List[PlannedOutput] = []
    for index, item in enumerate(planned, start=1):
        logger.info("Queue item %s/%s", index, len(planned))
        completed_plan.append(master_item(item, output_root, logger))

    finished_at = local_now_iso()
    final_summary = summarize_discovery(
        candidates=candidates,
        skipped=skipped,
        groups=groups,
        planned=completed_plan,
        output_root=output_root,
        validate_decode=not args.skip_decode_validation,
    )
    write_json(output_root / "reports" / "skipped_files.json", [asdict(item) for item in skipped])
    write_report(
        output_root=output_root,
        summary=final_summary,
        candidates=candidates,
        skipped=skipped,
        ambiguity=ambiguity,
        groups=groups,
        planned=completed_plan,
        started_at=started_at,
        finished_at=finished_at,
    )
    write_json(
        output_root / "manifests" / "manifest.json",
        manifest_payload(
            output_root=output_root,
            summary=final_summary,
            candidates=candidates,
            skipped=skipped,
            ambiguity=ambiguity,
            groups=groups,
            planned=completed_plan,
            started_at=started_at,
            finished_at=finished_at,
        ),
    )
    logger.info("Finished run. Manifest and report written.")

    completed = [item for item in completed_plan if item.status == "completed"]
    failed = [item for item in completed_plan if item.status == "failed"]
    print("\nFINAL SUMMARY")
    print(
        json.dumps(
            {
                "output_folder_created": str(output_root),
                "mastering_pipeline_used": final_summary["tool_pipeline_chosen"],
                "realtime_ai_used": any(item.mastering_mode == "realtime_ai" for item in completed),
                "total_candidate_files_found": len(candidates),
                "total_skipped_files": len(skipped),
                "total_song_groups_processed": len(groups),
                "total_masters_created": len(completed),
                "total_failures": len(failed),
                "manual_listening_check": (
                    "Level-match each source and master, then audition hook/verse in stereo and mono for "
                    "sub centering, transient punch, non-brittle highs, and width that does not collapse."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if not failed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and master oldest/newest source versions from AuralMind2 C and D drive roots."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print discovery and plan only.")
    parser.add_argument("--output-dir", help="Explicit output directory for this run.")
    parser.add_argument(
        "--skip-decode-validation",
        action="store_true",
        help="Skip ffprobe validation. Not recommended for production runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
