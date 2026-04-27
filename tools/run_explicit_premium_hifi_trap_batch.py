"""MCP-only explicit premium hi-fi trap batch runner.

Purpose: connects to the AuralMind2 FastMCP server and renders Christopher's
fixed 10-song queue using MCP tools only. The client exports returned artifact
bytes, but all analysis, planning, mastering, phase alignment, and final
metrics come from the MCP server. Optional delivery-format exports are file
encoding copies from the MCP artifact, not extra local mastering.
Data shapes: `TrackPlanItem` rows persist to `manifest.json` with source
metadata, MCP handles (`aud_*`, `job_*`, `art_*`), selected settings, phase
alignment metrics, final artifact/export paths, and optional 24/32-bit
delivery export metadata.
Syntax: `python tools/run_explicit_premium_hifi_trap_batch.py --poll-seconds 3`
or one-off `--source "data/FaceTime (6).wav" --delivery-formats 24,32`.
Important functions: `main` near line 716, `run_one` near line 425,
`export_artifact` near line 222, and `mcp_call` near line 154.
Possible bugs: if the Python process stops, in-memory MCP job handles can be
lost; resume keeps completed exports but must rerun unfinished jobs.
Enhance next: move delivery format conversion into an MCP server tool; add a
server-side batch queue tool so this script becomes a thin launcher.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import Client

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import server  # noqa: E402


SOURCE_FILENAMES = [
    "Family Operations.wav",
    "DaddysGirls (2).wav",
    "Why they Bitin Me Momma.wav",
    "IDGAF.wav",
    "Hot shit (1).wav",
    "Best of me (1).wav",
    "difference (10).wav",
    "Last Time (14) - AuralMind v7.3 hi fi DrakeRef no stems m021 24bit.wav",
    "Vegas - top teir (22).wav",
    "Walking_In_Rain.mp3",
]

DEFAULT_PRESET = "competitive_trap"
DEFAULT_PLATFORM = "spotify"
DEFAULT_STEM_MODE = "off"
DEFAULT_TARGET_LUFS = -12.2
DEFAULT_TRUE_PEAK = -1.0
DEFAULT_PHASE_CUTOFF_HZ = 155.0
DEFAULT_CONTROL_PROFILE = {
    "spatial_width": 0.10,
    "brightness_tilt": 0.18,
    "harshness_control": 0.34,
    "movement_amount": 0.26,
    "low_end_focus": 0.72,
}
REQ_WRAPPED_TOOLS = {
    "register_audio_from_path",
    "analyze_audio",
    "plan_mastering_strategy",
    "propose_master_settings",
    "run_master_job",
    "job_status",
    "job_result",
    "premium_phase_align",
}


@dataclass
class TrackPlanItem:
    index: int
    filename: str
    display_name: str
    source_path: str
    source_exists: bool
    source_size_bytes: int
    source_modified_at: str
    preset_name: str = DEFAULT_PRESET
    stem_mode: str = DEFAULT_STEM_MODE
    target_lufs: float = DEFAULT_TARGET_LUFS
    true_peak_dbtp: float = DEFAULT_TRUE_PEAK
    warmth: float = 0.30
    transient_boost_db: float = 2.20
    control_profile: Dict[str, float] = None  # type: ignore[assignment]
    audio_id: str = ""
    job_id: str = ""
    raw_artifact_id: str = ""
    phase_artifact_id: str = ""
    final_export_path: str = ""
    delivery_exports: Dict[str, Any] = None  # type: ignore[assignment]
    source_metrics: Dict[str, Any] = None  # type: ignore[assignment]
    final_metrics: Dict[str, Any] = None  # type: ignore[assignment]
    phase_alignment: Dict[str, Any] = None  # type: ignore[assignment]
    plan_reasoning: List[str] = None  # type: ignore[assignment]
    plan_warnings: List[str] = None  # type: ignore[assignment]
    status: str = "planned"
    error: str = ""
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        if self.control_profile is None:
            self.control_profile = dict(DEFAULT_CONTROL_PROFILE)
        if self.source_metrics is None:
            self.source_metrics = {}
        if self.final_metrics is None:
            self.final_metrics = {}
        if self.phase_alignment is None:
            self.phase_alignment = {}
        if self.delivery_exports is None:
            self.delivery_exports = {}
        if self.plan_reasoning is None:
            self.plan_reasoning = []
        if self.plan_warnings is None:
            self.plan_warnings = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = f"[{utc_now()}] {line}"
    log_path.open("a", encoding="utf-8").write(rendered + "\n")
    print(rendered, flush=True)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-+", "-", slug) or "untitled"


async def mcp_call(client: Client, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"req": arguments} if name in REQ_WRAPPED_TOOLS else arguments
    result = await client.call_tool(name, payload)
    if getattr(result, "is_error", False):
        raise RuntimeError(f"mcp_tool_error:{name}: {result}")
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        if isinstance(structured, dict):
            return structured
        return json.loads(json.dumps(structured, default=str))
    content = getattr(result, "content", None) or []
    if not content:
        return {}
    text = getattr(content[0], "text", "")
    return json.loads(text) if text else {}


async def read_mcp_resource_text(client: Client, uri: str) -> str:
    result = await client.read_resource(uri)
    content = getattr(result, "content", None) or result
    if isinstance(content, list) and content:
        return str(getattr(content[0], "text", content[0]))
    return str(content)


async def read_mcp_prompt_text(client: Client, name: str, arguments: Dict[str, Any]) -> str:
    try:
        result = await client.get_prompt(name, arguments)
    except Exception as exc:
        return f"prompt_unavailable:{name}: {exc}"
    messages = getattr(result, "messages", None) or []
    parts: List[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is not None:
            parts.append(str(getattr(content, "text", content)))
    return "\n".join(parts)


def lock_path(out_root: Path) -> Path:
    return out_root / ".auralmind2_master.lock"


def acquire_lock(out_root: Path, *, force: bool) -> None:
    lp = lock_path(out_root)
    if lp.exists() and not force:
        raise RuntimeError(f"lock_exists: {lp}")
    lp.write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "pid": os.getpid(),
                "script": str(Path(__file__).resolve()),
                "mode": "mcp_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def release_lock(out_root: Path) -> None:
    try:
        lock_path(out_root).unlink(missing_ok=True)
    except Exception:
        return None


async def export_artifact(client: Client, artifact_id: str, destination: Path) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    expected_sha = ""
    size_bytes = 0
    sha = hashlib.sha256()
    with destination.open("wb") as handle:
        while True:
            chunk = await mcp_call(
                client,
                "read_artifact",
                {"artifact_id": artifact_id, "offset": offset, "length": server.MAX_READ_BYTES},
            )
            data = base64.b64decode(chunk["data_b64"])
            handle.write(data)
            sha.update(data)
            offset += int(chunk["length"])
            expected_sha = str(chunk["sha256"])
            size_bytes = int(chunk["size_bytes"])
            if bool(chunk["is_last"]):
                break

    actual_sha = sha.hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(f"sha256_mismatch: expected {expected_sha}, got {actual_sha}")
    if destination.stat().st_size != size_bytes:
        raise RuntimeError(f"size_mismatch: expected {size_bytes}, got {destination.stat().st_size}")
    return {
        "path": str(destination),
        "size_bytes": size_bytes,
        "sha256": actual_sha,
    }


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def run_process(command: List[str]) -> str:
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"command_failed:{command[0]} exit={proc.returncode}: {stderr}")
    return (proc.stdout or "").strip()


def probe_audio_file(path: Path) -> Dict[str, Any]:
    raw = run_process(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,sample_fmt,bits_per_sample,bits_per_raw_sample,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(raw or "{}")
    streams = payload.get("streams") or []
    return dict(streams[0]) if streams else {}


def parse_delivery_formats(value: str) -> List[str]:
    if not value.strip():
        return []
    parsed: List[str] = []
    for part in re.split(r"[, ]+", value.strip().lower()):
        if not part:
            continue
        normalized = part.replace("-bit", "").replace("bit", "").replace("float", "")
        if normalized in {"24", "32"} and normalized not in parsed:
            parsed.append(normalized)
        else:
            raise ValueError(f"unsupported_delivery_format: {part}; expected 24 and/or 32")
    return parsed


def delivery_path_for(raw_path: Path, output_root: Path, fmt: str) -> Path:
    if fmt == "24":
        return output_root / "delivery_24bit" / f"{raw_path.stem}__24bit.wav"
    if fmt == "32":
        return output_root / "delivery_32bitfloat" / f"{raw_path.stem}__32bitfloat.wav"
    raise ValueError(f"unsupported_delivery_format: {fmt}")


def write_delivery_exports(
    source_path: Path,
    output_root: Path,
    formats: List[str],
    *,
    log_path: Path,
) -> Dict[str, Any]:
    exports: Dict[str, Any] = {}
    if not formats:
        return exports
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    for fmt in formats:
        codec = "pcm_s24le" if fmt == "24" else "pcm_f32le"
        destination = delivery_path_for(source_path, output_root, fmt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_process(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-map_metadata",
                "0",
                "-c:a",
                codec,
                str(destination),
            ]
        )
        exports[fmt] = {
            "path": str(destination),
            "codec": codec,
            "size_bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "probe": probe_audio_file(destination),
            "source_master_path": str(source_path),
            "note": "Delivery encoding only; MCP phase-aligned artifact remains the master source.",
        }
        append_log(log_path, f"delivery_{fmt}bit {destination}")
    return exports


def normalize_source_path(source: str) -> Path:
    candidate = source.strip().strip('"')
    if not candidate:
        raise ValueError("empty_source")
    p = Path(candidate)
    if p.is_absolute():
        return p
    normalized = candidate.replace("\\", "/")
    if normalized.lower().startswith("./"):
        normalized = normalized[2:]
    if normalized.lower().startswith("data/"):
        normalized = normalized[5:]
    return REPO_ROOT / "data" / normalized


def build_plan(out_root: Path, sources: Optional[List[str]] = None) -> List[TrackPlanItem]:
    final_dir = out_root / "final"
    plan: List[TrackPlanItem] = []
    source_values = sources if sources else SOURCE_FILENAMES
    for index, source in enumerate(source_values, start=1):
        path = normalize_source_path(source)
        filename = path.name
        exists = path.exists()
        stat = path.stat() if exists else None
        display_name = path.stem
        export_path = final_dir / f"{index:02d}_{safe_slug(display_name)}__mcp_premium_hifi_trap_phase_aligned.wav"
        plan.append(
            TrackPlanItem(
                index=index,
                filename=filename,
                display_name=display_name,
                source_path=str(path),
                source_exists=exists,
                source_size_bytes=int(stat.st_size) if stat else 0,
                source_modified_at=iso_from_ts(float(stat.st_mtime)) if stat else "",
                final_export_path=str(export_path),
            )
        )
    return plan


def coerce_plan_item(raw: Dict[str, Any]) -> TrackPlanItem:
    allowed = set(TrackPlanItem.__dataclass_fields__.keys())
    clean = {key: raw.get(key) for key in allowed if key in raw}
    return TrackPlanItem(**clean)


def write_manifest(path: Path, manifest: Dict[str, Any], plan: List[TrackPlanItem]) -> None:
    manifest["last_updated_at"] = utc_now()
    manifest["items"] = [asdict(item) for item in plan]
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def select_master_artifact(result: Dict[str, Any]) -> str:
    artifacts = result.get("artifacts") or []
    wavs = [item for item in artifacts if str(item.get("filename", "")).lower().endswith(".wav")]
    if not wavs:
        raise RuntimeError("job_result_missing_wav_artifact")
    return str(max(wavs, key=lambda item: int(item.get("size_bytes") or 0))["artifact_id"])


async def run_one(
    client: Client,
    item: TrackPlanItem,
    *,
    total_count: int,
    delivery_formats: List[str],
    output_root: Path,
    poll_s: float,
    max_wait_s: float,
    log_path: Path,
) -> None:
    item.started_at = utc_now()
    item.status = "running"
    item.error = ""

    if not item.source_exists:
        raise FileNotFoundError(item.source_path)

    append_log(log_path, f"register {item.index}/{total_count} {item.filename}")
    registered = await mcp_call(client, "register_audio_from_path", {"path": item.source_path})
    item.audio_id = str(registered["audio_id"])

    analysis = await mcp_call(client, "analyze_audio", {"audio_id": item.audio_id})
    item.source_metrics = dict(analysis["metrics"])

    prompt_text = await read_mcp_prompt_text(
        client,
        "premium_trap_mastering_session",
        {
            "file_uri": item.source_path,
            "goal": "premium hi-fidelity AI-integrated trap master with premium phase alignment",
            "platform": DEFAULT_PLATFORM,
            "intensity": "balanced",
        },
    )

    plan = await mcp_call(
        client,
        "plan_mastering_strategy",
        {
            "audio_id": item.audio_id,
            "goal": (
                "Premium hi-fidelity AI-integrated trap master. Prioritize centered 808/sub "
                "translation, premium phase alignment, vocal clarity, punch, hook lift, clean "
                "high-end sheen, and release-ready mono compatibility."
            ),
            "platform": DEFAULT_PLATFORM,
            "control_profile": item.control_profile,
            "stem_mode": item.stem_mode,
        },
    )
    item.plan_reasoning = list(plan.get("reasoning") or [])
    item.plan_warnings = list(plan.get("warnings") or [])

    proposed = await mcp_call(
        client,
        "propose_master_settings",
        {
            "preset_name": item.preset_name,
            "target_lufs": item.target_lufs,
            "warmth": item.warmth,
            "transient_boost_db": item.transient_boost_db,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "bit_depth": "float32",
            "control_profile": item.control_profile,
            "governor_search_steps": 8,
            "governor_gr_limit_db": -1.25,
            "stem_mode": item.stem_mode,
        },
    )
    settings = dict(proposed["settings"])
    item.preset_name = str(settings["preset_name"])

    launch_payload = {"audio_id": item.audio_id, **settings}
    append_log(
        log_path,
        (
            f"launch {item.index}/{total_count} {item.display_name} via_mcp preset={item.preset_name} "
            f"stem={item.stem_mode} target={item.target_lufs} phase=server_premium_phase_align"
        ),
    )
    launch = await mcp_call(client, "run_master_job", launch_payload)
    item.job_id = str(launch["job_id"])
    item.plan_reasoning.append(f"Prompt guidance loaded: {len(prompt_text)} chars")

    started = time.time()
    last_progress = -1
    while True:
        status = await mcp_call(client, "job_status", {"job_id": item.job_id})
        progress = int(status.get("progress") or 0)
        if progress != last_progress:
            last_progress = progress
            append_log(log_path, f"progress {item.index}/{total_count} {item.display_name} {status['status']} {progress}%")
        if status["status"] in ("done", "error", "cancelled"):
            break
        if time.time() - started > max_wait_s:
            raise RuntimeError(f"timeout waiting for {item.job_id}")
        time.sleep(poll_s)

    if status["status"] != "done":
        raise RuntimeError(json.dumps(status.get("error") or status))

    result = await mcp_call(client, "job_result", {"job_id": item.job_id})
    item.raw_artifact_id = select_master_artifact(result)

    phase = await mcp_call(
        client,
        "premium_phase_align",
        {"audio_id": item.raw_artifact_id, "cutoff_hz": DEFAULT_PHASE_CUTOFF_HZ},
    )
    item.phase_alignment = dict(phase)
    item.phase_artifact_id = str(phase["artifact_id"])

    final_analysis = await mcp_call(client, "analyze_audio", {"audio_id": item.phase_artifact_id})
    item.final_metrics = dict(final_analysis["metrics"])
    export_info = await export_artifact(client, item.phase_artifact_id, Path(item.final_export_path))
    item.final_metrics["export"] = export_info
    item.delivery_exports = write_delivery_exports(
        Path(item.final_export_path),
        output_root,
        delivery_formats,
        log_path=log_path,
    )
    item.status = "done"
    item.completed_at = utc_now()
    append_log(
        log_path,
        (
            f"complete {item.index}/{total_count} {item.display_name} -> {item.final_export_path} "
            f"LUFS={item.final_metrics.get('integrated_lufs')} "
            f"TP={item.final_metrics.get('true_peak_dbtp')} "
            f"phase_low={item.phase_alignment.get('low_corr_before')}->{item.phase_alignment.get('low_corr_after')}"
        ),
    )


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "masters" / f"mcp_premium_hifi_trap_explicit_{stamp}"


def ensure_delivery_exports(
    plan: List[TrackPlanItem],
    output_root: Path,
    formats: List[str],
    *,
    log_path: Path,
) -> None:
    if not formats:
        return
    for item in plan:
        if item.status != "done":
            continue
        existing = item.delivery_exports or {}
        missing = [
            fmt
            for fmt in formats
            if fmt not in existing or not Path(str(existing.get(fmt, {}).get("path", ""))).exists()
        ]
        if not missing:
            continue
        item.delivery_exports.update(
            write_delivery_exports(
                Path(item.final_export_path),
                output_root,
                missing,
                log_path=log_path,
            )
        )


async def async_main(args: argparse.Namespace) -> int:
    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    log_path = out_root / "run.log"
    delivery_formats = parse_delivery_formats(str(args.delivery_formats or ""))

    acquire_lock(out_root, force=bool(args.force_lock))
    append_log(log_path, f"lock_acquired output_root={out_root}")

    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            plan = [coerce_plan_item(item) for item in manifest.get("items", [])]
            append_log(log_path, f"resume_loaded items={len(plan)}")
        else:
            plan = build_plan(out_root, args.source or None)
            manifest = {
                "generated_at": utc_now(),
                "repo_root": str(REPO_ROOT),
                "output_root": str(out_root),
                "runner": str(Path(__file__).resolve()),
                "mode": "mcp_only",
                "mcp_server": "AuralMind2",
                "mcp_guidance": [
                    "auralmind://connect-kit",
                    "auralmind://premium-trap-workflow",
                    "auralmind://control-surface",
                    "premium_trap_mastering_session",
                ],
                "premium_phase_alignment": {
                    "tool": "premium_phase_align",
                    "enabled_for_all_tracks": True,
                    "cutoff_hz": DEFAULT_PHASE_CUTOFF_HZ,
                },
                "settings_defaults": {
                    "preset_name": DEFAULT_PRESET,
                    "stem_mode": DEFAULT_STEM_MODE,
                    "target_lufs": DEFAULT_TARGET_LUFS,
                    "true_peak_dbtp": DEFAULT_TRUE_PEAK,
                    "control_profile": DEFAULT_CONTROL_PROFILE,
                },
                "source_overrides": list(args.source or []),
                "delivery_formats": delivery_formats,
                "delivery_note": (
                    "24/32-bit files are delivery encodes from the MCP phase-aligned master artifact; "
                    "they do not add local mastering."
                ),
            }
            write_manifest(manifest_path, manifest, plan)
            append_log(log_path, f"fresh_plan_written items={len(plan)}")
        manifest["delivery_formats"] = delivery_formats
        manifest["delivery_note"] = (
            "24/32-bit files are delivery encodes from the MCP phase-aligned master artifact; "
            "they do not add local mastering."
        )

        if args.dry_run:
            total_count = len(plan)
            for item in plan:
                append_log(log_path, f"dry_run {item.index}/{total_count} {item.filename} exists={item.source_exists}")
            return 0

        async with Client(server.mcp, name="AuralMind2-explicit-premium-trap", timeout=args.call_timeout) as client:
            bootstrap = await mcp_call(client, "bootstrap", {})
            connect_kit = await read_mcp_resource_text(client, "auralmind://connect-kit")
            trap_workflow = await read_mcp_resource_text(client, "auralmind://premium-trap-workflow")
            manifest["bootstrap_counts"] = {
                "tools": len(bootstrap.get("tools") or []),
                "resources": len(bootstrap.get("resources") or []),
                "prompts": len(bootstrap.get("prompts") or []),
            }
            manifest["guidance_loaded"] = {
                "connect_kit_chars": len(connect_kit),
                "premium_trap_workflow_chars": len(trap_workflow),
            }
            write_manifest(manifest_path, manifest, plan)

            to_run = [
                item
                for item in plan
                if item.status != "done" and (item.status != "error" or bool(args.retry_errors))
            ]
            append_log(log_path, f"queue_start total={len(plan)} to_run={len(to_run)}")
            total_count = len(plan)

            for item in to_run:
                try:
                    await run_one(
                        client,
                        item,
                        total_count=total_count,
                        delivery_formats=delivery_formats,
                        output_root=out_root,
                        poll_s=float(args.poll_seconds),
                        max_wait_s=float(args.max_wait_seconds),
                        log_path=log_path,
                    )
                except Exception as exc:
                    item.status = "error"
                    item.error = str(exc)
                    append_log(log_path, f"error {item.index}/10 {item.display_name}: {item.error}")
                finally:
                    write_manifest(manifest_path, manifest, plan)

        ensure_delivery_exports(plan, out_root, delivery_formats, log_path=log_path)
        write_manifest(manifest_path, manifest, plan)

        done = sum(1 for item in plan if item.status == "done")
        errors = sum(1 for item in plan if item.status == "error")
        append_log(log_path, f"queue_done done={done} error={errors} manifest={manifest_path}")
        return 0 if errors == 0 else 1
    finally:
        release_lock(out_root)
        append_log(log_path, "lock_released")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render explicit premium hi-fi trap masters through MCP only.")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Render this source path or data/ filename instead of the fixed 10-song queue. May be repeated.",
    )
    parser.add_argument(
        "--delivery-formats",
        type=str,
        default="",
        help="Comma-separated delivery formats to encode from the MCP artifact, e.g. 24,32.",
    )
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--max-wait-seconds", type=float, default=60 * 60 * 4)
    parser.add_argument("--call-timeout", type=float, default=60 * 60 * 4)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--force-lock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import asyncio

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
