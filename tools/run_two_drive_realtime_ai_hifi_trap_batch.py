from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = {
    "C": Path(r"C:\Users\goku\Documents\AuralMind2\data"),
    "D": Path(r"D:\music"),
}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a"}
SERVER_NATIVE_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif"}
SKIP_MARKERS = ("auralmind", "master", "trapgod")
TARGET_SAMPLE_RATE = 48_000
TARGET_BIT_DEPTH = "float32"
TARGET_CODEC = "pcm_f32le"
MOVEMENT_AMOUNT = 0.26
PLATFORM = "spotify"
RUN_PREFIX = "AuralMind_Premium_HiFi_Trap_Masters"


def new_output_root() -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = Path.home() / "Desktop" / f"{RUN_PREFIX}_{stamp}"
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = Path(f"{base}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


OUTPUT_ROOT = new_output_root()
STAGING_DIR = OUTPUT_ROOT / "staging"

# server.py resolves the filesystem allowlist at import time, so include the
# second drive and this run's staging folder before importing server.
allowed_roots = [str(path) for path in SOURCE_ROOTS.values()] + [str(STAGING_DIR)]
if os.environ.get("AURALMIND_AUDIO_ROOTS"):
    allowed_roots.append(os.environ["AURALMIND_AUDIO_ROOTS"])
os.environ["AURALMIND_AUDIO_ROOTS"] = os.pathsep.join(allowed_roots)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server  # noqa: E402


class BatchContext:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    async def report_progress(self, current: int, total: int, message: str = "") -> None:
        return None


CTX = BatchContext(f"two_drive_realtime_ai_hifi_trap_{int(time.time())}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def safe_slug(value: str) -> str:
    chars: List[str] = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_", ".", "(", ")"}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:120] or "untitled-song"


def normalize_song(path: Path, assumptions: List[Dict[str, str]]) -> Tuple[str, str]:
    text = re.sub(r"[_\-]+", " ", path.stem.lower())
    text = re.sub(r"\s+", " ", text).strip()
    before_counter = text
    text = re.sub(r"\s*\((\d{1,3})\)\s*$", "", text).strip()
    if text != before_counter:
        assumptions.append(
            {
                "path": str(path),
                "assumption": "Trailing numeric parenthetical treated as export/version clutter.",
                "before": before_counter,
                "after": text,
            }
        )
    for pattern in (
        r"\b(pre[ -]?master|premaster|unmastered|rough|bounce|export|copy)\b",
        r"\b(vocal mix|full mix|mixdown|mix)\b$",
        r"\b(final|new final|latest|version)\b$",
        r"\bv\d{1,3}\b$",
    ):
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip() or path.stem.lower()
    return safe_slug(text), text


def run_cmd(command: List[str], timeout_s: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_s)


def ffprobe_audio(path: Path) -> Dict[str, Any]:
    result = run_cmd(
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
        ],
        timeout_s=45,
    )
    streams = json.loads(result.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe_no_audio_stream")
    return streams[0]


def ensure_dirs() -> None:
    for name in ("manifests", "reports", "logs", "outputs", "staging"):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def log_line(message: str) -> None:
    log_path = OUTPUT_ROOT / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")
    print(message)


def scan_sources() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    assumptions: List[Dict[str, str]] = []
    ignored_dirs = {"$recycle.bin", "system volume information", ".git", ".venv", "venv", "node_modules", "__pycache__"}

    for drive, root in sorted(SOURCE_ROOTS.items()):
        dir_count = 0
        audio_seen = 0
        log_line(f"Scanning source root {drive}: {root}")
        if not root.exists():
            skipped.append({"source_drive": drive, "path": str(root), "reason": "source_root_missing"})
            continue
        for current_root, dirs, files in os.walk(root):
            dir_count += 1
            if dir_count % 500 == 0:
                log_line(f"Scan progress {drive}: {dir_count} folders, {audio_seen} audio-like files seen.")
            dirs[:] = [d for d in dirs if d.lower() not in ignored_dirs]
            for filename in sorted(files, key=str.lower):
                path = Path(current_root) / filename
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                audio_seen += 1
                if audio_seen % 500 == 0:
                    log_line(f"Scan progress {drive}: {dir_count} folders, {audio_seen} audio-like files seen.")
                lowered = path.name.lower()
                marker = next((m for m in SKIP_MARKERS if m in lowered), None)
                if marker:
                    skipped.append({"source_drive": drive, "path": str(path), "reason": f"filename_contains_{marker}"})
                    continue
                try:
                    stat = path.stat()
                except OSError as exc:
                    skipped.append({"source_drive": drive, "path": str(path), "reason": "stat_failed", "error": str(exc)})
                    continue
                if stat.st_size <= 0:
                    skipped.append({"source_drive": drive, "path": str(path), "reason": "zero_byte_file"})
                    continue
                song, display = normalize_song(path, assumptions)
                candidates.append(
                    {
                        "source_drive": drive,
                        "path": str(path),
                        "modified_time": stat.st_mtime,
                        "modified_iso": to_utc_iso(stat.st_mtime),
                        "size_bytes": stat.st_size,
                        "normalized_song_name": song,
                        "display_identity": display,
                    }
                )

        log_line(f"Finished scanning {drive}: {dir_count} folders, {audio_seen} audio-like files seen.")

    candidates.sort(key=lambda item: (item["source_drive"], item["normalized_song_name"], item["modified_time"], item["path"].lower()))
    return candidates, skipped, assumptions


def group_candidates(candidates: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for item in candidates:
        grouped.setdefault(item["source_drive"], {}).setdefault(item["normalized_song_name"], []).append(item)
    for by_song in grouped.values():
        for files in by_song.values():
            files.sort(key=lambda item: (item["modified_time"], item["path"].lower()))
    return grouped


def decode_valid(item: Dict[str, Any], skipped: List[Dict[str, Any]]) -> bool:
    try:
        ffprobe_audio(Path(item["path"]))
        return True
    except Exception as exc:
        skipped.append(
            {
                "source_drive": item["source_drive"],
                "path": item["path"],
                "reason": "decode_validation_failed",
                "error": str(exc),
            }
        )
        return False


def first_valid(files: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in files:
        if decode_valid(item, skipped):
            return item
    return None


def last_valid(files: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in reversed(files):
        if decode_valid(item, skipped):
            return item
    return None


def build_plan(grouped: Dict[str, Dict[str, List[Dict[str, Any]]]], skipped: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for drive in sorted(grouped):
        for song in sorted(grouped[drive]):
            files = grouped[drive][song]
            out_dir = OUTPUT_ROOT / "outputs" / song / f"from_{drive}"
            oldest = first_valid(files, skipped)
            newest = last_valid(files, skipped)
            if oldest is None or newest is None:
                skipped.append({"source_drive": drive, "normalized_song_name": song, "reason": "group_has_no_decode_valid_sources"})
                continue
            if oldest["path"] == newest["path"]:
                for variant in ("A", "B"):
                    plan.append({**oldest, "selection_reason": "single-file-variant", "mastering_variant": variant, "output_file_path": str(out_dir / f"{song}__from_{drive}__single__variant{variant}.wav")})
            else:
                for reason, variant, source in (("oldest", "A", oldest), ("newest", "B", newest)):
                    plan.append({**source, "selection_reason": reason, "mastering_variant": variant, "output_file_path": str(out_dir / f"{song}__from_{drive}__{reason}__variant{variant}.wav")})
    return plan


def variant_profile(variant: str) -> Dict[str, Any]:
    if variant == "A":
        return {
            "preset_name": "hi_fi_streaming",
            "target_lufs": -12.8,
            "warmth": 0.34,
            "transient_boost_db": 2.1,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -1.2,
            "goal": "Hi-fidelity premium modern trap master with clean expensive polish, mono-sub anchor, widened mono-safe top-end detail, hooklifted chorus energy, movement 0.26, controlled 808 weight, and no brittle highs.",
            "control_profile": {
                "spatial_width": 0.14,
                "brightness_tilt": 0.16,
                "harshness_control": 0.44,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.74,
            },
        }
    return {
        "preset_name": "competitive_trap",
        "target_lufs": -12.2,
        "warmth": 0.30,
        "transient_boost_db": 2.8,
        "governor_search_steps": 6,
        "governor_gr_limit_db": -1.0,
        "goal": "Energized premium modern trap master with forward impact, wider polished air, mono-sub anchor, hooklifted hooks, movement 0.26, controlled sub discipline, punchy transients, and competitive loudness without cheap limiting.",
        "control_profile": {
            "spatial_width": 0.24,
            "brightness_tilt": 0.20,
            "harshness_control": 0.50,
            "movement_amount": MOVEMENT_AMOUNT,
            "low_end_focus": 0.78,
        },
    }


def stage_source_if_needed(source_path: Path, drive: str, song: str) -> Path:
    if source_path.suffix.lower() in SERVER_NATIVE_EXTENSIONS:
        return source_path
    staged = STAGING_DIR / drive / song / f"{safe_slug(source_path.stem)}__staged_48k_float.wav"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists() and staged.stat().st_size > 0:
        return staged
    run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            TARGET_CODEC,
            str(staged),
        ],
        timeout_s=300,
    )
    return staged


def artifact_path(artifact_id: str) -> Path:
    session_key, session_dir = server._get_session_info(CTX)
    entry = server._load_artifact(session_key, session_dir, artifact_id)
    if entry is None:
        raise RuntimeError(f"Could not resolve artifact {artifact_id}")
    return Path(session_dir) / entry.data_filename


def finalize_to_48k_float(source: Path, destination: Path) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            TARGET_CODEC,
            str(destination),
        ],
        timeout_s=600,
    )
    return ffprobe_audio(destination)


def metrics_dict(metrics: Any) -> Dict[str, Any]:
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    return dict(metrics) if isinstance(metrics, dict) else {}


async def realtime_ai_render(audio_id: str, profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    control = server.MasteringControlProfile(**profile["control_profile"])
    plan = server.plan_mastering_strategy(
        server.StrategyPlanIn(
            audio_id=audio_id,
            goal=profile["goal"],
            platform=PLATFORM,
            control_profile=control,
            governor_search_steps=profile["governor_search_steps"],
            governor_gr_limit_db=profile["governor_gr_limit_db"],
            stem_mode="off",
        ),
        ctx=CTX,
    )
    stage1 = await server.start_interactive_mastering(
        server.StartInteractiveMasteringIn(
            audio_id=audio_id,
            preset_name=profile["preset_name"],
            control_profile=control,
            stem_mode="off",
        ),
        ctx=CTX,
    )
    final = await server.commit_interactive_mastering(
        server.CommitInteractiveMasteringIn(
            session_token=stage1.session_token,
            warmth=profile["warmth"],
            transient_boost_db=profile["transient_boost_db"],
            control_profile=control,
        ),
        ctx=CTX,
    )
    snapshot = {
        "chosen_preset": plan.chosen_preset,
        "reasoning": plan.reasoning,
        "warnings": plan.warnings,
        "planned_settings": plan.settings.model_dump(),
        "stage1_settings": stage1.stage1_settings.model_dump(),
        "stage1_metrics": stage1.metrics.model_dump(),
        "interactive_console": final.ascii_console,
    }
    return final.artifact_id, metrics_dict(final.final_metrics), snapshot


def fallback_local_render(audio_id: str, profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    control = server.MasteringControlProfile(**profile["control_profile"])
    request = server.MasterRequest(
        audio_id=audio_id,
        preset_name=profile["preset_name"],
        target_lufs=profile["target_lufs"],
        warmth=profile["warmth"],
        transient_boost_db=profile["transient_boost_db"],
        enable_harshness_limiter=True,
        enable_masking_eq=True,
        enable_air_motion=True,
        enable_hooklift=True,
        bit_depth=TARGET_BIT_DEPTH,
        control_profile=control,
        governor_search_steps=profile["governor_search_steps"],
        governor_gr_limit_db=profile["governor_gr_limit_db"],
        stem_mode="off",
    )
    result = server.master_audio(request, ctx=CTX)
    return result.master_wav_id, metrics_dict(result.metrics_after), {"fallback_request": request.model_dump()}


async def process_one(item: Dict[str, Any]) -> Dict[str, Any]:
    profile = variant_profile(item["mastering_variant"])
    settings = {
        "preset_name": profile["preset_name"],
        "target_lufs": profile["target_lufs"],
        "warmth": profile["warmth"],
        "transient_boost_db": profile["transient_boost_db"],
        "stem_mode": "off",
        "movement": MOVEMENT_AMOUNT,
        "hooklift": True,
        "mono_sub": True,
        "widened_top_end": True,
        "enable_air_motion": True,
        "bit_depth_target": TARGET_BIT_DEPTH,
        "sample_rate_target": TARGET_SAMPLE_RATE,
        "control_profile": profile["control_profile"],
        "governor_search_steps": profile["governor_search_steps"],
        "governor_gr_limit_db": profile["governor_gr_limit_db"],
    }
    manifest = {
        "normalized_song_name": item["normalized_song_name"],
        "source_drive": item["source_drive"],
        "source_file_path": item["path"],
        "source_file_modified_time": item["modified_iso"],
        "selection_reason": item["selection_reason"],
        "output_file_path": item["output_file_path"],
        "mastering_variant": item["mastering_variant"],
        "mastering_mode": "realtime_ai",
        "tools_or_script_used": "server.py:start_interactive_mastering -> commit_interactive_mastering",
        "key_settings": settings,
        "sample_rate_out": None,
        "bit_depth_out": None,
        "status": "planned",
        "error_message": "",
    }
    try:
        source = stage_source_if_needed(Path(item["path"]), item["source_drive"], item["normalized_song_name"])
        manifest["server_registered_source_path"] = str(source)
        registration = server.register_audio_from_path(str(source), ctx=CTX)
        source_metrics = server.analyze_audio(registration.audio_id, ctx=CTX)
        manifest["source_audio_id"] = registration.audio_id
        manifest["source_metrics"] = metrics_dict(source_metrics)
        try:
            artifact_id, output_metrics, planner = await realtime_ai_render(registration.audio_id, profile)
        except Exception as exc:
            log_line(f"Realtime AI failed for {item['normalized_song_name']} from {item['source_drive']} variant{item['mastering_variant']}; fallback local: {exc}")
            manifest["mastering_mode"] = "fallback_local"
            manifest["tools_or_script_used"] = "server.py:master_audio fallback after realtime AI failure"
            manifest["realtime_ai_error"] = str(exc)
            artifact_id, output_metrics, planner = fallback_local_render(registration.audio_id, profile)

        raw_path = Path(item["output_file_path"]).with_name(Path(item["output_file_path"]).stem + "__raw_server.wav")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_path(artifact_id), raw_path)
        format_info = finalize_to_48k_float(raw_path, Path(item["output_file_path"]))
        manifest.update(
            {
                "artifact_id": artifact_id,
                "raw_server_output_path": str(raw_path),
                "output_metrics": output_metrics,
                "planner_snapshot": planner,
                "output_format_probe": format_info,
                "sample_rate_out": int(format_info.get("sample_rate") or TARGET_SAMPLE_RATE),
                "bit_depth_out": "32-bit float" if format_info.get("sample_fmt") == "flt" else format_info.get("sample_fmt"),
                "status": "completed",
            }
        )
        log_line(f"Completed {item['normalized_song_name']} from {item['source_drive']} {item['selection_reason']} variant{item['mastering_variant']}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_message"] = str(exc)
        log_line(f"Failed {item['normalized_song_name']} from {item['source_drive']} {item['selection_reason']} variant{item['mastering_variant']}: {exc}")
    return manifest


def build_summary(
    candidates: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]],
    plan: List[Dict[str, Any]],
) -> Dict[str, Any]:
    duplicate_ids = sorted(set(grouped.get("C", {})) & set(grouped.get("D", {})))
    return {
        "output_folder_path": str(OUTPUT_ROOT),
        "candidate_files": len(candidates),
        "skipped_files": len(skipped),
        "song_groups_on_C": len(grouped.get("C", {})),
        "song_groups_on_D": len(grouped.get("D", {})),
        "duplicate_song_identities_across_drives": len(duplicate_ids),
        "duplicate_song_identities": duplicate_ids,
        "total_expected_mastered_outputs": len(plan),
        "pipeline_chosen": "server.py realtime interactive AI path with server.master_audio fallback",
        "realtime_ai_enabled": True,
        "exact_32bit_48000_supported": True,
        "format_reason": "server.py/auralmind_maestro supports float32 WAV and this runner finalizes completed outputs to pcm_f32le at 48000 Hz.",
    }


def print_dry_run(summary: Dict[str, Any]) -> None:
    print("\nDRY-RUN PREVIEW")
    print("=" * 80)
    for key, value in summary.items():
        if key == "duplicate_song_identities":
            shown = ", ".join(value[:20])
            suffix = " ..." if len(value) > 20 else ""
            print(f"{key}: {shown}{suffix}")
        else:
            print(f"{key}: {value}")
    print("=" * 80)


def planned_manifest(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in plan:
        rows.append(
            {
                "normalized_song_name": item["normalized_song_name"],
                "source_drive": item["source_drive"],
                "source_file_path": item["path"],
                "source_file_modified_time": item["modified_iso"],
                "selection_reason": item["selection_reason"],
                "output_file_path": item["output_file_path"],
                "mastering_variant": item["mastering_variant"],
                "mastering_mode": "realtime_ai",
                "tools_or_script_used": "server.py:start_interactive_mastering -> commit_interactive_mastering",
                "key_settings": variant_profile(item["mastering_variant"]),
                "sample_rate_out": None,
                "bit_depth_out": None,
                "status": "planned",
                "error_message": "",
            }
        )
    return rows


def write_manifest(
    outputs: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    assumptions: List[Dict[str, str]],
    summary: Dict[str, Any],
) -> None:
    payload = {
        "generated_at": utc_now(),
        "scanned_roots": {drive: str(path) for drive, path in SOURCE_ROOTS.items()},
        "summary": summary,
        "candidate_files": candidates,
        "grouping_assumptions": assumptions,
        "outputs": outputs,
    }
    (OUTPUT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifests" / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "reports" / "skipped_files.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")


def write_report(outputs: List[Dict[str, Any]], skipped: List[Dict[str, Any]], assumptions: List[Dict[str, str]], summary: Dict[str, Any]) -> None:
    completed = [row for row in outputs if row["status"] == "completed"]
    failed = [row for row in outputs if row["status"] == "failed"]
    per_drive: Dict[str, Dict[str, int]] = {}
    per_song: Dict[str, List[Dict[str, Any]]] = {}
    for row in outputs:
        drive = row["source_drive"]
        per_drive.setdefault(drive, {"planned": 0, "completed": 0, "failed": 0})
        per_drive[drive]["planned"] += 1
        if row["status"] in {"completed", "failed"}:
            per_drive[drive][row["status"]] += 1
        per_song.setdefault(row["normalized_song_name"], []).append(row)

    lines = [
        "# AuralMind Premium HiFi Trap Two-Drive Processing Report",
        "",
        "## Overview",
        f"- Generated at: `{utc_now()}`",
        f"- Output folder: `{OUTPUT_ROOT}`",
        f"- Planned masters: `{len(outputs)}`",
        f"- Completed masters: `{len(completed)}`",
        f"- Failed masters: `{len(failed)}`",
        "",
        "## Scanned Roots",
        *[f"- {drive}: `{path}`" for drive, path in SOURCE_ROOTS.items()],
        "",
        "## Tooling Discovered",
        "- `server.py`: MCP-native registration, analysis, semantic planning, async jobs, and interactive stage-1/stage-2 mastering.",
        "- `ai_mastering_tool.py`: realtime AI wrapper showing live analysis, monitoring, commentary, and interactive mastering flow.",
        "- `tools/auralmind_maestro.py`: DSP engine under `server.py`, including mono-sub, air motion, hooklift, movement shaping, and float WAV export.",
        "",
        "## Whether Real-Time AI Mastering Was Used",
        f"- Real-time AI available: `{summary['realtime_ai_enabled']}`",
        f"- Completed realtime AI masters: `{len([r for r in completed if r['mastering_mode'] == 'realtime_ai'])}`",
        f"- Fallback local masters: `{len([r for r in completed if r['mastering_mode'] == 'fallback_local'])}`",
        "",
        "## Skip Rules Applied",
        "- Extensions scanned: `.wav`, `.mp3`, `.flac`, `.aiff`, `.aif`, `.m4a`",
        "- Filename exclusion markers: `auralmind`, `master`, `trapgod`",
        "- Zero-byte files and ffprobe decode failures were skipped and logged.",
        f"- Total skipped files: `{len(skipped)}`",
        "",
        "## Grouping Logic Summary",
        "- Lowercase filenames, strip extension, normalize separators, collapse spaces, remove safe trailing numeric export counters, then slug for path safety.",
        "- Grouping is per drive; C and D are never merged into one source pool.",
        "- Oldest and newest are selected by modified time ascending within each drive/song group.",
        "",
        "## Ambiguity Handling Notes",
    ]
    if assumptions:
        lines.extend(f"- `{note['path']}`: {note['assumption']} `{note.get('before')}` -> `{note.get('after')}`" for note in assumptions[:80])
    else:
        lines.append("- No ambiguity notes were recorded.")

    lines.extend(["", "## Per-Song Breakdown"])
    for song in sorted(per_song):
        lines.append(f"### {song}")
        for row in per_song[song]:
            lines.append(f"- from_{row['source_drive']} {row['selection_reason']} variant{row['mastering_variant']}: `{row['status']}` -> `{row['output_file_path']}`")
            if row.get("error_message"):
                lines.append(f"  - Error: `{row['error_message']}`")

    lines.extend(["", "## Per-Drive Breakdown"])
    for drive in sorted(per_drive):
        stats = per_drive[drive]
        lines.append(f"- {drive}: planned `{stats['planned']}`, completed `{stats['completed']}`, failed `{stats['failed']}`")

    lines.extend(
        [
            "",
            "## Output Summary",
            f"- Candidate files: `{summary['candidate_files']}`",
            f"- C song groups: `{summary['song_groups_on_C']}`",
            f"- D song groups: `{summary['song_groups_on_D']}`",
            f"- Duplicate identities across drives: `{summary['duplicate_song_identities_across_drives']}`",
            f"- Expected masters: `{summary['total_expected_mastered_outputs']}`",
            f"- Completed masters: `{len(completed)}`",
            "",
            "## Failures and Recoveries",
        ]
    )
    if failed:
        lines.extend(f"- `{row['normalized_song_name']}` from {row['source_drive']} variant{row['mastering_variant']}: `{row['error_message']}`" for row in failed)
    else:
        lines.append("- No final mastering failures recorded.")

    lines.extend(
        [
            f"- Realtime-to-local fallback recoveries: `{len([r for r in completed if r['mastering_mode'] == 'fallback_local'])}`",
            "",
            "## Chosen Mastering Chain and Why",
            "- Primary route: `server.py` interactive realtime AI path because it is repo-native, MCP-facing, supports semantic planning/control profiles, and renders a two-stage final pass.",
            "- Fallback route: `server.py master_audio` because it uses the same AuralMind engine with explicit settings if interactive state fails.",
            "- Stems were forced off to honor the no-stems constraint.",
            "",
            "## Export Format Results",
            f"- Target sample rate: `{TARGET_SAMPLE_RATE}`",
            "- Target bit depth: `32-bit float WAV`",
            f"- Exact target supported: `{summary['exact_32bit_48000_supported']}`",
            "- Every completed output was finalized through ffmpeg as `pcm_f32le` at `48000 Hz` and then probed.",
            "",
            "## Final Totals",
            f"- Candidate files found: `{summary['candidate_files']}`",
            f"- Skipped files: `{summary['skipped_files']}`",
            f"- Song groups processed: `{summary['song_groups_on_C'] + summary['song_groups_on_D']}`",
            f"- Masters created: `{len(completed)}`",
            f"- Failures: `{len(failed)}`",
            "",
            "## Manual Verification Checklist",
            "1. Level-match each A/B pair before judging tone.",
            "2. Check sub mono compatibility from 30-120 Hz on headphones and monitors.",
            "3. Listen to hooks for lift without vocal harshness or cymbal brittleness.",
            "4. Fold to mono and confirm the kick/808 center remains solid.",
            "5. Compare variant A vs B for polish versus forward energy, then keep the better emotional lane per song.",
        ]
    )
    (OUTPUT_ROOT / "reports" / "processing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_batch(dry_run: bool, limit: Optional[int]) -> int:
    ensure_dirs()
    log_line("Starting two-drive realtime AI hi-fi trap batch.")
    candidates, skipped, assumptions = scan_sources()
    grouped = group_candidates(candidates)
    plan = build_plan(grouped, skipped)
    if limit is not None:
        plan = plan[:limit]
    summary = build_summary(candidates, skipped, grouped, plan)
    print_dry_run(summary)
    log_line(f"Dry-run summary: {json.dumps(summary, sort_keys=True)}")

    planned = planned_manifest(plan)
    if dry_run:
        write_manifest(planned, candidates, skipped, assumptions, summary)
        write_report(planned, skipped, assumptions, summary)
        log_line("Dry-run only requested; no mastering jobs executed.")
        return 0

    outputs: List[Dict[str, Any]] = []
    for idx, item in enumerate(plan, start=1):
        log_line(f"Rendering {idx}/{len(plan)}: {item['normalized_song_name']} from {item['source_drive']} {item['selection_reason']} variant{item['mastering_variant']}")
        outputs.append(await process_one(item))
        write_manifest(outputs + planned[len(outputs):], candidates, skipped, assumptions, summary)

    write_manifest(outputs, candidates, skipped, assumptions, summary)
    write_report(outputs, skipped, assumptions, summary)
    log_line("Batch complete.")

    completed = len([row for row in outputs if row["status"] == "completed"])
    failed = len([row for row in outputs if row["status"] == "failed"])
    print("\nFINAL COMPLETION SUMMARY")
    print("=" * 80)
    print(f"output folder created: {OUTPUT_ROOT}")
    print("mastering pipeline used: server.py realtime interactive AI path with server.master_audio fallback")
    print("whether real-time AI mastering was used: yes")
    print(f"total candidate files found: {summary['candidate_files']}")
    print(f"total skipped files: {summary['skipped_files']}")
    print(f"total song groups processed: {summary['song_groups_on_C'] + summary['song_groups_on_D']}")
    print(f"total masters created: {completed}")
    print(f"total failures: {failed}")
    print("manual listening check: level-match A/B pairs, then verify mono sub focus, hook lift, and top-end width in stereo and mono fold-down.")
    print("=" * 80)
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-drive realtime AI premium hi-fi trap mastering batch.")
    parser.add_argument("--dry-run", action="store_true", help="Scan, group, and write planned manifests without mastering.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of planned outputs to process.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_batch(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
