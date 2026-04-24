from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server

RUN_LABEL = "premium_trap_master_batch_20260417"
PLATFORM = "spotify"
FINAL_TARGET_LUFS = -13.5
FINAL_TRUE_PEAK = -0.4
FINAL_SAMPLE_RATE = 44_100

OUTPUT_DIR = ROOT / "masters" / RUN_LABEL
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw_auralmind"
DELIVERY_24_DIR = OUTPUT_DIR / "delivery_24bit_44k1"
REPORTS_DIR = OUTPUT_DIR / "reports"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"

PROFILE_REFERENCE_SCRIPT = ROOT / "tools" / "render_dont_push_me_premium_quad_20260415.py"
PROFILE_REFERENCE_MANIFEST = ROOT / "masters" / "dont_push_me_premium_quad_20260415" / "manifest.json"

SOURCE_SONGS: List[Dict[str, Any]] = [
    {"display_name": "FaceTime (5)", "source_path": ROOT / "data" / "FaceTime (5).wav"},
    {"display_name": "New Project (18)", "source_path": ROOT / "data" / "New Project (18).wav"},
    {"display_name": "New Project (19)", "source_path": ROOT / "data" / "New Project (19).wav"},
    {"display_name": "Truthy (1)", "source_path": ROOT / "data" / "Truthy (1).wav"},
    {"display_name": "Stilll KOTS", "source_path": ROOT / "data" / "Stilll KOTS.wav"},
    {"display_name": "Best of me (1)", "source_path": ROOT / "data" / "Best of me (1).wav"},
]

VARIANT_PROFILES: List[Dict[str, Any]] = [
    {
        "variant_key": "industry-standard__stem-on",
        "variant_label": "Industry Standard (Stem-On)",
        "goal": (
            "Industry-standard premium trap master with maximum clarity, punchy transients, polished "
            "commercial finish, mono sub-bass discipline, hooklifted hooks, and movement 0.26."
        ),
        "preset_name": "competitive_trap",
        "warmth": 0.28,
        "transient_boost_db": 2.4,
        "governor_search_steps": 6,
        "governor_gr_limit_db": -1.2,
        "stem_mode": "on",
        "stem_gains_db": {"vocals": 0.2, "drums": 0.35, "bass": -0.1, "other": 0.05},
        "control_profile": {
            "spatial_width": 0.10,
            "brightness_tilt": 0.18,
            "harshness_control": 0.34,
            "movement_amount": 0.26,
            "low_end_focus": 0.72,
        },
    },
    {
        "variant_key": "industry-standard__stem-off",
        "variant_label": "Industry Standard (Stem-Off)",
        "goal": (
            "Industry-standard premium trap master with maximum clarity, punchy transients, polished "
            "commercial finish, mono sub-bass discipline, hooklifted hooks, and movement 0.26."
        ),
        "preset_name": "competitive_trap",
        "warmth": 0.28,
        "transient_boost_db": 2.4,
        "governor_search_steps": 6,
        "governor_gr_limit_db": -1.2,
        "stem_mode": "off",
        "stem_gains_db": None,
        "control_profile": {
            "spatial_width": 0.10,
            "brightness_tilt": 0.18,
            "harshness_control": 0.34,
            "movement_amount": 0.26,
            "low_end_focus": 0.72,
        },
    },
    {
        "variant_key": "deep-and-wide__stem-off",
        "variant_label": "Deep & Wide (Stem-Off)",
        "goal": (
            "Depth-first premium trap master with massive low-end foundation, wide spatial imaging, "
            "mono sub-bass discipline, hooklifted hooks, and movement 0.26."
        ),
        "preset_name": "cinematic",
        "warmth": 0.42,
        "transient_boost_db": 1.5,
        "governor_search_steps": 5,
        "governor_gr_limit_db": -0.9,
        "stem_mode": "off",
        "stem_gains_db": None,
        "control_profile": {
            "spatial_width": 0.42,
            "brightness_tilt": -0.04,
            "harshness_control": 0.28,
            "movement_amount": 0.26,
            "low_end_focus": 0.88,
        },
    },
    {
        "variant_key": "deep-and-wide__stem-on",
        "variant_label": "Deep & Wide (Stem-On)",
        "goal": (
            "Depth-first premium trap master with massive low-end foundation, wide spatial imaging, "
            "mono sub-bass discipline, hooklifted hooks, and movement 0.26."
        ),
        "preset_name": "cinematic",
        "warmth": 0.42,
        "transient_boost_db": 1.5,
        "governor_search_steps": 5,
        "governor_gr_limit_db": -0.9,
        "stem_mode": "on",
        "stem_gains_db": None,
        "control_profile": {
            "spatial_width": 0.42,
            "brightness_tilt": -0.04,
            "harshness_control": 0.28,
            "movement_amount": 0.26,
            "low_end_focus": 0.88,
        },
    },
]


class _DummyContext:
    session_id = f"{RUN_LABEL}-{int(datetime.now(timezone.utc).timestamp())}"

    async def report_progress(self, *_args: Any, **_kwargs: Any) -> None:
        return None


DUMMY_CTX = _DummyContext()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_name(value: str) -> str:
    safe: List[str] = []
    for char in value:
        if char.isalnum():
            safe.append(char.lower())
        elif char in {" ", "-", "_", "(", ")", "."}:
            safe.append("-")
    joined = "".join(safe).strip("-")
    while "--" in joined:
        joined = joined.replace("--", "-")
    return joined or "master"


def run_ffmpeg(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def run_ffprobe(path: Path) -> Dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,sample_rate,bits_per_sample,bits_per_raw_sample,sample_fmt,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    return streams[0] if streams else {}


def parse_loudnorm_json(stderr_text: str) -> Dict[str, Any]:
    start = stderr_text.rfind("{")
    end = stderr_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Could not locate loudnorm JSON in ffmpeg output.")
    return json.loads(stderr_text[start : end + 1])


def analyze_loudnorm(path: Path) -> Dict[str, Any]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(path),
        "-af",
        f"loudnorm=I={FINAL_TARGET_LUFS}:TP={FINAL_TRUE_PEAK}:LRA=7:print_format=json",
        "-f",
        "null",
        "-",
    ]
    result = run_ffmpeg(command)
    return parse_loudnorm_json(result.stderr)


def finalize_master(raw_path: Path, final_path: Path) -> Dict[str, Any]:
    analysis_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        (
            f"aresample={FINAL_SAMPLE_RATE},"
            f"loudnorm=I={FINAL_TARGET_LUFS}:TP={FINAL_TRUE_PEAK}:LRA=7:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    analysis_run = run_ffmpeg(analysis_cmd)
    measured = parse_loudnorm_json(analysis_run.stderr)

    render_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        (
            f"aresample={FINAL_SAMPLE_RATE},"
            f"loudnorm=I={FINAL_TARGET_LUFS}:TP={FINAL_TRUE_PEAK}:LRA=7:"
            f"measured_I={measured['input_i']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:"
            "linear=true:print_format=summary"
        ),
        "-ar",
        str(FINAL_SAMPLE_RATE),
        "-c:a",
        "pcm_s24le",
        str(final_path),
    ]
    render_run = run_ffmpeg(render_cmd)
    verification = analyze_loudnorm(final_path)
    stream_info = run_ffprobe(final_path)
    return {
        "analysis": measured,
        "verification": verification,
        "stream_info": stream_info,
        "render_summary": render_run.stderr.strip().splitlines()[-12:],
    }


def artifact_source_path(artifact_id: str) -> Path:
    session_key, session_dir = server._get_session_info(DUMMY_CTX)
    entry = server._load_artifact(session_key, session_dir, artifact_id)
    if entry is None:
        raise RuntimeError(f"Could not resolve artifact {artifact_id}.")
    return Path(session_dir) / entry.data_filename


def copy_artifact_to_path(artifact_id: str, destination: Path) -> None:
    source = artifact_source_path(artifact_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def metrics_dict(metrics: Any) -> Dict[str, Any]:
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    return dict(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the April 15 premium trap master profile across a fixed six-song batch "
            "with fresh AuralMind2 analysis, raw exports, and 44.1k / 24-bit delivery WAVs."
        )
    )
    parser.add_argument(
        "--song",
        action="append",
        dest="song_filters",
        default=[],
        help="Optional song name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variant_filters",
        default=[],
        help="Optional variant key filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if a 24-bit delivery file already exists.",
    )
    return parser.parse_args()


def filter_songs(song_filters: List[str]) -> List[Dict[str, Any]]:
    if not song_filters:
        return SOURCE_SONGS
    wanted = {value.strip().lower() for value in song_filters if value.strip()}
    return [song for song in SOURCE_SONGS if song["display_name"].lower() in wanted]


def filter_variants(variant_filters: List[str]) -> List[Dict[str, Any]]:
    if not variant_filters:
        return VARIANT_PROFILES
    wanted = {value.strip().lower() for value in variant_filters if value.strip()}
    return [variant for variant in VARIANT_PROFILES if variant["variant_key"].lower() in wanted]


def register_and_analyze_sources(
    selected_songs: List[Dict[str, Any]], selected_variants: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    registered: Dict[str, Dict[str, Any]] = {}
    for song in selected_songs:
        source_path = Path(song["source_path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")

        key = str(source_path)
        registration = server.register_audio_from_path(str(source_path), ctx=DUMMY_CTX)
        source_metrics = server.analyze_audio(registration.audio_id, ctx=DUMMY_CTX)

        planning: Dict[str, Any] = {}
        governor_by_preset: Dict[str, Any] = {}
        for variant in selected_variants:
            plan = server.plan_mastering_strategy(
                server.StrategyPlanIn(
                    audio_id=registration.audio_id,
                    goal=variant["goal"],
                    platform=PLATFORM,
                    control_profile=server.MasteringControlProfile(**variant["control_profile"]),
                    governor_search_steps=variant["governor_search_steps"],
                    governor_gr_limit_db=variant["governor_gr_limit_db"],
                    stem_gains_db=variant["stem_gains_db"],
                    stem_mode=variant["stem_mode"],
                ),
                ctx=DUMMY_CTX,
            )
            planning[variant["variant_key"]] = {
                "chosen_preset": plan.chosen_preset,
                "reasoning": plan.reasoning,
                "warnings": plan.warnings,
                "settings": plan.settings.model_dump(),
            }

            preset_name = variant["preset_name"]
            if preset_name not in governor_by_preset:
                governor_result = asyncio.run(
                    server.analyze_and_optimize_governor(
                        server.AnalyzeAndOptimizeGovernorIn(
                            audio_id=registration.audio_id,
                            preset_name=preset_name,
                        ),
                        ctx=DUMMY_CTX,
                    )
                )
                governor_by_preset[preset_name] = governor_result.model_dump()

        registered[key] = {
            "display_name": song["display_name"],
            "source_path": str(source_path),
            "audio_id": registration.audio_id,
            "source_metrics": metrics_dict(source_metrics),
            "planning": planning,
            "governor_by_preset": governor_by_preset,
        }
        print(
            f"Analyzed {song['display_name']} -> {registration.audio_id} "
            f"(LUFS {registered[key]['source_metrics']['integrated_lufs']:.2f}, "
            f"TP {registered[key]['source_metrics']['true_peak_dbtp']:.2f})"
        )
    return registered


def write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    verify = payload["delivery_24_verification"]["verification"]
    report_lines = [
        f"# {payload['display_name']} :: {payload['variant_label']}",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Source path: `{payload['source_path']}`",
        f"- Source audio id: `{payload['source_audio_id']}`",
        f"- Working audio id: `{payload['working_audio_id']}`",
        f"- Preset: `{payload['preset_name']}`",
        f"- Stem mode: `{payload['stem_mode']}`",
        f"- Final target: `{FINAL_TARGET_LUFS} LUFS`, `{FINAL_TRUE_PEAK} dBTP`, `{FINAL_SAMPLE_RATE} Hz`",
        "",
        "## Goal",
        payload["goal"],
        "",
        "## Source Analysis",
        "```json",
        json.dumps(payload["source_metrics"], indent=2),
        "```",
        "",
        "## Planner Snapshot",
        f"- Chosen preset: `{payload['planner_snapshot']['chosen_preset']}`",
        f"- Reasoning: {payload['planner_snapshot']['reasoning']}",
        f"- Warnings: {payload['planner_snapshot']['warnings'] or ['None']}",
        "",
        "## Governor Snapshot",
        "```json",
        json.dumps(payload["governor_snapshot"], indent=2),
        "```",
        "",
        "## Final Request",
        "```json",
        json.dumps(payload["final_request"], indent=2),
        "```",
        "",
        "## Metrics",
        f"- Before: `{json.dumps(payload['metrics_before'], indent=2)}`",
        f"- After: `{json.dumps(payload['metrics_after'], indent=2)}`",
        "",
        "## Delivery Output",
        f"- Raw AuralMind export: `{payload['raw_wav_path']}`",
        f"- 24-bit 44.1k delivery: `{payload['delivery_24_path']}`",
        "",
        "## Delivery Verification",
        f"- Verified LUFS: `{verify.get('input_i')} LUFS`",
        f"- Verified true peak: `{verify.get('input_tp')} dBTP`",
        "```json",
        json.dumps(payload["delivery_24_verification"], indent=2),
        "```",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def render_profile(
    song_info: Dict[str, Any],
    variant: Dict[str, Any],
    force: bool,
) -> Dict[str, Any]:
    source_slug = sanitize_name(song_info["display_name"])
    output_stub = f"{source_slug}__{variant['variant_key']}"
    raw_destination = RAW_OUTPUT_DIR / f"{output_stub}__raw.wav"
    delivery_24_path = DELIVERY_24_DIR / f"{output_stub}__24bit_44k1.wav"
    report_path = REPORTS_DIR / f"{output_stub}.md"

    if delivery_24_path.exists() and not force:
        print(f"Skipping existing delivery: {delivery_24_path}")
        return {
            "display_name": song_info["display_name"],
            "variant_label": variant["variant_label"],
            "skipped_existing": True,
            "delivery_24_path": str(delivery_24_path),
            "report_path": str(report_path),
        }

    planner_snapshot = song_info["planning"][variant["variant_key"]]
    explicit_settings = planner_snapshot["settings"]
    explicit_settings.update(
        {
            "preset_name": variant["preset_name"],
            "target_lufs": FINAL_TARGET_LUFS,
            "warmth": variant["warmth"],
            "transient_boost_db": variant["transient_boost_db"],
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "bit_depth": "float32",
            "control_profile": variant["control_profile"],
            "governor_search_steps": variant["governor_search_steps"],
            "governor_gr_limit_db": variant["governor_gr_limit_db"],
            "stem_gains_db": variant["stem_gains_db"],
            "stem_mode": variant["stem_mode"],
        }
    )
    normalized = server.propose_master_settings(server.MasterSettings(**explicit_settings)).settings
    request = server.MasterRequest(audio_id=song_info["audio_id"], **normalized.model_dump())
    result = server.master_audio(request, ctx=DUMMY_CTX)

    copy_artifact_to_path(result.master_wav_id, raw_destination)
    delivery_24 = finalize_master(raw_destination, delivery_24_path)

    payload = {
        "generated_at": utc_now(),
        "display_name": song_info["display_name"],
        "source_path": song_info["source_path"],
        "source_audio_id": song_info["audio_id"],
        "working_audio_id": song_info["audio_id"],
        "variant_key": variant["variant_key"],
        "variant_label": variant["variant_label"],
        "goal": variant["goal"],
        "preset_name": variant["preset_name"],
        "stem_mode": variant["stem_mode"],
        "source_metrics": song_info["source_metrics"],
        "planner_snapshot": planner_snapshot,
        "governor_snapshot": song_info["governor_by_preset"][variant["preset_name"]],
        "final_request": request.model_dump(),
        "metrics_before": result.metrics_before.model_dump(),
        "metrics_after": result.metrics_after.model_dump(),
        "master_wav_id": result.master_wav_id,
        "tuning_trace_id": result.tuning_trace_id,
        "artifacts": result.artifacts,
        "raw_wav_path": str(raw_destination),
        "delivery_24_path": str(delivery_24_path),
        "delivery_24_verification": delivery_24,
        "report_path": str(report_path),
        "skipped_existing": False,
    }
    write_report(report_path, payload)
    print(
        f"Rendered {song_info['display_name']} :: {variant['variant_label']} -> "
        f"{delivery_24_path}"
    )
    return payload


def write_summary(manifest: Dict[str, Any]) -> None:
    lines = [
        f"# {RUN_LABEL}",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Project root: `{manifest['project_root']}`",
        f"- Output root: `{manifest['output_dir']}`",
        f"- Profile reference script: `{manifest['profile_reference_script']}`",
        f"- Profile reference manifest: `{manifest['profile_reference_manifest']}`",
        f"- Final target: `{FINAL_TARGET_LUFS} LUFS`, `{FINAL_TRUE_PEAK} dBTP`, `{FINAL_SAMPLE_RATE} Hz`",
        "",
        "## Source Summary",
    ]
    for source in manifest["sources"]:
        metrics = source["source_metrics"]
        lines.extend(
            [
                f"### {source['display_name']}",
                f"- Audio id: `{source['audio_id']}`",
                f"- Source path: `{source['source_path']}`",
                f"- Integrated LUFS: `{metrics['integrated_lufs']:.2f}`",
                f"- True peak: `{metrics['true_peak_dbtp']:.2f} dBTP`",
                f"- Crest factor: `{metrics['crest_db']:.2f} dB`",
                f"- Stereo correlation: `{metrics['stereo_correlation']:.3f}`",
                "",
            ]
        )
    lines.append("## Renders")
    for render in manifest["renders"]:
        if render.get("skipped_existing"):
            lines.extend(
                [
                    f"### {render['display_name']} :: {render['variant_label']}",
                    f"- Skipped existing delivery: `{render['delivery_24_path']}`",
                    "",
                ]
            )
            continue
        verify = render["delivery_24_verification"]["verification"]
        lines.extend(
            [
                f"### {render['display_name']} :: {render['variant_label']}",
                f"- Raw export: `{render['raw_wav_path']}`",
                f"- 24-bit delivery: `{render['delivery_24_path']}`",
                f"- Report: `{render['report_path']}`",
                f"- Post-master LUFS: `{render['metrics_after']['integrated_lufs']:.2f}`",
                f"- Final verified LUFS: `{verify.get('input_i')} LUFS`",
                f"- Final verified TP: `{verify.get('input_tp')} dBTP`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    args = parse_args()
    selected_songs = filter_songs(args.song_filters)
    selected_variants = filter_variants(args.variant_filters)

    if not selected_songs:
        raise ValueError("No songs selected. Check the --song filters.")
    if not selected_variants:
        raise ValueError("No variants selected. Check the --variant filters.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERY_24_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    registered = register_and_analyze_sources(selected_songs, selected_variants)
    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "run_label": RUN_LABEL,
        "project_root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "platform": PLATFORM,
        "profile_reference_script": str(PROFILE_REFERENCE_SCRIPT),
        "profile_reference_manifest": str(PROFILE_REFERENCE_MANIFEST),
        "final_target_lufs": FINAL_TARGET_LUFS,
        "final_true_peak_dbtp": FINAL_TRUE_PEAK,
        "final_sample_rate_hz": FINAL_SAMPLE_RATE,
        "sources": list(registered.values()),
        "selected_variants": [variant["variant_key"] for variant in selected_variants],
        "renders": [],
    }

    for song in selected_songs:
        song_key = str(Path(song["source_path"]).resolve())
        song_info = registered[song_key]
        for variant in selected_variants:
            render_summary = render_profile(song_info, variant, force=args.force)
            manifest["renders"].append(render_summary)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest)
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
