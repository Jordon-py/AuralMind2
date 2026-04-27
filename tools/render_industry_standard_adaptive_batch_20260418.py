from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server

RUN_LABEL = "industry_standard_adaptive_batch_20260418"
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

BATCH_BASE_SCRIPT = ROOT / "tools" / "render_premium_trap_master_batch_20260417.py"
SONIC_REFERENCE_MANIFEST = ROOT / "masters" / "dont_push_me_premium_quad_20260415" / "manifest.json"

REFERENCE_SOURCE_METRICS: Dict[str, float] = {
    "integrated_lufs": -22.377421904756357,
    "true_peak_dbtp": 0.019233707758001053,
    "crest_db": 18.71931920167703,
    "stereo_correlation": 0.8816834375836028,
    "centroid_hz": 762.6115647391013,
}

LOSSY_EXTENSIONS = {".mp3", ".m4a", ".ogg"}

SOURCE_SONGS: List[Dict[str, Any]] = [
    {"display_name": "New Project (18)", "source_path": ROOT / "data" / "New Project (18).wav"},
    {"display_name": "New Project (19)", "source_path": ROOT / "data" / "New Project (19).wav"},
    {"display_name": "Truthy (1)", "source_path": ROOT / "data" / "Truthy (1).wav"},
    {"display_name": "FaceTime (5)", "source_path": ROOT / "data" / "FaceTime (5).wav"},
    {"display_name": "Hot shit (1)", "source_path": ROOT / "data" / "Hot shit (1).wav"},
    {"display_name": "difference (10)", "source_path": ROOT / "data" / "difference (10).wav"},
    {"display_name": "Stilll KOTS", "source_path": ROOT / "data" / "Stilll KOTS.wav"},
    {"display_name": "FaceTime (4)", "source_path": ROOT / "data" / "FaceTime (4).wav"},
    {"display_name": "Vegas - top teir (22)", "source_path": ROOT / "data" / "Vegas - top teir (22).wav"},
    {"display_name": "Walking_In_Rain", "source_path": ROOT / "data" / "Walking_In_Rain.mp3"},
]

VARIANT_PROFILES: List[Dict[str, Any]] = [
    {
        "variant_key": "industry-standard__adaptive",
        "variant_label": "Industry Standard (Adaptive)",
        "goal": (
            "Industry-standard premium trap master matched to the approved "
            "Industry Standard 24bit 44.1k Premium lane, but adapted per song from "
            "source metrics so bright or dense mixes get safer control while darker "
            "or more open mixes keep punch and finish."
        ),
        "preset_name": "competitive_trap",
        "warmth": 0.28,
        "transient_boost_db": 2.4,
        "governor_search_steps": 6,
        "governor_gr_limit_db": -1.2,
        "stem_mode": "on",
        "stem_gains_db": {"vocals": 0.2, "drums": 0.35, "bass": -0.1, "other": 0.05},
        "enable_harshness_limiter": True,
        "enable_masking_eq": True,
        "enable_air_motion": True,
        "enable_hooklift": True,
        "control_profile": {
            "spatial_width": 0.10,
            "brightness_tilt": 0.18,
            "harshness_control": 0.34,
            "movement_amount": 0.26,
            "low_end_focus": 0.72,
        },
    }
]


class _DummyContext:
    session_id = f"{RUN_LABEL}-{int(datetime.now(timezone.utc).timestamp())}"

    async def report_progress(self, *_args: Any, **_kwargs: Any) -> None:
        return None


DUMMY_CTX = _DummyContext()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
            "Render an adaptive Industry Standard premium trap batch by reusing the "
            "2026-04-17 batch runner structure and tuning each song against the approved "
            "Don't Push Me Industry Standard reference metrics."
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


def adapt_stem_gains(base_gains: Dict[str, float], source_metrics: Dict[str, Any]) -> Dict[str, float]:
    tuned = dict(base_gains)
    centroid = float(source_metrics.get("centroid_hz", REFERENCE_SOURCE_METRICS["centroid_hz"]))
    crest = float(source_metrics.get("crest_db", REFERENCE_SOURCE_METRICS["crest_db"]))
    lufs = float(source_metrics.get("integrated_lufs", REFERENCE_SOURCE_METRICS["integrated_lufs"]))
    corr = float(source_metrics.get("stereo_correlation", REFERENCE_SOURCE_METRICS["stereo_correlation"]))

    if centroid > 1800.0:
        tuned["vocals"] -= 0.05
        tuned["drums"] -= 0.10
        tuned["other"] -= 0.05
        tuned["bass"] -= 0.02
    elif centroid < 850.0 and crest > 17.5 and lufs < -21.0:
        tuned["vocals"] += 0.03
        tuned["drums"] += 0.05

    if lufs > -18.0 or crest < 13.0:
        tuned["vocals"] -= 0.03
        tuned["drums"] -= 0.07
        tuned["bass"] -= 0.04

    if corr < 0.75:
        tuned["other"] -= 0.04

    return {name: round(clamp(value, -0.25, 0.40), 3) for name, value in tuned.items()}


def build_adaptive_variant(
    song: Dict[str, Any],
    variant: Dict[str, Any],
    source_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    tuned = deepcopy(variant)
    tuned["control_profile"] = dict(variant["control_profile"])
    notes: List[str] = []

    centroid = float(source_metrics.get("centroid_hz", REFERENCE_SOURCE_METRICS["centroid_hz"]))
    crest = float(source_metrics.get("crest_db", REFERENCE_SOURCE_METRICS["crest_db"]))
    corr = float(source_metrics.get("stereo_correlation", REFERENCE_SOURCE_METRICS["stereo_correlation"]))
    lufs = float(source_metrics.get("integrated_lufs", REFERENCE_SOURCE_METRICS["integrated_lufs"]))
    true_peak = float(source_metrics.get("true_peak_dbtp", REFERENCE_SOURCE_METRICS["true_peak_dbtp"]))
    ext = Path(song["source_path"]).suffix.lower()

    ref_centroid = REFERENCE_SOURCE_METRICS["centroid_hz"]
    brightness_adjust = clamp(-(centroid - ref_centroid) / 1400.0 * 0.18, -0.24, 0.12)
    tuned["control_profile"]["brightness_tilt"] = round(
        clamp(variant["control_profile"]["brightness_tilt"] + brightness_adjust, -0.10, 0.24),
        3,
    )
    if centroid > 1700.0:
        notes.append("Bright source: trimmed brightness tilt and increased harshness control.")
    elif centroid < 700.0:
        notes.append("Darker source: kept a small top-end lift so the batch does not get too cloudy.")

    harshness = variant["control_profile"]["harshness_control"]
    if centroid > 1700.0:
        harshness += clamp((centroid - 1700.0) / 1500.0 * 0.14, 0.05, 0.16)
    if true_peak > 0.5:
        harshness += clamp((true_peak - 0.5) / 1.5 * 0.10, 0.03, 0.10)
    if ext in LOSSY_EXTENSIONS:
        harshness += 0.08
    tuned["control_profile"]["harshness_control"] = round(clamp(harshness, 0.34, 0.62), 3)

    spatial = variant["control_profile"]["spatial_width"]
    if corr > 0.93:
        spatial += 0.05
        notes.append("Highly correlated stereo image: opened width slightly to avoid a boxed-in finish.")
    elif corr < 0.75:
        spatial -= 0.06
        notes.append("Loose stereo image: tightened width to protect mono translation.")
    if corr < 0.65:
        spatial -= 0.03
    if ext in LOSSY_EXTENSIONS:
        spatial -= 0.03
    tuned["control_profile"]["spatial_width"] = round(clamp(spatial, -0.02, 0.18), 3)

    movement = variant["control_profile"]["movement_amount"]
    if lufs > -18.0 or crest < 13.0:
        movement -= 0.04
    if centroid > 2200.0:
        movement -= 0.02
    if corr < 0.65:
        movement -= 0.03
    if ext in LOSSY_EXTENSIONS:
        movement -= 0.02
    tuned["control_profile"]["movement_amount"] = round(clamp(movement, 0.16, 0.28), 3)

    low_end_focus = variant["control_profile"]["low_end_focus"]
    if corr < 0.75:
        low_end_focus += 0.05
    if lufs > -18.0:
        low_end_focus += 0.02
    if centroid > 2300.0:
        low_end_focus -= 0.02
    tuned["control_profile"]["low_end_focus"] = round(clamp(low_end_focus, 0.64, 0.82), 3)

    warmth = variant["warmth"]
    if centroid > 1800.0:
        warmth += 0.04
    if centroid > 2500.0:
        warmth += 0.03
    if centroid < 700.0:
        warmth -= 0.03
    if lufs > -17.0:
        warmth -= 0.02
    tuned["warmth"] = round(clamp(warmth, 0.18, 0.38), 3)

    transient = variant["transient_boost_db"]
    if lufs > -18.0 or crest < 13.0:
        transient -= 0.45
        notes.append("Dense source: eased transient push and backed off the raw loudness target slightly.")
    if lufs > -16.0 or crest < 12.0:
        transient -= 0.25
    if true_peak > 0.8:
        transient -= 0.20
        notes.append("Hot input peak: added more protection before final limiting.")
    tuned["transient_boost_db"] = round(clamp(transient, 1.2, 2.5), 3)

    engine_target_lufs = FINAL_TARGET_LUFS
    if lufs > -18.0 or crest < 13.0:
        engine_target_lufs = -13.8
    if lufs > -16.0 or crest < 12.0:
        engine_target_lufs = -14.0
    if lufs < -23.0 and crest > 18.0 and true_peak < -0.3:
        engine_target_lufs = -13.35
        notes.append("Open source with headroom: allowed a slightly stronger raw target before final loudnorm.")
    tuned["engine_target_lufs"] = round(clamp(engine_target_lufs, -14.0, -13.3), 2)

    governor_limit = variant["governor_gr_limit_db"]
    if lufs > -18.0 or crest < 13.0:
        governor_limit += 0.20
    if lufs > -16.0 or crest < 12.0:
        governor_limit += 0.10
    if true_peak > 0.8:
        governor_limit += 0.10
    if lufs < -23.0 and crest > 18.0 and true_peak < -0.3:
        governor_limit -= 0.10
    tuned["governor_gr_limit_db"] = round(clamp(governor_limit, -1.4, -0.8), 2)

    governor_steps = variant["governor_search_steps"]
    if ext in LOSSY_EXTENSIONS or lufs > -18.0 or crest < 13.0:
        governor_steps = 5
    tuned["governor_search_steps"] = governor_steps

    enable_air_motion = variant["enable_air_motion"]
    if ext in LOSSY_EXTENSIONS or corr < 0.65 or centroid > 2800.0:
        enable_air_motion = False
        notes.append("Air motion disabled for a fragile top-end or already unstable stereo image.")
    tuned["enable_air_motion"] = enable_air_motion

    stem_mode = variant["stem_mode"]
    stem_gains = deepcopy(variant["stem_gains_db"])
    if ext in LOSSY_EXTENSIONS:
        stem_mode = "off"
        stem_gains = None
        notes.append("Lossy source: disabled stems to avoid lifting codec artifacts.")
    elif corr < 0.62 and crest < 12.5:
        stem_mode = "off"
        stem_gains = None
        notes.append("Very wide dense source: disabled stems to avoid exaggerating separation artifacts.")
    elif stem_gains is not None:
        stem_gains = adapt_stem_gains(stem_gains, source_metrics)
    tuned["stem_mode"] = stem_mode
    tuned["stem_gains_db"] = stem_gains

    tuned["reference_delta"] = {
        "integrated_lufs": round(lufs - REFERENCE_SOURCE_METRICS["integrated_lufs"], 3),
        "true_peak_dbtp": round(true_peak - REFERENCE_SOURCE_METRICS["true_peak_dbtp"], 3),
        "crest_db": round(crest - REFERENCE_SOURCE_METRICS["crest_db"], 3),
        "stereo_correlation": round(corr - REFERENCE_SOURCE_METRICS["stereo_correlation"], 3),
        "centroid_hz": round(centroid - REFERENCE_SOURCE_METRICS["centroid_hz"], 3),
    }
    tuned["adaptive_notes"] = notes or [
        "Source landed close to the reference, so only minor tuning offsets were applied."
    ]
    tuned["adaptive_summary"] = {
        "engine_target_lufs": tuned["engine_target_lufs"],
        "warmth": tuned["warmth"],
        "transient_boost_db": tuned["transient_boost_db"],
        "governor_search_steps": tuned["governor_search_steps"],
        "governor_gr_limit_db": tuned["governor_gr_limit_db"],
        "stem_mode": tuned["stem_mode"],
        "stem_gains_db": tuned["stem_gains_db"],
        "enable_air_motion": tuned["enable_air_motion"],
        "enable_hooklift": tuned["enable_hooklift"],
        "control_profile": tuned["control_profile"],
    }
    return tuned


async def register_and_analyze_sources(
    selected_songs: List[Dict[str, Any]], selected_variants: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    registered: Dict[str, Dict[str, Any]] = {}
    for song in selected_songs:
        source_path = Path(song["source_path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")

        key = str(source_path)
        registration = server.register_audio_from_path(str(source_path), ctx=DUMMY_CTX)
        source_metrics = metrics_dict(server.analyze_audio(registration.audio_id, ctx=DUMMY_CTX))

        planning: Dict[str, Any] = {}
        adaptive_profiles: Dict[str, Any] = {}
        governor_by_preset: Dict[str, Any] = {}
        for variant in selected_variants:
            tuned = build_adaptive_variant(song, variant, source_metrics)
            adaptive_profiles[variant["variant_key"]] = tuned

            plan = server.plan_mastering_strategy(
                server.StrategyPlanIn(
                    audio_id=registration.audio_id,
                    goal=tuned["goal"],
                    platform=PLATFORM,
                    control_profile=server.MasteringControlProfile(**tuned["control_profile"]),
                    governor_search_steps=tuned["governor_search_steps"],
                    governor_gr_limit_db=tuned["governor_gr_limit_db"],
                    stem_gains_db=tuned["stem_gains_db"],
                    stem_mode=tuned["stem_mode"],
                ),
                ctx=DUMMY_CTX,
            )
            planning[variant["variant_key"]] = {
                "chosen_preset": plan.chosen_preset,
                "reasoning": plan.reasoning,
                "warnings": plan.warnings,
                "settings": plan.settings.model_dump(),
                "adaptive_summary": tuned["adaptive_summary"],
                "adaptive_notes": tuned["adaptive_notes"],
                "reference_delta": tuned["reference_delta"],
            }

            preset_name = tuned["preset_name"]
            if preset_name not in governor_by_preset:
                governor_result = await server.analyze_and_optimize_governor(
                    server.AnalyzeAndOptimizeGovernorIn(
                        audio_id=registration.audio_id,
                        preset_name=preset_name,
                    ),
                    ctx=DUMMY_CTX,
                )
                governor_by_preset[preset_name] = governor_result.model_dump()

        registered[key] = {
            "display_name": song["display_name"],
            "source_path": str(source_path),
            "audio_id": registration.audio_id,
            "source_metrics": source_metrics,
            "planning": planning,
            "adaptive_profiles": adaptive_profiles,
            "governor_by_preset": governor_by_preset,
        }
        print(
            f"Analyzed {song['display_name']} -> {registration.audio_id} "
            f"(LUFS {source_metrics['integrated_lufs']:.2f}, "
            f"TP {source_metrics['true_peak_dbtp']:.2f}, "
            f"centroid {source_metrics['centroid_hz']:.0f} Hz)"
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
        f"- Base script: `{payload['batch_base_script']}`",
        f"- Sonic reference manifest: `{payload['sonic_reference_manifest']}`",
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
        "## Adaptive Tuning",
        "```json",
        json.dumps(payload["adaptive_profile"]["adaptive_summary"], indent=2),
        "```",
        "",
        "## Adaptive Notes",
        *[f"- {note}" for note in payload["adaptive_profile"]["adaptive_notes"]],
        "",
        "## Reference Delta",
        "```json",
        json.dumps(payload["adaptive_profile"]["reference_delta"], indent=2),
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
        "### Before",
        "```json",
        json.dumps(payload["metrics_before"], indent=2),
        "```",
        "",
        "### After",
        "```json",
        json.dumps(payload["metrics_after"], indent=2),
        "```",
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
    adaptive_profile = song_info["adaptive_profiles"][variant["variant_key"]]
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
    explicit_settings = deepcopy(planner_snapshot["settings"])
    explicit_settings.update(
        {
            "preset_name": adaptive_profile["preset_name"],
            "target_lufs": adaptive_profile["engine_target_lufs"],
            "warmth": adaptive_profile["warmth"],
            "transient_boost_db": adaptive_profile["transient_boost_db"],
            "enable_harshness_limiter": adaptive_profile["enable_harshness_limiter"],
            "enable_masking_eq": adaptive_profile["enable_masking_eq"],
            "enable_air_motion": adaptive_profile["enable_air_motion"],
            "enable_hooklift": adaptive_profile["enable_hooklift"],
            "bit_depth": "float32",
            "control_profile": adaptive_profile["control_profile"],
            "governor_search_steps": adaptive_profile["governor_search_steps"],
            "governor_gr_limit_db": adaptive_profile["governor_gr_limit_db"],
            "stem_gains_db": adaptive_profile["stem_gains_db"],
            "stem_mode": adaptive_profile["stem_mode"],
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
        "goal": adaptive_profile["goal"],
        "preset_name": adaptive_profile["preset_name"],
        "stem_mode": adaptive_profile["stem_mode"],
        "batch_base_script": str(BATCH_BASE_SCRIPT),
        "sonic_reference_manifest": str(SONIC_REFERENCE_MANIFEST),
        "source_metrics": song_info["source_metrics"],
        "adaptive_profile": adaptive_profile,
        "planner_snapshot": planner_snapshot,
        "governor_snapshot": song_info["governor_by_preset"][adaptive_profile["preset_name"]],
        "final_request": request.model_dump(),
        "metrics_before": metrics_dict(result.metrics_before),
        "metrics_after": metrics_dict(result.metrics_after),
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
        f"- Batch base script: `{manifest['batch_base_script']}`",
        f"- Sonic reference manifest: `{manifest['sonic_reference_manifest']}`",
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
                f"- Spectral centroid: `{metrics['centroid_hz']:.0f} Hz`",
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
        adaptive = render["adaptive_profile"]["adaptive_summary"]
        lines.extend(
            [
                f"### {render['display_name']} :: {render['variant_label']}",
                f"- Raw export: `{render['raw_wav_path']}`",
                f"- 24-bit delivery: `{render['delivery_24_path']}`",
                f"- Report: `{render['report_path']}`",
                f"- Engine target LUFS: `{adaptive['engine_target_lufs']}`",
                f"- Stem mode: `{adaptive['stem_mode']}`",
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

    registered = await register_and_analyze_sources(selected_songs, selected_variants)
    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "run_label": RUN_LABEL,
        "project_root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "platform": PLATFORM,
        "batch_base_script": str(BATCH_BASE_SCRIPT),
        "sonic_reference_manifest": str(SONIC_REFERENCE_MANIFEST),
        "reference_source_metrics": REFERENCE_SOURCE_METRICS,
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
