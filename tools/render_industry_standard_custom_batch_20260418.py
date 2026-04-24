from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server

RUN_LABEL = "industry_standard_custom_batch_20260418"
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

PROFILE_BASE_SCRIPT = ROOT / "tools" / "render_premium_trap_master_batch_20260417.py"
PROFILE_REFERENCE_SCRIPT = ROOT / "tools" / "render_dont_push_me_premium_quad_20260415.py"
PROFILE_REFERENCE_MANIFEST = ROOT / "masters" / "dont_push_me_premium_quad_20260415" / "manifest.json"

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

REFERENCE_PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "competitive_trap": {
        "preset_name": "competitive_trap",
        "variant_label": "Industry Standard Custom",
        "goal": (
            "Industry-standard premium trap master with strong low-end control, punchy transients, "
            "clean vocal presence, polished commercial finish, stable center image, and movement 0.26."
        ),
        "warmth": 0.28,
        "transient_boost_db": 2.4,
        "governor_search_steps": 6,
        "governor_gr_limit_db": -1.2,
        "stem_gains_db": {"vocals": 0.2, "drums": 0.35, "bass": -0.1, "other": 0.05},
        "control_profile": {
            "spatial_width": 0.10,
            "brightness_tilt": 0.18,
            "harshness_control": 0.34,
            "movement_amount": 0.26,
            "low_end_focus": 0.72,
        },
    },
    "radio_loud": {
        "preset_name": "radio_loud",
        "variant_label": "Industry Standard Custom",
        "goal": (
            "Industry-standard premium master with vocal-forward presence, glued modern loudness, "
            "tight low end, clean hook lift, and a stable commercial stereo picture."
        ),
        "warmth": 0.24,
        "transient_boost_db": 1.9,
        "governor_search_steps": 6,
        "governor_gr_limit_db": -1.0,
        "stem_gains_db": {"vocals": 0.7, "drums": 0.05, "bass": -0.2, "other": -0.05},
        "control_profile": {
            "spatial_width": 0.06,
            "brightness_tilt": 0.16,
            "harshness_control": 0.22,
            "movement_amount": 0.26,
            "low_end_focus": 0.58,
        },
    },
    "club_clean": {
        "preset_name": "club_clean",
        "variant_label": "Industry Standard Custom",
        "goal": (
            "Industry-standard premium master with hot but controlled impact, disciplined highs, "
            "clean translation, focused low mids, and confident transient definition."
        ),
        "warmth": 0.34,
        "transient_boost_db": 2.8,
        "governor_search_steps": 7,
        "governor_gr_limit_db": -1.5,
        "stem_gains_db": {"vocals": 0.24, "drums": 0.12, "bass": -0.18, "other": 0.0},
        "control_profile": {
            "spatial_width": 0.14,
            "brightness_tilt": 0.12,
            "harshness_control": 0.18,
            "movement_amount": 0.26,
            "low_end_focus": 0.76,
        },
    },
}

TITLE_PRESET_HINTS = {
    "hot shit": "competitive_trap",
    "truthy": "competitive_trap",
    "vegas top teir": "competitive_trap",
    "vegas - top teir": "competitive_trap",
    "facetime": "radio_loud",
    "walking in rain": "radio_loud",
    "stilll kots": "club_clean",
}


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


def canonical_title(value: str) -> str:
    lowered = value.lower().replace("_", " ").strip()
    lowered = lowered.replace("-", " ")
    while "  " in lowered:
        lowered = lowered.replace("  ", " ")
    if lowered.endswith(" mp3") or lowered.endswith(" wav"):
        lowered = lowered.rsplit(" ", 1)[0]
    if lowered.endswith(")"):
        base, _, suffix = lowered.rpartition("(")
        if suffix[:-1].strip().isdigit():
            lowered = base.strip()
    return lowered.strip()


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
            "Render a customized Industry Standard premium batch using the April 17 batch runner "
            "shape plus metric-aware tuning derived from the April 15 reference master."
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


def choose_preset(song_name: str, source_metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    canonical = canonical_title(song_name)
    for needle, preset_name in TITLE_PRESET_HINTS.items():
        if needle in canonical:
            notes.append(f"title hint -> {preset_name}")
            return preset_name, notes

    centroid = float(source_metrics.get("centroid_hz") or 0.0)
    lufs = float(source_metrics.get("integrated_lufs") or -24.0)
    corr = float(source_metrics.get("stereo_correlation") or 0.9)

    if lufs > -16.0 and corr < 0.75:
        notes.append("already hot + unstable stereo -> club_clean")
        return "club_clean", notes
    if centroid > 2400.0:
        notes.append("very bright source -> radio_loud")
        return "radio_loud", notes
    if centroid < 800.0:
        notes.append("dark/low-mid heavy source -> radio_loud")
        return "radio_loud", notes

    notes.append("default trap lane -> competitive_trap")
    return "competitive_trap", notes


def choose_stem_mode(source_path: Path, source_metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    extension = source_path.suffix.lower()
    lufs = float(source_metrics.get("integrated_lufs") or -24.0)
    corr = float(source_metrics.get("stereo_correlation") or 0.9)
    tp = float(source_metrics.get("true_peak_dbtp") or -1.0)

    if extension == ".mp3":
        notes.append("lossy source -> stem_mode off")
        return "off", notes
    if corr < 0.72:
        notes.append("phasey/wide source -> stem_mode off")
        return "off", notes
    if lufs > -16.0 and tp > 0.5:
        notes.append("already hot source -> stem_mode off")
        return "off", notes

    notes.append("lossless source with room for separation -> stem_mode on")
    return "on", notes


def build_tuned_profile(song: Dict[str, Any], source_metrics: Dict[str, Any]) -> Dict[str, Any]:
    preset_name, preset_notes = choose_preset(song["display_name"], source_metrics)
    stem_mode, stem_notes = choose_stem_mode(Path(song["source_path"]), source_metrics)
    tuned = copy.deepcopy(REFERENCE_PRESET_PROFILES[preset_name])

    centroid = float(source_metrics.get("centroid_hz") or 0.0)
    crest = float(source_metrics.get("crest_db") or 16.0)
    lufs = float(source_metrics.get("integrated_lufs") or -24.0)
    corr = float(source_metrics.get("stereo_correlation") or 0.9)
    tp = float(source_metrics.get("true_peak_dbtp") or -1.0)

    cp = tuned["control_profile"]
    tuning_notes: List[str] = []
    tuning_notes.extend(preset_notes)
    tuning_notes.extend(stem_notes)

    if centroid > 1800.0:
        cp["brightness_tilt"] = round(clamp(cp["brightness_tilt"] - 0.08, -0.04, 0.18), 2)
        cp["harshness_control"] = round(clamp(cp["harshness_control"] + 0.08, 0.18, 0.52), 2)
        cp["spatial_width"] = round(clamp(min(cp["spatial_width"], 0.08), 0.02, 0.18), 2)
        tuned["warmth"] = round(clamp(tuned["warmth"] - 0.04, 0.18, 0.48), 2)
        tuning_notes.append("bright source -> darker tilt and stronger harshness control")
    elif centroid < 800.0:
        cp["brightness_tilt"] = round(clamp(cp["brightness_tilt"] + 0.03, -0.04, 0.18), 2)
        cp["low_end_focus"] = round(clamp(cp["low_end_focus"] + 0.06, 0.52, 0.9), 2)
        tuned["warmth"] = round(clamp(tuned["warmth"] + 0.06, 0.18, 0.48), 2)
        tuning_notes.append("dark source -> added light and low-end focus")

    if crest > 18.0:
        tuned["transient_boost_db"] = round(clamp(tuned["transient_boost_db"] + 0.2, 1.6, 3.0), 2)
        tuned["governor_gr_limit_db"] = round(clamp(tuned["governor_gr_limit_db"] - 0.1, -1.6, -0.8), 2)
        tuning_notes.append("high crest factor -> slightly more transient push")
    elif crest < 12.0:
        tuned["transient_boost_db"] = round(clamp(tuned["transient_boost_db"] - 0.4, 1.4, 3.0), 2)
        tuned["governor_gr_limit_db"] = round(clamp(tuned["governor_gr_limit_db"] + 0.2, -1.6, -0.8), 2)
        tuning_notes.append("already dense source -> softer transient push and gentler governor")

    if corr < 0.80:
        cp["spatial_width"] = round(clamp(min(cp["spatial_width"], 0.03), 0.0, 0.18), 2)
        cp["low_end_focus"] = round(clamp(cp["low_end_focus"] + 0.05, 0.52, 0.9), 2)
        cp["harshness_control"] = round(clamp(cp["harshness_control"] + 0.04, 0.18, 0.55), 2)
        tuning_notes.append("low stereo correlation -> narrowed width and tightened low end")
    elif corr > 0.92 and stem_mode == "on":
        cp["spatial_width"] = round(clamp(cp["spatial_width"] + 0.02, 0.02, 0.14), 2)
        tuning_notes.append("stable stereo field -> allowed a touch more width")

    if lufs > -16.0 or tp > 0.5:
        tuned["governor_gr_limit_db"] = round(clamp(tuned["governor_gr_limit_db"] + 0.2, -1.6, -0.8), 2)
        tuned["transient_boost_db"] = round(clamp(tuned["transient_boost_db"] - 0.2, 1.4, 3.0), 2)
        tuning_notes.append("already loud/hot source -> eased aggression before final loudnorm")

    tuned["stem_mode"] = stem_mode
    tuned["stem_gains_db"] = tuned["stem_gains_db"] if stem_mode == "on" else None
    tuned["target_lufs"] = FINAL_TARGET_LUFS
    tuned["tuning_notes"] = tuning_notes
    return tuned


def register_and_analyze_sources(selected_songs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    registered: Dict[str, Dict[str, Any]] = {}
    for song in selected_songs:
        source_path = Path(song["source_path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")

        registration = server.register_audio_from_path(str(source_path), ctx=DUMMY_CTX)
        source_metrics = metrics_dict(server.analyze_audio(registration.audio_id, ctx=DUMMY_CTX))
        tuned_profile = build_tuned_profile(song, source_metrics)

        registered[str(source_path)] = {
            "display_name": song["display_name"],
            "source_path": str(source_path),
            "audio_id": registration.audio_id,
            "source_metrics": source_metrics,
            "tuned_profile": tuned_profile,
        }
        print(
            f"Analyzed {song['display_name']} -> {registration.audio_id} "
            f"(LUFS {source_metrics['integrated_lufs']:.2f}, TP {source_metrics['true_peak_dbtp']:.2f}, "
            f"preset {tuned_profile['preset_name']}, stem_mode {tuned_profile['stem_mode']})"
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
        f"- Preset: `{payload['preset_name']}`",
        f"- Stem mode: `{payload['stem_mode']}`",
        f"- Final target: `{FINAL_TARGET_LUFS} LUFS`, `{FINAL_TRUE_PEAK} dBTP`, `{FINAL_SAMPLE_RATE} Hz`",
        "",
        "## Tuning Notes",
        *[f"- {note}" for note in payload["tuning_notes"]],
        "",
        "## Source Metrics",
        "```json",
        json.dumps(payload["source_metrics"], indent=2),
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


def render_song(song_info: Dict[str, Any], force: bool) -> Dict[str, Any]:
    tuned = song_info["tuned_profile"]
    source_slug = sanitize_name(song_info["display_name"])
    output_stub = f"{source_slug}__industry-standard-custom__{tuned['preset_name']}__stem-{tuned['stem_mode']}"
    raw_destination = RAW_OUTPUT_DIR / f"{output_stub}__raw.wav"
    delivery_24_path = DELIVERY_24_DIR / f"{output_stub}__24bit_44k1.wav"
    report_path = REPORTS_DIR / f"{output_stub}.md"

    if delivery_24_path.exists() and not force:
        print(f"Skipping existing delivery: {delivery_24_path}")
        return {
            "display_name": song_info["display_name"],
            "variant_label": tuned["variant_label"],
            "skipped_existing": True,
            "delivery_24_path": str(delivery_24_path),
            "report_path": str(report_path),
        }

    settings_payload = {
        "preset_name": tuned["preset_name"],
        "target_lufs": FINAL_TARGET_LUFS,
        "warmth": tuned["warmth"],
        "transient_boost_db": tuned["transient_boost_db"],
        "enable_harshness_limiter": True,
        "enable_masking_eq": True,
        "enable_air_motion": True,
        "enable_hooklift": True,
        "bit_depth": "float32",
        "control_profile": tuned["control_profile"],
        "governor_search_steps": tuned["governor_search_steps"],
        "governor_gr_limit_db": tuned["governor_gr_limit_db"],
        "stem_gains_db": tuned["stem_gains_db"],
        "stem_mode": tuned["stem_mode"],
    }
    normalized = server.propose_master_settings(server.MasterSettings(**settings_payload)).settings
    request = server.MasterRequest(audio_id=song_info["audio_id"], **normalized.model_dump())
    result = server.master_audio(request, ctx=DUMMY_CTX)

    copy_artifact_to_path(result.master_wav_id, raw_destination)
    delivery_24 = finalize_master(raw_destination, delivery_24_path)

    payload = {
        "generated_at": utc_now(),
        "display_name": song_info["display_name"],
        "source_path": song_info["source_path"],
        "source_audio_id": song_info["audio_id"],
        "variant_label": tuned["variant_label"],
        "preset_name": tuned["preset_name"],
        "stem_mode": tuned["stem_mode"],
        "tuning_notes": tuned["tuning_notes"],
        "source_metrics": song_info["source_metrics"],
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
    print(f"Rendered {song_info['display_name']} -> {delivery_24_path}")
    return payload


def write_summary(manifest: Dict[str, Any]) -> None:
    lines = [
        f"# {RUN_LABEL}",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Project root: `{manifest['project_root']}`",
        f"- Output root: `{manifest['output_dir']}`",
        f"- Base runner: `{manifest['profile_base_script']}`",
        f"- Reference runner: `{manifest['profile_reference_script']}`",
        f"- Reference manifest: `{manifest['profile_reference_manifest']}`",
        f"- Final target: `{FINAL_TARGET_LUFS} LUFS`, `{FINAL_TRUE_PEAK} dBTP`, `{FINAL_SAMPLE_RATE} Hz`",
        "",
        "## Source Summary",
    ]
    for source in manifest["sources"]:
        metrics = source["source_metrics"]
        tuned = source["tuned_profile"]
        lines.extend(
            [
                f"### {source['display_name']}",
                f"- Audio id: `{source['audio_id']}`",
                f"- Source path: `{source['source_path']}`",
                f"- Integrated LUFS: `{metrics['integrated_lufs']:.2f}`",
                f"- True peak: `{metrics['true_peak_dbtp']:.2f} dBTP`",
                f"- Crest factor: `{metrics['crest_db']:.2f} dB`",
                f"- Stereo correlation: `{metrics['stereo_correlation']:.3f}`",
                f"- Chosen preset: `{tuned['preset_name']}`",
                f"- Chosen stem mode: `{tuned['stem_mode']}`",
                *[f"- Note: {note}" for note in tuned["tuning_notes"]],
                "",
            ]
        )
    lines.append("## Renders")
    for render in manifest["renders"]:
        if render.get("skipped_existing"):
            lines.extend(
                [
                    f"### {render['display_name']}",
                    f"- Skipped existing delivery: `{render['delivery_24_path']}`",
                    "",
                ]
            )
            continue
        verify = render["delivery_24_verification"]["verification"]
        lines.extend(
            [
                f"### {render['display_name']}",
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
    if not selected_songs:
        raise ValueError("No songs selected. Check the --song filters.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERY_24_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    registered = register_and_analyze_sources(selected_songs)
    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "run_label": RUN_LABEL,
        "project_root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "platform": PLATFORM,
        "profile_base_script": str(PROFILE_BASE_SCRIPT),
        "profile_reference_script": str(PROFILE_REFERENCE_SCRIPT),
        "profile_reference_manifest": str(PROFILE_REFERENCE_MANIFEST),
        "final_target_lufs": FINAL_TARGET_LUFS,
        "final_true_peak_dbtp": FINAL_TRUE_PEAK,
        "final_sample_rate_hz": FINAL_SAMPLE_RATE,
        "sources": list(registered.values()),
        "renders": [],
    }

    for song in selected_songs:
        song_key = str(Path(song["source_path"]).resolve())
        render_summary = render_song(registered[song_key], force=args.force)
        manifest["renders"].append(render_summary)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest)
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
