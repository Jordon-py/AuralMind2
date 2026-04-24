from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server

RUN_LABEL = "dont_push_me_premium_quad_20260415"
PLATFORM = "spotify"
SOURCE_PATH = ROOT / "data" / "Don't Push Me.wav"
OUTPUT_DIR = ROOT / "masters" / RUN_LABEL
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw_auralmind"
DELIVERY_24_DIR = OUTPUT_DIR / "delivery_24bit_44k1"
DELIVERY_32_DIR = OUTPUT_DIR / "delivery_32bitfloat_44k1"
REPORTS_DIR = OUTPUT_DIR / "reports"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
FINAL_TARGET_LUFS = -13.5
FINAL_TRUE_PEAK = -0.4
FINAL_SAMPLE_RATE = 44_100


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


def finalize_master(raw_path: Path, final_path: Path, codec: str) -> Dict[str, Any]:
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
        codec,
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


def source_metrics_dict(metrics: Any) -> Dict[str, Any]:
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    return dict(metrics)


def build_profiles() -> List[Dict[str, Any]]:
    return [
        {
            "variant_label": 'Industry Standard (Stem-On)',
            "output_stub": "dont-push-me__industry-standard__stem-on",
            "goal": (
                "Industry-standard premium trap master with maximum clarity, punchy transients, "
                "polished commercial finish, mono sub-bass discipline, hooklifted hooks, and movement 0.26."
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
            "variant_label": 'Deep & Wide (Stem-Off)',
            "output_stub": "dont-push-me__deep-and-wide__stem-off",
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
            "variant_label": 'Vocal Forward (Stem-On)',
            "output_stub": "dont-push-me__vocal-forward__stem-on",
            "goal": (
                "Vocal-forward premium trap master with crystal clear vocal presence, intimate mid-range detail, "
                "mono sub-bass discipline, hooklifted hooks, and movement 0.26."
            ),
            "preset_name": "radio_loud",
            "warmth": 0.24,
            "transient_boost_db": 1.9,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -1.0,
            "stem_mode": "on",
            "stem_gains_db": {"vocals": 0.7, "drums": 0.05, "bass": -0.2, "other": -0.05},
            "control_profile": {
                "spatial_width": 0.06,
                "brightness_tilt": 0.16,
                "harshness_control": 0.22,
                "movement_amount": 0.26,
                "low_end_focus": 0.58,
            },
        },
        {
            "variant_label": 'Experimental Punch (Stem-Off)',
            "output_stub": "dont-push-me__experimental-punch__stem-off",
            "goal": (
                "Aggressive premium trap master with enhanced harmonic saturation, high-energy punch, "
                "mono sub-bass discipline, hooklifted hooks, and movement 0.26."
            ),
            "preset_name": "club_clean",
            "warmth": 0.34,
            "transient_boost_db": 2.8,
            "governor_search_steps": 7,
            "governor_gr_limit_db": -1.5,
            "stem_mode": "off",
            "stem_gains_db": None,
            "control_profile": {
                "spatial_width": 0.18,
                "brightness_tilt": 0.12,
                "harshness_control": 0.18,
                "movement_amount": 0.26,
                "low_end_focus": 0.76,
            },
            "harmonic_excitation": {
                "drive_amount": 24.0,
                "harmonics": "both",
            },
        },
    ]


async def maybe_prepare_audio(profile: Dict[str, Any], base_audio_id: str) -> Dict[str, Any]:
    excitation = profile.get("harmonic_excitation")
    if not excitation:
        return {
            "audio_id": base_audio_id,
            "preprocess": None,
        }

    result = await server.apply_harmonic_excitation(
        server.HarmonicExcitationIn(
            audio_id=base_audio_id,
            drive_amount=float(excitation["drive_amount"]),
            harmonics=str(excitation["harmonics"]),
        ),
        ctx=DUMMY_CTX,
    )
    session_key, session_dir = server._get_session_info(DUMMY_CTX)
    processed_path = artifact_source_path(result.artifact_id)
    prepared_audio_id = server._new_id("aud")
    server._store_file_from_path(
        session_key,
        session_dir,
        artifact_id=prepared_audio_id,
        kind="audio",
        filename=f"{sanitize_name(profile['variant_label'])}__prep.wav",
        source_path=str(processed_path),
        media_type="audio/wav",
    )
    return {
        "audio_id": prepared_audio_id,
        "preprocess": {
            "artifact_id": result.artifact_id,
            "prepared_audio_id": prepared_audio_id,
            "message": result.message,
            "meter": result.meter,
            "artifact_path": str(processed_path),
        },
    }


def write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    report_lines = [
        f"# {payload['variant_label']}",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Source audio id: `{payload['source_audio_id']}`",
        f"- Working audio id: `{payload['working_audio_id']}`",
        f"- Preset: `{payload['preset_name']}`",
        f"- Stem mode: `{payload['stem_mode']}`",
        f"- Final target: `{FINAL_TARGET_LUFS} LUFS`, `{FINAL_TRUE_PEAK} dBTP`, `{FINAL_SAMPLE_RATE} Hz`",
        "",
        "## Goal",
        payload["goal"],
        "",
        "## Planner",
        f"- Chosen preset: `{payload['chosen_preset_from_planner']}`",
        f"- Reasoning: {payload['plan_reasoning']}",
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
        "## Delivery Outputs",
        f"- 24-bit: `{payload['delivery_24_path']}`",
        f"- 32-bit float: `{payload['delivery_32_path']}`",
        "",
        "## Verification",
        "```json",
        json.dumps(
            {
                "delivery_24": payload["delivery_24_verification"],
                "delivery_32": payload["delivery_32_verification"],
            },
            indent=2,
        ),
        "```",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


async def render_profile(profile: Dict[str, Any], source_audio_id: str) -> Dict[str, Any]:
    prepared = await maybe_prepare_audio(profile, source_audio_id)
    working_audio_id = prepared["audio_id"]
    control_profile = server.MasteringControlProfile(**profile["control_profile"])
    plan = server.plan_mastering_strategy(
        server.StrategyPlanIn(
            audio_id=working_audio_id,
            goal=profile["goal"],
            platform=PLATFORM,
            control_profile=control_profile,
            governor_search_steps=profile["governor_search_steps"],
            governor_gr_limit_db=profile["governor_gr_limit_db"],
            stem_gains_db=profile["stem_gains_db"],
            stem_mode=profile["stem_mode"],
        ),
        ctx=DUMMY_CTX,
    )

    explicit_settings = plan.settings.model_dump()
    explicit_settings.update(
        {
            "preset_name": profile["preset_name"],
            "target_lufs": FINAL_TARGET_LUFS,
            "warmth": profile["warmth"],
            "transient_boost_db": profile["transient_boost_db"],
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "bit_depth": "float32",
            "control_profile": profile["control_profile"],
            "governor_search_steps": profile["governor_search_steps"],
            "governor_gr_limit_db": profile["governor_gr_limit_db"],
            "stem_gains_db": profile["stem_gains_db"],
            "stem_mode": profile["stem_mode"],
        }
    )
    normalized = server.propose_master_settings(server.MasterSettings(**explicit_settings)).settings
    request = server.MasterRequest(audio_id=working_audio_id, **normalized.model_dump())
    result = server.master_audio(request, ctx=DUMMY_CTX)

    raw_destination = RAW_OUTPUT_DIR / f"{profile['output_stub']}__raw.wav"
    delivery_24_path = DELIVERY_24_DIR / f"{profile['output_stub']}__24bit_44k1.wav"
    delivery_32_path = DELIVERY_32_DIR / f"{profile['output_stub']}__32bitfloat_44k1.wav"
    report_path = REPORTS_DIR / f"{sanitize_name(profile['variant_label'])}.md"

    copy_artifact_to_path(result.master_wav_id, raw_destination)
    delivery_24 = finalize_master(raw_destination, delivery_24_path, "pcm_s24le")
    delivery_32 = finalize_master(raw_destination, delivery_32_path, "pcm_f32le")

    payload = {
        "generated_at": utc_now(),
        "variant_label": profile["variant_label"],
        "goal": profile["goal"],
        "source_audio_id": source_audio_id,
        "working_audio_id": working_audio_id,
        "preset_name": profile["preset_name"],
        "stem_mode": profile["stem_mode"],
        "plan_reasoning": plan.reasoning,
        "plan_warnings": plan.warnings,
        "chosen_preset_from_planner": plan.chosen_preset,
        "final_request": request.model_dump(),
        "metrics_before": result.metrics_before.model_dump(),
        "metrics_after": result.metrics_after.model_dump(),
        "master_wav_id": result.master_wav_id,
        "tuning_trace_id": result.tuning_trace_id,
        "artifacts": result.artifacts,
        "raw_wav_path": str(raw_destination),
        "delivery_24_path": str(delivery_24_path),
        "delivery_32_path": str(delivery_32_path),
        "delivery_24_verification": delivery_24,
        "delivery_32_verification": delivery_32,
        "preprocess": prepared["preprocess"],
        "report_path": str(report_path),
    }
    write_report(report_path, payload)
    return payload


def write_summary(manifest: Dict[str, Any]) -> None:
    lines = [
        f"# {RUN_LABEL}",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Source path: `{manifest['source_path']}`",
        f"- Source audio id: `{manifest['source_audio_id']}`",
        f"- Baseline metrics: `{json.dumps(manifest['source_metrics'], indent=2)}`",
        f"- Delivery folders: `{DELIVERY_24_DIR}` and `{DELIVERY_32_DIR}`",
        "",
        "## Variants",
    ]
    for render in manifest["renders"]:
        verify_24 = render["delivery_24_verification"]["verification"]
        verify_32 = render["delivery_32_verification"]["verification"]
        lines.extend(
            [
                f"### {render['variant_label']}",
                f"- Raw: `{render['raw_wav_path']}`",
                f"- 24-bit: `{render['delivery_24_path']}`",
                f"- 32-bit: `{render['delivery_32_path']}`",
                f"- Post-master metrics: `{json.dumps(render['metrics_after'], indent=2)}`",
                f"- 24-bit verify: `I={verify_24.get('input_i')} LUFS, TP={verify_24.get('input_tp')} dBTP`",
                f"- 32-bit verify: `I={verify_32.get('input_i')} LUFS, TP={verify_32.get('input_tp')} dBTP`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERY_24_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERY_32_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    registration = server.register_audio_from_path(str(SOURCE_PATH), ctx=DUMMY_CTX)
    source_metrics = server.analyze_audio(registration.audio_id, ctx=DUMMY_CTX)

    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "run_label": RUN_LABEL,
        "project_root": str(ROOT),
        "source_path": str(SOURCE_PATH),
        "source_audio_id": registration.audio_id,
        "source_metrics": source_metrics_dict(source_metrics),
        "final_target_lufs": FINAL_TARGET_LUFS,
        "final_true_peak_dbtp": FINAL_TRUE_PEAK,
        "final_sample_rate_hz": FINAL_SAMPLE_RATE,
        "renders": [],
    }

    for profile in build_profiles():
        render_summary = await render_profile(profile, registration.audio_id)
        manifest["renders"].append(render_summary)
        print(
            f"Rendered {profile['variant_label']} -> "
            f"{render_summary['delivery_24_path']} | {render_summary['delivery_32_path']}"
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest)
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
