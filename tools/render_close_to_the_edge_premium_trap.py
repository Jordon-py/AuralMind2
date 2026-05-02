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

RUN_LABEL = "close_to_the_edge_premium_trap_20260414"
PLATFORM = "spotify"
SOURCE_PATH = ROOT / "data" / "Close to the edge.wav"
OUTPUT_DIR = ROOT / "masters" / RUN_LABEL
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw_auralmind"
MASTERS_DIR = OUTPUT_DIR / "masters"
REPORTS_DIR = OUTPUT_DIR / "reports"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
FINAL_TARGET_LUFS = -14.0
FINAL_TRUE_PEAK = -0.75


class _DummyContext:
    session_id = f"{RUN_LABEL}-{int(datetime.now(timezone.utc).timestamp())}"

    async def report_progress(self, *_args: Any, **_kwargs: Any) -> None:
        return None


DUMMY_CTX = _DummyContext()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum():
            safe.append(char.lower())
        elif char in {" ", "-", "_", "(", ")", "."}:
            safe.append("-")
    joined = "".join(safe).strip("-")
    while "--" in joined:
        joined = joined.replace("--", "-")
    return joined or "master"


def raw_output_path(profile: Dict[str, Any]) -> Path:
    return RAW_OUTPUT_DIR / f"{profile['output_stub']}__raw.wav"


def final_output_path(profile: Dict[str, Any]) -> Path:
    return MASTERS_DIR / f"{profile['output_stub']}.wav"


def report_output_path(profile: Dict[str, Any]) -> Path:
    return REPORTS_DIR / f"{sanitize_name(profile['variant_label'])}.md"


def copy_artifact_to_path(master_wav_id: str, destination: Path) -> None:
    session_key, session_dir_raw = server._get_session_info(DUMMY_CTX)
    entry = server._load_artifact(session_key, session_dir_raw, master_wav_id)
    if entry is None:
        raise RuntimeError(f"Could not resolve artifact for {master_wav_id}.")
    source = Path(session_dir_raw) / entry.data_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def run_ffmpeg(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def parse_loudnorm_json(stderr_text: str) -> Dict[str, Any]:
    start = stderr_text.rfind("{")
    end = stderr_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Could not locate loudnorm JSON in ffmpeg output.")
    return json.loads(stderr_text[start : end + 1])


def finalize_master(raw_path: Path, final_path: Path, target_lufs: float, true_peak: float) -> Dict[str, Any]:
    analysis_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7:print_format=json",
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
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7:"
            f"measured_I={measured['input_i']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:"
            "linear=true:print_format=summary"
        ),
        "-c:a",
        "pcm_s24le",
        str(final_path),
    ]
    render_run = run_ffmpeg(render_cmd)
    return {
        "analysis": measured,
        "render_summary": render_run.stderr.strip().splitlines()[-12:],
    }


def build_profiles() -> List[Dict[str, Any]]:
    return [
        {
            "variant_label": "HiFi Trap Motion",
            "output_stub": "close-to-the-edge__premium-trap-motion-m030",
            "goal": (
                "Premium higher-fidelity trap master with no stem separation, mono-sub discipline, "
                "preserved crest, cleaner transient definition, center-stable low end, and controlled polish."
            ),
            "preset_name": "club_clean",
            "target_lufs": -14.8,
            "warmth": 0.22,
            "transient_boost_db": 1.0,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "stem_mode": "off",
            "control_profile": {
                "spatial_width": 0.02,
                "brightness_tilt": 0.04,
                "harshness_control": 0.28,
                "movement_amount": 0.30,
                "low_end_focus": 0.54,
            },
        },
        {
            "variant_label": "HiFi Trap Clean",
            "output_stub": "close-to-the-edge__premium-trap-clean",
            "goal": (
                "Premium higher-fidelity trap master with no stem separation, mono-sub discipline, "
                "clean transient edges, preserved openness, and polished streaming translation."
            ),
            "preset_name": "hi_fi_streaming",
            "target_lufs": -15.1,
            "warmth": 0.18,
            "transient_boost_db": 1.4,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "stem_mode": "off",
            "control_profile": {
                "spatial_width": 0.06,
                "brightness_tilt": 0.08,
                "harshness_control": 0.22,
                "movement_amount": 0.10,
                "low_end_focus": 0.46,
            },
        },
    ]


def register_source() -> Dict[str, Any]:
    reg = server.register_audio_from_path(str(SOURCE_PATH.resolve()), DUMMY_CTX)
    analysis = server.analyze_audio(reg.audio_id, DUMMY_CTX).model_dump()
    return {
        "audio_id": reg.audio_id,
        "source_path": str(SOURCE_PATH.resolve()),
        "analysis": analysis,
    }


def optimize_governor(audio_id: str, preset_name: str) -> Dict[str, Any]:
    result = asyncio.run(
        server.analyze_and_optimize_governor(
            server.AnalyzeAndOptimizeGovernorIn(audio_id=audio_id, preset_name=preset_name),
            DUMMY_CTX,
        )
    )
    return result.model_dump()


def render_profile(profile: Dict[str, Any], source_info: Dict[str, Any]) -> Dict[str, Any]:
    control_profile = server.MasteringControlProfile(**profile["control_profile"])
    plan = server.plan_mastering_strategy(
        server.StrategyPlanIn(
            audio_id=source_info["audio_id"],
            goal=profile["goal"],
            platform=PLATFORM,
            control_profile=control_profile,
            stem_mode=profile["stem_mode"],
        ),
        DUMMY_CTX,
    )
    governor = optimize_governor(source_info["audio_id"], profile["preset_name"])

    explicit_settings = plan.settings.model_dump()
    explicit_settings.update(
        {
            "preset_name": profile["preset_name"],
            "target_lufs": profile["target_lufs"],
            "warmth": profile["warmth"],
            "transient_boost_db": profile["transient_boost_db"],
            "enable_harshness_limiter": profile["enable_harshness_limiter"],
            "enable_masking_eq": profile["enable_masking_eq"],
            "enable_air_motion": profile["enable_air_motion"],
            "enable_hooklift": profile["enable_hooklift"],
            "bit_depth": "float32",
            "control_profile": profile["control_profile"],
            "governor_search_steps": governor["recommended_governor_steps"],
            "governor_gr_limit_db": governor["recommended_governor_gr_limit_db"],
            "stem_gains_db": None,
            "stem_mode": profile["stem_mode"],
        }
    )
    normalized = server.propose_master_settings(server.MasterSettings(**explicit_settings)).settings
    request = server.MasterRequest(audio_id=source_info["audio_id"], **normalized.model_dump())
    result = server.master_audio(request, DUMMY_CTX)

    raw_destination = raw_output_path(profile)
    final_destination = final_output_path(profile)
    copy_artifact_to_path(result.master_wav_id, raw_destination)
    loudnorm_summary = finalize_master(raw_destination, final_destination, FINAL_TARGET_LUFS, FINAL_TRUE_PEAK)

    report_path = report_output_path(profile)
    report_lines = [
        f"# {profile['variant_label']}",
        "",
        f"- Rendered: {utc_now()}",
        f"- Source: `{source_info['source_path']}`",
        f"- Raw AuralMind render: `{raw_destination}`",
        f"- Final master: `{final_destination}`",
        f"- Goal: {profile['goal']}",
        f"- Requested stem_mode: `{profile['stem_mode']}`",
        f"- Requested final loudness: `{FINAL_TARGET_LUFS}` LUFS / `{FINAL_TRUE_PEAK}` dBTP",
        "",
        "## Source Analysis",
        f"- Integrated LUFS: `{source_info['analysis']['integrated_lufs']:.2f}`",
        f"- True Peak dBTP: `{source_info['analysis']['true_peak_dbtp']:.2f}`",
        f"- Crest dB: `{source_info['analysis']['crest_db']:.2f}`",
        f"- Stereo correlation: `{source_info['analysis']['stereo_correlation']:.3f}`",
        "",
        "## Planner Notes",
        *[f"- {line}" for line in plan.reasoning],
        "",
        "## Planner Warnings",
        *([f"- {line}" for line in plan.warnings] if plan.warnings else ["- None"]),
        "",
        "## Governor Recommendation",
        f"- Crest factor dB: `{governor['crest_factor_db']}`",
        f"- Search steps: `{governor['recommended_governor_steps']}`",
        f"- GR limit dB: `{governor['recommended_governor_gr_limit_db']}`",
        "",
        "## AuralMind Metrics",
        f"- Integrated LUFS: `{result.metrics_after.integrated_lufs:.2f}`",
        f"- True Peak dBTP: `{result.metrics_after.true_peak_dbtp:.2f}`",
        f"- Crest dB: `{result.metrics_after.crest_db:.2f}`",
        f"- Stereo correlation: `{result.metrics_after.stereo_correlation:.3f}`",
        "",
        "## Loudnorm Analysis",
        "```json",
        json.dumps(loudnorm_summary["analysis"], indent=2),
        "```",
        "",
        "## Final Request",
        "```json",
        json.dumps(request.model_dump(), indent=2),
        "```",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "variant_label": profile["variant_label"],
        "goal": profile["goal"],
        "planner_preset": plan.chosen_preset,
        "final_request": request.model_dump(),
        "metrics_before": result.metrics_before.model_dump(),
        "metrics_after": result.metrics_after.model_dump(),
        "raw_exported_wav_path": str(raw_destination),
        "exported_wav_path": str(final_destination),
        "report_path": str(report_path),
        "loudnorm_summary": loudnorm_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASTERS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    profiles = build_profiles()
    source_info = register_source()
    renders = []
    for profile in profiles:
        final_path = final_output_path(profile)
        report_path = report_output_path(profile)
        if final_path.exists() and report_path.exists():
            print(f"Skipping existing render: {final_path.name}")
            continue
        renders.append(render_profile(profile, source_info))

    manifest = {
        "generated_at": utc_now(),
        "run_label": RUN_LABEL,
        "platform": PLATFORM,
        "source": source_info,
        "final_target_lufs": FINAL_TARGET_LUFS,
        "final_true_peak": FINAL_TRUE_PEAK,
        "renders": renders,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_lines = [
        f"# {RUN_LABEL}",
        "",
        f"- Generated: {manifest['generated_at']}",
        f"- Source: `{source_info['source_path']}`",
        f"- Final masters directory: `{MASTERS_DIR}`",
        f"- Final loudness target: `{FINAL_TARGET_LUFS}` LUFS",
        f"- Final true peak target: `{FINAL_TRUE_PEAK}` dBTP",
        "",
        "## Outputs",
    ]
    for render in renders:
        summary_lines.extend(
            [
                f"### {render['variant_label']}",
                f"- Final master: `{render['exported_wav_path']}`",
                f"- Report: `{render['report_path']}`",
                f"- Raw AuralMind render: `{render['raw_exported_wav_path']}`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
