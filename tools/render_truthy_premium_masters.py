from __future__ import annotations

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

MOVEMENT_AMOUNT = 0.23
PLATFORM = "spotify"
SOURCE_PATH = ROOT / "data" / "Truthy.wav"
OUTPUT_DIR = Path.home() / "Desktop" / "AuralMind2 Premium Masters" / "Truthy"
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw_auralmind"
MANIFEST_PATH = OUTPUT_DIR / "render_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def copy_artifact_to_path(master_wav_id: str, destination: Path) -> None:
    session_key, session_dir_raw = server._get_session_info(None)
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


def finalize_master(raw_path: Path, final_path: Path, target_lufs: float) -> Dict[str, Any]:
    analysis_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP=-1.0:LRA=7:print_format=json",
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
            f"loudnorm=I={target_lufs}:TP=-1.0:LRA=7:"
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
            "variant_label": "808 Authority",
            "output_stub": "truthy__stems__808-authority",
            "goal": (
                "Premium stem-separated trap master with dominant 808 control, fast punch, clear vocal center, "
                "and modern commercial aggression without harsh hats."
            ),
            "preset_name": "competitive_trap",
            "target_lufs": -10.8,
            "warmth": 0.36,
            "transient_boost_db": 2.8,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -1.8,
            "stem_mode": "on",
            "stem_gains_db": {"vocals": 0.42, "drums": 0.28, "bass": -0.12, "other": 0.05},
            "control_profile": {
                "spatial_width": 0.07,
                "brightness_tilt": 0.1,
                "harshness_control": 0.42,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.84,
            },
        },
        {
            "variant_label": "Night Pressure",
            "output_stub": "truthy__stems__night-pressure",
            "goal": (
                "Premium stem-separated trap master with darker luxury tone, tighter kick-to-sub lock, "
                "clean vocal detail, and expensive controlled impact."
            ),
            "preset_name": "club_clean",
            "target_lufs": -11.0,
            "warmth": 0.48,
            "transient_boost_db": 2.4,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -1.7,
            "stem_mode": "on",
            "stem_gains_db": {"vocals": 0.24, "drums": 0.18, "bass": -0.2, "other": 0.02},
            "control_profile": {
                "spatial_width": 0.05,
                "brightness_tilt": -0.02,
                "harshness_control": 0.36,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.82,
            },
        },
        {
            "variant_label": "Velvet Wide",
            "output_stub": "truthy__nostems__velvet-wide",
            "goal": (
                "Premium no-stems trap master with glued full-mix energy, wide hook presentation, "
                "smooth top-end polish, and deep but tidy low-end translation."
            ),
            "preset_name": "radio_loud",
            "target_lufs": -11.2,
            "warmth": 0.4,
            "transient_boost_db": 2.0,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -1.5,
            "stem_mode": "off",
            "stem_gains_db": None,
            "control_profile": {
                "spatial_width": 0.11,
                "brightness_tilt": 0.07,
                "harshness_control": 0.52,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.72,
            },
        },
    ]


def register_source() -> Dict[str, Any]:
    reg = server.register_audio_from_path(str(SOURCE_PATH.resolve()))
    analysis = server.analyze_audio(reg.audio_id).model_dump()
    return {
        "audio_id": reg.audio_id,
        "source_label": "Truthy",
        "source_path": str(SOURCE_PATH.resolve()),
        "analysis": analysis,
    }


def render_profile(profile: Dict[str, Any], source_info: Dict[str, Any]) -> Dict[str, Any]:
    control_profile = server.MasteringControlProfile(**profile["control_profile"])
    plan = server.plan_mastering_strategy(
        server.StrategyPlanIn(
            audio_id=source_info["audio_id"],
            goal=profile["goal"],
            platform=PLATFORM,
            control_profile=control_profile,
            governor_search_steps=profile["governor_search_steps"],
            governor_gr_limit_db=profile["governor_gr_limit_db"],
            stem_gains_db=profile["stem_gains_db"],
            stem_mode=profile["stem_mode"],
        )
    )

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
            "governor_search_steps": profile["governor_search_steps"],
            "governor_gr_limit_db": profile["governor_gr_limit_db"],
            "stem_gains_db": profile["stem_gains_db"],
            "stem_mode": profile["stem_mode"],
        }
    )
    normalized = server.propose_master_settings(server.MasterSettings(**explicit_settings)).settings
    request = server.MasterRequest(audio_id=source_info["audio_id"], **normalized.model_dump())
    result = server.master_audio(request)

    raw_destination = RAW_OUTPUT_DIR / f"{profile['output_stub']}__raw.wav"
    final_destination = OUTPUT_DIR / f"{profile['output_stub']}.wav"
    copy_artifact_to_path(result.master_wav_id, raw_destination)
    loudnorm_summary = finalize_master(raw_destination, final_destination, profile["target_lufs"])

    return {
        "source_label": "Truthy",
        "source_path": source_info["source_path"],
        "variant_label": profile["variant_label"],
        "goal": profile["goal"],
        "plan_reasoning": plan.reasoning,
        "plan_warnings": plan.warnings,
        "chosen_preset_from_planner": plan.chosen_preset,
        "final_request": request.model_dump(),
        "metrics_before": result.metrics_before.model_dump(),
        "metrics_after": result.metrics_after.model_dump(),
        "master_wav_id": result.master_wav_id,
        "tuning_trace_id": result.tuning_trace_id,
        "artifacts": result.artifacts,
        "raw_exported_wav_path": str(raw_destination),
        "exported_wav_path": str(final_destination),
        "loudnorm_summary": loudnorm_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = build_profiles()
    source_info = register_source()

    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "project_root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "raw_output_dir": str(RAW_OUTPUT_DIR),
        "movement_amount_locked": MOVEMENT_AMOUNT,
        "source": source_info,
        "renders": [],
    }

    for profile in profiles:
        render_summary = render_profile(profile, source_info)
        manifest["renders"].append(render_summary)
        print(
            f"Rendered Truthy :: {profile['variant_label']} -> "
            f"{render_summary['exported_wav_path']}"
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
