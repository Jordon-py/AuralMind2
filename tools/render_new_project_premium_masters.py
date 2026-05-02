from __future__ import annotations

import json
import shutil
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
OUTPUT_DIR = Path.home() / "Desktop" / "AuralMind2 Premium Masters" / "New Project 8-9"
MANIFEST_PATH = OUTPUT_DIR / "render_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum():
            safe.append(char.lower())
        elif char in {" ", "-", "_", "(", ")"}:
            safe.append("-")
    joined = "".join(safe).strip("-")
    while "--" in joined:
        joined = joined.replace("--", "-")
    return joined or "master"


def copy_artifact_to_path(master_wav_id: str, destination: Path) -> None:
    session_key, session_dir_raw = server._get_session_info(None)
    entry = server._load_artifact(session_key, session_dir_raw, master_wav_id)
    if entry is None:
        raise RuntimeError(f"Could not resolve artifact for {master_wav_id}.")
    source = Path(session_dir_raw) / entry.data_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_profiles() -> List[Dict[str, Any]]:
    data_dir = ROOT / "data"
    return [
        {
            "source_label": "New Project (8)",
            "source_path": data_dir / "New Project (8).wav",
            "variant_label": "Trap Spine",
            "output_stub": "new-project-8__stems__trap-spine",
            "goal": (
                "Premium stem-separated trap master with expensive 808 discipline, forward vocal focus, "
                "crisp transient crack, premium low-end pressure, and controlled high-end shine."
            ),
            "preset_name": "competitive_trap",
            "target_lufs": -10.8,
            "warmth": 0.42,
            "transient_boost_db": 2.7,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -1.8,
            "stem_mode": "on",
            "stem_gains_db": {"vocals": 0.55, "drums": 0.25, "bass": -0.1, "other": 0.08},
            "control_profile": {
                "spatial_width": 0.08,
                "brightness_tilt": 0.12,
                "harshness_control": 0.44,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.82,
            },
        },
        {
            "source_label": "New Project (8)",
            "source_path": data_dir / "New Project (8).wav",
            "variant_label": "Radio Gloss",
            "output_stub": "new-project-8__nostems__radio-gloss",
            "goal": (
                "Premium full-mix trap master with stable center image, radio-grade hook clarity, "
                "sleek top-end gloss, and heavy but tidy low-end support."
            ),
            "preset_name": "radio_loud",
            "target_lufs": -11.0,
            "warmth": 0.34,
            "transient_boost_db": 2.1,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -2.2,
            "stem_mode": "off",
            "stem_gains_db": None,
            "control_profile": {
                "spatial_width": 0.09,
                "brightness_tilt": 0.08,
                "harshness_control": 0.5,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.7,
            },
        },
        {
            "source_label": "New Project (9)",
            "source_path": data_dir / "New Project (9).wav",
            "variant_label": "Dark Luxe",
            "output_stub": "new-project-9__stems__dark-luxe",
            "goal": (
                "Premium stem-separated trap master with darker luxury tone, dense chest-hit sub, "
                "clean vocal detail, and elite modern punch without brittle glare."
            ),
            "preset_name": "club_clean",
            "target_lufs": -10.7,
            "warmth": 0.46,
            "transient_boost_db": 2.8,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "governor_search_steps": 6,
            "governor_gr_limit_db": -1.7,
            "stem_mode": "on",
            "stem_gains_db": {"vocals": 0.38, "drums": 0.18, "bass": -0.18, "other": 0.05},
            "control_profile": {
                "spatial_width": 0.06,
                "brightness_tilt": -0.03,
                "harshness_control": 0.38,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.8,
            },
        },
        {
            "source_label": "New Project (9)",
            "source_path": data_dir / "New Project (9).wav",
            "variant_label": "Night Cinema",
            "output_stub": "new-project-9__nostems__night-cinema",
            "goal": (
                "Premium full-mix trap master with wide luxury space, smooth upper mids, "
                "deep sub authority, and an immersive late-night finish that still translates."
            ),
            "preset_name": "cinematic",
            "target_lufs": -11.5,
            "warmth": 0.4,
            "transient_boost_db": 1.8,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "governor_search_steps": 5,
            "governor_gr_limit_db": -1.3,
            "stem_mode": "off",
            "stem_gains_db": None,
            "control_profile": {
                "spatial_width": 0.12,
                "brightness_tilt": -0.08,
                "harshness_control": 0.54,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.74,
            },
        },
    ]


def register_sources(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    registered: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        source_path = Path(profile["source_path"]).resolve()
        source_key = str(source_path)
        if source_key in registered:
            continue
        reg = server.register_audio_from_path(str(source_path))
        analysis = server.analyze_audio(reg.audio_id).model_dump()
        registered[source_key] = {
            "audio_id": reg.audio_id,
            "source_label": profile["source_label"],
            "source_path": str(source_path),
            "analysis": analysis,
        }
    return registered


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

    destination = OUTPUT_DIR / f"{profile['output_stub']}.wav"
    copy_artifact_to_path(result.master_wav_id, destination)

    return {
        "source_label": profile["source_label"],
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
        "exported_wav_path": str(destination),
    }


def main() -> None:
    profiles = build_profiles()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registered = register_sources(profiles)
    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "project_root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "movement_amount_locked": MOVEMENT_AMOUNT,
        "sources": list(registered.values()),
        "renders": [],
    }

    for profile in profiles:
        source_key = str(Path(profile["source_path"]).resolve())
        render_summary = render_profile(profile, registered[source_key])
        manifest["renders"].append(render_summary)
        print(
            f"Rendered {profile['source_label']} :: {profile['variant_label']} -> "
            f"{render_summary['exported_wav_path']}"
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
