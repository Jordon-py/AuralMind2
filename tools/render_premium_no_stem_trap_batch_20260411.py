from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server

RUN_LABEL = "premium_no_stem_trap_batch_20260411"
MOVEMENT_AMOUNT = 0.24
PLATFORM = "spotify"
RUN_SESSION_ID = f"{RUN_LABEL}-{int(datetime.now(timezone.utc).timestamp())}"

OUTPUT_ROOT = ROOT / "masters" / RUN_LABEL
MASTERS_DIR = OUTPUT_ROOT / "masters"
REPORTS_DIR = OUTPUT_ROOT / "reports"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
SUMMARY_PATH = OUTPUT_ROOT / "batch_summary.md"


class _DummyContext:
    session_id: Optional[str] = RUN_SESSION_ID

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


def copy_artifact_to_path(artifact_id: str, destination: Path) -> None:
    session_key, session_dir_raw = server._get_session_info(DUMMY_CTX)
    entry = server._load_artifact(session_key, session_dir_raw, artifact_id)
    if entry is None:
        raise RuntimeError(f"Could not resolve artifact for {artifact_id}.")
    source = Path(session_dir_raw) / entry.data_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_profiles() -> List[Dict[str, Any]]:
    data_dir = ROOT / "data"
    return [
        {
            "source_path": data_dir / "Stilll KOTS.wav",
            "display_name": "Stilll KOTS",
            "output_name": "Stilll KOTS - AuralMind2 Premium No-Stem Trap.wav",
            "goal": (
                "Premium next-gen no-stem trap master with dark-luxe weight, tightened stereo discipline, "
                "expensive sub pressure, center-stable vocal energy, and polished air without brittle hats."
            ),
            "preset_name": "club_clean",
            "target_lufs": -10.8,
            "warmth": 0.46,
            "transient_boost_db": 2.4,
            "harmonic_drive": 8.0,
            "harmonics": "both",
            "control_profile": {
                "spatial_width": -0.10,
                "brightness_tilt": 0.12,
                "harshness_control": 0.18,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.60,
            },
        },
        {
            "source_path": data_dir / "Vegas - This Life I Lead.wav",
            "display_name": "Vegas - This Life I Lead",
            "output_name": "Vegas - This Life I Lead - AuralMind2 Premium No-Stem Trap.wav",
            "goal": (
                "Premium next-gen no-stem trap master with hook-led commercial gloss, articulate vocal focus, "
                "sleek top-end air, and tight low-end support that feels premium rather than bloated."
            ),
            "preset_name": "radio_loud",
            "target_lufs": -11.0,
            "warmth": 0.34,
            "transient_boost_db": 2.1,
            "harmonic_drive": 4.0,
            "harmonics": "even",
            "control_profile": {
                "spatial_width": 0.06,
                "brightness_tilt": -0.14,
                "harshness_control": 0.38,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.42,
            },
        },
        {
            "source_path": data_dir / "Why they Bitin me.wav",
            "display_name": "Why they Bitin me",
            "output_name": "Why they Bitin me - AuralMind2 Premium No-Stem Trap.wav",
            "goal": (
                "Premium next-gen no-stem trap master with aggressive vocal focus, hostile modern punch, "
                "tight transient edge, and a bright top end kept controlled against fatigue."
            ),
            "preset_name": "club_clean",
            "target_lufs": -11.0,
            "warmth": 0.32,
            "transient_boost_db": 2.5,
            "harmonic_drive": 3.5,
            "harmonics": "even",
            "control_profile": {
                "spatial_width": 0.02,
                "brightness_tilt": -0.10,
                "harshness_control": 0.34,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.48,
            },
        },
        {
            "source_path": data_dir / "FaceTime (3).wav",
            "display_name": "FaceTime (3)",
            "output_name": "FaceTime (3) - AuralMind2 Premium No-Stem Trap.wav",
            "goal": (
                "Premium next-gen no-stem melodic trap master with intimate vocal focus, late-night width kept disciplined, "
                "rich sub support, and premium sheen without a sibilant or brittle edge."
            ),
            "preset_name": "radio_loud",
            "target_lufs": -11.1,
            "warmth": 0.42,
            "transient_boost_db": 2.3,
            "harmonic_drive": 6.5,
            "harmonics": "both",
            "control_profile": {
                "spatial_width": -0.02,
                "brightness_tilt": 0.10,
                "harshness_control": 0.18,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.54,
            },
        },
        {
            "source_path": data_dir / "FaceTime (4).wav",
            "display_name": "FaceTime (4)",
            "output_name": "FaceTime (4) - AuralMind2 Premium No-Stem Trap.wav",
            "goal": (
                "Premium next-gen no-stem melodic trap master with intimate vocal focus, late-night width kept disciplined, "
                "rich sub support, and premium sheen without a sibilant or brittle edge."
            ),
            "preset_name": "radio_loud",
            "target_lufs": -11.1,
            "warmth": 0.42,
            "transient_boost_db": 2.3,
            "harmonic_drive": 6.5,
            "harmonics": "both",
            "control_profile": {
                "spatial_width": -0.02,
                "brightness_tilt": 0.10,
                "harshness_control": 0.18,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.54,
            },
        },
    ]


def register_sources(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    registered: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        source_path = Path(profile["source_path"]).resolve()
        key = str(source_path)
        if key in registered:
            continue
        reg = server.register_audio_from_path(str(source_path), DUMMY_CTX)
        analysis = server.analyze_audio(reg.audio_id, DUMMY_CTX).model_dump()
        registered[key] = {
            "audio_id": reg.audio_id,
            "source_path": str(source_path),
            "source_analysis": analysis,
        }
    return registered


def run_harmonic_prepass(audio_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    drive = float(profile.get("harmonic_drive") or 0.0)
    if drive <= 0.0:
        return {"working_audio_id": audio_id, "prepass": None}
    result = asyncio.run(
        server.apply_harmonic_excitation(
            server.HarmonicExcitationIn(
                audio_id=audio_id,
                drive_amount=drive,
                harmonics=profile["harmonics"],
            ),
            DUMMY_CTX,
        )
    )
    session_key, session_dir = server._get_session_info(DUMMY_CTX)
    prepass_entry = server._load_artifact(session_key, session_dir, result.artifact_id)
    if prepass_entry is None:
        raise RuntimeError(f"Could not resolve harmonic prepass artifact for {profile['display_name']}.")
    working_audio_id = server._new_id("aud")
    server._register_existing_file(
        session_key,
        session_dir,
        artifact_id=working_audio_id,
        kind="audio",
        filename=f"{sanitize_name(profile['display_name'])}__harmonic_prepass.wav",
        data_filename=prepass_entry.data_filename,
        media_type=prepass_entry.media_type,
    )
    processed_analysis = server.analyze_audio(working_audio_id, DUMMY_CTX).model_dump()
    return {
        "working_audio_id": working_audio_id,
        "prepass": {
            "artifact_id": result.artifact_id,
            "working_audio_id": working_audio_id,
            "message": result.message,
            "meter": result.meter,
            "analysis": processed_analysis,
            "drive_amount": drive,
            "harmonics": profile["harmonics"],
        },
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
    prepass_info = run_harmonic_prepass(source_info["audio_id"], profile)
    working_audio_id = prepass_info["working_audio_id"]

    plan = server.plan_mastering_strategy(
        server.StrategyPlanIn(
            audio_id=working_audio_id,
            goal=profile["goal"],
            platform=PLATFORM,
            control_profile=control_profile,
            stem_mode="off",
        ),
        DUMMY_CTX,
    )

    governor = optimize_governor(working_audio_id, profile["preset_name"])
    explicit_settings = plan.settings.model_dump()
    explicit_settings.update(
        {
            "preset_name": profile["preset_name"],
            "target_lufs": profile["target_lufs"],
            "warmth": profile["warmth"],
            "transient_boost_db": profile["transient_boost_db"],
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": True,
            "enable_hooklift": True,
            "bit_depth": "float32",
            "control_profile": profile["control_profile"],
            "governor_search_steps": governor["recommended_governor_steps"],
            "governor_gr_limit_db": governor["recommended_governor_gr_limit_db"],
            "stem_mode": "off",
            "stem_gains_db": None,
        }
    )
    normalized = server.propose_master_settings(server.MasterSettings(**explicit_settings)).settings
    request = server.MasterRequest(audio_id=working_audio_id, **normalized.model_dump())
    result = server.master_audio(request, DUMMY_CTX)

    destination = MASTERS_DIR / profile["output_name"]
    copy_artifact_to_path(result.master_wav_id, destination)

    report_path = REPORTS_DIR / f"{sanitize_name(profile['display_name'])}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"# {profile['display_name']}",
        "",
        f"- Rendered: {utc_now()}",
        f"- Source: `{source_info['source_path']}`",
        f"- Output: `{destination}`",
        f"- Goal: {profile['goal']}",
        f"- Locked movement_amount: `{MOVEMENT_AMOUNT}`",
        f"- Locked stem_mode: `off`",
        f"- Chosen preset: `{profile['preset_name']}`",
        "",
        "## Source Analysis",
        f"- Integrated LUFS: `{source_info['source_analysis']['integrated_lufs']:.2f}`",
        f"- True Peak dBTP: `{source_info['source_analysis']['true_peak_dbtp']:.2f}`",
        f"- Crest dB: `{source_info['source_analysis']['crest_db']:.2f}`",
        f"- Stereo correlation: `{source_info['source_analysis']['stereo_correlation']:.3f}`",
        f"- Spectral centroid Hz: `{source_info['source_analysis']['centroid_hz']:.2f}`",
        "",
    ]
    if prepass_info["prepass"] is not None:
        pre = prepass_info["prepass"]
        report_lines.extend(
            [
                "## Harmonic Prepass",
                f"- Drive amount: `{pre['drive_amount']}`",
                f"- Harmonics: `{pre['harmonics']}`",
                f"- Meter: `{pre['meter']}`",
                f"- Post-prepass LUFS: `{pre['analysis']['integrated_lufs']:.2f}`",
                f"- Post-prepass True Peak dBTP: `{pre['analysis']['true_peak_dbtp']:.2f}`",
                "",
            ]
        )
    report_lines.extend(
        [
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
            f"- Reasoning: {governor['music_theory_reasoning']}",
            "",
            "## Final Metrics",
            f"- Integrated LUFS: `{result.metrics_after.integrated_lufs:.2f}`",
            f"- True Peak dBTP: `{result.metrics_after.true_peak_dbtp:.2f}`",
            f"- Crest dB: `{result.metrics_after.crest_db:.2f}`",
            f"- Stereo correlation: `{result.metrics_after.stereo_correlation:.3f}`",
            f"- Spectral centroid Hz: `{result.metrics_after.centroid_hz:.2f}`",
            "",
            "## Final Request",
            "```json",
            json.dumps(request.model_dump(), indent=2),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "display_name": profile["display_name"],
        "source_path": source_info["source_path"],
        "source_audio_id": source_info["audio_id"],
        "working_audio_id": working_audio_id,
        "source_analysis": source_info["source_analysis"],
        "prepass": prepass_info["prepass"],
        "goal": profile["goal"],
        "chosen_preset": profile["preset_name"],
        "plan_reasoning": plan.reasoning,
        "plan_warnings": plan.warnings,
        "governor": governor,
        "final_request": request.model_dump(),
        "metrics_before": result.metrics_before.model_dump(),
        "metrics_after": result.metrics_after.model_dump(),
        "master_wav_id": result.master_wav_id,
        "tuning_trace_id": result.tuning_trace_id,
        "artifacts": result.artifacts,
        "exported_wav_path": str(destination),
        "report_path": str(report_path),
    }


def write_summary(render_summaries: List[Dict[str, Any]]) -> None:
    lines = [
        f"# {RUN_LABEL}",
        "",
        f"- Generated: {utc_now()}",
        f"- Output root: `{OUTPUT_ROOT}`",
        f"- Locked movement_amount: `{MOVEMENT_AMOUNT}`",
        "- Locked stem_mode: `off`",
        "",
        "## Batch Notes",
        "- This batch stays in one premium no-stem trap lane with controlled width and elevated masking/harshness management.",
        "- Harmonic excitation was used as the creative enhancement pass because reliable key/BPM detection is not available in this repo for safe musical EQ or tempo-synced dynamics.",
        "- FaceTime (3) and FaceTime (4) were intentionally held to the same target aesthetic for pair consistency.",
        "",
        "## Results",
    ]
    for summary in render_summaries:
        after = summary["metrics_after"]
        lines.extend(
            [
                f"### {summary['display_name']}",
                f"- Output: `{summary['exported_wav_path']}`",
                f"- Preset: `{summary['chosen_preset']}`",
                f"- Final LUFS: `{after['integrated_lufs']:.2f}`",
                f"- Final True Peak dBTP: `{after['true_peak_dbtp']:.2f}`",
                f"- Final Crest dB: `{after['crest_db']:.2f}`",
                f"- Final Stereo correlation: `{after['stereo_correlation']:.3f}`",
                f"- Report: `{summary['report_path']}`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    profiles = build_profiles()
    MASTERS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    registered = register_sources(profiles)
    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "project_root": str(ROOT),
        "output_root": str(OUTPUT_ROOT),
        "movement_amount_locked": MOVEMENT_AMOUNT,
        "stem_mode_locked": "off",
        "renders": [],
    }

    for profile in profiles:
        source_key = str(Path(profile["source_path"]).resolve())
        render_summary = render_profile(profile, registered[source_key])
        manifest["renders"].append(render_summary)
        print(f"Rendered {profile['display_name']} -> {render_summary['exported_wav_path']}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest["renders"])
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
