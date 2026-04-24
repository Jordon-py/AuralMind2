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

RUN_LABEL = "premium_no_stem_trap_selective_hotter_pass_20260411_v2"
MOVEMENT_AMOUNT = 0.24
PLATFORM = "spotify"
RUN_SESSION_ID = f"{RUN_LABEL}-{int(datetime.now(timezone.utc).timestamp())}"

OUTPUT_ROOT = ROOT / "masters" / RUN_LABEL
MASTERS_DIR = OUTPUT_ROOT / "masters"
REPORTS_DIR = OUTPUT_ROOT / "reports"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
SUMMARY_PATH = OUTPUT_ROOT / "batch_summary.md"
BASELINE_BATCH_ROOT = ROOT / "masters" / "premium_no_stem_trap_batch_20260411" / "masters"


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
            "source_path": data_dir / "Vegas - This Life I Lead.wav",
            "baseline_path": BASELINE_BATCH_ROOT / "Vegas - This Life I Lead - AuralMind2 Premium No-Stem Trap.wav",
            "display_name": "Vegas - This Life I Lead",
            "output_name": "Vegas - This Life I Lead - AuralMind2 Premium No-Stem Trap HOTTER PASS.wav",
            "goal": (
                "Hotter and more aggressive premium no-stem trap master with harder commercial density, firmer low-end grip, "
                "more forward punch, and premium hook presence while preventing brittle top-end splash."
            ),
            "preset_name": "radio_loud",
            "target_lufs": -10.0,
            "warmth": 0.27,
            "transient_boost_db": 1.8,
            "harmonic_drive": 18.0,
            "harmonics": "even",
            "tempo_bpm": 140.0,
            "tempo_note_division": "1/16",
            "minimum_governor_steps": 8,
            "maximum_gr_limit_db": -2.6,
            "control_profile": {
                "spatial_width": 0.03,
                "brightness_tilt": 0.0,
                "harshness_control": 0.30,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.56,
            },
        },
        {
            "source_path": data_dir / "Why they Bitin me.wav",
            "baseline_path": BASELINE_BATCH_ROOT / "Why they Bitin me - AuralMind2 Premium No-Stem Trap.wav",
            "display_name": "Why they Bitin me",
            "output_name": "Why they Bitin me - AuralMind2 Premium No-Stem Trap HOTTER PASS.wav",
            "goal": (
                "Hotter and more aggressive premium no-stem trap master with denser impact, more hostile vocal cut, "
                "tighter low-end aggression, and harder commercial loudness without hashy hats or smeared hooks."
            ),
            "preset_name": "competitive_trap",
            "target_lufs": -9.9,
            "warmth": 0.26,
            "transient_boost_db": 1.4,
            "harmonic_drive": 30.0,
            "harmonics": "both",
            "tempo_bpm": 150.0,
            "tempo_note_division": "1/16",
            "minimum_governor_steps": 8,
            "maximum_gr_limit_db": -2.8,
            "control_profile": {
                "spatial_width": 0.0,
                "brightness_tilt": 0.0,
                "harshness_control": 0.30,
                "movement_amount": MOVEMENT_AMOUNT,
                "low_end_focus": 0.62,
            },
        },
    ]


def register_audio(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    try:
        reg = server.register_audio_from_path(str(resolved), DUMMY_CTX)
        audio_id = reg.audio_id
    except Exception:
        session_key, session_dir = server._get_session_info(DUMMY_CTX)
        audio_id = server._new_id("aud")
        media_type = server._guess_media_type(resolved.name, fallback="audio/wav")
        server._store_file_from_path(
            session_key,
            session_dir,
            artifact_id=audio_id,
            kind="audio",
            filename=resolved.name,
            source_path=str(resolved),
            media_type=media_type,
        )
    analysis = server.analyze_audio(audio_id, DUMMY_CTX).model_dump()
    return {"audio_id": audio_id, "path": str(resolved), "analysis": analysis}


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
        filename=f"{sanitize_name(profile['display_name'])}__hotter_prepass.wav",
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


def run_tempo_prepass(audio_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    bpm = profile.get("tempo_bpm")
    if not bpm:
        return {"working_audio_id": audio_id, "prepass": None}
    note_division = profile.get("tempo_note_division", "1/16")
    result = asyncio.run(
        server.apply_tempo_dynamics(
            server.TempoDynamicsIn(
                audio_id=audio_id,
                bpm=float(bpm),
                note_division=note_division,
            ),
            DUMMY_CTX,
        )
    )
    session_key, session_dir = server._get_session_info(DUMMY_CTX)
    prepass_entry = server._load_artifact(session_key, session_dir, result.artifact_id)
    if prepass_entry is None:
        raise RuntimeError(f"Could not resolve tempo prepass artifact for {profile['display_name']}.")
    working_audio_id = server._new_id("aud")
    server._register_existing_file(
        session_key,
        session_dir,
        artifact_id=working_audio_id,
        kind="audio",
        filename=f"{sanitize_name(profile['display_name'])}__tempo_lock.wav",
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
            "pulse_grid": result.pulse_grid,
            "analysis": processed_analysis,
            "bpm": float(bpm),
            "note_division": note_division,
        },
    }


def optimize_governor(audio_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    base = asyncio.run(
        server.analyze_and_optimize_governor(
            server.AnalyzeAndOptimizeGovernorIn(audio_id=audio_id, preset_name=profile["preset_name"]),
            DUMMY_CTX,
        )
    ).model_dump()
    return {
        "base_recommendation": base,
        "recommended_governor_steps": max(base["recommended_governor_steps"], profile["minimum_governor_steps"]),
        "recommended_governor_gr_limit_db": min(base["recommended_governor_gr_limit_db"], profile["maximum_gr_limit_db"]),
    }


def compare_to_baseline(baseline_audio_id: str, new_master_id: str) -> Dict[str, Any]:
    result = server.compare_audio_metrics(
        server.CompareMetricsIn(audio_id_a=baseline_audio_id, audio_id_b=new_master_id),
        DUMMY_CTX,
    )
    return result.model_dump()


def render_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    source_info = register_audio(Path(profile["source_path"]))
    baseline_info = register_audio(Path(profile["baseline_path"]))

    control_profile = server.MasteringControlProfile(**profile["control_profile"])
    prepass_info = run_harmonic_prepass(source_info["audio_id"], profile)
    tempo_prepass_info = run_tempo_prepass(prepass_info["working_audio_id"], profile)
    working_audio_id = tempo_prepass_info["working_audio_id"]

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

    governor = optimize_governor(working_audio_id, profile)
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
    comparison = compare_to_baseline(baseline_info["audio_id"], result.master_wav_id)

    destination = MASTERS_DIR / profile["output_name"]
    copy_artifact_to_path(result.master_wav_id, destination)

    report_path = REPORTS_DIR / f"{sanitize_name(profile['display_name'])}.md"
    report_lines = [
        f"# {profile['display_name']}",
        "",
        f"- Rendered: {utc_now()}",
        f"- Source: `{source_info['path']}`",
        f"- Baseline master: `{baseline_info['path']}`",
        f"- Output: `{destination}`",
        f"- Goal: {profile['goal']}",
        f"- Locked movement_amount: `{MOVEMENT_AMOUNT}`",
        "- Locked stem_mode: `off`",
        "",
        "## Source Analysis",
        f"- Integrated LUFS: `{source_info['analysis']['integrated_lufs']:.2f}`",
        f"- True Peak dBTP: `{source_info['analysis']['true_peak_dbtp']:.2f}`",
        f"- Crest dB: `{source_info['analysis']['crest_db']:.2f}`",
        f"- Stereo correlation: `{source_info['analysis']['stereo_correlation']:.3f}`",
        f"- Spectral centroid Hz: `{source_info['analysis']['centroid_hz']:.2f}`",
        "",
        "## Baseline Master Analysis",
        f"- Integrated LUFS: `{baseline_info['analysis']['integrated_lufs']:.2f}`",
        f"- True Peak dBTP: `{baseline_info['analysis']['true_peak_dbtp']:.2f}`",
        f"- Crest dB: `{baseline_info['analysis']['crest_db']:.2f}`",
        f"- Stereo correlation: `{baseline_info['analysis']['stereo_correlation']:.3f}`",
        f"- Spectral centroid Hz: `{baseline_info['analysis']['centroid_hz']:.2f}`",
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
    if tempo_prepass_info["prepass"] is not None:
        tempo_pre = tempo_prepass_info["prepass"]
        report_lines.extend(
            [
                "## Tempo Dynamics Prepass",
                f"- BPM: `{tempo_pre['bpm']}`",
                f"- Note division: `{tempo_pre['note_division']}`",
                f"- Pulse grid: `{tempo_pre['pulse_grid']}`",
                f"- Post-tempo-lock LUFS: `{tempo_pre['analysis']['integrated_lufs']:.2f}`",
                f"- Post-tempo-lock True Peak dBTP: `{tempo_pre['analysis']['true_peak_dbtp']:.2f}`",
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
            "## Governor",
            f"- Base recommendation steps: `{governor['base_recommendation']['recommended_governor_steps']}`",
            f"- Base recommendation GR limit: `{governor['base_recommendation']['recommended_governor_gr_limit_db']}`",
            f"- Forced second-pass steps: `{governor['recommended_governor_steps']}`",
            f"- Forced second-pass GR limit: `{governor['recommended_governor_gr_limit_db']}`",
            "",
            "## Final Metrics",
            f"- Integrated LUFS: `{result.metrics_after.integrated_lufs:.2f}`",
            f"- True Peak dBTP: `{result.metrics_after.true_peak_dbtp:.2f}`",
            f"- Crest dB: `{result.metrics_after.crest_db:.2f}`",
            f"- Stereo correlation: `{result.metrics_after.stereo_correlation:.3f}`",
            f"- Spectral centroid Hz: `{result.metrics_after.centroid_hz:.2f}`",
            "",
            "## Baseline Comparison",
            f"- LUFS delta: `{comparison['delta']['lufs_delta']:.2f}`",
            f"- True peak delta: `{comparison['delta']['true_peak_delta']:.2f}`",
            f"- Crest delta: `{comparison['delta']['crest_delta']:.2f}`",
            f"- Correlation delta: `{comparison['delta']['correlation_delta']:.3f}`",
            "",
            "## Final Request",
            "```json",
            json.dumps(request.model_dump(), indent=2),
            "```",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "display_name": profile["display_name"],
        "source_path": source_info["path"],
        "baseline_path": baseline_info["path"],
        "goal": profile["goal"],
        "preset_name": profile["preset_name"],
        "source_analysis": source_info["analysis"],
        "baseline_analysis": baseline_info["analysis"],
        "prepass": prepass_info["prepass"],
        "tempo_prepass": tempo_prepass_info["prepass"],
        "plan_reasoning": plan.reasoning,
        "plan_warnings": plan.warnings,
        "governor": governor,
        "final_request": request.model_dump(),
        "metrics_before": result.metrics_before.model_dump(),
        "metrics_after": result.metrics_after.model_dump(),
        "comparison_to_baseline": comparison["delta"],
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
        "- This is a targeted hotter/aggressive second pass for the two under-shot tracks from the previous no-stem batch.",
        "- Both rerenders use a denser AuralMind2 pre-conditioning chain: stronger harmonic excitation, tempo-synced dynamics, and then a more aggressive governor while preserving masking EQ, harshness control, air, and hooklift features.",
        "",
        "## Results",
    ]
    for summary in render_summaries:
        after = summary["metrics_after"]
        delta = summary["comparison_to_baseline"]
        lines.extend(
            [
                f"### {summary['display_name']}",
                f"- Output: `{summary['exported_wav_path']}`",
                f"- Preset: `{summary['preset_name']}`",
                f"- Final LUFS: `{after['integrated_lufs']:.2f}`",
                f"- Final True Peak dBTP: `{after['true_peak_dbtp']:.2f}`",
                f"- Final Crest dB: `{after['crest_db']:.2f}`",
                f"- LUFS delta vs baseline: `{delta['lufs_delta']:.2f}`",
                f"- Crest delta vs baseline: `{delta['crest_delta']:.2f}`",
                f"- Correlation delta vs baseline: `{delta['correlation_delta']:.3f}`",
                f"- Report: `{summary['report_path']}`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    profiles = build_profiles()
    MASTERS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "project_root": str(ROOT),
        "output_root": str(OUTPUT_ROOT),
        "movement_amount_locked": MOVEMENT_AMOUNT,
        "stem_mode_locked": "off",
        "renders": [],
    }

    for profile in profiles:
        render_summary = render_profile(profile)
        manifest["renders"].append(render_summary)
        print(f"Rendered {profile['display_name']} -> {render_summary['exported_wav_path']}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest["renders"])
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
