from __future__ import annotations

import json
import re
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server

OUTPUT_ROOT = ROOT / "masters" / "premium_melodic_vocal_no_stem_batch"
MANIFESTS_DIR = OUTPUT_ROOT / "_manifests"
REPORTS_DIR = OUTPUT_ROOT / "_reports"
RENDS_DIR = OUTPUT_ROOT / "masters"

CATALOG_PATH = MANIFESTS_DIR / "catalog_manifest.json"
RUN_RESULTS_PATH = MANIFESTS_DIR / "run_results.json"
QA_SUMMARY_PATH = MANIFESTS_DIR / "qa_summary.json"
RUN_LOG_PATH = REPORTS_DIR / "run_log.md"

MOVEMENT_AMOUNT = 0.21
MAX_CONCURRENT_JOBS = 3
PLATFORM = "spotify"

TRACK_RELATIVE_PATHS = [
    "(edit) Slidin.wav",
    "Been winning .wav",
    "Best of me (1).wav",
    "Bigger picture  (1).wav",
    "chef jeff ft vegas - Somebody_v002.wav",
    "Cleanthie (1).wav",
    "DaddysGirls.wav",
    "difference (10).wav",
    "Don't let me down(9)_PREMASTER_.wav",
    "FaceTime (15)_PREMASTER_-6dB.wav",
    "Fall in Love.wav",
    "Fire (4).wav",
    "FM Vegas - Consistent (1).wav",
    "Got too.wav",
    "Hold it down (8).wav",
    "Hot shit (2).wav",
    "I'm Him (7).wav",
    "In the moment .wav",
    "Its still love baby (1).wav",
    "Last Time (14).wav",
    "M.O (3).wav",
    "Missin You (1).wav",
    "New Project (6).wav",
    "Newskie (1).wav",
    "Nipssey.wav",
    "No_good.mp3",
    "Ride.wav",
    "SB.wav",
    "SOMEBODY.wav",
    "Stilll_KOTS.wav",
    "stressed out.wav",
    "Track_5.m4a",
    "Truthy.wav",
    "Untitled Dec 3, 2022 6_12 PM.wav",
    "Vegaas (3).wav",
    "Vegas - top teir (22).wav",
    "Vegas - Weeknd (3).wav",
    "Vegas__FaceTime.m4a",
    "Walking_In_Rain.mp3",
    "Why they Bitin me.wav",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return deepcopy(default)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_log(message: str) -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"- [{utc_now()}] {message}\n")


def sanitize_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum():
            safe.append(char.lower())
        elif char in (" ", "-", "_", "."):
            safe.append("-")
    joined = "".join(safe).strip("-")
    while "--" in joined:
        joined = joined.replace("--", "-")
    return joined or "untitled"


def canonical_title_from_path(relative_path: str) -> str:
    stem = Path(relative_path).stem
    stem = re.sub(r"^\(edit\)\s*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_PREMASTER_?-?\d*dB", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_v\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(\d+\)$", "", stem).strip()
    stem = stem.replace("__", " ").replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" .-_")
    if not stem:
        stem = Path(relative_path).stem
    return stem


def build_tasks() -> Dict[str, Any]:
    tracks: List[Dict[str, Any]] = []
    missing: List[str] = []
    for relative_path in TRACK_RELATIVE_PATHS:
        source_path = ROOT / "data" / relative_path
        if not source_path.exists():
            missing.append(relative_path)
            continue
        tracks.append(
            {
                "key": relative_path,
                "relative_path": relative_path,
                "source_path": str(source_path.resolve()),
                "source_file_name": Path(relative_path).name,
                "canonical_title": canonical_title_from_path(relative_path),
            }
        )
    manifest = {
        "generated_at": utc_now(),
        "movement_amount": MOVEMENT_AMOUNT,
        "stem_mode": "off",
        "intent": "premium professional trap master with melodic clarity and vocal presence",
        "tracks_requested": len(TRACK_RELATIVE_PATHS),
        "tracks_found": len(tracks),
        "tracks_missing": missing,
        "tracks": tracks,
    }
    return manifest


def build_goal(canonical_title: str) -> str:
    return (
        f"Premium professional trap master for {canonical_title} with no stems, mono-sub discipline, "
        "forward lead vocals, melodic clarity, smooth upper mids, stable center image, and confident but controlled low end. "
        "Favor hook intelligibility and vocal presence over brute-force aggression."
    )


def choose_preset(metrics: Dict[str, Any]) -> str:
    centroid = float(metrics.get("centroid_hz") or 0.0)
    crest = float(metrics.get("crest_db") or 0.0)
    if centroid > 1850.0 or crest < 8.0:
        return "club_clean"
    return "radio_loud"


def build_control_profile(metrics: Dict[str, Any]) -> server.MasteringControlProfile:
    centroid = float(metrics.get("centroid_hz") or 0.0)
    corr = float(metrics.get("stereo_correlation") or 0.9)

    # Keep the vocal center and sub image tighter after early passes showed spready outputs.
    spatial_width = 0.04
    if corr < 0.80:
        spatial_width = 0.02
    elif corr > 0.93:
        spatial_width = 0.05

    brightness_tilt = 0.05
    if centroid < 900.0:
        brightness_tilt = 0.06
    elif centroid > 1700.0:
        brightness_tilt = 0.02

    harshness_control = 0.48
    if centroid > 1800.0:
        harshness_control = 0.54

    low_end_focus = 0.71
    if centroid < 900.0:
        low_end_focus = 0.74
    elif centroid > 1700.0:
        low_end_focus = 0.66

    return server.MasteringControlProfile(
        spatial_width=round(spatial_width, 2),
        brightness_tilt=round(brightness_tilt, 2),
        harshness_control=round(harshness_control, 2),
        movement_amount=MOVEMENT_AMOUNT,
        low_end_focus=round(low_end_focus, 2),
    )


def apply_house_overrides(
    settings: server.MasterSettings,
    preset_name: str,
    metrics: Dict[str, Any],
    control_profile: server.MasteringControlProfile,
) -> server.MasterSettings:
    centroid = float(metrics.get("centroid_hz") or 0.0)
    crest = float(metrics.get("crest_db") or 16.0)

    if preset_name == "club_clean":
        target_lufs = -10.6
        governor_gr_limit_db = -2.2
    else:
        target_lufs = -10.4
        governor_gr_limit_db = -2.4

    if crest > 19.0:
        target_lufs = -10.3
    elif crest < 12.5:
        target_lufs = -10.6

    warmth = 0.07
    if centroid < 850.0:
        warmth = 0.09
    elif centroid > 1700.0:
        warmth = 0.05

    transient_boost_db = 2.2
    if crest > 18.0:
        transient_boost_db = 2.5
    elif crest < 12.5:
        transient_boost_db = 1.8

    return settings.model_copy(
        update={
            "preset_name": preset_name,
            "target_lufs": target_lufs,
            "warmth": warmth,
            "transient_boost_db": transient_boost_db,
            "enable_harshness_limiter": True,
            "enable_masking_eq": True,
            "enable_air_motion": False,
            "enable_hooklift": True,
            "bit_depth": "float32",
            "control_profile": control_profile,
            "governor_search_steps": 6,
            "governor_gr_limit_db": governor_gr_limit_db,
            "stem_mode": "off",
            "stem_gains_db": None,
        }
    )


def build_request_for_track(task: Dict[str, Any]) -> Dict[str, Any]:
    reg = server.register_audio_from_path(task["source_path"])
    audio_id = reg.audio_id
    metrics = server.analyze_audio(audio_id).model_dump()
    control_profile = build_control_profile(metrics)

    plan_req = server.StrategyPlanIn(
        audio_id=audio_id,
        goal=build_goal(task["canonical_title"]),
        platform=PLATFORM,
        control_profile=control_profile,
        governor_search_steps=5,
        governor_gr_limit_db=-2.0,
        stem_gains_db=None,
        stem_mode="off",
    )
    strategy = server.plan_mastering_strategy(plan_req)
    preset_name = choose_preset(metrics)
    tuned = apply_house_overrides(strategy.settings, preset_name, metrics, control_profile)
    normalized = server.propose_master_settings(tuned).settings
    master_req = server.MasterRequest(audio_id=audio_id, **normalized.model_dump())

    return {
        "audio_id": audio_id,
        "source_metrics": metrics,
        "strategy": strategy.model_dump(),
        "request": master_req,
    }


def quality_gate(final_metrics: Dict[str, Any]) -> Dict[str, Any]:
    lufs = float(final_metrics.get("integrated_lufs") or -99.0)
    tp = float(final_metrics.get("true_peak_dbtp") or 0.0)
    crest = float(final_metrics.get("crest_db") or 0.0)
    corr = float(final_metrics.get("stereo_correlation") or 0.0)

    gates = {
        "loudness_ok": lufs >= -15.5,
        "true_peak_ok": tp <= -0.95,
        "crest_ok": 7.0 <= crest <= 14.5,
        "stereo_ok": corr >= 0.72,
    }
    gates["passed"] = all(gates.values())
    return gates


def export_artifact(session_dir: Path, artifact_filename: str, destination: Path) -> None:
    source = session_dir / artifact_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RENDS_DIR.mkdir(parents=True, exist_ok=True)

    catalog_manifest = build_tasks()
    write_json(CATALOG_PATH, catalog_manifest)

    session_key, session_dir_raw = server._get_session_info(None)
    session_dir = Path(session_dir_raw)

    default_results = {
        "generated_at": utc_now(),
        "session_key": session_key,
        "storage_dir": str(server.STORAGE_DIR),
        "tracks": {},
    }
    run_results = load_json(RUN_RESULTS_PATH, default_results)
    run_results.setdefault("tracks", {})
    run_results["session_key"] = session_key
    run_results["storage_dir"] = str(server.STORAGE_DIR)
    run_results["updated_at"] = utc_now()

    append_log(
        f"Premium melodic/vocal no-stem batch started in session `{session_key}` using storage `{server.STORAGE_DIR}`."
    )
    append_log(
        f"Tracks requested `{catalog_manifest['tracks_requested']}`, found `{catalog_manifest['tracks_found']}`, "
        f"missing `{len(catalog_manifest['tracks_missing'])}`."
    )
    if catalog_manifest["tracks_missing"]:
        append_log(f"Missing source paths: {', '.join(catalog_manifest['tracks_missing'])}")

    completed = {
        key for key, value in run_results["tracks"].items() if value.get("status") == "completed"
    }
    queue = [task for task in catalog_manifest["tracks"] if task["key"] not in completed]
    pending: List[Dict[str, Any]] = []

    while queue or pending:
        while queue and len(pending) < MAX_CONCURRENT_JOBS:
            task = queue.pop(0)
            append_log(f"Preparing `{task['source_file_name']}`.")
            prepared = build_request_for_track(task)
            launch = server.run_master_job(prepared["request"])
            pending.append(
                {
                    **task,
                    **prepared,
                    "job_id": launch.job_id,
                }
            )
            run_results["tracks"][task["key"]] = {
                "status": "running",
                "job_id": launch.job_id,
                "source_file_name": task["source_file_name"],
                "relative_path": task["relative_path"],
                "canonical_title": task["canonical_title"],
                "started_at": utc_now(),
                "request": prepared["request"].model_dump(),
                "strategy": prepared["strategy"],
                "source_metrics": prepared["source_metrics"],
            }
            run_results["updated_at"] = utc_now()
            write_json(RUN_RESULTS_PATH, run_results)
            append_log(f"Launched job `{launch.job_id}` for `{task['source_file_name']}`.")

        if not pending:
            break

        time.sleep(15)
        still_pending: List[Dict[str, Any]] = []

        for item in pending:
            status = server.job_status(item["job_id"])
            if status.status in {"queued", "running"}:
                still_pending.append(item)
                continue

            track_record = run_results["tracks"][item["key"]]

            if status.status == "error":
                error_message = status.error.message if status.error else "unknown_job_error"
                append_log(f"Job `{item['job_id']}` failed for `{item['source_file_name']}`: {error_message}")
                track_record.update(
                    {
                        "status": "failed",
                        "error": error_message,
                        "finished_at": utc_now(),
                    }
                )
                run_results["updated_at"] = utc_now()
                write_json(RUN_RESULTS_PATH, run_results)
                continue

            result = server.job_result(item["job_id"])
            artifacts = [artifact.model_dump() for artifact in result.artifacts]
            audio_artifact = next(
                (artifact for artifact in artifacts if artifact.get("media_type", "").startswith("audio/")),
                artifacts[0],
            )
            final_metrics = result.metrics.model_dump()
            compare = server.compare_audio_metrics(
                server.CompareMetricsIn(audio_id_a=item["audio_id"], audio_id_b=audio_artifact["artifact_id"]),
                None,
            )
            delta = compare.delta.model_dump()
            qa = quality_gate(final_metrics)

            song_dir = RENDS_DIR / sanitize_name(item["canonical_title"])
            final_name = (
                f"{sanitize_name(item['canonical_title'])}"
                f"__src-{sanitize_name(Path(item['source_file_name']).stem)}"
                f"__nostem-melodic-vocal"
                f"__movement-{MOVEMENT_AMOUNT:.2f}"
                "__mastered.wav"
            )
            summary_name = final_name.replace("__mastered.wav", "__summary.json")
            export_artifact(session_dir, audio_artifact["filename"], song_dir / final_name)

            summary_payload = {
                "completed_at": utc_now(),
                "session_key": session_key,
                "job_id": item["job_id"],
                "source_file_name": item["source_file_name"],
                "relative_path": item["relative_path"],
                "canonical_title": item["canonical_title"],
                "audio_id": item["audio_id"],
                "request": item["request"].model_dump(),
                "strategy": item["strategy"],
                "source_metrics": item["source_metrics"],
                "final_metrics": final_metrics,
                "metrics_delta": delta,
                "qa": qa,
                "artifacts": artifacts,
                "exported_master_path": str(song_dir / final_name),
            }
            write_json(song_dir / summary_name, summary_payload)

            track_record.update(
                {
                    "status": "completed",
                    "job_id": item["job_id"],
                    "finished_at": utc_now(),
                    "request": item["request"].model_dump(),
                    "strategy": item["strategy"],
                    "source_metrics": item["source_metrics"],
                    "final_metrics": final_metrics,
                    "metrics_delta": delta,
                    "qa": qa,
                    "artifacts": artifacts,
                    "exported_master_path": str(song_dir / final_name),
                    "summary_path": str(song_dir / summary_name),
                }
            )
            run_results["updated_at"] = utc_now()
            write_json(RUN_RESULTS_PATH, run_results)
            append_log(
                f"Completed `{item['source_file_name']}` -> `{final_name}` "
                f"(LUFS {final_metrics['integrated_lufs']:.2f}, TP {final_metrics['true_peak_dbtp']:.2f}, "
                f"crest {final_metrics['crest_db']:.2f}, corr {final_metrics['stereo_correlation']:.3f})."
            )

        pending = still_pending

    qa_records = run_results["tracks"]
    completed_records = [value for value in qa_records.values() if value.get("status") == "completed"]
    failed_records = [value for value in qa_records.values() if value.get("status") == "failed"]

    qa_summary = {
        "generated_at": utc_now(),
        "session_key": session_key,
        "completed_tracks": len(completed_records),
        "failed_tracks": len(failed_records),
        "average_integrated_lufs": (
            round(
                sum(track["final_metrics"]["integrated_lufs"] for track in completed_records)
                / len(completed_records),
                3,
            )
            if completed_records
            else None
        ),
        "average_true_peak_dbtp": (
            round(
                sum(track["final_metrics"]["true_peak_dbtp"] for track in completed_records)
                / len(completed_records),
                3,
            )
            if completed_records
            else None
        ),
        "average_stereo_correlation": (
            round(
                sum(track["final_metrics"]["stereo_correlation"] for track in completed_records)
                / len(completed_records),
                3,
            )
            if completed_records
            else None
        ),
        "tracks": qa_records,
    }
    write_json(QA_SUMMARY_PATH, qa_summary)
    append_log(
        f"Batch finished with {len(completed_records)} completed and {len(failed_records)} failed."
    )


if __name__ == "__main__":
    main()
