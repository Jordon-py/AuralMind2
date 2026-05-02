from __future__ import annotations

import json
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

OUTPUT_ROOT = ROOT / "Ignorance Is Bliss"
MANIFESTS_DIR = OUTPUT_ROOT / "_manifests"
REPORTS_DIR = OUTPUT_ROOT / "_reports"
LOGS_DIR = OUTPUT_ROOT / "_logs"
MASTERS_DIR = OUTPUT_ROOT / "masters"

SELECTED_VERSIONS_PATH = MANIFESTS_DIR / "selected_versions.json"
RUN_RESULTS_PATH = MANIFESTS_DIR / "run_results.json"
QA_SUMMARY_PATH = MANIFESTS_DIR / "qa_summary.json"
RUN_LOG_PATH = REPORTS_DIR / "run_log.md"

MOVEMENT_AMOUNT = 0.32
MAX_CONCURRENT_JOBS = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum():
            safe.append(char.lower())
        elif char in (" ", "-", "_"):
            safe.append("-")
    joined = "".join(safe).strip("-")
    while "--" in joined:
        joined = joined.replace("--", "-")
    return joined or "untitled"


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


def title_goal(canonical_title: str, batch_mode: str) -> str:
    lower = canonical_title.lower()
    if lower in {"fire", "hot shit", "vegas top teir", "i'm him", "nipssey"}:
        base = (
            "Hard-hitting modern trap master with authoritative 808 weight, punchy drums, "
            "mono-compatible sub, and aggressive but controlled energy."
        )
    elif lower in {"fall in love", "last time", "facetime", "somebody"}:
        base = (
            "High-fidelity melodic trap master with vocal clarity, emotional openness, "
            "smooth upper mids, and strong low-end support without brittleness."
        )
    else:
        base = (
            "High-fidelity modern trap master with tight low end, controlled width, "
            "clean transients, and competitive streaming impact."
        )
    if batch_mode == "stems":
        return (
            base
            + " Use stems only when they improve low-end discipline, vocal separation, or hook lift."
        )
    return base + " Keep the center image stable in full-mix mode."


def choose_preset(canonical_title: str, batch_mode: str, metrics: Dict[str, Any]) -> str:
    lower = canonical_title.lower()
    centroid = float(metrics.get("centroid_hz") or 0.0)
    if lower in {"fire", "hot shit", "vegas top teir", "i'm him", "nipssey"}:
        return "competitive_trap"
    if lower in {"last time", "fall in love", "facetime", "somebody"}:
        return "radio_loud"
    if batch_mode == "stems" and centroid > 1250.0:
        return "club_clean"
    return "competitive_trap" if batch_mode == "stems" else "radio_loud"


def build_control_profile(batch_mode: str, metrics: Dict[str, Any]) -> server.MasteringControlProfile:
    stems = batch_mode == "stems"
    centroid = float(metrics.get("centroid_hz") or 0.0)
    corr = float(metrics.get("stereo_correlation") or 0.9)
    crest = float(metrics.get("crest_db") or 16.0)

    spatial_width = 0.12 if stems else 0.14
    brightness_tilt = 0.04 if stems else 0.05
    harshness_control = 0.36 if stems else 0.34
    low_end_focus = 0.66 if stems else 0.62

    if centroid > 1300.0:
        brightness_tilt -= 0.03
        harshness_control += 0.05
    elif centroid < 800.0:
        brightness_tilt += 0.02
        low_end_focus += 0.03

    if corr < 0.82:
        spatial_width = min(spatial_width, 0.10)

    if crest > 19.0:
        low_end_focus += 0.02
        harshness_control += 0.02

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
    batch_mode: str,
    metrics: Dict[str, Any],
) -> server.MasterSettings:
    centroid = float(metrics.get("centroid_hz") or 0.0)
    crest = float(metrics.get("crest_db") or 16.0)

    settings = settings.model_copy(
        update={
            "preset_name": preset_name,
            "enable_air_motion": True,
            "enable_harshness_limiter": True,
            "enable_hooklift": True,
            "enable_masking_eq": True,
            "warmth": 0.04 if centroid > 1200.0 else (0.07 if centroid < 700.0 else 0.06),
            "transient_boost_db": (
                2.7 if crest >= 20.0 else 2.5 if crest >= 17.0 else 1.8 if crest <= 13.0 else 2.2
            ),
            "bit_depth": "float32",
        }
    )

    if preset_name == "competitive_trap":
        target_lufs = -9.9 if batch_mode == "stems" else -9.8
        gr_limit = -2.2 if batch_mode == "stems" else -3.0
    elif preset_name == "club_clean":
        target_lufs = -10.6
        gr_limit = -1.9
    elif preset_name == "radio_loud":
        target_lufs = -10.4 if batch_mode == "stems" else -10.2
        gr_limit = -2.0 if batch_mode == "stems" else -2.4
    else:
        target_lufs = -11.2
        gr_limit = -1.8

    if crest > 20.0:
        gr_limit = -1.9 if batch_mode == "stems" else -2.2

    updates: Dict[str, Any] = {
        "target_lufs": target_lufs,
        "governor_search_steps": 6 if preset_name == "competitive_trap" and batch_mode != "stems" else 5,
        "governor_gr_limit_db": gr_limit,
        "stem_mode": "on" if batch_mode == "stems" else "off",
        "control_profile": settings.control_profile.model_copy(update={"movement_amount": MOVEMENT_AMOUNT})
        if settings.control_profile
        else None,
    }
    if batch_mode == "stems":
        updates["stem_gains_db"] = {"vocals": 0.6, "drums": 0.25, "bass": -0.15, "other": 0.1}
    else:
        updates["stem_gains_db"] = None

    return settings.model_copy(update=updates)


def build_request_for_track(track: Dict[str, Any]) -> Dict[str, Any]:
    reg = server.register_audio_from_path(track["relative_path"])
    audio_id = reg.audio_id
    metrics = server.analyze_audio(audio_id).model_dump()
    control_profile = build_control_profile(track["planned_mode"], metrics)
    plan_req = server.StrategyPlanIn(
        audio_id=audio_id,
        goal=title_goal(track["canonical_title"], track["planned_mode"]),
        platform="spotify",
        control_profile=control_profile,
        governor_search_steps=5,
        governor_gr_limit_db=-2.0 if track["planned_mode"] == "stems" else -2.4,
        stem_gains_db={"vocals": 0.6, "drums": 0.25, "bass": -0.15, "other": 0.1}
        if track["planned_mode"] == "stems"
        else None,
        stem_mode="on" if track["planned_mode"] == "stems" else "off",
    )
    strategy = server.plan_mastering_strategy(plan_req)
    preset_name = choose_preset(track["canonical_title"], track["planned_mode"], metrics)
    tuned = apply_house_overrides(strategy.settings, preset_name, track["planned_mode"], metrics)
    normalized = server.propose_master_settings(tuned).settings
    master_req = server.MasterRequest(audio_id=audio_id, **normalized.model_dump())

    return {
        "audio_id": audio_id,
        "source_metrics": metrics,
        "strategy": strategy.model_dump(),
        "request": master_req,
    }


def quality_gate(
    final_metrics: Dict[str, Any],
    batch_mode: str,
) -> Dict[str, Any]:
    lufs = float(final_metrics.get("integrated_lufs") or -99.0)
    tp = float(final_metrics.get("true_peak_dbtp") or 0.0)
    crest = float(final_metrics.get("crest_db") or 0.0)
    corr = float(final_metrics.get("stereo_correlation") or 0.0)

    gates = {
        "loudness_ok": lufs >= (-15.8 if batch_mode == "stems" else -16.2),
        "true_peak_ok": tp <= -0.95,
        "crest_ok": 7.0 <= crest <= 13.5,
        "stereo_ok": corr >= (0.68 if batch_mode == "stems" else 0.72),
    }
    gates["passed"] = all(gates.values())
    return gates


def export_artifact(
    session_dir: Path,
    artifact_filename: str,
    destination: Path,
) -> None:
    source = session_dir / artifact_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_tasks() -> List[Dict[str, Any]]:
    selected = load_json(SELECTED_VERSIONS_PATH, {})
    tasks: List[Dict[str, Any]] = []
    for group in selected.get("selected_groups", []):
        for file_info in group.get("selected_files", []):
            key = f"{group['canonical_title']}::{file_info['file_name']}"
            tasks.append(
                {
                    "key": key,
                    "canonical_title": group["canonical_title"],
                    "normalized_title": group["normalized_title"],
                    "group_order": group["group_order"],
                    "planned_mode": group["batch_mode"],
                    "batch_half": group["batch_half"],
                    "relative_path": file_info["relative_path"],
                    "source_file_name": file_info["file_name"],
                    "group_confidence": group.get("group_confidence"),
                }
            )
    return tasks


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MASTERS_DIR.mkdir(parents=True, exist_ok=True)

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

    append_log(f"Batch runner started in session `{session_key}` using storage `{server.STORAGE_DIR}`.")
    append_log(f"Server preset inventory: {', '.join(server.list_presets().presets.keys())}")

    completed = {
        key
        for key, value in run_results["tracks"].items()
        if value.get("status") == "completed"
    }
    queue = [task for task in prepare_tasks() if task["key"] not in completed]

    pending: List[Dict[str, Any]] = []

    while queue or pending:
        while queue and len(pending) < MAX_CONCURRENT_JOBS:
            task = queue.pop(0)
            append_log(f"Preparing `{task['key']}` with planned mode `{task['planned_mode']}`.")
            prepared = build_request_for_track(task)
            launch = server.run_master_job(prepared["request"])
            pending.append(
                {
                    **task,
                    **prepared,
                    "job_id": launch.job_id,
                    "actual_mode": prepared["request"].stem_mode,
                    "attempt": 1,
                }
            )
            run_results["tracks"][task["key"]] = {
                "status": "running",
                "job_id": launch.job_id,
                "planned_mode": task["planned_mode"],
                "actual_mode": prepared["request"].stem_mode,
                "source_file_name": task["source_file_name"],
                "relative_path": task["relative_path"],
                "started_at": utc_now(),
                "request": prepared["request"].model_dump(),
                "strategy": prepared["strategy"],
                "source_metrics": prepared["source_metrics"],
            }
            write_json(RUN_RESULTS_PATH, run_results)
            append_log(f"Launched job `{launch.job_id}` for `{task['key']}`.")

        if not pending:
            break

        time.sleep(20)
        still_pending: List[Dict[str, Any]] = []

        for item in pending:
            status = server.job_status(item["job_id"])
            if status.status in {"queued", "running"}:
                still_pending.append(item)
                continue

            track_record = run_results["tracks"][item["key"]]

            if status.status == "error":
                error_message = status.error.message if status.error else "unknown_job_error"
                append_log(f"Job `{item['job_id']}` failed for `{item['key']}`: {error_message}")
                if item["planned_mode"] == "stems" and item["actual_mode"] == "on":
                    append_log(f"Falling back `{item['key']}` to no-stems after stems failure.")
                    fallback_request = item["request"].model_copy(
                        update={"stem_mode": "off", "stem_gains_db": None}
                    )
                    launch = server.run_master_job(fallback_request)
                    still_pending.append(
                        {
                            **item,
                            "job_id": launch.job_id,
                            "request": fallback_request,
                            "actual_mode": "off",
                            "attempt": item["attempt"] + 1,
                            "fallback_reason": error_message,
                        }
                    )
                    track_record.update(
                        {
                            "status": "running",
                            "job_id": launch.job_id,
                            "actual_mode": "off",
                            "fallback_reason": error_message,
                            "attempt": item["attempt"] + 1,
                        }
                    )
                    write_json(RUN_RESULTS_PATH, run_results)
                    continue

                track_record.update({"status": "failed", "error": error_message, "finished_at": utc_now()})
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
            qa = quality_gate(final_metrics, item["actual_mode"])

            if item["planned_mode"] == "stems" and item["actual_mode"] == "on" and not qa["passed"]:
                append_log(
                    f"Stems QA miss for `{item['key']}` (lufs={final_metrics['integrated_lufs']:.2f}, "
                    f"corr={final_metrics['stereo_correlation']:.3f}); rerunning no-stems."
                )
                fallback_request = item["request"].model_copy(update={"stem_mode": "off", "stem_gains_db": None})
                launch = server.run_master_job(fallback_request)
                still_pending.append(
                    {
                        **item,
                        "job_id": launch.job_id,
                        "request": fallback_request,
                        "actual_mode": "off",
                        "attempt": item["attempt"] + 1,
                        "fallback_reason": "stems_quality_gate_failed",
                    }
                )
                track_record.update(
                    {
                        "status": "running",
                        "job_id": launch.job_id,
                        "actual_mode": "off",
                        "attempt": item["attempt"] + 1,
                        "fallback_reason": "stems_quality_gate_failed",
                    }
                )
                write_json(RUN_RESULTS_PATH, run_results)
                continue

            source_basename = Path(item["source_file_name"]).stem
            song_dir = MASTERS_DIR / item["canonical_title"]
            final_name = (
                f"{sanitize_name(item['canonical_title'])}"
                f"__src-{sanitize_name(source_basename)}"
                f"__mode-{'stems' if item['actual_mode'] == 'on' else 'no-stems'}"
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
                "planned_mode": item["planned_mode"],
                "actual_mode": item["actual_mode"],
                "attempt": item["attempt"],
                "fallback_reason": item.get("fallback_reason"),
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
                    "planned_mode": item["planned_mode"],
                    "actual_mode": item["actual_mode"],
                    "attempt": item["attempt"],
                    "fallback_reason": item.get("fallback_reason"),
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
            write_json(RUN_RESULTS_PATH, run_results)
            append_log(
                f"Completed `{item['key']}` -> `{final_name}` "
                f"(LUFS {final_metrics['integrated_lufs']:.2f}, TP {final_metrics['true_peak_dbtp']:.2f}, "
                f"crest {final_metrics['crest_db']:.2f}, corr {final_metrics['stereo_correlation']:.3f})."
            )

        pending = still_pending

    qa_records = run_results["tracks"]
    completed_records = [value for value in qa_records.values() if value.get("status") == "completed"]
    failed_records = [value for value in qa_records.values() if value.get("status") == "failed"]
    fallback_records = [value for value in completed_records if value.get("planned_mode") == "stems" and value.get("actual_mode") == "off"]

    qa_summary = {
        "generated_at": utc_now(),
        "session_key": session_key,
        "completed_tracks": len(completed_records),
        "failed_tracks": len(failed_records),
        "fallback_to_no_stems": len(fallback_records),
        "average_integrated_lufs": (
            round(sum(track["final_metrics"]["integrated_lufs"] for track in completed_records) / len(completed_records), 3)
            if completed_records
            else None
        ),
        "average_true_peak_dbtp": (
            round(sum(track["final_metrics"]["true_peak_dbtp"] for track in completed_records) / len(completed_records), 3)
            if completed_records
            else None
        ),
        "average_stereo_correlation": (
            round(sum(track["final_metrics"]["stereo_correlation"] for track in completed_records) / len(completed_records), 3)
            if completed_records
            else None
        ),
        "tracks": qa_records,
    }
    write_json(QA_SUMMARY_PATH, qa_summary)
    append_log(
        f"Batch runner finished with {len(completed_records)} completed, "
        f"{len(failed_records)} failed, {len(fallback_records)} stems fallbacks."
    )


if __name__ == "__main__":
    main()
