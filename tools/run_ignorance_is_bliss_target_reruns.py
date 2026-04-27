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
MASTERS_DIR = OUTPUT_ROOT / "masters"

PASS1_QA_PATH = MANIFESTS_DIR / "qa_summary.json"
TARGETS_PATH = MANIFESTS_DIR / "qa_failed_rerun_targets.json"
RERUN_RESULTS_PATH = MANIFESTS_DIR / "run_results_pass2.json"
RERUN_QA_PATH = MANIFESTS_DIR / "qa_summary_pass2.json"
RERUN_LOG_PATH = REPORTS_DIR / "run_log_pass2.md"

MOVEMENT_AMOUNT = 0.32
MAX_CONCURRENT_JOBS = 2
PASS_LABEL = "pass2"
RERUN_TAG = "qa-rerun01"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def reset_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Ignorance Is Bliss Pass-2 Rerun Log\n\n"
        "## Execution Metadata\n"
        f"- Date: {utc_now()}\n"
        "- Project: AuralMind2\n"
        f"- Output root: `{OUTPUT_ROOT}`\n"
        "- Scope: pass-1 QA failures only\n"
        f"- Rerun label: `{PASS_LABEL}` / `{RERUN_TAG}`\n"
        f"- Concurrency cap: `{MAX_CONCURRENT_JOBS}` active jobs\n"
        "- Status: pass-2 targeted reruns initialized\n\n"
        "## Events\n",
        encoding="utf-8",
    )


def append_log(message: str) -> None:
    with RERUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"- [{utc_now()}] {message}\n")


def quality_gate(final_metrics: Dict[str, Any], batch_mode: str) -> Dict[str, Any]:
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


def export_artifact(session_dir: Path, artifact_filename: str, destination: Path) -> None:
    source = session_dir / artifact_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def failure_reasons(record: Dict[str, Any]) -> List[str]:
    qa = record.get("qa", {})
    reasons: List[str] = []
    if not qa.get("loudness_ok", True):
        reasons.append("loudness")
    if not qa.get("stereo_ok", True):
        reasons.append("stereo")
    if not qa.get("true_peak_ok", True):
        reasons.append("true_peak")
    if not qa.get("crest_ok", True):
        reasons.append("crest")
    return reasons


def conservative_stem_gains(song_title: str) -> Dict[str, float]:
    if song_title == "In The Moment":
        return {"vocals": 0.45, "drums": 0.15, "bass": -0.08, "other": 0.05}
    return {"vocals": 0.35, "drums": 0.15, "bass": -0.10, "other": 0.05}


def build_targets() -> List[Dict[str, Any]]:
    qa_summary = load_json(PASS1_QA_PATH, {})
    pass1_tracks = qa_summary.get("tracks", {})
    targets: List[Dict[str, Any]] = []
    grouped: Dict[str, List[str]] = {}

    for track_key, record in pass1_tracks.items():
        if record.get("status") != "completed":
            continue
        if record.get("qa", {}).get("passed", False):
            continue

        canonical_title, _ = track_key.split("::", 1)
        grouped.setdefault(canonical_title, []).append(track_key)
        targets.append(
            {
                "track_key": track_key,
                "canonical_title": canonical_title,
                "source_file_name": record["source_file_name"],
                "relative_path": record["relative_path"],
                "planned_mode": record["planned_mode"],
                "previous_actual_mode": record["actual_mode"],
                "movement_amount": MOVEMENT_AMOUNT,
                "failure_reasons": failure_reasons(record),
                "pass1_record": record,
            }
        )

    targets.sort(key=lambda item: (sanitize_name(item["canonical_title"]), item["source_file_name"].lower()))

    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_of_truth": {
            "qa_summary": str(PASS1_QA_PATH),
        },
        "rerun_scope": "qa_failed_only",
        "track_count": len(targets),
        "groups": grouped,
        "tracks": [
            {
                "track_key": item["track_key"],
                "canonical_title": item["canonical_title"],
                "source_file_name": item["source_file_name"],
                "relative_path": item["relative_path"],
                "planned_mode": item["planned_mode"],
                "previous_actual_mode": item["previous_actual_mode"],
                "movement_amount": MOVEMENT_AMOUNT,
                "failure_reasons": item["failure_reasons"],
            }
            for item in targets
        ],
    }
    write_json(TARGETS_PATH, manifest)
    return targets


def normalize_request(request_dict: Dict[str, Any]) -> server.MasterRequest:
    payload = deepcopy(request_dict)
    audio_id = payload.pop("audio_id")
    settings = server.propose_master_settings(server.MasterSettings(**payload)).settings.model_dump()
    settings["audio_id"] = audio_id
    return server.MasterRequest(**settings)


def stereo_width_from_correlation(corr: float) -> float:
    if corr < 0.60:
        return 0.01
    if corr < 0.65:
        return 0.02
    if corr < 0.69:
        return 0.04
    if corr < 0.72:
        return 0.05
    return 0.06


def apply_stereo_recipe(task: Dict[str, Any], request_dict: Dict[str, Any]) -> List[str]:
    record = task["pass1_record"]
    song = task["canonical_title"]
    corr = float(record["final_metrics"]["stereo_correlation"])
    cp = deepcopy(request_dict.get("control_profile") or {})
    notes = ["stereo-recovery"]

    request_dict["stem_mode"] = "off"
    request_dict["stem_gains_db"] = None
    request_dict["enable_air_motion"] = False
    request_dict["governor_search_steps"] = max(int(request_dict.get("governor_search_steps") or 5), 6)
    request_dict["warmth"] = round(clamp(float(request_dict.get("warmth") or 0.04) + 0.03, 0.0, 0.14), 2)
    request_dict["transient_boost_db"] = round(
        clamp(float(request_dict.get("transient_boost_db") or 2.0) - 0.4, 1.5, 3.0),
        2,
    )

    cp["movement_amount"] = MOVEMENT_AMOUNT
    cp["spatial_width"] = stereo_width_from_correlation(corr)
    cp["brightness_tilt"] = round(
        clamp(min(float(cp.get("brightness_tilt") or 0.0), 0.02), -0.04, 0.08),
        2,
    )
    cp["harshness_control"] = round(
        clamp(float(cp.get("harshness_control") or 0.34) + 0.04, 0.2, 0.55),
        2,
    )
    cp["low_end_focus"] = round(
        clamp(float(cp.get("low_end_focus") or 0.62) + 0.02, 0.4, 0.8),
        2,
    )

    if song == "Got Too":
        request_dict["preset_name"] = "hi_fi_streaming"
        request_dict["target_lufs"] = -10.6
        request_dict["governor_gr_limit_db"] = -2.2
        cp["spatial_width"] = 0.02
        notes.append("got-too-hi-fi-width-clamp")
    elif song == "Last Time":
        request_dict["preset_name"] = "hi_fi_streaming"
        request_dict["target_lufs"] = -10.4
        request_dict["governor_gr_limit_db"] = -2.1
        cp["spatial_width"] = min(cp["spatial_width"], 0.03)
        notes.append("last-time-correlation-priority")
    elif song == "I'm Him":
        request_dict["preset_name"] = "radio_loud"
        request_dict["target_lufs"] = -10.2
        request_dict["governor_gr_limit_db"] = -2.6
        cp["spatial_width"] = min(cp["spatial_width"], 0.03)
        notes.append("im-him-full-mix-rescue")
    elif song == "Fire":
        request_dict["preset_name"] = "competitive_trap"
        request_dict["target_lufs"] = -9.7
        request_dict["governor_gr_limit_db"] = -3.1
        notes.append("fire-keep-aggression-tighten-image")
    else:
        request_dict["governor_gr_limit_db"] = min(float(request_dict.get("governor_gr_limit_db") or -2.4), -2.4)

    request_dict["control_profile"] = cp
    return notes


def apply_loudness_recipe(task: Dict[str, Any], request_dict: Dict[str, Any]) -> List[str]:
    record = task["pass1_record"]
    song = task["canonical_title"]
    lufs = float(record["final_metrics"]["integrated_lufs"])
    cp = deepcopy(request_dict.get("control_profile") or {})
    notes = ["loudness-recovery"]

    cp["movement_amount"] = MOVEMENT_AMOUNT
    cp["spatial_width"] = round(clamp(min(float(cp.get("spatial_width") or 0.1), 0.08), 0.02, 0.08), 2)
    cp["brightness_tilt"] = round(clamp(float(cp.get("brightness_tilt") or 0.02), -0.03, 0.04), 2)
    cp["harshness_control"] = round(
        clamp(float(cp.get("harshness_control") or 0.34) + 0.02, 0.2, 0.55),
        2,
    )
    cp["low_end_focus"] = round(
        clamp(float(cp.get("low_end_focus") or 0.62) + 0.01, 0.4, 0.8),
        2,
    )

    request_dict["warmth"] = round(clamp(float(request_dict.get("warmth") or 0.04) + 0.02, 0.0, 0.14), 2)
    request_dict["transient_boost_db"] = round(
        clamp(float(request_dict.get("transient_boost_db") or 2.0) - 0.5, 1.5, 3.0),
        2,
    )
    request_dict["governor_search_steps"] = max(int(request_dict.get("governor_search_steps") or 5), 6)

    if song == "Fall In Love":
        request_dict["preset_name"] = "radio_loud"
        request_dict["target_lufs"] = -9.8
        request_dict["governor_gr_limit_db"] = -2.8
        request_dict["stem_mode"] = "off"
        request_dict["stem_gains_db"] = None
        notes.append("fall-in-love-gentle-hotter")
    elif song == "Hot Shit":
        request_dict["preset_name"] = "competitive_trap"
        request_dict["target_lufs"] = -9.4
        request_dict["governor_gr_limit_db"] = -3.4
        request_dict["governor_search_steps"] = 7
        request_dict["stem_mode"] = "off"
        request_dict["stem_gains_db"] = None
        notes.append("hot-shit-density-push")
    elif song == "In The Moment":
        request_dict["preset_name"] = "radio_loud"
        request_dict["target_lufs"] = -10.0
        request_dict["governor_gr_limit_db"] = -2.8
        request_dict["stem_mode"] = "on"
        request_dict["stem_gains_db"] = conservative_stem_gains(song)
        request_dict["enable_air_motion"] = False
        cp["spatial_width"] = min(cp["spatial_width"], 0.07)
        notes.append("in-the-moment-stems-retry")
    elif song == "Project":
        request_dict["preset_name"] = "radio_loud"
        request_dict["target_lufs"] = -10.0
        request_dict["governor_gr_limit_db"] = -2.8
        request_dict["stem_mode"] = "off"
        request_dict["stem_gains_db"] = None
        request_dict["enable_air_motion"] = False
        cp["spatial_width"] = min(cp["spatial_width"], 0.06)
        notes.append("project-off-stems-hotter")
    else:
        # Mild generic push for anything else that slipped under the loudness gate.
        loudness_push = 1.2 if lufs <= -17.5 else 0.8
        request_dict["target_lufs"] = round(float(request_dict.get("target_lufs") or -10.4) + loudness_push, 2)
        request_dict["governor_gr_limit_db"] = min(float(request_dict.get("governor_gr_limit_db") or -2.4) - 0.4, -2.8)

    request_dict["control_profile"] = cp
    return notes


def build_requests_for_target(task: Dict[str, Any]) -> Dict[str, Any]:
    record = task["pass1_record"]
    reg = server.register_audio_from_path(task["relative_path"])
    audio_id = reg.audio_id
    source_metrics = server.analyze_audio(audio_id).model_dump()

    request_dict = deepcopy(record["request"])
    request_dict["audio_id"] = audio_id
    request_dict["bit_depth"] = "float32"
    request_dict["enable_harshness_limiter"] = True
    request_dict["enable_masking_eq"] = True
    request_dict["enable_hooklift"] = True
    recipe_notes: List[str] = []

    reasons = task["failure_reasons"]
    if "stereo" in reasons:
        recipe_notes.extend(apply_stereo_recipe(task, request_dict))
    if "loudness" in reasons:
        recipe_notes.extend(apply_loudness_recipe(task, request_dict))

    request_dict["control_profile"]["movement_amount"] = MOVEMENT_AMOUNT
    primary_request = normalize_request(request_dict)

    fallback_request = None
    if primary_request.stem_mode == "on":
        fallback_payload = deepcopy(primary_request.model_dump())
        fallback_payload["audio_id"] = audio_id
        fallback_payload["stem_mode"] = "off"
        fallback_payload["stem_gains_db"] = None
        fallback_cp = deepcopy(fallback_payload.get("control_profile") or {})
        fallback_cp["movement_amount"] = MOVEMENT_AMOUNT
        fallback_cp["spatial_width"] = round(clamp(min(float(fallback_cp.get("spatial_width") or 0.06), 0.05), 0.02, 0.05), 2)
        fallback_payload["control_profile"] = fallback_cp
        fallback_request = normalize_request(fallback_payload)

    return {
        "audio_id": audio_id,
        "source_metrics": source_metrics,
        "primary_request": primary_request,
        "fallback_request": fallback_request,
        "recipe_notes": sorted(set(recipe_notes)),
    }


def summarize_results(run_results: Dict[str, Any], session_key: str) -> Dict[str, Any]:
    completed_records = [value for value in run_results["tracks"].values() if value.get("status") == "completed"]
    failed_records = [value for value in run_results["tracks"].values() if value.get("status") == "failed"]
    fallback_records = [
        value
        for value in completed_records
        if value.get("requested_mode") == "stems" and value.get("actual_mode") == "off"
    ]
    return {
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
        "tracks": run_results["tracks"],
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MASTERS_DIR.mkdir(parents=True, exist_ok=True)

    session_key, session_dir_raw = server._get_session_info(None)
    session_dir = Path(session_dir_raw)

    reset_log(RERUN_LOG_PATH)
    append_log(f"Pass-2 reruns started in session `{session_key}` using storage `{server.STORAGE_DIR}`.")
    append_log("Stems-capable reruns are limited because stems renders are treated as ~7-minute jobs.")

    tasks = build_targets()
    append_log(f"Loaded {len(tasks)} pass-1 QA failures from `{PASS1_QA_PATH.name}`.")

    default_results = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "running",
        "pass_label": PASS_LABEL,
        "rerun_tag": RERUN_TAG,
        "source_pass_qa_summary": str(PASS1_QA_PATH),
        "session_key": session_key,
        "storage_dir": str(server.STORAGE_DIR),
        "tracks": {},
    }
    run_results = load_json(RERUN_RESULTS_PATH, default_results)
    run_results["status"] = "running"
    run_results["session_key"] = session_key
    run_results["storage_dir"] = str(server.STORAGE_DIR)
    run_results["updated_at"] = utc_now()

    pending: List[Dict[str, Any]] = []
    queue = tasks[:]

    while queue or pending:
        while queue and len(pending) < MAX_CONCURRENT_JOBS:
            task = queue.pop(0)
            prepared = build_requests_for_target(task)
            request = prepared["primary_request"]
            launch = server.run_master_job(request)
            pending.append(
                {
                    **task,
                    **prepared,
                    "job_id": launch.job_id,
                    "active_request": request,
                    "actual_mode": request.stem_mode,
                    "attempt": 1,
                }
            )
            run_results["tracks"][task["track_key"]] = {
                "status": "running",
                "pass_label": PASS_LABEL,
                "rerun_tag": RERUN_TAG,
                "job_id": launch.job_id,
                "track_key": task["track_key"],
                "canonical_title": task["canonical_title"],
                "source_file_name": task["source_file_name"],
                "relative_path": task["relative_path"],
                "failure_reasons": task["failure_reasons"],
                "requested_mode": "stems" if request.stem_mode == "on" else "no-stems",
                "actual_mode": request.stem_mode,
                "started_at": utc_now(),
                "recipe_notes": prepared["recipe_notes"],
                "source_metrics": prepared["source_metrics"],
                "request": request.model_dump(),
                "pass1_summary_path": task["pass1_record"].get("summary_path"),
                "pass1_exported_master_path": task["pass1_record"].get("exported_master_path"),
                "pass1_final_metrics": task["pass1_record"].get("final_metrics"),
            }
            write_json(RERUN_RESULTS_PATH, run_results)
            append_log(
                f"Launched rerun job `{launch.job_id}` for `{task['track_key']}` "
                f"using `{request.preset_name}` in mode `{'stems' if request.stem_mode == 'on' else 'no-stems'}`."
            )

        if not pending:
            break

        time.sleep(20)
        still_pending: List[Dict[str, Any]] = []

        for item in pending:
            status = server.job_status(item["job_id"])
            if status.status in {"queued", "running"}:
                still_pending.append(item)
                continue

            track_record = run_results["tracks"][item["track_key"]]

            if status.status == "error":
                error_message = status.error.message if status.error else "unknown_job_error"
                if item["fallback_request"] is not None and item["actual_mode"] == "on":
                    launch = server.run_master_job(item["fallback_request"])
                    item["job_id"] = launch.job_id
                    item["active_request"] = item["fallback_request"]
                    item["actual_mode"] = "off"
                    item["attempt"] += 1
                    item["fallback_reason"] = error_message
                    still_pending.append(item)
                    track_record.update(
                        {
                            "status": "running",
                            "job_id": launch.job_id,
                            "actual_mode": "off",
                            "attempt": item["attempt"],
                            "fallback_reason": error_message,
                            "request": item["fallback_request"].model_dump(),
                        }
                    )
                    write_json(RERUN_RESULTS_PATH, run_results)
                    append_log(
                        f"Stems rerun errored for `{item['track_key']}`; "
                        f"falling back to no-stems via job `{launch.job_id}`."
                    )
                    continue

                track_record.update({"status": "failed", "error": error_message, "finished_at": utc_now()})
                write_json(RERUN_RESULTS_PATH, run_results)
                append_log(f"Rerun job `{item['job_id']}` failed for `{item['track_key']}`: {error_message}")
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
            qa = quality_gate(final_metrics, "stems" if item["actual_mode"] == "on" else "no-stems")

            if item["fallback_request"] is not None and item["actual_mode"] == "on" and not qa["passed"]:
                launch = server.run_master_job(item["fallback_request"])
                item["job_id"] = launch.job_id
                item["active_request"] = item["fallback_request"]
                item["actual_mode"] = "off"
                item["attempt"] += 1
                item["fallback_reason"] = "pass2_stems_quality_gate_failed"
                still_pending.append(item)
                track_record.update(
                    {
                        "status": "running",
                        "job_id": launch.job_id,
                        "actual_mode": "off",
                        "attempt": item["attempt"],
                        "fallback_reason": "pass2_stems_quality_gate_failed",
                        "request": item["fallback_request"].model_dump(),
                    }
                )
                write_json(RERUN_RESULTS_PATH, run_results)
                append_log(
                    f"Pass-2 stems QA miss for `{item['track_key']}` "
                    f"(LUFS {final_metrics['integrated_lufs']:.2f}, corr {final_metrics['stereo_correlation']:.3f}); "
                    f"falling back to no-stems."
                )
                continue

            source_basename = Path(item["source_file_name"]).stem
            song_dir = MASTERS_DIR / item["canonical_title"]
            final_name = (
                f"{sanitize_name(item['canonical_title'])}"
                f"__src-{sanitize_name(source_basename)}"
                f"__mode-{'stems' if item['actual_mode'] == 'on' else 'no-stems'}"
                f"__movement-{MOVEMENT_AMOUNT:.2f}"
                f"__{PASS_LABEL}"
                f"__{RERUN_TAG}"
                "__mastered.wav"
            )
            summary_name = final_name.replace("__mastered.wav", "__summary.json")
            exported_path = song_dir / final_name
            export_artifact(session_dir, audio_artifact["filename"], exported_path)

            summary_payload = {
                "completed_at": utc_now(),
                "pass_label": PASS_LABEL,
                "rerun_tag": RERUN_TAG,
                "session_key": session_key,
                "job_id": item["job_id"],
                "track_key": item["track_key"],
                "source_file_name": item["source_file_name"],
                "relative_path": item["relative_path"],
                "canonical_title": item["canonical_title"],
                "requested_mode": track_record["requested_mode"],
                "actual_mode": item["actual_mode"],
                "attempt": item["attempt"],
                "failure_reasons": item["failure_reasons"],
                "fallback_reason": item.get("fallback_reason"),
                "recipe_notes": item["recipe_notes"],
                "audio_id": item["audio_id"],
                "request": item["active_request"].model_dump(),
                "source_metrics": item["source_metrics"],
                "final_metrics": final_metrics,
                "metrics_delta": delta,
                "qa": qa,
                "artifacts": artifacts,
                "pass1_summary_path": item["pass1_record"].get("summary_path"),
                "pass1_exported_master_path": item["pass1_record"].get("exported_master_path"),
                "pass1_final_metrics": item["pass1_record"].get("final_metrics"),
                "exported_master_path": str(exported_path),
            }
            write_json(song_dir / summary_name, summary_payload)

            track_record.update(
                {
                    "status": "completed",
                    "finished_at": utc_now(),
                    "attempt": item["attempt"],
                    "actual_mode": item["actual_mode"],
                    "fallback_reason": item.get("fallback_reason"),
                    "request": item["active_request"].model_dump(),
                    "final_metrics": final_metrics,
                    "metrics_delta": delta,
                    "qa": qa,
                    "artifacts": artifacts,
                    "summary_path": str(song_dir / summary_name),
                    "exported_master_path": str(exported_path),
                }
            )
            write_json(RERUN_RESULTS_PATH, run_results)
            append_log(
                f"Completed pass-2 rerun for `{item['track_key']}` -> `{final_name}` "
                f"(LUFS {final_metrics['integrated_lufs']:.2f}, TP {final_metrics['true_peak_dbtp']:.2f}, "
                f"crest {final_metrics['crest_db']:.2f}, corr {final_metrics['stereo_correlation']:.3f})."
            )

        pending = still_pending

    run_results["status"] = "completed"
    run_results["updated_at"] = utc_now()
    write_json(RERUN_RESULTS_PATH, run_results)

    qa_summary = summarize_results(run_results, session_key)
    write_json(RERUN_QA_PATH, qa_summary)
    append_log(
        f"Pass-2 reruns finished with {qa_summary['completed_tracks']} completed, "
        f"{qa_summary['failed_tracks']} failed, {qa_summary['fallback_to_no_stems']} stems fallbacks."
    )


if __name__ == "__main__":
    main()
