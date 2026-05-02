from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import auralmind_maestro as am

SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff", ".m4a"}
LOSSLESS_EXTS = {".wav", ".flac", ".aif", ".aiff"}
EXCLUDED_ROOT_DIRS = {"codex_mastered"}

EXCLUDE_PATTERNS = [
    re.compile(r"\bprobe\b", re.IGNORECASE),
    re.compile(r"\breport\b", re.IGNORECASE),
    re.compile(r"\banalysis\b", re.IGNORECASE),
    re.compile(r"\bauralmind\d*\b", re.IGNORECASE),
    re.compile(r"(?<!pre)master(?:ed)?(?:\d+)?\b", re.IGNORECASE),
    re.compile(r"\bcompat\b", re.IGNORECASE),
    re.compile(r"\bpass\d+\b", re.IGNORECASE),
    re.compile(r"\bfloat(?:32|64)?\b", re.IGNORECASE),
    re.compile(r"\b(?:24|32)bit\b", re.IGNORECASE),
    re.compile(r"\bauto sub\b", re.IGNORECASE),
    re.compile(r"\bno stems\b", re.IGNORECASE),
    re.compile(r"\bpremium\b", re.IGNORECASE),
    re.compile(r"\b(?:clean|punchy)\b", re.IGNORECASE),
    re.compile(r"\bmov\d+\b", re.IGNORECASE),
    re.compile(r"\bhl\d+\b", re.IGNORECASE),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().replace(" ", "_")
    return cleaned or "track"


def slugify(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-")
    return lowered or "track"


def find_ffmpeg() -> Optional[Path]:
    from_path = shutil.which("ffmpeg")
    if from_path:
        return Path(from_path)

    user_profile = Path.home()
    winget_root = user_profile / "AppData" / "Local" / "Microsoft" / "WinGet"
    direct_candidates = [winget_root / "Links" / "ffmpeg.exe"]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    pkg_root = winget_root / "Packages"
    if pkg_root.exists():
        for candidate in pkg_root.glob("Gyan.FFmpeg.*/*/bin/ffmpeg.exe"):
            if candidate.exists():
                return candidate
    return None


def convert_to_wav(source_path: Path, ffmpeg_path: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{sanitize_name(source_path.stem)}.wav"
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg conversion failed for '{source_path}': {(proc.stderr or '').strip()}")
    return out_path


def is_excluded_source(path: Path, data_dir: Path) -> Tuple[bool, Optional[str]]:
    rel = path.relative_to(data_dir)
    if rel.parts and rel.parts[0].lower() in EXCLUDED_ROOT_DIRS:
        return True, f"excluded_root:{rel.parts[0]}"

    lowered = path.stem.lower()
    normalized = re.sub(r"[_\-]+", " ", lowered)
    for pattern in EXCLUDE_PATTERNS:
        if pattern.search(lowered) or pattern.search(normalized):
            return True, f"matched:{pattern.pattern}"
    return False, None


def canonical_title_from_name(name: str) -> str:
    cleaned = name
    cleaned = re.sub(r"^\(edit\)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_ -]*premaster.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_ -]*v\d{2,}$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-")
    return cleaned or name.strip()


def candidate_rank(path: Path) -> Tuple[int, int, int]:
    lowered = path.stem.lower()
    ext = path.suffix.lower()

    ext_score = 0
    if ext == ".wav":
        ext_score = 40
    elif ext == ".flac":
        ext_score = 36
    elif ext in {".aif", ".aiff"}:
        ext_score = 34
    elif ext == ".m4a":
        ext_score = 18
    elif ext == ".mp3":
        ext_score = 10
    elif ext == ".ogg":
        ext_score = 8

    intent_score = 0
    if "premaster" in lowered:
        intent_score += 18
    if lowered.startswith("(edit)"):
        intent_score += 4
    if re.search(r"\(\d+\)\s*$", path.stem):
        intent_score -= 2

    size_score = int(path.stat().st_size // 1024)
    return ext_score + intent_score, int(path.stat().st_mtime), size_score


def discover_candidates(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_by_title: Dict[str, Dict[str, Any]] = {}
    excluded: List[Dict[str, Any]] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        excluded_flag, reason = is_excluded_source(path, data_dir)
        if excluded_flag:
            excluded.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(data_dir)),
                    "reason": reason,
                }
            )
            continue

        canonical_title = canonical_title_from_name(path.stem)
        candidate = {
            "canonical_title": canonical_title,
            "source_file_name": path.name,
            "relative_path": str(path.relative_to(data_dir)),
            "path": path,
            "rank": candidate_rank(path),
            "size_bytes": path.stat().st_size,
        }

        previous = selected_by_title.get(canonical_title)
        if previous is None or candidate["rank"] > previous["rank"]:
            if previous is not None:
                excluded.append(
                    {
                        "path": str(previous["path"]),
                        "relative_path": previous["relative_path"],
                        "reason": f"shadowed_by_higher_rank:{path.name}",
                    }
                )
            selected_by_title[canonical_title] = candidate
        else:
            excluded.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(data_dir)),
                    "reason": f"lower_rank_than_selected:{selected_by_title[canonical_title]['source_file_name']}",
                }
            )

    selected = sorted(selected_by_title.values(), key=lambda item: item["canonical_title"].lower())
    return selected, excluded


def analyze_metrics(audio_path: Path) -> Dict[str, float]:
    y, sr = am.load_audio(str(audio_path))
    return am.analyze_track_features(y, sr)


def choose_base_preset(source_metrics: Dict[str, float]) -> str:
    centroid = float(source_metrics.get("centroid_hz", 1800.0))
    crest = float(source_metrics.get("crest_db", 10.0))
    lufs = float(source_metrics.get("lufs", -16.0))

    if centroid >= 2200.0 or crest <= 9.2 or lufs >= -12.5:
        return "hi_fi_streaming"
    return "radio_loud"


def build_preset(
    source_metrics: Dict[str, float],
    pass_index: int,
    prev_out_metrics: Optional[Dict[str, float]] = None,
) -> am.Preset:
    presets = am.get_presets()
    base_name = choose_base_preset(source_metrics)
    base = presets[base_name]

    centroid = float(source_metrics.get("centroid_hz", 1800.0))
    crest = float(source_metrics.get("crest_db", 10.0))
    corr_lo = float(source_metrics.get("corr_lo", 0.9))

    if base_name == "radio_loud":
        target_lufs = -10.5 if crest >= 10.0 else -10.3
        governor_gr_limit = -2.2 if crest >= 11.5 else -2.5
    else:
        target_lufs = -10.9 if crest >= 10.0 else -10.7
        governor_gr_limit = -1.8 if crest >= 11.5 else -2.1

    warmth = 0.10 if centroid < 1200.0 else 0.07 if centroid < 1800.0 else 0.03
    width_hi = 1.18 if corr_lo < 0.9 else 1.22
    width_mid = 1.02
    microshift_mix = 0.13 if base_name == "hi_fi_streaming" else 0.15
    air_mix = 0.08 if centroid > 2400.0 else 0.10
    harshness_mix = 0.68 if centroid > 2200.0 else 0.58
    transient_boost = 1.9 if crest > 12.5 else 2.1 if crest > 9.0 else 2.3

    if prev_out_metrics is not None and pass_index > 1:
        out_lufs = float(prev_out_metrics.get("lufs", target_lufs))
        out_tp = float(prev_out_metrics.get("tp_dbfs", -1.0))
        out_crest = float(prev_out_metrics.get("crest_db", crest))
        out_corr_lo = float(prev_out_metrics.get("corr_lo", corr_lo))
        out_corr_hi = float(prev_out_metrics.get("corr_hi", 0.8))

        if out_lufs < target_lufs - 0.8:
            target_lufs = clamp(target_lufs + 0.4, -11.4, -10.0)
        elif out_lufs > target_lufs + 0.6:
            target_lufs = clamp(target_lufs - 0.3, -11.4, -10.0)

        if out_tp > -0.9:
            target_lufs = clamp(target_lufs - 0.3, -11.4, -10.0)
            governor_gr_limit = clamp(governor_gr_limit + 0.2, -2.8, -1.4)

        if out_crest < 8.0:
            transient_boost = clamp(transient_boost + 0.3, 1.8, 2.8)
            governor_gr_limit = clamp(governor_gr_limit + 0.2, -2.8, -1.4)
        elif out_crest > 12.8:
            transient_boost = clamp(transient_boost - 0.2, 1.6, 2.8)

        if out_corr_lo < 0.88:
            width_hi = clamp(width_hi - 0.03, 1.14, 1.22)
            microshift_mix = clamp(microshift_mix - 0.02, 0.10, 0.16)

        if out_corr_hi < 0.08:
            width_hi = clamp(width_hi - 0.03, 1.14, 1.22)
        elif out_corr_hi > 0.96:
            width_hi = clamp(width_hi + 0.02, 1.14, 1.24)

    return replace(
        base,
        name="premium_no_stems_melodic",
        stem_mode="off",
        target_lufs=target_lufs,
        governor_gr_limit_db=governor_gr_limit,
        governor_iters=4,
        warmth=warmth,
        width_mid=width_mid,
        width_hi=width_hi,
        microshift_ms=0.18,
        microshift_mix=microshift_mix,
        air_motion_mix=air_mix,
        air_motion_corr_floor=0.84,
        harshness_mix=harshness_mix,
        transient_sculpt_boost_db=transient_boost,
        transient_sculpt_mix=0.33,
        hooklift_mix=0.19,
        movement_amount=0.21,
        mono_sub_base_mix=0.60,
        demucs_device="cpu",
    )


def evaluate_master_metrics(metrics: Dict[str, float], target_lufs: float) -> Tuple[bool, Dict[str, bool], float]:
    lufs = float(metrics.get("lufs", -99.0))
    tp = float(metrics.get("tp_dbfs", 0.0))
    crest = float(metrics.get("crest_db", 0.0))
    corr_hi = float(metrics.get("corr_hi", 0.0))
    corr_lo = float(metrics.get("corr_lo", 0.0))

    checks = {
        "lufs_window": abs(lufs - target_lufs) <= 0.9,
        "true_peak_safe": tp <= -0.9,
        "crest_window": 8.0 <= crest <= 12.8,
        "mono_sub_safe": corr_lo >= 0.88,
        "stereo_safe": 0.08 <= corr_hi <= 0.96,
    }
    passed = all(checks.values())

    score = 0.0
    score += abs(lufs - target_lufs) * 2.0
    score += max(0.0, tp + 0.9) * 6.0
    if crest < 8.0:
        score += (8.0 - crest) * 1.8
    if crest > 12.8:
        score += (crest - 12.8) * 1.2
    if corr_lo < 0.88:
        score += (0.88 - corr_lo) * 18.0
    if corr_hi < 0.08:
        score += (0.08 - corr_hi) * 10.0
    if corr_hi > 0.96:
        score += (corr_hi - 0.96) * 8.0
    return passed, checks, score


def export_manifest(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def upsert_track_record(summary: Dict[str, Any], record: Dict[str, Any]) -> None:
    relative_path = str(record.get("relative_path") or "")
    tracks = list(summary.get("tracks") or [])
    replaced = False
    for idx, existing in enumerate(tracks):
        if str(existing.get("relative_path") or "") == relative_path:
            tracks[idx] = record
            replaced = True
            break
    if not replaced:
        tracks.append(record)
    summary["tracks"] = tracks


def run_batch(
    data_dir: Path,
    out_dir: Path,
    max_passes: int,
    limit: Optional[int],
    select_only: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / ".tmp_inputs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = find_ffmpeg()
    selection_path = out_dir / "selection_manifest.json"
    existing_selection = load_json(selection_path, None)
    if existing_selection and existing_selection.get("selected"):
        selected = []
        for item in existing_selection["selected"]:
            selected.append(
                {
                    "canonical_title": item["canonical_title"],
                    "source_file_name": item["source_file_name"],
                    "relative_path": item["relative_path"],
                    "path": data_dir / item["relative_path"],
                    "rank": tuple(item.get("rank") or (0, 0, 0)),
                    "size_bytes": item.get("size_bytes") or 0,
                }
            )
        excluded = list(existing_selection.get("excluded") or [])
    else:
        selected, excluded = discover_candidates(data_dir)
        export_manifest(
            selection_path,
            {
                "generated_at": utc_now(),
                "selected": [
                    {
                        "canonical_title": item["canonical_title"],
                        "source_file_name": item["source_file_name"],
                        "relative_path": item["relative_path"],
                        "rank": list(item["rank"]),
                        "size_bytes": item["size_bytes"],
                    }
                    for item in selected
                ],
                "excluded": excluded,
            },
        )

    summary_path = out_dir / "run_summary.json"
    summary: Dict[str, Any] = load_json(
        summary_path,
        {
            "profile": "premium_no_stems_melodic",
            "generated_at": utc_now(),
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "constraints": {
                "stem_mode": "off",
                "movement_amount": 0.21,
                "target": "melodic_clarity_vocal_presence",
                "mono_sub_discipline": True,
                "max_passes": int(max_passes),
            },
            "tracks_total": len(selected),
            "tracks": [],
            "excluded_count": len(excluded),
        },
    )
    summary["tracks_total"] = len(selected)
    summary["excluded_count"] = len(excluded)
    summary["constraints"]["max_passes"] = int(max_passes)

    completed_rel_paths = {
        str(track.get("relative_path") or "")
        for track in summary.get("tracks") or []
        if str(track.get("status") or "") == "completed"
    }
    selected = [item for item in selected if item["relative_path"] not in completed_rel_paths]
    if limit is not None:
        selected = selected[:limit]

    if select_only:
        summary["selection_only"] = True
        return summary

    if not selected:
        raise RuntimeError(f"No candidate source songs found in '{data_dir}'.")

    for idx, item in enumerate(selected, start=1):
        source_path = Path(item["path"])
        print(f"[{idx}/{len(selected)}] processing: {source_path.name}")
        input_path = source_path
        converted = False
        pass_records: List[Dict[str, Any]] = []
        track_slug = slugify(item["canonical_title"])

        try:
            if source_path.suffix.lower() == ".m4a":
                if ffmpeg_path is None:
                    upsert_track_record(
                        summary,
                        {
                            "source": str(source_path),
                            "relative_path": item["relative_path"],
                            "canonical_title": item["canonical_title"],
                            "status": "failed",
                            "error": "ffmpeg_not_found_for_m4a",
                        },
                    )
                    export_manifest(summary_path, summary)
                    continue
                input_path = convert_to_wav(source_path, ffmpeg_path, temp_dir)
                converted = True

            source_metrics = analyze_metrics(input_path)
            best_score = float("inf")
            best_pass: Optional[int] = None
            best_pass_path: Optional[Path] = None
            best_metrics: Optional[Dict[str, float]] = None
            best_result: Optional[Dict[str, Any]] = None
            prev_metrics: Optional[Dict[str, float]] = None

            for pass_index in range(1, max_passes + 1):
                preset = build_preset(source_metrics, pass_index, prev_metrics)
                pass_out_path = out_dir / "raw_passes" / f"{track_slug}__pass{pass_index}.wav"
                pass_report_path = out_dir / "reports" / f"{track_slug}__pass{pass_index}.md"
                pass_out_path.parent.mkdir(parents=True, exist_ok=True)
                pass_report_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    result = am.master(
                        str(input_path),
                        str(pass_out_path),
                        preset,
                        report_path=str(pass_report_path),
                        out_subtype="PCM_24",
                        dither=False,
                        dither_seed=0,
                    )
                    out_metrics = analyze_metrics(pass_out_path)
                    passed, checks, score = evaluate_master_metrics(out_metrics, float(preset.target_lufs))
                    pass_record = {
                        "pass": pass_index,
                        "preset_base": choose_base_preset(source_metrics),
                        "preset_used": {
                            "target_lufs": preset.target_lufs,
                            "governor_gr_limit_db": preset.governor_gr_limit_db,
                            "movement_amount": preset.movement_amount,
                            "warmth": preset.warmth,
                            "width_mid": preset.width_mid,
                            "width_hi": preset.width_hi,
                            "microshift_mix": preset.microshift_mix,
                            "air_motion_mix": preset.air_motion_mix,
                            "hooklift_mix": preset.hooklift_mix,
                            "mono_sub_base_mix": preset.mono_sub_base_mix,
                            "stem_mode": preset.stem_mode,
                        },
                        "master_result": result,
                        "post_metrics": out_metrics,
                        "checks": checks,
                        "passed": passed,
                        "score": score,
                        "pass_output": str(pass_out_path),
                        "pass_report": str(pass_report_path),
                    }
                except Exception as exc:
                    pass_record = {
                        "pass": pass_index,
                        "status": "failed",
                        "error": str(exc),
                        "pass_output": str(pass_out_path),
                        "pass_report": str(pass_report_path),
                    }
                    pass_records.append(pass_record)
                    continue

                pass_records.append(pass_record)

                if score < best_score:
                    best_score = score
                    best_pass = pass_index
                    best_pass_path = pass_out_path
                    best_metrics = out_metrics
                    best_result = result

                prev_metrics = out_metrics
                if passed:
                    break

            if best_pass is None or best_pass_path is None or best_metrics is None or best_result is None:
                upsert_track_record(
                    summary,
                    {
                        "source": str(source_path),
                        "relative_path": item["relative_path"],
                        "canonical_title": item["canonical_title"],
                        "status": "failed",
                        "error": "no_successful_pass",
                        "passes": pass_records,
                    },
                )
                export_manifest(summary_path, summary)
                continue

            song_dir = out_dir / "masters" / track_slug
            song_dir.mkdir(parents=True, exist_ok=True)
            final_name = (
                f"{track_slug}__src-{slugify(source_path.stem)}"
                "__profile-premium-no-stems-melodic"
                "__movement-0.21"
                "__mastered.wav"
            )
            final_out = song_dir / final_name
            shutil.copyfile(best_pass_path, final_out)

            compat_src = best_pass_path.with_name(best_pass_path.stem + "_compat.wav")
            compat_dst = final_out.with_name(final_out.stem + "_compat.wav")
            if compat_src.exists():
                shutil.copyfile(compat_src, compat_dst)

            summary_record = {
                "source": str(source_path),
                "relative_path": item["relative_path"],
                "canonical_title": item["canonical_title"],
                "input_used": str(input_path),
                "input_was_converted": converted,
                "source_metrics": source_metrics,
                "best_pass": best_pass,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "best_result": best_result,
                "final_output": str(final_out),
                "compat_output": str(compat_dst) if compat_dst.exists() else None,
                "passes": pass_records,
                "status": "completed",
            }
            upsert_track_record(summary, summary_record)
            export_manifest(song_dir / f"{final_out.stem}__summary.json", summary_record)
            export_manifest(summary_path, summary)
        except Exception as exc:
            upsert_track_record(
                summary,
                {
                    "source": str(source_path),
                    "relative_path": item["relative_path"],
                    "canonical_title": item["canonical_title"],
                    "status": "failed",
                    "error": str(exc),
                    "passes": pass_records,
                },
            )
            export_manifest(summary_path, summary)

    summary["remaining_tracks"] = max(
        0,
        len(
            [
                item
                for item in load_json(selection_path, {}).get("selected", [])
                if str(item.get("relative_path") or "")
                not in {
                    str(track.get("relative_path") or "")
                    for track in summary.get("tracks") or []
                    if str(track.get("status") or "") == "completed"
                }
            ]
        ),
    )
    export_manifest(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch premium no-stems melodic mastering run across curated songs in data/."
    )
    parser.add_argument("--data-dir", default="data", help="Input data directory to scan for songs.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to masters/<timestamped-batch>.")
    parser.add_argument("--max-passes", type=int, default=1, help="Maximum mastering passes per song.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for the number of selected songs.")
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Only build the curated source selection manifest without rendering masters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Input data directory does not exist: {data_dir}", file=sys.stderr)
        return 2

    if args.out_dir:
        out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (root / "masters" / f"premium_no_stems_melodic_batch_{stamp}").resolve()

    summary = run_batch(
        data_dir=data_dir,
        out_dir=out_dir,
        max_passes=max(1, int(args.max_passes)),
        limit=args.limit,
        select_only=bool(args.select_only),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
