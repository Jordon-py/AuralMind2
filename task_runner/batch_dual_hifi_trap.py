from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import auralmind_maestro as am


DEFAULT_MANIFEST = ROOT_DIR / "task_runner" / "dual_hifi_manifest_20260311.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "masters" / "Dual_HiFi_Trap_Batch_20260311"
VERSION_DIRS = {
    "A": "version_A_reference",
    "B": "version_B_experiment",
}
VERSION_LABELS = {
    "A": "AuralMind2_HiFi_Ref",
    "B": "AuralMind2_HiFi_Exp",
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().replace(" ", "_")
    return cleaned or "track"


def load_manifest(path: Path) -> List[Path]:
    tracks: List[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tracks.append(Path(line))
    return tracks


def analyze_metrics(audio_path: Path) -> Dict[str, float]:
    y, sr = am.load_audio(str(audio_path))
    metrics = am.analyze_track_features(y, sr)
    return {key: float(value) for key, value in metrics.items()}


def build_track_base(index: int, source_path: Path, version_key: str) -> str:
    return f"{index:03d}__{sanitize_name(source_path.stem)}__{VERSION_LABELS[version_key]}"


def summarize_preset(preset: am.Preset) -> Dict[str, Any]:
    keys = [
        "name",
        "target_lufs",
        "ceiling_dbfs",
        "movement_amount",
        "hooklift_mix",
        "hooklift_auto_percentile",
        "warmth",
        "width_mid",
        "width_hi",
        "microshift_ms",
        "microshift_mix",
        "air_motion_rate_hz",
        "air_motion_depth_ms",
        "air_motion_mix",
        "air_motion_corr_floor",
        "transient_sculpt_boost_db",
        "transient_sculpt_mix",
        "transient_sculpt_crest_guard_db",
        "transient_sculpt_decay_ms",
        "microdetail_amount",
        "microdetail_mix",
        "governor_gr_limit_db",
        "governor_search_steps",
        "softclip_mix",
        "softclip_drive_db",
        "limiter_release_ms",
        "limiter_stereo_link",
        "harshness_threshold_db",
        "harshness_max_cut_db",
        "harshness_mix",
        "enable_stem_separation",
    ]
    return {key: getattr(preset, key) for key in keys}


def build_reference_preset(source_metrics: Dict[str, float]) -> Tuple[am.Preset, Dict[str, Any]]:
    base = am.get_presets()["competitive_trap"]
    crest = float(source_metrics["crest_db"])
    centroid = float(source_metrics["centroid_hz"])
    corr_hi = float(source_metrics["corr_hi"])
    tp_dbfs = float(source_metrics["tp_dbfs"])

    target_lufs = -13.0
    width_hi = 1.28
    microshift_mix = 0.17
    air_mix = 0.12
    air_corr_floor = 0.82
    if corr_hi > 0.90:
        width_hi = 1.30
        microshift_mix = 0.18
        air_mix = 0.14
        air_corr_floor = 0.80
    elif corr_hi < 0.28:
        width_hi = 1.24
        microshift_mix = 0.15
        air_mix = 0.10
        air_corr_floor = 0.84

    harshness_threshold = -14.2
    harshness_cut = 2.0
    harshness_mix = 0.60
    if centroid > 4200.0:
        harshness_threshold = -14.8
        harshness_cut = 2.4
        harshness_mix = 0.66

    if tp_dbfs > 2.0 or crest >= 13.5:
        enable_microdetail = False
        hooklift_mix = 0.12
        hooklift_percentile = 92.0
        transient_boost = 0.0
        transient_mix = 0.0
        transient_guard = 16.8
        transient_decay = 5.6
        warmth = 0.22
        governor_gr_limit = -1.6
    elif tp_dbfs > 1.0 or crest >= 11.5:
        enable_microdetail = False
        hooklift_mix = 0.14
        hooklift_percentile = 90.0
        transient_boost = 1.2
        transient_mix = 0.22
        transient_guard = 16.8
        transient_decay = 5.6
        warmth = 0.24
        governor_gr_limit = -1.4
    else:
        enable_microdetail = True
        hooklift_mix = 0.16
        hooklift_percentile = 90.0
        transient_boost = 1.6
        transient_mix = 0.26
        transient_guard = 17.2
        transient_decay = 5.8
        warmth = 0.26
        governor_gr_limit = -1.3

    microdetail_amount = clamp(0.20 + ((2600.0 - centroid) / 12000.0), 0.18, 0.23)
    microdetail_mix = clamp(0.60 + ((2600.0 - centroid) / 18000.0), 0.58, 0.64)

    preset = replace(
        base,
        name="dual_hifi_reference",
        bit_depth="float32",
        enable_stem_separation=False,
        target_lufs=target_lufs,
        ceiling_dbfs=-1.2,
        limiter_mode="v2",
        fir_streaming="auto",
        enable_mono_sub_v2=True,
        mono_sub_base_mix=0.56,
        enable_masking_eq=True,
        enable_microdetail=enable_microdetail,
        microdetail_amount=microdetail_amount,
        microdetail_mix=microdetail_mix,
        width_mid=1.06,
        width_hi=width_hi,
        microshift_ms=0.22,
        microshift_mix=microshift_mix,
        enable_air_motion=True,
        air_motion_rate_hz=0.35,
        air_motion_depth_ms=0.15,
        air_motion_mix=air_mix,
        air_motion_corr_floor=air_corr_floor,
        enable_movement=True,
        movement_amount=0.23,
        enable_hooklift=True,
        hooklift_auto=True,
        hooklift_auto_percentile=hooklift_percentile,
        hooklift_mix=hooklift_mix,
        enable_transient_sculpt=True,
        transient_sculpt_boost_db=transient_boost,
        transient_sculpt_mix=transient_mix,
        transient_sculpt_crest_guard_db=transient_guard,
        transient_sculpt_decay_ms=transient_decay,
        warmth=warmth,
        enable_harshness_limiter=True,
        harshness_threshold_db=harshness_threshold,
        harshness_max_cut_db=harshness_cut,
        harshness_mix=harshness_mix,
        governor_gr_limit_db=governor_gr_limit,
        governor_search_steps=11,
        softclip_mix=0.22,
        softclip_drive_db=1.0,
        limiter_release_ms=62.0,
        limiter_stereo_link=0.93,
    )
    details = {
        "focus": "Reference hi-fi trap line based on the proven competitive-trap hi-fi override lane.",
        "base_preset": "competitive_trap",
    }
    return preset, details


def build_experiment_preset(source_metrics: Dict[str, float]) -> Tuple[am.Preset, Dict[str, Any]]:
    base = am.get_presets()["competitive_trap"]
    crest = float(source_metrics["crest_db"])
    centroid = float(source_metrics["centroid_hz"])
    corr_hi = float(source_metrics["corr_hi"])
    tp_dbfs = float(source_metrics["tp_dbfs"])

    if tp_dbfs > 2.0 or crest >= 13.5:
        target_lufs = -12.4
        governor_gr_limit = -1.8
        transient_boost = 1.4
        transient_mix = 0.24
        movement_amount = 0.24
        hooklift_mix = 0.20
        hooklift_percentile = 84.0
        warmth = 0.18
    elif tp_dbfs > 1.0 or crest >= 11.5:
        target_lufs = -12.2
        governor_gr_limit = -1.9
        transient_boost = 1.8
        transient_mix = 0.30
        movement_amount = 0.25
        hooklift_mix = 0.24
        hooklift_percentile = 82.0
        warmth = 0.22
    else:
        target_lufs = -12.0
        governor_gr_limit = -2.1
        transient_boost = 2.2
        transient_mix = 0.34
        movement_amount = 0.27
        hooklift_mix = 0.26
        hooklift_percentile = 80.0
        warmth = 0.24

    if corr_hi > 0.90:
        movement_amount += 0.01
    elif corr_hi < 0.28:
        movement_amount -= 0.01
    movement_amount = clamp(movement_amount, 0.23, 0.29)

    if centroid > 4200.0:
        warmth = min(0.28, warmth + 0.03)

    microdetail_amount = clamp(0.20 + ((2600.0 - centroid) / 10000.0), 0.18, 0.26)
    microdetail_mix = clamp(0.58 + ((2600.0 - centroid) / 15000.0), 0.56, 0.66)

    width_hi = 1.30
    microshift_mix = 0.19
    air_mix = 0.15
    air_corr_floor = 0.80
    if corr_hi > 0.90:
        width_hi = 1.34
        microshift_mix = 0.22
        air_mix = 0.17
        air_corr_floor = 0.78
    elif corr_hi < 0.28:
        width_hi = 1.26
        microshift_mix = 0.16
        air_mix = 0.13
        air_corr_floor = 0.83

    harshness_threshold = -14.4
    harshness_cut = 2.2
    harshness_mix = 0.64
    if centroid > 4200.0:
        harshness_threshold = -15.0
        harshness_cut = 2.6
        harshness_mix = 0.70

    preset = replace(
        base,
        name="dual_hifi_experiment",
        bit_depth="float32",
        enable_stem_separation=False,
        target_lufs=target_lufs,
        ceiling_dbfs=-1.1,
        limiter_mode="v2",
        fir_streaming="auto",
        enable_mono_sub_v2=True,
        mono_sub_base_mix=0.52,
        enable_masking_eq=True,
        enable_microdetail=True,
        microdetail_amount=microdetail_amount,
        microdetail_mix=microdetail_mix,
        width_mid=1.08,
        width_hi=width_hi,
        microshift_ms=0.22,
        microshift_mix=microshift_mix,
        enable_air_motion=True,
        air_motion_rate_hz=0.33,
        air_motion_depth_ms=0.18,
        air_motion_mix=air_mix,
        air_motion_corr_floor=air_corr_floor,
        enable_movement=True,
        movement_amount=movement_amount,
        enable_hooklift=True,
        hooklift_auto=True,
        hooklift_auto_percentile=hooklift_percentile,
        hooklift_mix=hooklift_mix,
        enable_transient_sculpt=True,
        transient_sculpt_boost_db=transient_boost,
        transient_sculpt_mix=transient_mix,
        transient_sculpt_crest_guard_db=18.0,
        transient_sculpt_decay_ms=6.0,
        warmth=warmth,
        enable_harshness_limiter=True,
        harshness_threshold_db=harshness_threshold,
        harshness_max_cut_db=harshness_cut,
        harshness_mix=harshness_mix,
        governor_gr_limit_db=governor_gr_limit,
        governor_search_steps=11,
        softclip_mix=0.24,
        softclip_drive_db=1.1,
        limiter_release_ms=60.0,
        limiter_stereo_link=0.92,
        glow_mix=0.52,
        glow_drive_db=0.85,
    )
    details = {
        "focus": "Creative hi-fi experiment with adaptive width, air, hooklift, and slightly denser energy.",
        "base_preset": "competitive_trap",
    }
    return preset, details


def render_variant(
    index: int,
    source_path: Path,
    output_root: Path,
    version_key: str,
    source_metrics: Dict[str, float],
    force: bool,
) -> Dict[str, Any]:
    variant_dir = output_root / VERSION_DIRS[version_key]
    variant_dir.mkdir(parents=True, exist_ok=True)
    base = build_track_base(index, source_path, version_key)
    out_path = variant_dir / f"{base}.wav"
    report_path = variant_dir / f"{base}.md"

    if version_key == "A":
        preset, details = build_reference_preset(source_metrics)
    else:
        preset, details = build_experiment_preset(source_metrics)

    if out_path.exists() and report_path.exists() and not force:
        post_metrics = analyze_metrics(out_path)
        return {
            "version": version_key,
            "status": "skipped_existing",
            "focus": details["focus"],
            "base_preset": details["base_preset"],
            "preset_used": summarize_preset(preset),
            "output_path": str(out_path),
            "report_path": str(report_path),
            "post_metrics": post_metrics,
        }

    result = am.master(
        str(source_path),
        str(out_path),
        preset,
        report_path=str(report_path),
        out_subtype="FLOAT",
        dither=False,
        dither_seed=0,
    )
    post_metrics = analyze_metrics(out_path)
    return {
        "version": version_key,
        "status": "ok",
        "focus": details["focus"],
        "base_preset": details["base_preset"],
        "preset_used": summarize_preset(preset),
        "master_result": result,
        "output_path": str(out_path),
        "report_path": str(report_path),
        "post_metrics": post_metrics,
    }


def process_track(index: int, source_path_str: str, output_root_str: str, force: bool) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    source_path = Path(source_path_str)
    output_root = Path(output_root_str)

    if not source_path.exists():
        return {
            "index": index,
            "source": str(source_path),
            "status": "missing",
            "variants": [],
        }

    source_metrics = analyze_metrics(source_path)
    variants: List[Dict[str, Any]] = []
    status = "ok"
    for version_key in ("A", "B"):
        try:
            variants.append(render_variant(index, source_path, output_root, version_key, source_metrics, force))
        except Exception as exc:  # pragma: no cover - long-running batch error capture
            status = "partial" if variants else "error"
            variants.append(
                {
                    "version": version_key,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return {
        "index": index,
        "source": str(source_path),
        "status": status,
        "source_metrics": source_metrics,
        "variants": variants,
    }


def iter_tracks(tracks: Iterable[Path], max_tracks: int | None) -> List[Path]:
    track_list = list(tracks)
    if max_tracks is None or max_tracks <= 0:
        return track_list
    return track_list[:max_tracks]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual hi-fi trap batch mastering with reference + experiment variants.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Text manifest of source paths, one per line.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for rendered masters, reports, and summary JSON.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Track-level workers. Each worker renders both versions for one track.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="Optional smoke-test cap on tracks from the manifest. 0 means all tracks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if both output files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    tracks = iter_tracks(load_manifest(manifest_path), args.max_tracks)
    if not tracks:
        raise ValueError(f"No tracks found in manifest: {manifest_path}")

    summary: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "output_dir": str(output_root),
        "workers": int(args.workers),
        "tracks_total": len(tracks),
        "variants": VERSION_LABELS,
        "tracks": [],
    }

    future_map: Dict[cf.Future[Dict[str, Any]], Tuple[int, Path]] = {}
    # Windows workspace sandboxing can block ProcessPool IPC handles. Threads are
    # less isolated, but they work reliably here and still parallelize the heavy
    # NumPy/SciPy-backed DSP stages well enough for batch rendering.
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for index, source_path in enumerate(tracks, start=1):
            future = pool.submit(process_track, index, str(source_path), str(output_root), bool(args.force))
            future_map[future] = (index, source_path)

        for future in cf.as_completed(future_map):
            index, source_path = future_map[future]
            result = future.result()
            summary["tracks"].append(result)

            variant_states = ", ".join(
                f"{variant.get('version')}={variant.get('status')}" for variant in result.get("variants", [])
            )
            print(f"[done] {index:03d} {source_path.name} :: {result['status']} :: {variant_states}")

    summary["tracks"].sort(key=lambda item: int(item["index"]))
    summary_path = output_root / "dual_hifi_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
