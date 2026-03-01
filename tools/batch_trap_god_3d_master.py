from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import auralmind_maestro as am

SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff", ".m4a"}
NATIVE_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().replace(" ", "_")
    return cleaned or "track"


def find_ffmpeg() -> Optional[Path]:
    from_path = shutil.which("ffmpeg")
    if from_path:
        return Path(from_path)

    user_profile = Path.home()
    winget_root = user_profile / "AppData" / "Local" / "Microsoft" / "WinGet"
    direct_candidates = [
        winget_root / "Links" / "ffmpeg.exe",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    pkg_root = winget_root / "Packages"
    if pkg_root.exists():
        for candidate in pkg_root.glob("Gyan.FFmpeg.*/*/bin/ffmpeg.exe"):
            if candidate.exists():
                return candidate
    return None


def discover_audio_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        rel = path.relative_to(data_dir)
        if rel.parts and rel.parts[0].lower() == "codex_mastered":
            continue
        files.append(path)
    return files


def convert_to_wav(source_path: Path, ffmpeg_path: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_name = sanitize_name(source_path.stem) + ".wav"
    out_path = temp_dir / out_name
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
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg conversion failed for '{source_path}': {stderr}")
    return out_path


def analyze_metrics(audio_path: Path) -> Dict[str, float]:
    y, sr = am.load_audio(str(audio_path))
    return am.analyze_track_features(y, sr)


def evaluate_master_metrics(metrics: Dict[str, float]) -> Tuple[bool, Dict[str, bool], float]:
    checks = {
        "lufs_window": -12.2 <= float(metrics["lufs"]) <= -10.2,
        "true_peak_safe": float(metrics["tp_dbfs"]) <= -0.8,
        "crest_window": 7.2 <= float(metrics["crest_db"]) <= 13.5,
        "stereo_safe": 0.12 <= float(metrics["corr_hi"]) <= 0.97,
    }
    passed = all(checks.values())

    score = 0.0
    score += abs(float(metrics["lufs"]) - (-10.9)) * 2.2
    score += max(0.0, float(metrics["tp_dbfs"]) + 0.8) * 8.0
    if float(metrics["crest_db"]) < 7.2:
        score += (7.2 - float(metrics["crest_db"])) * 2.0
    if float(metrics["crest_db"]) > 13.5:
        score += (float(metrics["crest_db"]) - 13.5) * 1.0
    if float(metrics["corr_hi"]) < 0.12:
        score += (0.12 - float(metrics["corr_hi"])) * 10.0
    if float(metrics["corr_hi"]) > 0.97:
        score += (float(metrics["corr_hi"]) - 0.97) * 10.0
    return passed, checks, score


def build_trap_god_3d_preset(
    source_metrics: Dict[str, float],
    pass_index: int,
    prev_out_metrics: Optional[Dict[str, float]] = None,
) -> am.Preset:
    presets = am.get_presets()
    auto_name = am.auto_select_preset_name(source_metrics)
    if auto_name == "hi_fi_streaming":
        base = presets["hi_fi_streaming"]
    else:
        base = presets["competitive_trap"]

    centroid = float(source_metrics["centroid_hz"])
    crest = float(source_metrics["crest_db"])
    corr_hi = float(source_metrics["corr_hi"])

    if crest >= 12.0:
        target_lufs = -11.6
        governor_gr_limit = -1.0
    elif crest <= 8.5:
        target_lufs = -10.6
        governor_gr_limit = -2.1
    else:
        target_lufs = -10.9
        governor_gr_limit = -1.4

    warmth = clamp(0.26 + ((3000.0 - centroid) / 9000.0), 0.12, 0.42)
    microdetail_amount = clamp(0.20 + ((2600.0 - centroid) / 9000.0), 0.17, 0.30)
    microdetail_mix = clamp(0.60 + ((2600.0 - centroid) / 14000.0), 0.56, 0.70)

    width_hi = 1.30
    microshift_mix = 0.19
    air_mix = 0.13
    air_corr_floor = 0.80
    if corr_hi > 0.88:
        width_hi = 1.36
        microshift_mix = 0.22
        air_mix = 0.16
        air_corr_floor = 0.78
    elif corr_hi < 0.25:
        width_hi = 1.24
        microshift_mix = 0.15
        air_mix = 0.10
        air_corr_floor = 0.84

    harshness_threshold = -14.0
    harshness_cut = 2.0
    harshness_mix = 0.60
    if centroid > 4300.0:
        harshness_threshold = -15.0
        harshness_cut = 2.5
        harshness_mix = 0.70

    transient_boost = 2.2
    transient_mix = 0.36
    if crest < 8.5:
        transient_boost = 2.8
        transient_mix = 0.42
    elif crest > 12.0:
        transient_boost = 1.8
        transient_mix = 0.30

    if prev_out_metrics is not None and pass_index > 1:
        out_lufs = float(prev_out_metrics["lufs"])
        out_tp = float(prev_out_metrics["tp_dbfs"])
        out_crest = float(prev_out_metrics["crest_db"])
        out_corr = float(prev_out_metrics["corr_hi"])

        if out_lufs < -12.2:
            target_lufs = clamp(target_lufs + 0.5, -12.2, -10.2)
        elif out_lufs > -10.2:
            target_lufs = clamp(target_lufs - 0.5, -12.2, -10.2)

        if out_tp > -0.8:
            target_lufs = clamp(target_lufs - 0.3, -12.4, -10.2)

        if out_crest < 7.2:
            transient_boost = clamp(transient_boost + 0.5, 1.6, 3.4)
            transient_mix = clamp(transient_mix + 0.05, 0.28, 0.48)
            target_lufs = clamp(target_lufs - 0.3, -12.4, -10.2)
        elif out_crest > 13.5:
            transient_boost = clamp(transient_boost - 0.3, 1.4, 3.4)

        if out_corr < 0.12:
            width_hi = clamp(width_hi - 0.08, 1.18, 1.40)
            microshift_mix = clamp(microshift_mix - 0.03, 0.10, 0.24)
            air_mix = clamp(air_mix - 0.02, 0.07, 0.18)
        elif out_corr > 0.97:
            width_hi = clamp(width_hi + 0.06, 1.18, 1.40)
            microshift_mix = clamp(microshift_mix + 0.03, 0.10, 0.24)
            air_mix = clamp(air_mix + 0.02, 0.07, 0.18)

    return replace(
        base,
        name="trap_god_3d",
        bit_depth="float64",
        target_lufs=target_lufs,
        ceiling_dbfs=-1.0,
        governor_gr_limit_db=governor_gr_limit,
        movement_amount=0.22,
        enable_movement=True,
        enable_air_motion=True,
        air_motion_rate_hz=0.33,
        air_motion_depth_ms=0.18,
        air_motion_mix=air_mix,
        air_motion_corr_floor=air_corr_floor,
        width_mid=1.08,
        width_hi=width_hi,
        microshift_ms=0.22,
        microshift_mix=microshift_mix,
        warmth=warmth,
        hooklift_mix=0.24,
        hooklift_auto=True,
        hooklift_auto_percentile=74.0,
        microdetail_amount=microdetail_amount,
        microdetail_mix=microdetail_mix,
        enable_harshness_limiter=True,
        harshness_threshold_db=harshness_threshold,
        harshness_max_cut_db=harshness_cut,
        harshness_mix=harshness_mix,
        transient_sculpt_boost_db=transient_boost,
        transient_sculpt_mix=transient_mix,
        transient_sculpt_crest_guard_db=18.0,
        transient_sculpt_decay_ms=6.0,
        demucs_overlap=0.23,
        demucs_shifts=1,
        demucs_device="cpu",
    )


def relative_stem_name(data_dir: Path, source_path: Path) -> str:
    rel = source_path.relative_to(data_dir)
    stem_rel = rel.with_suffix("")
    return sanitize_name("__".join(stem_rel.parts))


def run_batch(data_dir: Path, out_dir: Path, max_passes: int) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / ".tmp_inputs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = find_ffmpeg()
    files = discover_audio_files(data_dir)
    if not files:
        raise RuntimeError(f"No supported audio files found in '{data_dir}'.")

    summary: Dict[str, Any] = {
        "profile": "trap_god_3d",
        "movement_amount_locked": 0.22,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "max_passes": int(max_passes),
        "tracks_total": len(files),
        "tracks": [],
    }

    for idx, source_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] processing: {source_path}")
        input_path = source_path
        converted = False
        if source_path.suffix.lower() not in NATIVE_EXTS:
            if source_path.suffix.lower() == ".m4a":
                if ffmpeg_path is None:
                    raise RuntimeError(
                        f"Cannot process '{source_path}': m4a requires ffmpeg, but ffmpeg was not found."
                    )
                input_path = convert_to_wav(source_path, ffmpeg_path, temp_dir)
                converted = True
            else:
                continue

        source_metrics = analyze_metrics(input_path)
        track_key = relative_stem_name(data_dir, source_path)
        best_score = float("inf")
        best_pass: Optional[int] = None
        best_pass_path: Optional[Path] = None
        best_metrics: Optional[Dict[str, float]] = None
        prev_metrics: Optional[Dict[str, float]] = None
        pass_records: List[Dict[str, Any]] = []

        for pass_index in range(1, max_passes + 1):
            preset = build_trap_god_3d_preset(source_metrics, pass_index, prev_metrics)
            pass_out_path = out_dir / f"{track_key}_trap_god_3d_pass{pass_index}.wav"
            pass_report_path = out_dir / f"{track_key}_trap_god_3d_pass{pass_index}.md"
            result = am.master(
                str(input_path),
                str(pass_out_path),
                preset,
                report_path=str(pass_report_path),
                out_subtype=None,
                dither=False,
                dither_seed=0,
            )
            out_metrics = analyze_metrics(pass_out_path)
            passed, checks, score = evaluate_master_metrics(out_metrics)
            pass_record = {
                "pass": pass_index,
                "preset_name": preset.name,
                "preset_used": {
                    "target_lufs": preset.target_lufs,
                    "governor_gr_limit_db": preset.governor_gr_limit_db,
                    "movement_amount": preset.movement_amount,
                    "warmth": preset.warmth,
                    "width_hi": preset.width_hi,
                    "microshift_mix": preset.microshift_mix,
                    "air_motion_mix": preset.air_motion_mix,
                    "air_motion_corr_floor": preset.air_motion_corr_floor,
                    "transient_sculpt_boost_db": preset.transient_sculpt_boost_db,
                    "transient_sculpt_mix": preset.transient_sculpt_mix,
                },
                "master_result": result,
                "post_metrics": out_metrics,
                "checks": checks,
                "passed": passed,
                "score": score,
                "pass_output": str(pass_out_path),
                "pass_report": str(pass_report_path),
            }
            pass_records.append(pass_record)

            if score < best_score:
                best_score = score
                best_pass = pass_index
                best_pass_path = pass_out_path
                best_metrics = out_metrics

            prev_metrics = out_metrics
            if passed:
                break

        if best_pass is None or best_pass_path is None or best_metrics is None:
            raise RuntimeError(f"No successful mastering pass produced output for '{source_path}'.")

        final_out = out_dir / f"{track_key}_trap_god_3d_master.wav"
        shutil.copyfile(best_pass_path, final_out)

        compat_src = best_pass_path.with_name(best_pass_path.stem + "_compat.wav")
        compat_dst = final_out.with_name(final_out.stem + "_compat.wav")
        if compat_src.exists():
            shutil.copyfile(compat_src, compat_dst)

        summary["tracks"].append(
            {
                "source": str(source_path),
                "input_used": str(input_path),
                "input_was_converted": converted,
                "source_metrics": source_metrics,
                "best_pass": best_pass,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "final_output": str(final_out),
                "compat_output": str(compat_dst) if compat_dst.exists() else None,
                "passes": pass_records,
            }
        )

    summary_path = out_dir / "trap_god_3d_batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Trap God 3D mastering loop using tools.auralmind_maestro."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Input data directory to scan recursively for songs.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/codex_mastered",
        help="Output directory for mastered files.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=4,
        help="Maximum mastering passes per track.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not data_dir.exists():
        print(f"Input data directory does not exist: {data_dir}", file=sys.stderr)
        return 2

    summary = run_batch(data_dir, out_dir, max_passes=max(1, int(args.max_passes)))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
