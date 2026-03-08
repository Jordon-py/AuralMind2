from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT_DIR / "task_runner" / "hi_fi_trapgod3d_manifest.txt"
MASTER_SCRIPT = ROOT_DIR / "tools" / "auralmind_maestro.py"
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\goku\Downloads\AuralMind2_HiFi_TrapGod3D_Masters")
DEFAULT_PROFILE_LABEL = "AuralMind2_HiFi_TrapGod3D"

# Fixed custom profile: trap-forward, hi-fi, 3D, streaming-safe, movement locked on.
FIXED_PROFILE_ARGS = [
    "--preset", "competitive_trap",
    "--out-subtype", "FLOAT",
    "--no-dither",
    "--no-stems",
    "--mono-sub",
    "--masking-eq",
    "--microdetail",
    "--target-lufs", "-13.0",
    "--ceiling", "-1.2",
    "--limiter", "v2",
    "--fir-stream", "auto",
    "--movement-amount", "0.23",
    "--hooklift-mix", "0.22",
    "--hooklift-percentile", "88",
    "--transient-boost", "2.2",
    "--transient-mix", "0.34",
    "--transient-guard", "18.5",
    "--transient-decay", "6.2",
    "--warmth", "0.36",
]


@dataclass(frozen=True)
class Job:
    index: int
    source: Path
    out_path: Path
    report_path: Path
    log_path: Path
    command: list[str]


def load_manifest(path: Path) -> list[Path]:
    tracks: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tracks.append(Path(line))
    return tracks


def build_track_stub(index: int, source: Path) -> str:
    name = source.name
    if name.lower().endswith(".wav") and len(name) > 4:
        base = name[:-4]
    else:
        base = source.stem or name
    base = base.strip()
    if not base or base in {".", ".."}:
        base = f"unnamed_{index:03d}"
    return f"{index:03d}__{base}"


def build_output_base(index: int, source: Path, profile_label: str) -> str:
    return f"{build_track_stub(index, source)}__{profile_label}"


def parse_report_json(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    text = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def build_report_tuned_args(source: Path, report_data: dict[str, Any] | None) -> list[str]:
    name_lc = source.name.lower()
    common = [
        "--out-subtype", "FLOAT",
        "--no-dither",
        "--no-stems",
        "--masking-eq",
        "--target-lufs", "-13.0",
        "--ceiling", "-1.2",
        "--limiter", "v2",
        "--fir-stream", "auto",
        "--movement-amount", "0.23",
    ]

    if not report_data:
        return [
            "--preset", "competitive_trap",
            *common,
            "--mono-sub",
            "--microdetail",
            "--hooklift-mix", "0.16",
            "--hooklift-percentile", "90",
            "--transient-boost", "1.6",
            "--transient-mix", "0.26",
            "--transient-guard", "17.2",
            "--transient-decay", "5.8",
            "--warmth", "0.26",
        ]

    lufs_post = float(report_data.get("lufs_post", 0.0))
    limiter_min = float(report_data.get("limiter_min_gain_db", 0.0))
    limiter_avg = float(report_data.get("limiter_avg_gr_db", 0.0))
    mono_mix = float(report_data.get("mono_sub_mix", 0.0))

    voice_like = (
        "voiceaudio" in name_lc
        or lufs_post <= -18.0
        or mono_mix >= 0.68
    )
    peak_stressed = (
        limiter_min <= -14.0
        or limiter_avg <= -4.5
        or lufs_post <= -15.5
    )
    moderately_stressed = (
        limiter_min <= -10.0
        or lufs_post <= -14.7
    )

    if voice_like:
        return [
            "--preset", "hi_fi_streaming",
            *common,
            "--no-mono-sub",
            "--no-microdetail",
            "--no-hooklift",
            "--transient-boost", "0.0",
            "--transient-mix", "0.0",
            "--transient-guard", "16.0",
            "--transient-decay", "5.0",
            "--warmth", "0.18",
        ]

    if peak_stressed:
        return [
            "--preset", "competitive_trap",
            *common,
            "--mono-sub",
            "--no-microdetail",
            "--hooklift-mix", "0.12",
            "--hooklift-percentile", "92",
            "--transient-boost", "0.0",
            "--transient-mix", "0.0",
            "--transient-guard", "16.8",
            "--transient-decay", "5.6",
            "--warmth", "0.22",
        ]

    if moderately_stressed:
        return [
            "--preset", "competitive_trap",
            *common,
            "--mono-sub",
            "--no-microdetail",
            "--hooklift-mix", "0.14",
            "--hooklift-percentile", "90",
            "--transient-boost", "1.2",
            "--transient-mix", "0.22",
            "--transient-guard", "16.8",
            "--transient-decay", "5.6",
            "--warmth", "0.24",
        ]

    return [
        "--preset", "competitive_trap",
        *common,
        "--mono-sub",
        "--microdetail",
        "--hooklift-mix", "0.16",
        "--hooklift-percentile", "90",
        "--transient-boost", "1.6",
        "--transient-mix", "0.26",
        "--transient-guard", "17.2",
        "--transient-decay", "5.8",
        "--warmth", "0.26",
    ]


def build_job(
    index: int,
    source: Path,
    output_dir: Path,
    profile_label: str,
    profile_mode: str,
    analysis_dir: Path,
    analysis_label: str,
) -> Job:
    base = build_output_base(index, source, profile_label)
    out_path = output_dir / f"{base}.wav"
    report_path = output_dir / f"{base}.md"
    log_path = output_dir / f"{base}.log"
    track_stub = build_track_stub(index, source)
    prior_report = analysis_dir / f"{track_stub}__{analysis_label}.md"
    prior_report_data = parse_report_json(prior_report) if profile_mode == "report_tuned" else None
    profile_args = FIXED_PROFILE_ARGS if profile_mode == "fixed" else build_report_tuned_args(source, prior_report_data)
    command = [
        sys.executable,
        str(MASTER_SCRIPT),
        "--target",
        str(source),
        "--out",
        str(out_path),
        "--report",
        str(report_path),
        *profile_args,
    ]
    return Job(
        index=index,
        source=source,
        out_path=out_path,
        report_path=report_path,
        log_path=log_path,
        command=command,
    )


def run_job(job: Job) -> dict[str, Any]:
    started = time.time()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    command_str = subprocess.list2cmdline(job.command)
    with job.log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"source={job.source}\n")
        handle.write(f"command={command_str}\n\n")
        completed = subprocess.run(
            job.command,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        handle.write(completed.stdout)
    return {
        "index": job.index,
        "source": str(job.source),
        "out_path": str(job.out_path),
        "report_path": str(job.report_path),
        "log_path": str(job.log_path),
        "returncode": completed.returncode,
        "elapsed_s": round(time.time() - started, 2),
        "status": "ok" if completed.returncode == 0 and job.out_path.exists() else "error",
    }


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch hi-fi TrapGod 3D mastering with fixed AuralMind2 settings.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for mastered WAVs, reports, logs, and summary JSON.",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Plain-text manifest of source WAV paths, one per line.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel mastering workers.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=["fixed", "report_tuned"],
        default="fixed",
        help="Use the original fixed profile or adapt settings from prior report JSON.",
    )
    parser.add_argument(
        "--profile-label",
        default=DEFAULT_PROFILE_LABEL,
        help="Label used in rendered output filenames and summary JSON.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing prior report markdown files used for report_tuned mode.",
    )
    parser.add_argument(
        "--analysis-label",
        default=DEFAULT_PROFILE_LABEL,
        help="Prior report label to read in report_tuned mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render tracks even if output WAV and report already exist.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not MASTER_SCRIPT.exists():
        raise FileNotFoundError(f"Master script not found: {MASTER_SCRIPT}")

    tracks = load_manifest(manifest_path)
    if not tracks:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    results: list[dict[str, Any]] = []
    jobs: list[Job] = []
    for index, source in enumerate(tracks, start=1):
        if not source.exists():
            results.append(
                {
                    "index": index,
                    "source": str(source),
                    "status": "missing",
                }
            )
            continue
        job = build_job(
            index=index,
            source=source,
            output_dir=output_dir,
            profile_label=str(args.profile_label),
            profile_mode=str(args.profile_mode),
            analysis_dir=analysis_dir,
            analysis_label=str(args.analysis_label),
        )
        if not args.force and job.out_path.exists() and job.report_path.exists():
            results.append(
                {
                    "index": index,
                    "source": str(source),
                    "out_path": str(job.out_path),
                    "report_path": str(job.report_path),
                    "log_path": str(job.log_path),
                    "status": "skipped_existing",
                }
            )
            continue
        jobs.append(job)

    print(f"Queued {len(jobs)} tracks from {len(tracks)} manifest entries.")
    if jobs:
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            future_map = {pool.submit(run_job, job): job for job in jobs}
            for future in cf.as_completed(future_map):
                result = future.result()
                results.append(result)
                print(
                    f"[{result['status']}] "
                    f"{result['index']:03d} "
                    f"{Path(result['source']).name} "
                    f"{result.get('elapsed_s', 0)}s"
                )

    results.sort(key=lambda row: row["index"])
    summary_path = output_dir / f"{args.profile_label}__summary.json"
    write_summary(summary_path, results)

    ok_count = sum(1 for row in results if row["status"] == "ok")
    missing_count = sum(1 for row in results if row["status"] == "missing")
    error_count = sum(1 for row in results if row["status"] == "error")
    skipped_count = sum(1 for row in results if row["status"] == "skipped_existing")
    print(
        f"Complete. ok={ok_count} skipped={skipped_count} "
        f"missing={missing_count} error={error_count} summary={summary_path}"
    )
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
