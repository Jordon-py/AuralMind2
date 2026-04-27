from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.run_two_drive_realtime_ai_hifi_trap_batch as batch

batch.MOVEMENT_AMOUNT = 0.28


def build_global_plan(candidates: Iterable[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[item["normalized_song_name"]].append(item)

    plan: List[Dict[str, Any]] = []
    for song in sorted(grouped):
        files = sorted(grouped[song], key=lambda item: (item["modified_time"], item["path"].lower()))
        oldest = batch.first_valid(files, skipped)
        newest = batch.last_valid(files, skipped)
        if oldest is None or newest is None:
            skipped.append({"normalized_song_name": song, "reason": "global_group_has_no_decode_valid_sources"})
            continue

        out_dir = batch.OUTPUT_ROOT / "outputs" / song / "global"
        if oldest["path"] == newest["path"]:
            plan.append(
                {
                    **oldest,
                    "selection_reason": "single-file-master",
                    "mastering_variant": "A",
                    "output_file_path": str(out_dir / f"{song}__global__single__variantA.wav"),
                }
            )
        else:
            for reason, variant, source in (("oldest", "A", oldest), ("newest", "B", newest)):
                plan.append(
                    {
                        **source,
                        "selection_reason": reason,
                        "mastering_variant": variant,
                        "output_file_path": str(out_dir / f"{song}__global__{reason}__variant{variant}.wav"),
                    }
                )
    return plan


def build_summary(candidates: List[Dict[str, Any]], skipped: List[Dict[str, Any]], plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped_by_drive: Dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        grouped_by_drive[item["source_drive"]].add(item["normalized_song_name"])

    duplicate_ids = sorted(set(grouped_by_drive.get("C", set())) & set(grouped_by_drive.get("D", set())))
    return {
        "output_folder_path": str(batch.OUTPUT_ROOT),
        "candidate_files": len(candidates),
        "skipped_files": len(skipped),
        "song_groups_on_C": len(grouped_by_drive.get("C", set())),
        "song_groups_on_D": len(grouped_by_drive.get("D", set())),
        "duplicate_song_identities_across_drives": len(duplicate_ids),
        "duplicate_song_identities": duplicate_ids,
        "total_expected_mastered_outputs": len(plan),
        "pipeline_chosen": "server.py realtime interactive AI path with server.master_audio fallback",
        "realtime_ai_enabled": True,
        "exact_32bit_48000_supported": True,
        "format_reason": "server.py/auralmind_maestro supports float32 WAV and this runner finalizes completed outputs to pcm_f32le at 48000 Hz.",
        "selection_scope": "global oldest and newest version per song identity across both drives",
    }


def print_dry_run(summary: Dict[str, Any]) -> None:
    print("\nDRY-RUN PREVIEW")
    print("=" * 80)
    for key, value in summary.items():
        if key == "duplicate_song_identities":
            shown = ", ".join(value[:20])
            suffix = " ..." if len(value) > 20 else ""
            print(f"{key}: {shown}{suffix}")
        else:
            print(f"{key}: {value}")
    print("=" * 80)


async def run_batch(dry_run: bool, limit: Optional[int]) -> int:
    batch.ensure_dirs()
    batch.log_line("Starting global realtime AI hi-fi trap batch.")
    candidates, skipped, assumptions = batch.scan_sources()
    plan = build_global_plan(candidates, skipped)
    if limit is not None:
        plan = plan[:limit]

    summary = build_summary(candidates, skipped, plan)
    print_dry_run(summary)
    batch.log_line(f"Dry-run summary: {batch.json.dumps(summary, sort_keys=True)}")

    planned = batch.planned_manifest(plan)
    if dry_run:
        batch.write_manifest(planned, candidates, skipped, assumptions, summary)
        batch.write_report(planned, skipped, assumptions, summary)
        batch.log_line("Dry-run only requested; no mastering jobs executed.")
        return 0

    outputs: List[Dict[str, Any]] = []
    for idx, item in enumerate(plan, start=1):
        batch.log_line(
            f"Rendering {idx}/{len(plan)}: {item['normalized_song_name']} from {item['source_drive']} "
            f"{item['selection_reason']} variant{item['mastering_variant']}"
        )
        outputs.append(await batch.process_one(item))
        batch.write_manifest(outputs + planned[len(outputs):], candidates, skipped, assumptions, summary)

    batch.write_manifest(outputs, candidates, skipped, assumptions, summary)
    batch.write_report(outputs, skipped, assumptions, summary)
    batch.log_line("Batch complete.")

    completed = len([row for row in outputs if row["status"] == "completed"])
    failed = len([row for row in outputs if row["status"] == "failed"])
    print("\nFINAL COMPLETION SUMMARY")
    print("=" * 80)
    print(f"output folder created: {batch.OUTPUT_ROOT}")
    print("mastering pipeline used: server.py realtime interactive AI path with server.master_audio fallback")
    print("whether real-time AI mastering was used: yes")
    print(f"total candidate files found: {summary['candidate_files']}")
    print(f"total skipped files: {summary['skipped_files']}")
    print(f"total song groups processed: {summary['song_groups_on_C'] + summary['song_groups_on_D']}")
    print(f"total masters created: {completed}")
    print(f"total failures: {failed}")
    print("manual listening check: level-match A/B pairs, then verify mono sub focus, hook lift, and top-end width in stereo and mono fold-down.")
    print("=" * 80)
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global oldest/newest realtime AI premium hi-fi trap mastering batch.")
    parser.add_argument("--dry-run", action="store_true", help="Scan, group, and write planned manifests without mastering.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of planned outputs to process.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_batch(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
