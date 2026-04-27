r"""Run Christopher's requested 8-song premium hi-fi trap batch.

Purpose: thin wrapper around `run_explicit_premium_hifi_trap_batch.py` that
locks the exact requested source list and output folder, avoiding Windows
background-launch quoting issues for filenames with spaces and parentheses.
Data shapes: passes CLI-style strings to the base runner, which writes a
manifest containing one `TrackPlanItem` per source plus MCP job/artifact IDs,
phase-alignment metrics, final export paths, and delivery encode metadata.
Syntax: `C:\Python313\python.exe tools/run_requested8_premium_hifi_trap_batch_20260425.py`
or add `--dry-run` to only verify the eight resolved source paths.
Important functions: `main` starts near line 39; the imported runner's
`run_one` performs the MCP render/phase-align/export lifecycle.
Possible bugs: this wrapper is intentionally fixed to one date-stamped batch;
rerunning it resumes or retries the same output folder rather than making a
new run folder.
Enhance next: add `--output-root` passthrough for future reuse; promote this
fixed-source pattern into the base runner as a `--source-list-json` option.
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_explicit_premium_hifi_trap_batch import main as run_base_main


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "masters" / "mcp_premium_hifi_trap_requested_8_20260425_py313"
SOURCES = [
    REPO_ROOT / "data" / "Ride.wav",
    REPO_ROOT / "data" / "SB.wav",
    REPO_ROOT / "data" / "M.O (3).wav",
    REPO_ROOT / "data" / "Not the Same (3).wav",
    REPO_ROOT / "data" / "Vegas - top teir (1).wav",
    REPO_ROOT / "data" / "Vegas - top teir (2).wav",
    REPO_ROOT / "data" / "FaceTime (6).wav",
    REPO_ROOT / "data" / "DaddysGirls (2).wav",
]


def main() -> int:
    user_args = set(sys.argv[1:])
    if "--help" in user_args or "-h" in user_args:
        print(__doc__)
        return 0

    argv = [
        "run_explicit_premium_hifi_trap_batch.py",
        "--output-root",
        str(OUTPUT_ROOT),
        "--delivery-formats",
        "24,32",
        "--poll-seconds",
        "3",
        "--retry-errors",
        "--force-lock",
    ]
    if "--dry-run" in user_args:
        argv.append("--dry-run")
    for source in SOURCES:
        argv.extend(["--source", str(source)])
    sys.argv = argv
    return run_base_main()


if __name__ == "__main__":
    raise SystemExit(main())
