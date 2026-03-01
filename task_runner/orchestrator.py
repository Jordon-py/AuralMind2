from __future__ import annotations

import os
import glob
import subprocess
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path



import shutil

def get_files() -> list[dict]:
    """
    Returns a list of files to process.
    Looks in the data folder relative to the script location.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    files = glob.glob(os.path.join(data_dir, "*"))
    num_files = []

    for k, file in enumerate(files):
        # Skip directories and non-audio files
        if os.path.isfile(file) and file.lower().endswith(('.wav', '.mp3', '.m4a')):
            num_files.append({"number": k, "file": file})

    return num_files


def run_script(task: dict) -> tuple[str, int, str]:
    """
    Runs ONE python file as a separate process with args.
    Returns (script_path, exit_code, original_target).
    """
    script = Path(task["script"]).resolve()
    target = task.get("target_original", "")
    cmd = [sys.executable, str(script)] + task.get("args", [])
    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd)
    return str(script), completed.returncode, target


def update_arg(args: list[str], flag: str, value: str) -> None:
    """
    Updates or appends a flag/value pair in the args list.
    """
    try:
        idx = args.index(flag)
        if idx + 1 < len(args):
            args[idx + 1] = value
        else:
            args.append(value)
    except ValueError:
        args.extend([flag, value])


def main() -> None:
    template_scripts = [
        {
            "script": r"C:\Users\goku\Documents\Projects\mcp_mind\AuralMind\auralmind_match_maestro_v7_3.py",
            "args": [
                "--preset", "club_clean",
                "--mono-sub",
                "--target-lufs", "-13.0",
                "--ceiling", "-1.2",
                "--limiter", "v2",
                "--fir-stream", "auto",
                "--microdetail",
                "--movement-amount", "0.22",
                "--hooklift-mix", "0.44",
                "--hooklift-percentile", "88",
                "--transient-boost", "2.0",
                "--transient-mix", "0.40",
                "--transient-guard", "24.0",
                "--transient-decay", "6.5",
                "--masking-eq"
            ]
        }
    ]

    all_files = get_files()
    max_files = 30  # BATCH SIZE
    num_files = all_files[:max_files]

    max_workers = 2
    song_list = []

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    complete_dir = os.path.join(data_dir, "complete")
    os.makedirs(complete_dir, exist_ok=True)

    for item in num_files:
        k = item["number"]
        file_path = item["file"]

        # Create a deep-ish copy of the task
        task = {
            "script": template_scripts[0]["script"],
            "args": list(template_scripts[0]["args"]),
            "target_original": file_path
        }

        # Update specific args
        update_arg(task["args"], "--target", file_path)
        update_arg(task["args"], "--out", f"C:\\Users\\goku\\Downloads\\{k}_god_3d.wav")
        update_arg(task["args"], "--report", f"C:\\Users\\goku\\Downloads\\{k}_god_3d_Report.md")

        song_list.append(task)

    print(f"Orchestrating {len(song_list)} tasks...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_script, s) for s in song_list]

        for fut in as_completed(futures):
            script, code, target = fut.result()
            print(f"{script} -> exit code {code} for {target}")

            if code == 0 and target:
                dest = os.path.join(complete_dir, os.path.basename(target))
                print(f"Moving {target} to {dest}")
                try:
                    shutil.move(target, dest)
                except Exception as e:
                    print(f"Error moving file: {e}")


if __name__ == "__main__":
    main()
