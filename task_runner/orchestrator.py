from __future__ import annotations

import os
import glob
import subprocess
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path



def get_files() -> list[str]:
    """
    Returns a list of files to process.
    """
    files = glob.glob(os.path.join(os.path.dirname(__file__), "data", "*"))
    num_files = []

    try:
        for k, file in enumerate(files):
            num_files.append({"number": k, "file": file})

    except ValueError:
        print("File name must be in the format 'file_1.wav'")
        exit(1)

    return num_files


def run_script(task: dict) -> list[str]:
    """
    Runs ONE python file as a separate process with args.
    Returns (script_path, exit_code).
    """
    script = Path(task["script"]).resolve()
    # sys.executable ensures we use the same Python that ran THIS runner script
    cmd = [sys.executable, str(script)] + task.get("args", [])
    completed = subprocess.run(cmd)  # waits for this one script to finish
    return str(script), completed.returncode


def main() -> None:
    scripts = [
        {"script": r"C:\\Users\\goku\\Documents\\Projects\\mcp_mind\\AuralMind\\auralmind_match_maestro_v7_3.py", "args": [
            "--target", "C:\\Users\\goku\\Downloads\\Vegas - top teir (21).wav",
            "--out", "C:\\Users\\goku\\Downloads\\Vegas - top teir (21) god_3d.wav",
            "--report", "C:\\Users\\goku\\Downloads\\Vegas - top teir (21) MASTER_trap_god_3d_Report.md",
            "--preset", "hi_fi_streaming",
            "--mono-sub",
            "--target-lufs", "-13.0",
            "--ceiling", "-1.2",
            "--limiter", "v2",
            "--fir-stream", "auto",
            "--microdetail",
            "--movement-amount", "0.23",
            "--hooklift-mix", "0.46",
            "--hooklift-percentile", "88",
            "--transient-boost", "2.2",
            "--transient-mix", "0.42",
            "--transient-guard", "23.0",
            "--transient-decay", "6.7",
            "--stems",
            "--masking-eq"]
            },

        {"script": r"C:\\Users\\goku\\Documents\\Projects\\mcp_mind\\AuralMind\\auralmind_match_maestro_v7_3.py", "args": [
            "--target", "C:\\Users\\goku\\Downloads\\Vegas - top teir (22).wav",
            "--out", "C:\\Users\\goku\\Downloads\\Vegas - top teir (22) god_3d.wav",
            "--report", "C:\\Users\\goku\\Downloads\\Vegas - top teir (22) MASTER_trap_god_3d_Report.md",
            "--preset", "hi_fi_streaming",
            "--mono-sub",
            "--target-lufs", "-13.0",
            "--ceiling", "-1.2",
            "--limiter", "v2",
            "--fir-stream", "auto",
            "--microdetail",
            "--movement-amount", "0.21",
            "--hooklift-mix", "0.46",
            "--hooklift-percentile", "88",
            "--transient-boost", "2.2",
            "--transient-mix", "0.42",
            "--transient-guard", "21.0",
            "--transient-decay", "6.7",
            "--masking-eq"]
            },
    ]

    num_files = get_files()
    max_workers = 2  # how many scripts to run at the same time
    song_list = []


    for k, file in enumerate(num_files):
       scripts[0]["args"]["target"] = file["file"]
       scripts[0]["args"]["out"] = "C:\\Users\\goku\\Downloads\\" + str(k) + "_god_3d.wav"
       scripts[0]["args"]["report"] = "C:\\Users\\goku\\Downloads\\" + str(k) + "_god_3d_Report.md"
       scripts[1]["args"]["target"] = file["file"]
       scripts[1]["args"]["out"] = "C:\\Users\\goku\\Downloads\\" + str(k) + "_god_3d.wav"
       scripts[1]["args"]["report"] = "C:\\Users\\goku\\Downloads\\" + str(k) + "_god_3d_Report.md"
       song_list.append(scripts)


    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_script, s) for s in song_list]

        for fut in as_completed(futures):
            script, code = fut.result()
            print(f"{script} -> exit code {code}")


if __name__ == "__main__":
    main()
