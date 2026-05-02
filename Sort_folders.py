from __future__ import annotations
from typing import Any

"""
Sort_folders.py
================

This file is intentionally written as a teaching scaffold rather than a finished
sorting script. The goal is to help you learn how to design a folder-sorting
tool in Python without hiding the thinking behind it.

Core mental model:
Treat folder sorting as a pipeline, not as one big script.

1. Scan
   Read what exists in the source folder.
2. Classify
   Decide how each item should be grouped.
3. Plan
   Compute the destination path before moving anything.
4. Validate
   Check for collisions, missing folders, weird names, or unsupported cases.
5. Execute
   Only then copy or move files.
6. Report
   Print or log what happened so the run is understandable and debuggable.

Why this mental model matters:
- It separates thinking from action.
- It makes bugs easier to isolate.
- It makes your script interview-friendly because each stage has a clear purpose.

Good beginner-to-intermediate design rule:
Build the script in "dry run" mode first.

That means your script should first learn how to say:
"Here is what I would move, and where I would move it."

Only after that works should you let it actually move files.

Second-pass coaching tips:
1. Design the data shape before the loop.
   Most beginner bugs in scripts like this come from messy data structures, not
   from the loop itself. If your category map is clean, the rest of the script
   gets much easier.
2. Solve one file cleanly before solving a whole folder.
   If you can classify one file correctly and predict its destination, scaling
   to 500 files is mostly just repetition.
3. Unsafe actions must happen late.
   Scanning and printing are cheap. Moving files is risky. Put the risky step at
   the end of the pipeline so earlier mistakes do less damage.

Recommended sorting strategies:
- By file extension: `.jpg` -> `images`, `.mp3` -> `audio`, `.pdf` -> `documents`
- By date: year/month folders using created or modified timestamps
- By naming pattern: files containing keywords go into topic folders
- By size or type family: large media vs small docs vs archives

Best practices:
- Prefer `pathlib.Path` over string-based paths. It is cleaner and safer.
- Normalize extensions with `.suffix.lower()` so `.JPG` and `.jpg` are treated the same.
- Skip directories unless you explicitly want recursive sorting.
- Decide early whether duplicates should be skipped, renamed, overwritten, or logged.
- Start with copying if the data is important. Moving is riskier.
- Keep one source of truth for your category rules, usually a dictionary.
- Log edge cases instead of silently ignoring them.
- Never assume every file has an extension.
- Handle hidden files and temp files intentionally.
- Make your script idempotent when possible, meaning rerunning it should not create chaos.

Common mistakes:
- Mixing path discovery, classification, and moving logic in one loop with no structure
- Hardcoding fragile Windows-style string paths everywhere
- Overwriting files with the same name without checking first
- Moving files before printing a plan
- Forgetting that folders may already exist
- Forgetting that some entries returned from a directory scan are subfolders, not files

A strong engineering pattern:
Think in terms of "input -> decision -> output".

For each file, ask:
- What is this file?
- What rule applies?
- Where should it go?
- Is that destination safe?
- What action should happen?

If you can answer those five questions cleanly, the rest of the script becomes much easier.

Suggested build order:
1. Hardcode a small test folder.
2. Print discovered files.
3. Print the category each file would get.
4. Print the destination path each file would use.
5. Add folder creation.
6. Add copy behavior.
7. Replace copy with move only when you trust the logic.
8. Add recursive behavior later, not first.

What to optimize for:
- predictability
- readability
- recoverability
- safe reruns

If you build it this way, you are not just writing a utility script. You are
practicing a reusable backend engineering pattern: inspect -> transform ->
validate -> apply.

How I want you to think about your current draft:
- `DL_folder` should represent one clear source directory.
- `sorting_cat` should represent one clean set of rules.
- `file_map(...)` should answer one question only:
  "Given one file, what category should it belong to?"
- The folder scan should be separate from the classification function.
- The move/copy step should be separate from both of those.

That separation is one of the biggest mental model upgrades you can make as a
developer. It keeps each part small, testable, and explainable.
"""

import logging
import shutil
from collections import defaultdict
from pathlib import Path


# --- CONFIGURATION ---
USER_HOME = Path.home()
SOURCE_FOLDERS = [
    USER_HOME / "Music",
    USER_HOME / "Downloads",
    USER_HOME / "Desktop",
    USER_HOME / "Documents",
]

DESTINATION_ROOT = Path(r"D:/  My_music")

SORTING_RULES = {
    "audio": [".wav"]
}
DRY_RUN = False



def file_map(path: Path, rules: dict[str, list[str]]) -> str:
    ext = path.suffix.lower()
    for category, extensions in rules.items():
        if ext in extensions:
            return category
    return "misc"

def gather_wav_files(dry_run=False) -> Any:
    print(f"🚀 Starting WAV gathering pipeline (Dry Run: {dry_run})")
    files_to_move = []

    for folder in SOURCE_FOLDERS:
        if not folder.exists():
            print(f"⚠️  Skipping missing folder: {folder}")
            continue
        print(f"Scanning: {folder}...")
        for item in folder.rglob("*"):
            if item.is_file():
                category = file_map(item, SORTING_RULES)
                if category == "audio":
                    dest_path = DESTINATION_ROOT / item.name
                    files_to_move.append((item, dest_path))

    if not files_to_move:
        print("No .wav files found.")
        return

    print(f"Found {len(files_to_move)} files to gather.")

    for src, dst in files_to_move:
        if dry_run:
            print(f"[DRY RUN] Would copy: {src.name} -> {dst}")
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                final_dst = dst
                counter = 1
                while final_dst.exists():
                    final_dst = dst.with_name(f"{dst.stem}_{counter}{dst.suffix}")
                    counter += 1
                shutil.copy2(src, final_dst)
                print(f"✅ Copied: {src.name} -> {final_dst.name}")
            except Exception as e:
                print(f"❌ Failed to copy {src.name}: {e}")

if __name__ == "__main__":
    gather_wav_files(dry_run=False)




# Step 4: Scan the source folder and inspect one item at a time.
# Mental model:
# A directory scan gives you entries; your job is to filter and classify them.
# First check:
# - Is this entry a file?
# - Does it have an extension?
# - Do I know how to categorize it?
# What I want you to do in code at this stage:
# - Loop through the source folder one entry at a time.
# - Ignore directories for now.
# - For each file, print:
#   - the file name
#   - the suffix
#   - the category your classifier returns
# This stage is about visibility, not action.
# If you cannot explain what the script sees, do not let it move anything yet.


# Step 5: Build a destination path before moving anything.
# Mental model:
# Never "just move the file."
# First compute:
# source path -> category -> destination folder -> final destination path
# Planning before execution is one of the biggest quality upgrades in scripting.
# What I want you to do here:
# - Create the destination path in memory first.
# - Think in pieces:
#   - source folder
#   - category folder
#   - original file name
# - Print the full destination path before doing any file operation.
# Strong mental model:
# Paths are outputs of decisions.
# If the destination path looks wrong, the problem is probably your classifier or
# your naming rules, not `shutil`.


# Step 6: Add a dry-run phase.
# Mental model:
# Your script should be able to explain its plan in plain English.
# Example thought process:
# "I found song.wav, classified it as audio, and would move it to sorted/audio/"
# If the plan looks wrong, your logic is wrong. Fix that before touching files.
# What I want you to do here:
# - Add a boolean like `dry_run = True`.
# - When dry run is on, only print or log the planned action.
# - When dry run is off, allow copy or move behavior.
# Why this matters:
# Dry run turns debugging into reading.
# Without dry run, debugging turns into damage control.


# Step 7: Handle collisions intentionally.
# Mental model:
# If `report.pdf` already exists in the destination, your script needs a policy.
# Typical strategies:
# - skip
# - overwrite
# - rename
# - move into a duplicates folder
# Choose one on purpose. Never let it happen by accident.
# What I want you to do here:
# - Before writing to the destination, check whether that path already exists.
# - Decide on one policy for version 1.
# Best beginner choice:
# - skip duplicates and print that you skipped them
# That keeps the behavior simple and safe.
# Later, if you want, you can add renaming logic.


# Step 8: Create missing folders only when needed.
# Mental model:
# Destination folders are part of the output contract.
# Your script should ensure the destination structure exists before file operations.
# What I want you to do here:
# - Create category folders only when the script actually needs them.
# - Keep folder creation near the execution step, not the scan step.
# Why:
# Scanning is about understanding input.
# Folder creation is part of producing output.
# Keeping those separate helps you reason clearly.


# Step 9: Only after validation, perform the file operation.
# Mental model:
# `shutil.copy2(...)` is safer while learning because the original file remains.
# `shutil.move(...)` is better later when you trust the logic.
# Start safe, then get aggressive only when the script proves itself.
# What I want you to do here:
# - Use copy first while you are learning.
# - Confirm the output folders look correct.
# - Only then switch to move if the goal is true reorganization.
# Strong mental model:
# First prove correctness.
# Then optimize for convenience.


# Step 10: Log results and edge cases.
# Mental model:
# A good script leaves evidence.
# You want to know:
# - what moved
# - what was skipped
# - what failed
# - why it failed
# This is how you debug real-world automation.
# What I want you to do here:
# - Print a short message for every important decision.
# - Also print when the script does nothing and why.
# Example cases worth logging:
# - unsupported extension
# - duplicate destination
# - skipped directory
# - dry-run action
# You are building evidence, not just behavior.


# Step 11: Add recursion only after the non-recursive version is stable.
# Mental model:
# Recursive scripts multiply risk.
# First prove your rules on a flat folder.
# Then decide whether subfolders should be preserved, ignored, or flattened.
# What I want you to do here:
# - Ignore subfolders in version 1.
# - Make the one-level script solid first.
# - Only then decide whether nested folders should be:
#   - skipped
#   - entered recursively
#   - preserved in their original structure
# Strong mental model:
# Complexity should be earned.
# Do not add recursion just because Python can do it.


# Step 12: Test like an engineer, not like a gambler.
# Mental model:
# Build a fake folder containing:
# - files with known extensions
# - files with uppercase extensions
# - files with no extension
# - duplicate names
# - hidden/temp files
# - nested folders
# If your logic survives this, it is much more trustworthy.
# What I want you to do here:
# Build a small fake test folder and intentionally include annoying cases.
# That teaches you faster than using a perfect folder.
# The goal of the test folder is not realism.
# The goal is to force your script to prove that its rules are clear.
