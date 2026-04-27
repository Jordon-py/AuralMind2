from __future__ import annotations

import argparse
import logging
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SOURCE_DIR = Path("D:/music")
DEST_DIR_NAME = "MAYBE_IIB"
AUDIO_EXTENSIONS = {".wav", ".mp3"}
KEEP_RECENT_COUNT = 4


@dataclass(frozen=True)
class RankedFile:
    path: Path
    prefix: str
    modified_time: float
    version_hint: tuple[int, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group .wav/.mp3 files in a folder by normalized filename prefix, "
            "keep the most recent versions for each group, and move them into "
            "a MAYBE_IIB folder."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DIR,
        help=f"Folder to scan. Defaults to {SOURCE_DIR}.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=(
            "Destination folder. Defaults to <source>/MAYBE_IIB. "
            "A relative path is resolved from the source folder."
        ),
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=KEEP_RECENT_COUNT,
        help="How many of the newest files to keep per prefix group.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Only move groups with at least this many matching files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would move without changing any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed grouping and move information.",
    )
    return parser


def resolve_destination(source_dir: Path, destination: Path | None) -> Path:
    if destination is None:
        return source_dir / DEST_DIR_NAME
    if destination.is_absolute():
        return destination
    return source_dir / destination


def iter_audio_files(source_dir: Path) -> list[Path]:
    return [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def extract_prefix(path: Path) -> str:
    stem = path.stem.strip()
    parts = [part.strip() for part in stem.split("__") if part.strip()]

    if parts:
        if re.fullmatch(r"\d+", parts[0]) and len(parts) >= 2:
            candidate = parts[1]
        elif len(parts) >= 2 and parts[1].lower() in {"stems", "nostems"}:
            candidate = parts[0]
        else:
            candidate = parts[0]
    else:
        candidate = stem

    candidate = candidate.lower()

    # Strip copy and revision markers without destroying the song title.
    cleanup_patterns = [
        r"\s+\(\d+\)$",                  # "Song (2)"
        r"_(\d+)$",                      # "Song_1"
        r"(?i)[ _-]v\d+(?:[-_]\d+)*$",   # "song_v3", "song-v3-15"
        r"(?i)[ _-]rev\d+$",             # "song REV4"
        r"(?i)-\d{3}$",                  # "song-010"
    ]
    changed = True
    while changed:
        changed = False
        for pattern in cleanup_patterns:
            updated = re.sub(pattern, "", candidate).strip()
            if updated != candidate:
                candidate = updated
                changed = True

    candidate = candidate.replace("_", " ")
    candidate = re.sub(r"^[^a-z0-9]+", "", candidate)
    candidate = re.sub(r"[^a-z0-9]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate or path.stem.lower()


def version_hint(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.stem)
    return tuple(int(number) for number in numbers[-4:])


def rank_files(paths: list[Path]) -> list[RankedFile]:
    ranked: list[RankedFile] = []
    for path in paths:
        ranked.append(
            RankedFile(
                path=path,
                prefix=extract_prefix(path),
                modified_time=path.stat().st_mtime,
                version_hint=version_hint(path),
            )
        )
    return ranked


def group_ranked_files(ranked_files: list[RankedFile]) -> dict[str, list[RankedFile]]:
    grouped: dict[str, list[RankedFile]] = defaultdict(list)
    for ranked_file in ranked_files:
        grouped[ranked_file.prefix].append(ranked_file)
    return grouped


def choose_recent_versions(
    grouped_files: dict[str, list[RankedFile]],
    keep: int,
    min_group_size: int,
) -> dict[str, list[RankedFile]]:
    selected: dict[str, list[RankedFile]] = {}
    for prefix, files in grouped_files.items():
        if len(files) < min_group_size:
            continue
        ordered = sorted(
            files,
            key=lambda item: (item.modified_time, item.version_hint, item.path.name.lower()),
            reverse=True,
        )
        selected[prefix] = ordered[:keep]
    return selected


def unique_destination(destination_dir: Path, source_file: Path) -> Path:
    target = destination_dir / source_file.name
    if not target.exists():
        return target

    counter = 1
    while True:
        candidate = destination_dir / f"{source_file.stem}_{counter}{source_file.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_selected_files(
    selected_files: dict[str, list[RankedFile]],
    destination_dir: Path,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    moved_count = 0
    group_count = 0

    if not dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)

    for prefix, files in sorted(selected_files.items()):
        group_count += 1
        logging.info("prefix=%s count=%s", prefix, len(files))
        for ranked_file in files:
            target = unique_destination(destination_dir, ranked_file.path)
            if dry_run:
                logging.info("DRY RUN move: %s -> %s", ranked_file.path, target)
            else:
                shutil.move(str(ranked_file.path), str(target))
                logging.info("Moved: %s -> %s", ranked_file.path, target)
            moved_count += 1

    return group_count, moved_count


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose or args.dry_run else logging.WARNING,
        format="%(message)s",
    )

    source_dir = args.source.expanduser().resolve()
    destination_dir = resolve_destination(source_dir, args.dest).expanduser().resolve()

    if not source_dir.exists():
        parser.error(f"Source folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        parser.error(f"Source path is not a folder: {source_dir}")
    if args.keep < 1:
        parser.error("--keep must be at least 1")
    if args.min_group_size < 2:
        parser.error("--min-group-size must be at least 2")
    if destination_dir == source_dir:
        parser.error("Destination folder must be different from the source folder")

    audio_files = iter_audio_files(source_dir)
    ranked_files = rank_files(audio_files)
    grouped_files = group_ranked_files(ranked_files)
    selected_files = choose_recent_versions(
        grouped_files,
        keep=args.keep,
        min_group_size=args.min_group_size,
    )

    if not selected_files:
        print("No matching prefix groups met the selection rules.")
        return 0

    group_count, moved_count = move_selected_files(
        selected_files,
        destination_dir,
        dry_run=args.dry_run,
    )

    action = "Would move" if args.dry_run else "Moved"
    print(
        f"{action} {moved_count} files from {group_count} prefix groups "
        f"into {destination_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
