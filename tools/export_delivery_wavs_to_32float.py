from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 44.1 kHz pcm_f32le delivery copies from an existing folder of "
            "mastered WAV files."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing the source mastered WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where the 32-bit float WAV files should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing 32-bit float files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    for source in wav_files:
        destination = output_dir / source.name.replace("__24bit_44k1", "__32float_44k1")
        if destination.exists() and not args.force:
            print(f"Skipping existing {destination.name}")
            continue

        command = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-ar",
            "44100",
            "-c:a",
            "pcm_f32le",
            str(destination),
        ]
        run_ffmpeg(command)
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
