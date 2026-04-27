from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_mastering_tool import AIIntegratedMasteringTool
import server

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3"}
EXCLUDED_NAME_PARTS = (
    "auralmind",
    "_compat",
    "_probe",
    "mastered",
    "__",
    "analysis_master",
)
OUTPUT_DIR = ROOT / "masters" / "ai_integrated_latest_two_20260414"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def sanitize_name(value: str) -> str:
    safe: list[str] = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {" ", "-", "_", "(", ")"}:
            safe.append("-")
    joined = "".join(safe).strip("-")
    while "--" in joined:
        joined = joined.replace("--", "-")
    return joined or "master"


def is_source_candidate(path: Path) -> bool:
    if path.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        return False
    lowered = path.name.lower()
    return not any(part in lowered for part in EXCLUDED_NAME_PARTS)


def find_two_newest_songs(data_dir: Path) -> List[Path]:
    candidates = [path for path in data_dir.iterdir() if path.is_file() and is_source_candidate(path)]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[:2]


def build_variants(song_path: Path) -> List[Dict[str, str]]:
    song_key = sanitize_name(song_path.stem)
    if song_key == "close-to-the-edge":
        return [
            {"label": "cinema-wide-stems", "preset": "cinematic", "stem_mode": "on"},
            {"label": "hifi-detail-stems", "preset": "hi_fi_streaming", "stem_mode": "on"},
            {"label": "cinema-wide-nostems", "preset": "cinematic", "stem_mode": "off"},
            {"label": "streaming-polish-nostems", "preset": "hi_fi_streaming", "stem_mode": "off"},
        ]

    return [
        {"label": "trap-focus-stems", "preset": "competitive_trap", "stem_mode": "on"},
        {"label": "dense-luxe-stems", "preset": "club_clean", "stem_mode": "on"},
        {"label": "radio-push-nostems", "preset": "radio_loud", "stem_mode": "off"},
        {"label": "hifi-lift-nostems", "preset": "hi_fi_streaming", "stem_mode": "off"},
    ]


def copy_artifact_to_path(ctx: Any, artifact_id: str, destination: Path) -> None:
    session_key, session_dir = server._get_session_info(ctx)
    entry = server._load_artifact(session_key, session_dir, artifact_id)
    if entry is None:
        raise RuntimeError(f"Could not resolve artifact for {artifact_id}")
    source = Path(session_dir) / entry.data_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


async def register_and_analyze(
    tool: AIIntegratedMasteringTool,
    song_path: Path,
) -> Dict[str, Any]:
    audio_id = await tool.register_audio(song_path.stem, str(song_path))
    if not audio_id:
        raise RuntimeError(f"Failed to register {song_path.name}")

    metrics = await tool.analyze_audio_deep(audio_id, song_path.stem)
    if metrics is None:
        raise RuntimeError(f"Failed to analyze {song_path.name}")

    return {
        "song_name": song_path.stem,
        "song_path": str(song_path),
        "audio_id": audio_id,
        "source_metrics": metrics.model_dump(),
    }


async def launch_variants(
    tool: AIIntegratedMasteringTool,
    source_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    launched: List[Dict[str, Any]] = []
    for variant in build_variants(Path(source_info["song_path"])):
        print("\n" + "-" * 80)
        print(
            f"[AI BATCH] Launching {source_info['song_name']} :: "
            f"{variant['label']} :: preset={variant['preset']} :: stem_mode={variant['stem_mode']}"
        )
        job_id = await tool.launch_mastering_job(
            source_info["audio_id"],
            variant["preset"],
            stem_mode=variant["stem_mode"],
        )
        if not job_id:
            raise RuntimeError(
                f"Failed to launch {source_info['song_name']} variant {variant['label']}"
            )
        launched.append(
            {
                **variant,
                "song_name": source_info["song_name"],
                "song_path": source_info["song_path"],
                "audio_id": source_info["audio_id"],
                "job_id": job_id,
            }
        )
    return launched


async def monitor_and_export(
    tool: AIIntegratedMasteringTool,
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    artifact_id = await tool.monitor_job(
        variant["job_id"],
        poll_interval=8,
        max_polls=270,
    )
    if not artifact_id:
        raise RuntimeError(
            f"Variant failed or timed out: {variant['song_name']} :: {variant['label']}"
        )

    output_stub = sanitize_name(f"{variant['song_name']}__{variant['label']}")
    output_path = OUTPUT_DIR / f"{output_stub}.wav"
    copy_artifact_to_path(tool.ctx, artifact_id, output_path)
    output_metrics = server.analyze_audio(artifact_id, tool.ctx)

    return {
        **variant,
        "artifact_id": artifact_id,
        "exported_wav_path": str(output_path),
        "output_metrics": output_metrics.model_dump(),
        "completed_at": utc_now(),
    }


def write_summary(manifest: Dict[str, Any]) -> None:
    lines = [
        "# AI Integrated Latest-Two Mastering Summary",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Session id: `{manifest['session_id']}`",
        "",
        "## Sources",
    ]

    for source in manifest["sources"]:
        metrics = source["source_metrics"]
        lines.extend(
            [
                f"- `{source['song_name']}`",
                f"  Path: `{source['song_path']}`",
                (
                    f"  Source metrics: LUFS `{metrics['integrated_lufs']:.2f}`, "
                    f"TP `{metrics['true_peak_dbtp']:.4f}`, "
                    f"Crest `{metrics['crest_db']:.2f}`, "
                    f"Corr `{metrics['stereo_correlation']:.3f}`"
                ),
            ]
        )

    lines.extend(["", "## Renders"])
    for render in manifest["renders"]:
        metrics = render["output_metrics"]
        lines.extend(
            [
                f"- `{render['song_name']}` :: `{render['label']}`",
                (
                    f"  Preset `{render['preset']}` | stem_mode `{render['stem_mode']}` | "
                    f"job `{render['job_id']}` | artifact `{render['artifact_id']}`"
                ),
                (
                    f"  Output metrics: LUFS `{metrics['integrated_lufs']:.2f}`, "
                    f"TP `{metrics['true_peak_dbtp']:.4f}`, "
                    f"Crest `{metrics['crest_db']:.2f}`, "
                    f"Corr `{metrics['stereo_correlation']:.3f}`"
                ),
                f"  Export: `{render['exported_wav_path']}`",
            ]
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    ensure_utf8_console()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session_id = "ai_integrated_latest_two_batch_20260414"
    tool = AIIntegratedMasteringTool(session_id=session_id)
    data_dir = ROOT / "data"
    songs = find_two_newest_songs(data_dir)
    if len(songs) < 2:
        raise RuntimeError("Could not find two suitable source songs in data/")

    print("=" * 100)
    print("AuralMind2 AI Integrated Mastering Batch")
    print("=" * 100)
    print("Selected newest songs:")
    for song in songs:
        print(f"  - {song.name} ({datetime.fromtimestamp(song.stat().st_mtime)})")

    source_infos = []
    for song in songs:
        source_infos.append(await register_and_analyze(tool, song))

    queued_variants: List[Dict[str, Any]] = []
    for source_info in source_infos:
        queued_variants.extend(await launch_variants(tool, source_info))

    completed_renders: List[Dict[str, Any]] = []
    for variant in queued_variants:
        completed_renders.append(await monitor_and_export(tool, variant))

    manifest = {
        "generated_at": utc_now(),
        "session_id": session_id,
        "project_root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "sources": source_infos,
        "renders": completed_renders,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest)

    print("\n" + "=" * 100)
    print("AI integrated mastering batch complete")
    print("=" * 100)
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Summary:  {SUMMARY_PATH}")
    for render in completed_renders:
        print(f"  ✓ {render['song_name']} :: {render['label']} -> {render['exported_wav_path']}")


if __name__ == "__main__":
    asyncio.run(main())
