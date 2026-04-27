from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import server  # noqa: E402


TARGET_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}

# Skip obvious non-source / derived / probe artifacts. This list is intentionally
# conservative: the user's request is "newest 20 songs, no duplicates".
SKIP_SUBSTRINGS = (
    "_probe",
    "smoke_",
    "analysis_master",
    "streaming",
    "trapgod",
    # These are usually exported masters or internal variants, not source songs.
    "auralmind",
    "mastered",
    "_compat",
    "__hi_fi",
    "__hifi",
    "float64",
    "mov",
    "pass",
)

PLATFORM = "spotify"
DEFAULT_PRESET = "competitive_trap"

# Premium trap defaults (can be overridden).
DEFAULT_MOVEMENT = 0.28
DEFAULT_TRUE_PEAK = -1.0
DEFAULT_LRA = 7
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CODEC = "pcm_f32le"

# Hard limit to avoid accidentally rendering huge batches.
MAX_TRACKS = 20


@dataclass(frozen=True)
class Candidate:
    rel_path: str
    abs_path: str
    display_name: str
    modified_ts: float
    size_bytes: int
    duration_s: Optional[float]


@dataclass
class MasterPlanItem:
    display_name: str
    rel_path: str
    key_guess: str
    mode_guess: str
    key_score: float
    preset_name: str
    target_lufs: float
    warmth: float
    transient_boost_db: float
    control_profile: Dict[str, float]
    stem_mode: str
    output_raw_wav: str
    output_final_wav: str
    status: str = "planned"
    job_id: str = ""
    audio_id: str = ""
    error: str = ""


class _DummyContext:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    async def report_progress(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_log(log_path: Path, line: str) -> None:
    ts = utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.open("a", encoding="utf-8").write(f"[{ts}] {line}\n")


def lock_path(out_root: Path) -> Path:
    return out_root / ".auralmind2_master.lock"


def acquire_lock(out_root: Path, *, force: bool) -> None:
    """
    Best-effort single-run lock to avoid multiple master processes racing the
    same manifest/output folder.

    We keep it intentionally simple: if the lock exists, we refuse unless force
    is requested. This avoids silent corruption.
    """
    lp = lock_path(out_root)
    if lp.exists() and not force:
        raise RuntimeError(f"lock_exists: {lp}")
    content = {
        "created_at": utc_now(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "script": str(Path(__file__).resolve()),
    }
    lp.write_text(json.dumps(content, indent=2), encoding="utf-8")


def release_lock(out_root: Path) -> None:
    try:
        lock_path(out_root).unlink(missing_ok=True)  # py3.8+: OK on 3.12
    except Exception:
        return None


def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "untitled-song"


_FAMILY_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")


def normalize_song_family(name: str) -> str:
    """
    Normalize names like "difference (8)" vs "difference (10)" so the batch can
    honor the user's "no duplicates" requirement even when bounced versions have
    different durations/sizes.

    This is intentionally conservative: we only strip a trailing "(digits)".
    """
    n = name.strip().lower()
    n = _FAMILY_SUFFIX_RE.sub("", n)
    n = re.sub(r"[_\\-\\s]+", " ", n).strip()
    return n


def run_ffmpeg(cmd: List[str], timeout_s: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)


def parse_loudnorm_json(stderr_text: str) -> Dict[str, Any]:
    start = stderr_text.rfind("{")
    end = stderr_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Could not locate loudnorm JSON in ffmpeg output.")
    return json.loads(stderr_text[start : end + 1])


def loudnorm_two_pass(
    raw_path: Path,
    final_path: Path,
    *,
    target_lufs: float,
    true_peak: float,
    lra: int,
    sample_rate: int,
    codec: str,
) -> Dict[str, Any]:
    # Pass 1: measure
    analysis_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        f"aresample={sample_rate},loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    analysis_run = run_ffmpeg(analysis_cmd)
    measured = parse_loudnorm_json(analysis_run.stderr)

    # Pass 2: render
    render_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        (
            f"aresample={sample_rate},"
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:"
            f"measured_I={measured['input_i']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:"
            "linear=true:print_format=summary"
        ),
        "-ar",
        str(sample_rate),
        "-c:a",
        codec,
        str(final_path),
    ]
    render_run = run_ffmpeg(render_cmd)
    return {
        "measured": measured,
        "render_summary_tail": render_run.stderr.strip().splitlines()[-14:],
    }


def ffprobe_duration(path: Path, timeout_seconds: int = 20) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return None
    if res.returncode != 0:
        return None
    try:
        return float((res.stdout or "").strip())
    except ValueError:
        return None


def iter_audio_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TARGET_EXTENSIONS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def is_skipped_name(name: str) -> bool:
    low = name.lower()
    if any(substr in low for substr in SKIP_SUBSTRINGS):
        return True
    # Many derived files in this repo use double-underscore naming.
    if "__" in low:
        return True
    return False


def pick_newest_unique(candidates: List[Candidate], limit: int = MAX_TRACKS) -> List[Candidate]:
    picked: List[Candidate] = []
    seen_signatures: set[Tuple[int, int]] = set()
    seen_families: set[str] = set()

    # Signature: (rounded duration_ms, size_bytes). This is a pragmatic "no duplicates"
    # heuristic that catches identical exports even when filenames differ.
    for item in candidates:
        family = normalize_song_family(item.display_name)
        if family in seen_families:
            continue
        dur_ms = int(round((item.duration_s or 0.0) * 1000))
        sig = (dur_ms, item.size_bytes)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        seen_families.add(family)
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def detect_key_fast(path: Path, seconds: float = 35.0) -> Tuple[str, str, float]:
    # Fast, approximate musical key detection. The output is used as intent context
    # for the semantic planner and for labeling, not as a guarantee.
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=11025, mono=True, duration=seconds)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    v = chroma.mean(axis=1)
    v = v / (float(v.sum()) + 1e-9)

    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    min_ = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    maj = maj / maj.sum()
    min_ = min_ / min_.sum()

    def corr(a: Any, b: Any) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = float((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)
        return float((a * b).sum() / denom)

    best = (-999.0, "C", "major")
    for i, k in enumerate(keys):
        best = max(best, (corr(v, np.roll(maj, i)), k, "major"))
        best = max(best, (corr(v, np.roll(min_, i)), k, "minor"))
    return best[1], best[2], float(best[0])


def choose_control_profile(metrics: server.AudioMetrics, *, mode: str) -> Dict[str, float]:
    # Keep this bounded and explainable. The semantic planner also adjusts internally.
    centroid = float(metrics.centroid_hz or 0.0)
    corr = float(metrics.stereo_correlation)
    crest = float(metrics.crest_db)

    # Width: widen if very mono, tighten if already risky wide.
    if corr > 0.65:
        width = 0.35
    elif corr > 0.25:
        width = 0.20
    elif corr > 0.05:
        width = 0.10
    else:
        width = -0.05

    # Brightness + harshness protection.
    if centroid >= 3200:
        brightness = 0.08
        harshness = 0.55
    elif centroid >= 2400:
        brightness = 0.12
        harshness = 0.40
    else:
        brightness = 0.16
        harshness = 0.30

    # Minor keys tend to tolerate slightly darker tilt.
    if mode == "minor":
        brightness = max(-1.0, brightness - 0.10)

    # Phase-risk: keep center stability over width. This also reduces the odds
    # of widening sub content in the master stage.
    if corr < 0.15:
        width = min(width, 0.0)
        harshness = min(1.0, harshness + 0.05)

    # Movement: premium hooklift but avoid over-animation on already dense masters.
    if crest < 8.0:
        movement = 0.20
    elif crest < 10.0:
        movement = 0.26
    else:
        movement = 0.32

    # Trap low-end: adapt per track. If the mix is already crushed (low crest),
    # pushing low-end focus can get boomy; if it has headroom, we can anchor the
    # 808 more firmly.
    if crest >= 12.0:
        low_end = 0.78
    elif crest >= 9.0:
        low_end = 0.70
    else:
        low_end = 0.62

    return {
        "spatial_width": round(width, 3),
        "brightness_tilt": round(brightness, 3),
        "harshness_control": round(harshness, 3),
        "movement_amount": round(movement, 3),
        "low_end_focus": round(low_end, 3),
    }


def choose_targets(metrics: server.AudioMetrics) -> Tuple[float, float, float]:
    # Return (target_lufs, warmth, transient_boost_db).
    crest = float(metrics.crest_db)
    centroid = float(metrics.centroid_hz or 0.0)
    corr = float(metrics.stereo_correlation)

    # Loudness: keep it competitive, but stay in a streaming-safe band.
    if crest < 8.0:
        target_lufs = -12.8
    elif crest < 10.0:
        target_lufs = -12.2
    else:
        target_lufs = -11.6

    # Bright + limited tracks get brittle fast if pushed; back off slightly.
    if centroid >= 3200 and crest < 10.0:
        target_lufs -= 0.3

    # Dark, spacious tracks can take a touch more loudness without sounding edgy.
    if centroid < 1800 and crest >= 11.0:
        target_lufs = max(target_lufs, -11.4)

    # Warmth: temper brightness.
    if centroid >= 3200:
        warmth = 0.36
    elif centroid >= 2400:
        warmth = 0.30
    else:
        warmth = 0.26

    # Transients: trap-friendly snap, but don't overdo for already punchy tracks.
    if crest >= 11.0:
        transient = 1.6
    else:
        transient = 2.3

    # Wide/phase-risk mixes can sound smeared when transient lift is too high.
    if corr < 0.15:
        transient = max(1.3, transient - 0.3)

    return float(target_lufs), float(warmth), float(transient)


def artifact_source_path(ctx: _DummyContext, artifact_id: str) -> Path:
    session_key, session_dir = server._get_session_info(ctx)
    entry = server._load_artifact(session_key, session_dir, artifact_id)
    if entry is None:
        raise RuntimeError(f"Could not resolve artifact {artifact_id}.")
    return Path(session_dir) / entry.data_filename


def run_one(item: MasterPlanItem, ctx: _DummyContext, *, poll_s: float, max_wait_s: float) -> None:
    # 1) Register
    reg = server.register_audio_from_path(server.RegisterAudioPathIn(path=item.rel_path), ctx=ctx)
    item.audio_id = reg.audio_id

    # 2) Analyze
    analysis = server.analyze_audio(server.AnalyzeIn(audio_id=item.audio_id), ctx=ctx)
    metrics = analysis.metrics
    item.control_profile = choose_control_profile(metrics, mode=item.mode_guess)
    item.target_lufs, item.warmth, item.transient_boost_db = choose_targets(metrics)

    # 3) Plan
    cp = server.MasteringControlProfile(**item.control_profile)
    goal = (
        f"Premium, industry-standard trap master. Key: {item.key_guess} {item.mode_guess}. "
        "Tight mono sub-bass discipline (808 anchor), punchy transients, polished commercial finish, "
        "wide hook lift, controlled upper-mids for long-listen comfort."
    )
    plan = server.plan_mastering_strategy(
        server.StrategyPlanIn(
            audio_id=item.audio_id,
            goal=goal,
            platform=PLATFORM,
            control_profile=cp,
            stem_mode=item.stem_mode,
        ),
        ctx=ctx,
    )
    item.preset_name = plan.chosen_preset

    # 4) Execute (explicitly pass settings we want to lock)
    settings = plan.settings.model_copy(
        update={
            "target_lufs": item.target_lufs,
            "warmth": item.warmth,
            "transient_boost_db": item.transient_boost_db,
            "control_profile": cp,
            "bit_depth": "float32",
            "stem_mode": item.stem_mode,
        }
    )
    req = server.MasterRequest(audio_id=item.audio_id, **settings.model_dump())
    launch = server.run_master_job(req, ctx=ctx)
    item.job_id = launch.job_id

    # 5) Poll
    started = time.time()
    while True:
        status = server.job_status(server.JobIdIn(job_id=item.job_id), ctx=ctx)
        if status.status in ("done", "error", "cancelled"):
            break
        if time.time() - started > max_wait_s:
            raise RuntimeError(f"timeout waiting for job {item.job_id}")
        time.sleep(poll_s)

    if status.status != "done":
        raise RuntimeError(status.error.message if status.error else f"job_{status.status}")

    result = server.job_result(server.JobIdIn(job_id=item.job_id), ctx=ctx)

    # Find a WAV artifact to export (the master WAV is expected to exist).
    wav_candidates = [a for a in result.artifacts if a.filename.lower().endswith(".wav")]
    if not wav_candidates:
        raise RuntimeError("job_result_missing_wav_artifact")
    master_art = max(wav_candidates, key=lambda a: a.size_bytes)

    raw_path = Path(item.output_raw_wav)
    final_path = Path(item.output_final_wav)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast local copy from session store.
    shutil.copy2(artifact_source_path(ctx, master_art.artifact_id), raw_path)

    # Two-pass loudnorm to land the delivery tightly.
    loudnorm_two_pass(
        raw_path,
        final_path,
        target_lufs=item.target_lufs,
        true_peak=DEFAULT_TRUE_PEAK,
        lra=DEFAULT_LRA,
        sample_rate=DEFAULT_SAMPLE_RATE,
        codec=DEFAULT_CODEC,
    )

    item.status = "done"


def coerce_plan_item(d: Dict[str, Any]) -> MasterPlanItem:
    # Allow forward-compatible manifests: ignore unknown keys.
    allowed = set(MasterPlanItem.__annotations__.keys())
    clean: Dict[str, Any] = {k: d.get(k) for k in allowed if k in d}
    # Required fields safety (for older manifests).
    clean.setdefault("status", "planned")
    clean.setdefault("job_id", "")
    clean.setdefault("audio_id", "")
    clean.setdefault("error", "")
    return MasterPlanItem(**clean)  # type: ignore[arg-type]


def build_candidates() -> List[Candidate]:
    items: List[Candidate] = []
    for p in iter_audio_files(DATA_ROOT):
        if is_skipped_name(p.name):
            continue
        rel = p.relative_to(DATA_ROOT).as_posix()
        items.append(
            Candidate(
                rel_path=rel,
                abs_path=str(p),
                display_name=p.stem,
                modified_ts=float(p.stat().st_mtime),
                size_bytes=int(p.stat().st_size),
                duration_s=ffprobe_duration(p),
            )
        )
    return items


def build_plan(selected: List[Candidate], out_root: Path) -> List[MasterPlanItem]:
    plan: List[MasterPlanItem] = []
    raw_dir = out_root / "raw"
    final_dir = out_root / "final"

    for cand in selected:
        abs_path = Path(cand.abs_path)
        key, mode, score = detect_key_fast(abs_path)

        # We don't have server metrics yet for control profile, so we will do a first analysis
        # per item and then compute profile/targets from that analysis.
        # Here we just stub; real values filled before execution.
        placeholder = {"spatial_width": 0.2, "brightness_tilt": 0.1, "harshness_control": 0.4, "movement_amount": DEFAULT_MOVEMENT, "low_end_focus": 0.7}
        slug = safe_slug(cand.display_name)
        plan.append(
            MasterPlanItem(
                display_name=cand.display_name,
                rel_path=cand.rel_path,
                key_guess=key,
                mode_guess=mode,
                key_score=score,
                preset_name=DEFAULT_PRESET,
                target_lufs=-12.2,
                warmth=0.30,
                transient_boost_db=2.3,
                control_profile=placeholder,
                stem_mode="auto",
                output_raw_wav=str(raw_dir / f"{slug}__raw.wav"),
                output_final_wav=str(final_dir / f"{slug}__premium_trap__{key}{'m' if mode == 'minor' else ''}.wav"),
            )
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Master newest 20 unique songs in data/ as premium trap masters.")
    parser.add_argument("--limit", type=int, default=MAX_TRACKS)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--max-wait-seconds", type=float, default=60 * 60 * 4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing manifest.json and rebuild plan from data/.")
    parser.add_argument("--retry-errors", action="store_true", help="When resuming, retry items with status=error.")
    parser.add_argument("--force-lock", action="store_true", help="Override existing lock file for output-root.")
    args = parser.parse_args()

    limit = max(1, min(int(args.limit), MAX_TRACKS))

    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else None
    if out_root is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_root = REPO_ROOT / f"masters_premium_trap20_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.json"
    log_path = out_root / "run.log"

    acquire_lock(out_root, force=bool(args.force_lock))
    append_log(log_path, f"lock_acquired pid={os.getpid()} out_root={out_root}")

    try:
        ctx = _DummyContext(f"newest20_premium_trap_{int(time.time())}")

        manifest: Dict[str, Any]
        plan: List[MasterPlanItem]

        if manifest_path.exists() and not args.fresh:
            # Resume: reuse the exact planned song list and derived metadata.
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_items = list(manifest.get("items", []))
            plan = [coerce_plan_item(x) for x in raw_items]
            append_log(log_path, f"resume_loaded items={len(plan)} from={manifest_path}")
        else:
            candidates = build_candidates()
            selected = pick_newest_unique(candidates, limit=limit)
            plan = build_plan(selected, out_root)

            # Persist plan early so an interrupted run can be inspected.
            manifest = {
                "generated_at": utc_now(),
                "repo_root": str(REPO_ROOT),
                "data_root": str(DATA_ROOT),
                "output_root": str(out_root),
                "limit": limit,
                "items": [asdict(item) for item in plan],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            append_log(log_path, f"fresh_plan_written items={len(plan)} to={manifest_path}")

        if args.dry_run:
            print(str(out_root))
            print(f"Dry-run: wrote {manifest_path}")
            return 0

        to_run: List[MasterPlanItem] = []
        for it in plan:
            if it.status == "done":
                continue
            if it.status == "error" and not bool(args.retry_errors):
                continue
            to_run.append(it)

        append_log(log_path, f"queue_start total={len(plan)} to_run={len(to_run)} retry_errors={bool(args.retry_errors)}")

        for idx, item in enumerate(to_run, start=1):
            try:
                append_log(log_path, f"start {idx}/{len(to_run)} rel_path={item.rel_path}")
                run_one(item, ctx, poll_s=float(args.poll_seconds), max_wait_s=float(args.max_wait_seconds))
            except Exception as exc:
                item.status = "error"
                item.error = str(exc)
                append_log(log_path, f"error {idx}/{len(to_run)} rel_path={item.rel_path} err={item.error}")
            finally:
                manifest["last_updated_at"] = utc_now()
                manifest["items"] = [asdict(x) for x in plan]
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            msg = f"{idx}/{len(to_run)} {item.display_name}: {item.status}"
            print(msg)
            append_log(log_path, msg)

        append_log(log_path, "queue_done")
        return 0
    finally:
        release_lock(out_root)
        append_log(log_path, "lock_released")


if __name__ == "__main__":
    raise SystemExit(main())
