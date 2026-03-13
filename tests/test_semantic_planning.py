import asyncio
import math
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict
from unittest import mock

import numpy as np

import server


@dataclass
class FakePreset:
    name: str
    target_lufs: float = -12.8
    ceiling_dbfs: float = -1.0
    limiter_mode: str = "v2"
    governor_gr_limit_db: float = -1.2
    match_strength: float = 0.5
    enable_harshness_limiter: bool = True
    enable_air_motion: bool = True
    bit_depth: str = "float32"
    warmth: float = 0.1
    transient_sculpt_boost_db: float = 1.8
    width_mid: float = 1.04
    width_hi: float = 1.24
    air_motion_mix: float = 0.12
    harshness_max_cut_db: float = 2.0
    movement_amount: float = 0.15
    hooklift_mix: float = 0.2
    mono_sub_base_mix: float = 0.55
    enable_stem_separation: bool = False
    governor_search_steps: int = 11
    microshift_mix: float = 0.14
    hi_factor: float = 0.72
    microdetail_amount: float = 0.2
    glow_mix: float = 0.5
    harshness_mix: float = 0.6
    deess_mix: float = 0.5
    transient_sculpt_mix: float = 0.34
    stem_gains_db: Dict[str, float] = field(default_factory=dict)
    enable_movement: bool = True
    enable_hooklift: bool = True


class FakeFuture:
    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        return None


class FakeContext:
    def __init__(self, session_id: str = "test-session") -> None:
        self.session_id = session_id

    async def report_progress(self, *_args, **_kwargs) -> None:
        return None

    def info(self, *_args, **_kwargs) -> None:
        return None

    def debug(self, *_args, **_kwargs) -> None:
        return None


class FakeMaestro:
    HAS_DEMUCS = True

    def __init__(self) -> None:
        self.rendered_presets = []

    def get_presets(self) -> Dict[str, FakePreset]:
        return {
            "hi_fi_streaming": FakePreset(name="hi_fi_streaming", target_lufs=-12.8, governor_gr_limit_db=-1.0),
            "cinematic": FakePreset(name="cinematic", target_lufs=-13.6, governor_gr_limit_db=-0.8, width_hi=1.22),
            "competitive_trap": FakePreset(name="competitive_trap", target_lufs=-11.0, governor_gr_limit_db=-1.3, width_hi=1.30),
            "club_clean": FakePreset(name="club_clean", target_lufs=-10.4, governor_gr_limit_db=-1.6, width_hi=1.28),
            "radio_loud": FakePreset(name="radio_loud", target_lufs=-11.2, governor_gr_limit_db=-1.3, width_hi=1.26),
        }

    def auto_select_preset_name(self, _features: Dict[str, float]) -> str:
        return "hi_fi_streaming"

    def master(self, target_path: str, out_path: str, preset: FakePreset) -> Dict[str, Any]:
        _ = target_path
        self.rendered_presets.append(preset)
        with open(out_path, "wb") as handle:
            handle.write(b"mastered-audio")
        return {"ok": True}

    def write_audio(self, path: str, _audio: np.ndarray, _sr: int, **_kwargs: Any) -> None:
        with open(path, "wb") as handle:
            handle.write(b"audio")

    def load_audio(self, _path: str):
        samples = np.linspace(-0.25, 0.25, 48000, dtype=np.float32)
        stereo = np.stack([samples, samples], axis=1)
        return stereo, 48000

    def analyze_track_features(self, _audio: np.ndarray, _sr: int) -> Dict[str, float]:
        return {
            "lufs": -13.0,
            "tp_dbfs": -1.1,
            "crest_db": 10.0,
            "corr_hi": 0.3,
            "peak_dbfs": -0.4,
            "rms_dbfs": -10.4,
            "centroid_hz": 3100.0,
        }

    def ensure_stereo(self, audio: np.ndarray) -> np.ndarray:
        return audio

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        return np.mean(audio, axis=1).astype(np.float32)

    def lin_to_db(self, value: float) -> float:
        return 20.0 * math.log10(max(abs(float(value)), 1e-12))

    def peak(self, audio: np.ndarray) -> float:
        return float(np.max(np.abs(audio)))

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))

    def integrated_loudness_lufs(self, _audio: np.ndarray, _sr: int) -> float:
        return -12.0

    def demucs_separate_stems(self, audio: np.ndarray, _sr: int, **_kwargs: Any):
        return {
            "vocals": audio * 0.7,
            "drums": audio * 0.5,
            "bass": audio * 0.4,
            "other": audio * 0.6,
        }, {"enabled": True}

    def true_peak_limiter_v2(self, audio: np.ndarray, _sr: int, **_kwargs: Any):
        return audio * 0.9, {"min_gain_db": -1.0, "avg_gr_db": -0.4, "tp_dbfs": -1.0}


class SemanticPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.ctx = FakeContext()
        self.fake_maestro = FakeMaestro()
        self.storage_patch = mock.patch.object(server, "STORAGE_DIR", self.tempdir.name)
        self.storage_patch.start()
        os.makedirs(server.STORAGE_DIR, exist_ok=True)
        server._ARTIFACTS.clear()
        server._JOBS.clear()
        server.INTERACTIVE_SESSIONS.clear()

    def tearDown(self) -> None:
        self.storage_patch.stop()
        server._ARTIFACTS.clear()
        server._JOBS.clear()
        server.INTERACTIVE_SESSIONS.clear()
        self.tempdir.cleanup()

    def _register_test_audio(self) -> str:
        session_key, session_dir = server._get_session_info(self.ctx)
        audio_id = "aud_1234567890ab"
        source_path = os.path.join(session_dir, "source.wav")
        with open(source_path, "wb") as handle:
            handle.write(b"source-audio")
        server._register_existing_file(
            session_key,
            session_dir,
            artifact_id=audio_id,
            kind="audio",
            filename="source.wav",
            data_filename="source.wav",
            media_type="audio/wav",
        )
        return audio_id

    def test_plan_mastering_strategy_changes_by_goal_and_platform(self) -> None:
        metrics = server.AudioMetrics(
            integrated_lufs=-14.0,
            true_peak_dbtp=-1.1,
            crest_db=10.5,
            stereo_correlation=0.18,
            duration_s=120.0,
            peak_dbfs=-0.4,
            rms_dbfs=-11.0,
            centroid_hz=3300.0,
        )
        with mock.patch.object(server, "_get_maestro", return_value=(self.fake_maestro, None)):
            with mock.patch.object(server, "_analyze_internal", return_value=metrics):
                cinematic = server.plan_mastering_strategy(
                    server.StrategyPlanIn(
                        audio_id="aud_1234567890ab",
                        goal="Wide cinematic depth with smooth highs",
                        platform="spotify",
                    ),
                    self.ctx,
                )
                club = server.plan_mastering_strategy(
                    server.StrategyPlanIn(
                        audio_id="aud_1234567890ab",
                        goal="Club-ready trap banger with heavy low end",
                        platform="club",
                    ),
                    self.ctx,
                )

        self.assertEqual(cinematic.chosen_preset, "cinematic")
        self.assertEqual(club.chosen_preset, "competitive_trap")
        self.assertNotEqual(cinematic.settings.target_lufs, club.settings.target_lufs)
        self.assertNotEqual(cinematic.settings.control_profile, club.settings.control_profile)

    def test_run_master_job_and_worker_preserve_control_profile_and_safe_overrides(self) -> None:
        audio_id = self._register_test_audio()
        before_metrics = server.AudioMetrics(
            integrated_lufs=-14.1,
            true_peak_dbtp=-1.2,
            crest_db=10.1,
            stereo_correlation=0.22,
            duration_s=120.0,
            peak_dbfs=-0.5,
            rms_dbfs=-11.3,
            centroid_hz=3000.0,
        )
        after_metrics = before_metrics.model_copy(update={"integrated_lufs": -12.3, "crest_db": 9.4})

        def analyze_side_effect(audio_handle: str, *_args, **_kwargs):
            return after_metrics if audio_handle.startswith("art_") else before_metrics

        req = server.MasterRequest(
            audio_id=audio_id,
            preset_name="hi_fi_streaming",
            control_profile=server.MasteringControlProfile(
                spatial_width=0.5,
                movement_amount=0.4,
                low_end_focus=0.6,
            ),
            governor_search_steps=4,
            governor_gr_limit_db=-2.1,
            stem_gains_db={"vocals": 1.0},
        )

        with mock.patch.object(server, "_get_maestro", return_value=(self.fake_maestro, None)):
            with mock.patch.object(server, "_analyze_internal", side_effect=analyze_side_effect):
                with mock.patch.object(server._JOB_EXECUTOR, "submit", return_value=FakeFuture()):
                    launch = server.run_master_job(req, self.ctx)
                queued = server._get_job(launch.job_id)
                self.assertIsNotNone(queued)
                self.assertGreater(queued.settings.warmth, 0.1)
                self.assertEqual(queued.settings.governor_search_steps, 4)
                self.assertEqual(queued.settings.stem_gains_db, {"vocals": 1.0})

                server._run_master_job_worker(launch.job_id)
                completed = server._get_job(launch.job_id)

        self.assertEqual(completed.status, "done")
        rendered = self.fake_maestro.rendered_presets[-1]
        self.assertEqual(rendered.governor_search_steps, 4)
        self.assertEqual(rendered.governor_gr_limit_db, -2.1)
        self.assertEqual(rendered.stem_gains_db, {"vocals": 1.0})
        self.assertGreater(rendered.width_hi, self.fake_maestro.get_presets()["hi_fi_streaming"].width_hi)

    def test_repaired_ai_tools_return_session_scoped_artifacts(self) -> None:
        audio_id = self._register_test_audio()
        before_metrics = server.AudioMetrics(
            integrated_lufs=-14.0,
            true_peak_dbtp=-1.1,
            crest_db=10.0,
            stereo_correlation=0.2,
            duration_s=100.0,
            peak_dbfs=-0.4,
            rms_dbfs=-10.4,
            centroid_hz=3200.0,
        )
        after_metrics = before_metrics.model_copy(update={"integrated_lufs": -12.1})

        def analyze_side_effect(audio_handle: str, *_args, **_kwargs):
            return after_metrics if audio_handle.startswith("art_") else before_metrics

        async def run_tools() -> None:
            stage1 = await server.start_interactive_mastering(
                server.StartInteractiveMasteringIn(audio_id=audio_id, preset_name="hi_fi_streaming"),
                self.ctx,
            )
            final = await server.commit_interactive_mastering(
                server.CommitInteractiveMasteringIn(
                    session_token=stage1.session_token,
                    warmth=0.3,
                    transient_boost_db=2.0,
                ),
                self.ctx,
            )
            ab = await server.semantic_a_b_mastering(
                server.SemanticABMasteringIn(
                    audio_id=audio_id,
                    preset_a="hi_fi_streaming",
                    preset_b="cinematic",
                ),
                self.ctx,
            )
            governor = await server.analyze_and_optimize_governor(
                server.AnalyzeAndOptimizeGovernorIn(audio_id=audio_id, preset_name="hi_fi_streaming"),
                self.ctx,
            )
            stems = await server.ai_stem_remix(server.AiStemRemixIn(audio_id=audio_id), self.ctx)

            self.assertTrue(final.artifact_id.startswith("art_"))
            self.assertTrue(ab.artifact_id_a.startswith("art_"))
            self.assertTrue(ab.artifact_id_b.startswith("art_"))
            self.assertGreaterEqual(governor.recommended_governor_steps, 3)
            self.assertIn("Example override", stems.mix_theory_advice)

        with mock.patch.object(server, "_get_maestro", return_value=(self.fake_maestro, None)):
            with mock.patch.object(server, "_analyze_internal", side_effect=analyze_side_effect):
                asyncio.run(run_tools())


if __name__ == "__main__":
    unittest.main()
