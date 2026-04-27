"""
Semantic planning, async job, and premium QC tests for AuralMind2.

Data shapes: fake Maestro presets, `AudioMetrics`, `MasterRequest`, job state,
artifact summaries, and quality reports.
Syntax: run with `python -m pytest tests/test_semantic_planning.py`.
Important tests: `SemanticPlanningTests` around line 160 covers intent planning,
job lifecycle, AI-assisted tools, validation, and stale persisted jobs.
Possible bugs: the fake DSP does not exercise full loudness rendering. Extend by
adding golden short-audio fixtures and trace JSON schema snapshots.
"""

import asyncio
import math
import os
import shutil
import unittest
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict
from unittest import mock

import numpy as np

import server
from pydantic import ValidationError


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
            "corr_lo": 0.94,
            "peak_dbfs": -0.4,
            "rms_dbfs": -10.4,
            "centroid_hz": 3100.0,
            "vocal_presence_db": -1.5,
            "low_mid_masking_db": 1.5,
            "sub_to_kick_balance_db": 2.0,
            "harshness_index_db": 0.6,
            "spectral_tilt_db": -4.0,
            "lra_proxy_db": 5.2,
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
        self.tempdir_path = os.path.join(os.getcwd(), "tests", "_runtime_semantic", uuid.uuid4().hex)
        os.makedirs(self.tempdir_path, exist_ok=True)
        self.ctx = FakeContext()
        self.fake_maestro = FakeMaestro()
        self.storage_patch = mock.patch.object(server, "STORAGE_DIR", self.tempdir_path)
        self.db_patch = mock.patch.object(server, "MAESTRO_DB_PATH", os.path.join(self.tempdir_path, "maestro_state.db"))
        self.storage_patch.start()
        self.db_patch.start()
        os.makedirs(server.STORAGE_DIR, exist_ok=True)
        server._init_db()
        server._ARTIFACTS.clear()
        server._JOBS.clear()
        server.INTERACTIVE_SESSIONS.clear()

    def tearDown(self) -> None:
        self.storage_patch.stop()
        self.db_patch.stop()
        server._ARTIFACTS.clear()
        server._JOBS.clear()
        server.INTERACTIVE_SESSIONS.clear()
        shutil.rmtree(self.tempdir_path, ignore_errors=True)

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
            low_band_correlation=0.88,
            high_band_correlation=0.18,
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

    def test_plan_mastering_strategy_uses_premium_material_metrics(self) -> None:
        metrics = server.AudioMetrics(
            integrated_lufs=-14.4,
            true_peak_dbtp=-1.4,
            crest_db=10.0,
            stereo_correlation=0.8,
            duration_s=142.0,
            peak_dbfs=-0.5,
            rms_dbfs=-11.2,
            centroid_hz=4300.0,
            low_band_correlation=0.52,
            high_band_correlation=0.22,
            vocal_presence_db=-6.2,
            low_mid_masking_db=7.1,
            sub_to_kick_balance_db=8.4,
            harshness_index_db=2.9,
            spectral_tilt_db=-5.6,
            lra_proxy_db=4.0,
        )
        with mock.patch.object(server, "_get_maestro", return_value=(self.fake_maestro, None)):
            with mock.patch.object(server, "_analyze_internal", return_value=metrics):
                plan = server.plan_mastering_strategy(
                    server.StrategyPlanIn(
                        audio_id="aud_1234567890ab",
                        goal="Melodic trap vocal master with clear hook and controlled 808",
                        platform="spotify",
                    ),
                    self.ctx,
                )

        self.assertEqual(plan.chosen_preset, "radio_loud")
        self.assertIsNotNone(plan.settings.control_profile)
        self.assertLessEqual(plan.settings.control_profile.spatial_width, 0.1)
        self.assertLessEqual(plan.settings.control_profile.low_end_focus, 0.35)
        self.assertTrue(any("Low-band correlation" in warning for warning in plan.warnings))
        self.assertTrue(any("Low-mid masking" in warning for warning in plan.warnings))

    def test_analyze_audio_surfaces_premium_material_metrics(self) -> None:
        audio_id = self._register_test_audio()
        with mock.patch.object(server, "_get_maestro", return_value=(self.fake_maestro, None)):
            result = server.analyze_audio(server.AnalyzeIn(audio_id=audio_id), self.ctx)

        self.assertEqual(result.metrics.low_band_correlation, 0.94)
        self.assertEqual(result.metrics.high_band_correlation, 0.3)
        self.assertEqual(result.metrics.vocal_presence_db, -1.5)
        self.assertEqual(result.metrics.sub_to_kick_balance_db, 2.0)

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
            low_band_correlation=0.93,
            high_band_correlation=0.24,
            vocal_presence_db=-1.8,
            low_mid_masking_db=1.8,
            sub_to_kick_balance_db=3.0,
            harshness_index_db=0.9,
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

        self.assertIsNotNone(completed)
        self.assertEqual(completed.status, "done")
        rendered = self.fake_maestro.rendered_presets[-1]
        self.assertEqual(rendered.governor_search_steps, 4)
        self.assertEqual(rendered.governor_gr_limit_db, -2.1)
        self.assertEqual(rendered.stem_gains_db, {"vocals": 1.0})
        self.assertGreater(rendered.width_hi, self.fake_maestro.get_presets()["hi_fi_streaming"].width_hi)
        self.assertGreater(rendered.mono_sub_base_mix, self.fake_maestro.get_presets()["hi_fi_streaming"].mono_sub_base_mix)

        result = server.job_result(server.JobIdIn(job_id=launch.job_id), self.ctx)
        artifact_kinds = {artifact.kind for artifact in result.artifacts}
        self.assertEqual(result.master_wav_id, completed.result.master_wav_id)
        self.assertEqual(result.tuning_trace_id, completed.result.tuning_trace_id)
        self.assertIn("mastered_audio", artifact_kinds)
        self.assertIn("trace", artifact_kinds)
        self.assertIsNotNone(result.quality_report)
        self.assertIn(result.quality_report.verdict, {"premium_ready", "premium_with_cautions", "review_recommended"})

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

    def test_master_settings_validation_rejects_unsafe_values(self) -> None:
        with self.assertRaises(ValidationError):
            server.MasterSettings(target_lufs=-5.0)
        with self.assertRaises(ValidationError):
            server.MasterSettings(stem_gains_db={"vocals": 10.0})
        with self.assertRaises(ValueError):
            server._as_bit_depth("24bit")
        with self.assertRaises(ValueError):
            server._as_stem_mode("maybe")

    def test_persisted_running_job_is_recovered_as_stale(self) -> None:
        session_key, session_dir = server._get_session_info(self.ctx)
        job = server.JobState(
            job_id="job_1234567890ab",
            audio_id="aud_1234567890ab",
            status="running",
            progress=45,
            settings=server.MasterSettings(),
            session_key=session_key,
            session_dir=session_dir,
        )
        server._set_job_in_db(job)
        server._JOBS.clear()

        recovered = server._get_job(job.job_id)
        persisted = server._get_job_from_db(job.job_id)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, "error")
        self.assertEqual(recovered.error.code, "stale_job_recovered")
        self.assertEqual(recovered.error.details["previous_status"], "running")
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, "error")


if __name__ == "__main__":
    unittest.main()
