"""
Premium DSP regression tests for the AuralMind2 Maestro engine.

Data shapes: float32 stereo arrays shaped `(samples, 2)` and mid/side vectors.
Syntax: run with `python -m pytest tests/test_maestro_premium_dsp.py`.
Important tests: `MaestroPremiumDSPTests` around line 22 validates mono-sub
translation behavior in `tools.auralmind_maestro.mono_sub_v2`.
Possible bugs: synthetic tones do not prove release quality on full songs.
Extend by adding short real-audio fixtures and loudness-governor golden checks.
"""

import math
import unittest

import numpy as np

from tools import auralmind_maestro as maestro


def _lowpass(signal: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    sos = maestro.sps.butter(2, cutoff_hz / (sr * 0.5), btype="lowpass", output="sos")
    return maestro.sps.sosfiltfilt(sos, signal).astype(np.float32)


class MaestroPremiumDSPTests(unittest.TestCase):
    def test_analyze_track_features_reports_premium_band_metrics(self) -> None:
        sr = 48_000
        t = np.arange(sr, dtype=np.float32) / sr
        mono = (
            0.25 * np.sin(2.0 * math.pi * 60.0 * t)
            + 0.15 * np.sin(2.0 * math.pi * 220.0 * t)
            + 0.06 * np.sin(2.0 * math.pi * 2_500.0 * t)
        ).astype(np.float32)
        audio = np.stack([mono, mono], axis=1)

        features = maestro.analyze_track_features(audio, sr)

        for key in (
            "corr_lo",
            "vocal_presence_db",
            "low_mid_masking_db",
            "sub_to_kick_balance_db",
            "harshness_index_db",
            "spectral_tilt_db",
            "lra_proxy_db",
        ):
            self.assertIn(key, features)
            self.assertTrue(np.isfinite(features[key]))

    def test_mono_sub_v2_reduces_low_band_side_energy_when_mix_increases(self) -> None:
        sr = 48_000
        t = np.arange(sr, dtype=np.float32) / sr
        side_only_sub = 0.35 * np.sin(2.0 * math.pi * 55.0 * t)
        audio = np.stack([side_only_sub, -side_only_sub], axis=1).astype(np.float32)

        _, side_before = maestro.mid_side_encode(audio)
        processed, cutoff_hz, mono_mix = maestro.mono_sub_v2(audio, sr, f0_hz=45.0, base_mix=0.65)
        _, side_after = maestro.mid_side_encode(processed)

        low_side_before = _lowpass(side_before, sr, cutoff_hz)
        low_side_after = _lowpass(side_after, sr, cutoff_hz)
        before_rms = float(np.sqrt(np.mean(low_side_before.astype(np.float64) ** 2)))
        after_rms = float(np.sqrt(np.mean(low_side_after.astype(np.float64) ** 2)))

        self.assertGreaterEqual(mono_mix, 0.65)
        self.assertLess(after_rms, before_rms * 0.55)


if __name__ == "__main__":
    unittest.main()
