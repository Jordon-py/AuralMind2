"""
AudioAnalysisEngine Agent
Parallel audio analysis and metrics computation
"""

import asyncio
import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy import signal
from scipy.fft import fft
import soundfile as sf
from dataclasses import dataclass
import json


@dataclass
class AudioMetrics:
    """Complete audio analysis metrics"""
    duration: float
    sample_rate: int
    channels: int

    # Loudness metrics
    rms: float
    lufs: float
    true_peak: float

    # Dynamic metrics
    crest_factor: float
    dynamic_range: float

    # Frequency metrics
    peak_freq: float
    spectral_centroid: float
    spectral_flatness: float

    # Stereo metrics
    stereo_correlation: float
    phase_coherence: float

    # Temporal metrics
    onset_count: int
    zero_crossing_rate: float

    @property
    def quality_score(self) -> float:
        """Predict mastering quality (0-100)"""
        # Composite scoring based on metrics
        loudness_score = min(100, (abs(self.lufs + 14) / 14) * 100)
        dynamics_score = min(100, (self.crest_factor / 24) * 100)
        freq_score = min(100, self.spectral_flatness * 100)
        stereo_score = min(100, self.stereo_correlation * 100)

        return (loudness_score * 0.35 + dynamics_score * 0.25 +
                freq_score * 0.25 + stereo_score * 0.15)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'loudness': {
                'rms': float(self.rms),
                'lufs': float(self.lufs),
                'true_peak': float(self.true_peak)
            },
            'dynamics': {
                'crest_factor': float(self.crest_factor),
                'dynamic_range': float(self.dynamic_range)
            },
            'frequency': {
                'peak_freq': float(self.peak_freq),
                'spectral_centroid': float(self.spectral_centroid),
                'spectral_flatness': float(self.spectral_flatness)
            },
            'stereo': {
                'correlation': float(self.stereo_correlation),
                'phase_coherence': float(self.phase_coherence)
            },
            'temporal': {
                'onset_count': int(self.onset_count),
                'zero_crossing_rate': float(self.zero_crossing_rate)
            },
            'quality_score': self.quality_score
        }


class AudioAnalysisEngine:
    """
    Expert async agent for parallel audio analysis

    Features:
    - Multi-threaded audio analysis (loudness, frequency, dynamics, stereo)
    - Efficient batch processing
    - Metadata caching
    - Support for various audio formats
    """

    # ITU-R BS.1770-4 loudness reference
    LOUDNESS_REF = -23.0  # LUFS

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize analysis engine

        Args:
            cache_enabled: Cache analysis results
        """
        self.cache_enabled = cache_enabled
        self.analysis_cache: Dict[str, AudioMetrics] = {}
        self.analysis_stats = {
            'analyses_run': 0,
            'cache_hits': 0,
            'avg_analysis_time': 0.0
        }

    async def analyze_audio_file(self, audio_path: str, cache_key: Optional[str] = None) -> AudioMetrics:
        """
        Comprehensive async audio analysis

        Args:
            audio_path: Path to audio file
            cache_key: Optional cache key

        Returns:
            AudioMetrics: Complete analysis results
        """
        # Check cache
        if self.cache_enabled and cache_key and cache_key in self.analysis_cache:
            self.analysis_stats['cache_hits'] += 1
            print(f"[AudioAnalysis] Cache hit for {cache_key}")
            return self.analysis_cache[cache_key]

        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=1)

            # Run parallel analysis tasks
            metrics = await self._analyze_parallel(audio, sr)

            # Cache result
            if self.cache_enabled and cache_key:
                self.analysis_cache[cache_key] = metrics

            self.analysis_stats['analyses_run'] += 1
            print(f"[AudioAnalysis] Analyzed {audio_path}: "
                  f"LUFS={metrics.lufs:.1f}, Quality={metrics.quality_score:.0f}/100")

            return metrics

        except Exception as e:
            print(f"[AudioAnalysis] Error analyzing {audio_path}: {e}")
            raise

    async def analyze_audio_data(self, audio: np.ndarray, sr: int) -> AudioMetrics:
        """Analyze audio data from memory"""
        return await self._analyze_parallel(audio, sr)

    async def _analyze_parallel(self, audio: np.ndarray, sr: int) -> AudioMetrics:
        """Run all analysis tasks in parallel"""
        # Create async tasks for independent analyses
        tasks = [
            asyncio.create_task(self._analyze_loudness(audio, sr)),
            asyncio.create_task(self._analyze_dynamics(audio, sr)),
            asyncio.create_task(self._analyze_frequency(audio, sr)),
            asyncio.create_task(self._analyze_stereo(audio, sr)),
            asyncio.create_task(self._analyze_temporal(audio, sr))
        ]

        # Wait for all to complete
        loudness, dynamics, frequency, stereo, temporal = await asyncio.gather(*tasks)

        # Combine results
        return AudioMetrics(
            duration=len(audio) / sr,
            sample_rate=sr,
            channels=audio.shape[1] if audio.ndim > 1 else 1,
            **loudness,
            **dynamics,
            **frequency,
            **stereo,
            **temporal
        )

    async def _analyze_loudness(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Loudness analysis: RMS, LUFS, true peak"""
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_loudness, audio, sr)

    def _compute_loudness(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute loudness metrics"""
        # Mono conversion if needed
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        # RMS
        rms = np.sqrt(np.mean(mono ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Simplified LUFS (true ITU-R BS.1770-4 would use K-weighting)
        lufs = rms_db - 0.691  # Approximate calibration

        # True peak
        true_peak = np.max(np.abs(audio))
        true_peak_db = 20 * np.log10(true_peak + 1e-10)

        return {
            'rms': rms,
            'lufs': lufs,
            'true_peak': true_peak_db
        }

    async def _analyze_dynamics(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Dynamic range analysis: crest factor, dynamic range"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_dynamics, audio)

    def _compute_dynamics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute dynamics metrics"""
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        peak = np.max(np.abs(mono))
        rms = np.sqrt(np.mean(mono ** 2))

        # Crest factor (peak to RMS ratio in dB)
        crest_factor = 20 * np.log10(peak / (rms + 1e-10))

        # Dynamic range (using percentiles)
        sorted_abs = np.sort(np.abs(mono))
        high = np.percentile(sorted_abs, 95)
        low = np.percentile(sorted_abs, 5)
        dynamic_range = 20 * np.log10(high / (low + 1e-10))

        return {
            'crest_factor': crest_factor,
            'dynamic_range': dynamic_range
        }

    async def _analyze_frequency(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Frequency analysis: spectrum, centroid, flatness"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_frequency, audio, sr)

    def _compute_frequency(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute frequency metrics"""
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        # FFT
        spectrum = np.abs(fft(mono))[:len(mono)//2]
        freqs = np.fft.fftfreq(len(mono), 1/sr)[:len(mono)//2]

        # Peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]

        # Spectral centroid (center of mass in frequency domain)
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        # Spectral flatness (Wiener entropy)
        normalized = spectrum / (np.sum(spectrum) + 1e-10)
        flatness = np.exp(-np.sum(normalized * np.log(normalized + 1e-10)))

        return {
            'peak_freq': peak_freq,
            'spectral_centroid': centroid,
            'spectral_flatness': flatness
        }

    async def _analyze_stereo(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Stereo analysis: correlation, phase coherence"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_stereo, audio)

    def _compute_stereo(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute stereo metrics"""
        if audio.ndim == 1 or audio.shape[1] == 1:
            # Mono
            return {
                'stereo_correlation': 1.0,
                'phase_coherence': 1.0
            }

        left = audio[:, 0]
        right = audio[:, 1] if audio.shape[1] > 1 else left

        # Correlation
        correlation = np.corrcoef(left, right)[0, 1]

        # Phase coherence (simplified - using FFT bins)
        left_fft = fft(left)[:len(left)//2]
        right_fft = fft(right)[:len(right)//2]

        phase_diff = np.angle(left_fft) - np.angle(right_fft)
        phase_coherence = np.mean(np.cos(phase_diff))

        return {
            'stereo_correlation': correlation,
            'phase_coherence': phase_coherence
        }

    async def _analyze_temporal(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Temporal analysis: onsets, zero crossing rate"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_temporal, audio, sr)

    def _compute_temporal(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute temporal metrics"""
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        # Zero crossing rate
        zero_crossings = np.abs(np.diff(np.sign(mono))).sum()
        zcr = zero_crossings / len(mono)

        # Onset detection (energy-based)
        frame_size = 2048
        frames = np.array_split(mono, len(mono) // frame_size)
        frame_energy = [np.mean(f ** 2) for f in frames]

        onset_count = 0
        for i in range(1, len(frame_energy)):
            if frame_energy[i] > frame_energy[i-1] * 1.5:  # 50% energy increase
                onset_count += 1

        return {
            'onset_count': onset_count,
            'zero_crossing_rate': zcr
        }

    async def batch_analyze(self, audio_files: list) -> list:
        """Analyze multiple files concurrently"""
        tasks = [
            self.analyze_audio_file(path, cache_key=path)
            for path in audio_files
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> Dict:
        """Get analysis engine statistics"""
        return {
            **self.analysis_stats,
            'cache_size': len(self.analysis_cache),
            'cache_efficiency': (
                self.analysis_stats['cache_hits'] /
                max(1, self.analysis_stats['analyses_run']) * 100
            )
        }

    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        print("[AudioAnalysis] Cache cleared")
