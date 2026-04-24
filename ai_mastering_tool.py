import asyncio
import inspect
from server import (
    register_audio_from_path,
    analyze_audio,
    job_status,
    job_result,
    run_master_job,
    list_session_state,
    read_artifact,
    plan_mastering_strategy,
    semantic_a_b_mastering,
    analyze_and_optimize_governor,
    start_interactive_mastering,
    commit_interactive_mastering,
    apply_harmonic_excitation,
    apply_musical_eq,
    apply_tempo_dynamics,
    ai_stem_remix,
    compare_audio_metrics,
    StrategyPlanIn,
    MasterRequest
)

class AIIntegratedMasteringTool:
    """
    Interactive AI-integrated mastering tool.
    Allows the AI assistant to guide, monitor, and participate in the mastering process.
    """

    def __init__(self, session_id='ai_integrated_master'):
        self.session_id = session_id
        self.ctx = self._create_context()
        self.active_jobs = {}
        self.audio_registry = {}
        self.mastering_log = []

    def _create_context(self):
        """Create a mock FastMCP context for tool calls"""
        class MockContext:
            def __init__(self, session_id):
                self.session_id = session_id
        return MockContext(self.session_id)

    async def safe_await(self, coro_or_result):
        """Safely await coroutines or return results as-is"""
        if inspect.iscoroutine(coro_or_result):
            return await coro_or_result
        return coro_or_result

    async def register_audio(self, name, path):
        """Register an audio file and track it"""
        print(f"\n[AI] Registering audio: {name}")
        try:
            res = register_audio_from_path(path, self.ctx)
            self.audio_registry[name] = res.audio_id
            print(f"✓ {name}: {res.audio_id}")
            print(f"  Size: {res.size_bytes / 1024 / 1024:.2f} MB")
            print(f"  Format: {res.format}")
            return res.audio_id
        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    async def analyze_audio_deep(self, audio_id, label=""):
        """Deep analysis of audio with AI commentary"""
        print(f"\n[AI] Analyzing audio{' - ' + label if label else ''}...")
        try:
            metrics = await self.safe_await(analyze_audio(audio_id, self.ctx))
            if metrics:
                print(f"\n📊 Audio Analysis Results:")
                print(f"  • Loudness (LUFS): {metrics.integrated_lufs:.2f}")
                print(f"  • True Peak (dBTP): {metrics.true_peak_dbtp:.4f}")
                print(f"  • Crest Factor: {metrics.crest_db:.2f} dB")
                print(f"  • Stereo Correlation: {metrics.stereo_correlation:.3f}")
                print(f"  • Duration: {metrics.duration_s:.1f}s")

                # AI Commentary
                self._provide_analysis_commentary(metrics)

                return metrics
        except Exception as e:
            print(f"✗ Analysis error: {e}")
        return None

    def _provide_analysis_commentary(self, metrics):
        """Provide intelligent AI commentary based on metrics"""
        print(f"\n[AI INSIGHT]:")

        # Loudness assessment
        if metrics.integrated_lufs < -18:
            print(f"  • Source is quite soft ({metrics.integrated_lufs:.1f} LUFS) - significant gain needed")
        elif metrics.integrated_lufs < -14:
            print(f"  • Source has moderate loudness ({metrics.integrated_lufs:.1f} LUFS) - manageable headroom")
        else:
            print(f"  • Source is already hot ({metrics.integrated_lufs:.1f} LUFS) - careful limiting required")

        # Crest factor assessment
        if metrics.crest_db > 20:
            print(f"  • High crest factor ({metrics.crest_db:.1f} dB) - dynamic material, can tolerate aggressive compression")
        elif metrics.crest_db > 15:
            print(f"  • Moderate crest factor ({metrics.crest_db:.1f} dB) - balanced dynamic range")
        else:
            print(f"  • Low crest factor ({metrics.crest_db:.1f} dB) - already compressed/limited source")

        # Stereo assessment
        if metrics.stereo_correlation > 0.85:
            print(f"  • High stereo correlation ({metrics.stereo_correlation:.2f}) - mostly mono-compatible, safe for width effects")
        elif metrics.stereo_correlation > 0.65:
            print(f"  • Good stereo correlation ({metrics.stereo_correlation:.2f}) - good stereo separation with mono safety")
        else:
            print(f"  • Wide stereo field ({metrics.stereo_correlation:.2f}) - careful with aggressive stereo effects")

    async def launch_mastering_job(self, audio_id, preset, stem_mode='auto'):
        """Launch a mastering job and track it"""
        print(f"\n[AI] Launching mastering job...")
        print(f"  • Audio ID: {audio_id}")
        print(f"  • Preset: {preset}")
        print(f"  • Stem Mode: {stem_mode}")

        try:
            req = MasterRequest(
                audio_id=audio_id,
                preset_name=preset,
                stem_mode=stem_mode
            )
            res = await self.safe_await(run_master_job(req, ctx=self.ctx))

            job_id = res.job_id
            self.active_jobs[job_id] = {
                'audio_id': audio_id,
                'preset': preset,
                'status': 'queued',
                'launched_at': asyncio.get_event_loop().time()
            }

            print(f"✓ Job launched: {job_id}")
            return job_id
        except Exception as e:
            print(f"✗ Job launch error: {e}")
            return None

    async def monitor_job(self, job_id, poll_interval=5, max_polls=120):
        """Monitor a job with live feedback"""
        print(f"\n[AI] Monitoring job {job_id}...")
        polls = 0

        while polls < max_polls:
            try:
                status_res = await self.safe_await(job_status(job_id, ctx=self.ctx))

                status = status_res.status
                progress = status_res.progress
                elapsed = status_res.elapsed_s

                print(f"  [{polls}] Status: {status} | Progress: {progress}% | Elapsed: {elapsed}s")

                if status in ('done', 'error'):
                    if status == 'error':
                        print(f"✗ Job failed: {status_res.error}")
                        return False
                    else:
                        print(f"✓ Job completed in {elapsed}s")
                        return await self._fetch_and_analyze_result(job_id)

                await asyncio.sleep(poll_interval)
                polls += 1

            except Exception as e:
                print(f"✗ Monitoring error: {e}")
                return False

        print(f"✗ Job monitoring timeout after {max_polls * poll_interval}s")
        return False

    async def _fetch_and_analyze_result(self, job_id):
        """Fetch and analyze mastering job result"""
        print(f"\n[AI] Fetching mastering results...")
        try:
            res = await self.safe_await(job_result(job_id, ctx=self.ctx))

            print(f"\n✓ Mastering Result:")
            print(f"  • Status: {res.status}")
            print(f"  • Artifacts: {len(res.artifacts)}")
            print(f"  • Precision: {res.precision}")

            if res.metrics:
                metrics = res.metrics
                print(f"\n📊 Output Metrics:")
                print(f"  • LUFS: {metrics.integrated_lufs:.2f}")
                print(f"  • True Peak: {metrics.true_peak_dbtp:.4f} dBTP")
                print(f"  • Crest: {metrics.crest_db:.2f} dB")
                print(f"  • Stereo Corr: {metrics.stereo_correlation:.3f}")

                # AI Commentary on mastering
                self._provide_mastering_commentary(metrics)

            # Store artifacts
            if res.artifacts:
                master_artifact = res.artifacts[0]
                return master_artifact.artifact_id

            return True
        except Exception as e:
            print(f"✗ Result fetch error: {e}")
            return False

    def _provide_mastering_commentary(self, metrics):
        """Provide AI commentary on mastering results"""
        print(f"\n[AI COMMENTARY]:")

        if -13 < metrics.integrated_lufs < -11:
            print(f"  ✓ Excellent loudness target ({metrics.integrated_lufs:.2f} LUFS) for streaming platforms")
        elif metrics.integrated_lufs < -13:
            print(f"  ⚠ Quieter master ({metrics.integrated_lufs:.2f} LUFS) - may leave competitive headroom unused")
        else:
            print(f"  ⚠ Hot master ({metrics.integrated_lufs:.2f} LUFS) - check peak levels")

        if metrics.true_peak_dbtp < 0:
            print(f"  ✓ Safe true peak ({metrics.true_peak_dbtp:.4f} dBTP) - no clipping risk")
        else:
            print(f"  ✗ TRUE PEAK EXCEEDED ({metrics.true_peak_dbtp:.4f} dBTP) - clipping detected")

    async def compare_presets(self, audio_id, preset_a, preset_b):
        """Compare two presets side-by-side"""
        print(f"\n[AI] Comparing presets: {preset_a} vs {preset_b}")

        try:
            comparison = await self.safe_await(semantic_a_b_mastering({
                'audio_id': audio_id,
                'preset_a': preset_a,
                'preset_b': preset_b
            }, ctx=self.ctx))

            print(f"✓ Comparison rendered")
            print(f"  Type: {type(comparison).__name__}")
            return comparison
        except Exception as e:
            print(f"✗ Comparison error: {e}")
            return None

    async def apply_effects_chain(self, audio_id, effects_config):
        """Apply a chain of musical effects"""
        print(f"\n[AI] Applying effects chain...")
        current_id = audio_id
        effects_applied = []

        # Harmonic Excitation
        if effects_config.get('harmonic'):
            print(f"  [1/3] Harmonic Excitation ({effects_config['harmonic']})...")
            try:
                result = await self.safe_await(apply_harmonic_excitation({
                    'audio_id': current_id,
                    'harmonics_ratio': effects_config['harmonic']
                }, ctx=self.ctx))
                if result and hasattr(result, 'artifact_id'):
                    current_id = result.artifact_id
                    effects_applied.append('harmonic')
                    print(f"  ✓ Harmonic applied")
            except Exception as e:
                print(f"  ℹ Harmonic skipped: {str(e)[:50]}")

        # Musical EQ
        if effects_config.get('eq_key'):
            print(f"  [2/3] Musical EQ (Key: {effects_config['eq_key']})...")
            try:
                result = await self.safe_await(apply_musical_eq({
                    'audio_id': current_id,
                    'key': effects_config['eq_key'],
                    'emphasis': effects_config.get('eq_emphasis', 0.5)
                }, ctx=self.ctx))
                if result and hasattr(result, 'artifact_id'):
                    current_id = result.artifact_id
                    effects_applied.append('musical_eq')
                    print(f"  ✓ Musical EQ applied")
            except Exception as e:
                print(f"  ℹ Musical EQ skipped: {str(e)[:50]}")

        # Tempo Dynamics
        if effects_config.get('tempo_dynamics'):
            print(f"  [3/3] Tempo Dynamics ({effects_config['tempo_dynamics']})...")
            try:
                result = await self.safe_await(apply_tempo_dynamics({
                    'audio_id': current_id,
                    'dynamics_ratio': effects_config['tempo_dynamics']
                }, ctx=self.ctx))
                if result and hasattr(result, 'artifact_id'):
                    current_id = result.artifact_id
                    effects_applied.append('tempo_dynamics')
                    print(f"  ✓ Tempo Dynamics applied")
            except Exception as e:
                print(f"  ℹ Tempo Dynamics skipped: {str(e)[:50]}")

        print(f"\n✓ Effects chain complete ({len(effects_applied)}/{len([k for k in effects_config if effects_config[k]])})")
        return current_id

    async def interactive_mastering_session(self, audio_id, preset, tweaks=None):
        """Run an interactive mastering session with AI control"""
        print(f"\n[AI] Starting interactive mastering session...")

        try:
            # Stage 1: Initial render
            print(f"\n  STAGE 1: Initial Preset Render")
            stage1 = await self.safe_await(start_interactive_mastering({
                'audio_id': audio_id,
                'preset_name': preset,
                'stem_mode': 'auto'
            }, ctx=self.ctx))

            session_token = stage1
            print(f"  ✓ Stage 1 complete")

            # Stage 2: Final tweaks
            if tweaks:
                print(f"\n  STAGE 2: AI-Guided Final Refinement")
                print(f"    • Warmth: {tweaks.get('warmth', 0.5)}")
                print(f"    • Transient Boost: {tweaks.get('transient_boost_db', 1.0)} dB")

                final = await self.safe_await(commit_interactive_mastering({
                    'session_token': session_token,
                    'warmth': tweaks.get('warmth', 0.5),
                    'transient_boost_db': tweaks.get('transient_boost_db', 1.0),
                    'control_profile': tweaks.get('control_profile')
                }, ctx=self.ctx))

                print(f"  ✓ Stage 2 complete")
                return final

            return stage1

        except Exception as e:
            print(f"✗ Session error: {e}")
            return None

    async def governer_optimization(self, audio_id, preset):
        """Optimize governor settings for best loudness/punch balance"""
        print(f"\n[AI] Analyzing governor optimization...")

        try:
            result = await self.safe_await(analyze_and_optimize_governor({
                'audio_id': audio_id,
                'preset_name': preset
            }, ctx=self.ctx))

            print(f"✓ Governor optimization complete")
            print(f"  Type: {type(result).__name__}")
            return result
        except Exception as e:
            print(f"✗ Governor error: {e}")
            return None

    async def stem_analysis(self, audio_id):
        """AI-powered stem mix analysis"""
        print(f"\n[AI] Analyzing vocal/bass/drums balance...")

        try:
            advice = await self.safe_await(ai_stem_remix({
                'audio_id': audio_id
            }, ctx=self.ctx))

            print(f"✓ Stem analysis complete")
            print(f"  Result: {type(advice).__name__}")
            return advice
        except Exception as e:
            print(f"✗ Stem analysis error: {e}")
            return None

    async def compare_masters(self, audio_id_before, audio_id_after):
        """Compare before/after masters"""
        print(f"\n[AI] Comparing masters (before vs after)...")

        try:
            comparison = await self.safe_await(compare_audio_metrics({
                'audio_id_a': audio_id_before,
                'audio_id_b': audio_id_after
            }, ctx=self.ctx))

            print(f"✓ Comparison complete")
            return comparison
        except Exception as e:
            print(f"✗ Comparison error: {e}")
            return None

    def print_session_summary(self):
        """Print AI-controlled mastering session summary"""
        print(f"\n{'='*100}")
        print(f"🤖 AI INTEGRATED MASTERING TOOL - SESSION SUMMARY")
        print(f"{'='*100}")
        print(f"\n📋 Registered Audio Files:")
        for name, audio_id in self.audio_registry.items():
            print(f"  • {name}: {audio_id}")

        print(f"\n🎯 Active Jobs:")
        for job_id, info in self.active_jobs.items():
            print(f"  • {job_id}")
            print(f"    - Audio: {info['audio_id']}")
            print(f"    - Preset: {info['preset']}")
            print(f"    - Status: {info['status']}")

# ============================================================================
# EXAMPLE USAGE: AI-GUIDED MASTERING WORKFLOW
# ============================================================================

async def ai_mastering_demo():
    """Demonstration of AI-integrated mastering"""

    tool = AIIntegratedMasteringTool(session_id='ai_demo_master')

    print(f"\n{'='*100}")
    print(f"🤖 AI-INTEGRATED MASTERING TOOL")
    print(f"{'='*100}")
    print(f"\nThis tool allows the AI assistant to:")
    print(f"  ✓ Register and analyze audio files")
    print(f"  ✓ Launch and monitor mastering jobs")
    print(f"  ✓ Make intelligent decisions about presets and effects")
    print(f"  ✓ Apply effects chains intelligently")
    print(f"  ✓ Run interactive mastering sessions")
    print(f"  ✓ Compare and optimize masters")
    print(f"  ✓ Provide real-time AI insights and commentary")

    # Register audio
    audio_id_1 = await tool.register_audio(
        'New Project (15)',
        'c:/Users/goku/Documents/AuralMind2/data/New Project (15).wav'
    )

    if audio_id_1:
        # Deep analysis
        metrics = await tool.analyze_audio_deep(audio_id_1, label="Initial Assessment")

        # Governor optimization
        gov = await tool.governer_optimization(audio_id_1, 'competitive_trap')

        # Preset comparison
        comparison = await tool.compare_presets(audio_id_1, 'competitive_trap', 'radio_loud')

        # Launch mastering job
        job_id = await tool.launch_mastering_job(audio_id_1, 'competitive_trap')

        if job_id:
            # Monitor job
            result = await tool.monitor_job(job_id, poll_interval=3)

            if result:
                # Apply effects if result is an artifact ID
                if isinstance(result, str):
                    effects_config = {
                        'harmonic': 0.6,
                        'eq_key': 'C',
                        'eq_emphasis': 0.5,
                        'tempo_dynamics': 2.0
                    }
                    final_id = await tool.apply_effects_chain(result, effects_config)

                    # Final analysis
                    if final_id:
                        final_metrics = await tool.analyze_audio_deep(final_id, label="Final Master")

                        # Compare before/after
                        comparison_result = await tool.compare_masters(audio_id_1, final_id)

    # Print summary
    tool.print_session_summary()

if __name__ == '__main__':
    print("🤖 AI-Integrated AuralMind2 Mastering Tool")
    print("=" * 100)
    print("\nUsage:")
    print("  from ai_mastering_tool import AIIntegratedMasteringTool")
    print("  tool = AIIntegratedMasteringTool()")
    print("  await tool.register_audio('Song', '/path/to/audio.wav')")
    print("  await tool.analyze_audio_deep(audio_id)")
    print("  job_id = await tool.launch_mastering_job(audio_id, 'preset_name')")
    print("  await tool.monitor_job(job_id)")
    print("\nTo run demo: asyncio.run(ai_mastering_demo())")
    print("=" * 100)
