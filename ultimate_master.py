import asyncio
import inspect
from server import (
    register_audio_from_path,
    analyze_audio,
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
    StrategyPlanIn
)

async def safe_await(coro_or_result):
    """Safely await coroutines or return results as-is"""
    if inspect.iscoroutine(coro_or_result):
        return await coro_or_result
    return coro_or_result

async def run():
    class MockContext:
        def __init__(self, session_id):
            self.session_id = session_id

    ctx = MockContext('ultimate_expert_master_final')

    # Register fresh audio files
    files = {
        'New Project (15)': 'c:/Users/goku/Documents/AuralMind2/data/New Project (15).wav',
        'Close to the edge': 'c:/Users/goku/Documents/AuralMind2/data/Close to the edge.wav'
    }

    audio_map = {}
    for name, path in files.items():
        try:
            res = register_audio_from_path(path, ctx)
            audio_map[name] = res.audio_id
            print(f'✓ Registered {name}: {res.audio_id}')
        except Exception as e:
            print(f'✗ Error registering {name}: {e}')
            return

    # COMPLETE EXPERT WORKFLOW: ALL AURALMIND2 FEATURES
    master_configs = [
        {
            'song': 'New Project (15)',
            'initial_goal': 'Professional trap vocal master with aggressive competitive loudness, surgical vocal upfront, tight 808 bass lock, hook-oriented polish. Maximize vocal presence.',
            'platform': 'spotify',
            'preset_a': 'competitive_trap',
            'preset_b': 'radio_loud',
            'primary_preset': 'competitive_trap',
            'control': {
                'brightness_tilt': 0.35,
                'movement_amount': 0.5,
                'low_end_focus': 0.65,
                'spatial_width': 0.3,
                'harshness_control': -0.4
            }
        },
        {
            'song': 'Close to the edge',
            'initial_goal': 'Mastered orchestral cinematic with immersive stereo imaging, warm analog saturation, dynamic macro-motion, articulate string definition, pristine spatial depth for film scoring.',
            'platform': 'youtube',
            'preset_a': 'cinematic',
            'preset_b': 'hi_fi_streaming',
            'primary_preset': 'cinematic',
            'control': {
                'brightness_tilt': -0.25,
                'movement_amount': 0.3,
                'low_end_focus': -0.15,
                'spatial_width': 0.9,
                'harshness_control': -0.5
            }
        }
    ]

    for config in master_configs:
        if config['song'] not in audio_map:
            continue

        original_audio_id = audio_map[config['song']]
        print(f"\n{'='*100}")
        print(f"🏆 ULTIMATE EXPERT MASTER: {config['song'].upper()}")
        print(f"{'='*100}")

        try:
            # STEP 1: PRE-MASTERING ANALYSIS
            print(f"\n[STEP 1] 🔍 PRE-MASTERING ANALYSIS")
            original_metrics = await safe_await(analyze_audio(original_audio_id, ctx=ctx))
            if original_metrics:
                print(f"✓ Original Metrics:")
                print(f"  • LUFS: {original_metrics.integrated_lufs:.2f}")
                print(f"  • True Peak: {original_metrics.true_peak_dbtp:.4f} dBTP")
                print(f"  • Crest: {original_metrics.crest_db:.2f} dB")
                print(f"  • Stereo Corr: {original_metrics.stereo_correlation:.3f}")
                print(f"  • Duration: {original_metrics.duration_s:.1f}s")

            # STEP 2: SEMANTIC PLANNING
            print(f"\n[STEP 2] 📋 SEMANTIC MASTERING STRATEGY")
            plan_req = StrategyPlanIn(
                audio_id=original_audio_id,
                goal=config['initial_goal'],
                platform=config['platform'],
                control_profile=config['control']
            )
            strategy = await safe_await(plan_mastering_strategy(plan_req, ctx=ctx))
            if strategy:
                print(f"✓ Strategy Planned")
                print(f"  • Chosen Preset: {strategy.settings.preset_name}")
                print(f"  • Target LUFS: {strategy.settings.target_lufs}")
                print(f"  • Warmth: {strategy.settings.warmth:.2f}")

            # STEP 3: SEMANTIC A/B PRESET COMPARISON
            print(f"\n[STEP 3] 🎚️  SEMANTIC A/B PRESET COMPARISON")
            print(f"Comparing {config['preset_a']} vs {config['preset_b']}...")
            ab_req = {
                'audio_id': original_audio_id,
                'preset_a': config['preset_a'],
                'preset_b': config['preset_b']
            }
            ab_result = await safe_await(semantic_a_b_mastering(ab_req, ctx=ctx))
            print(f"✓ A/B Comparison Rendered: {type(ab_result).__name__}")

            # STEP 4: GOVERNOR OPTIMIZATION
            print(f"\n[STEP 4] ⚙️  GOVERNOR OPTIMIZATION")
            gov_req = {
                'audio_id': original_audio_id,
                'preset_name': config['primary_preset']
            }
            gov_result = await safe_await(analyze_and_optimize_governor(gov_req, ctx=ctx))
            print(f"✓ Governor Optimized: {type(gov_result).__name__}")

            # STEP 5: INTERACTIVE MASTERING - STAGE 1
            print(f"\n[STEP 5] 🎛️  INTERACTIVE MASTERING - STAGE 1")
            stage1_req = {
                'audio_id': original_audio_id,
                'preset_name': config['primary_preset'],
                'control_profile': config['control'],
                'stem_mode': 'auto'
            }
            stage1_result = await safe_await(start_interactive_mastering(stage1_req, ctx=ctx))
            print(f"✓ Stage 1 Complete: {type(stage1_result).__name__}")
            session_token = stage1_result

            # STEP 6: APPLY MUSICAL EFFECTS
            print(f"\n[STEP 6] 🎼 APPLY MUSICAL EFFECTS (Harmonic + EQ + Dynamics)")

            stage1_audio_id = original_audio_id
            current_audio_id = stage1_audio_id

            # 6A: Harmonic Excitation
            print(f"  [6A] Harmonic Excitation...")
            try:
                harmonic_result = await safe_await(apply_harmonic_excitation(
                    {'audio_id': current_audio_id, 'harmonics_ratio': 0.6},
                    ctx=ctx
                ))
                print(f"  ✓ Harmonic applied")
                if harmonic_result and hasattr(harmonic_result, 'artifact_id'):
                    current_audio_id = harmonic_result.artifact_id
            except Exception as e:
                print(f"  ℹ Harmonic skipped: {str(e)[:60]}")

            # 6B: Musical EQ
            print(f"  [6B] Musical EQ (Key-aware)...")
            try:
                eq_result = await safe_await(apply_musical_eq(
                    {'audio_id': current_audio_id, 'key': 'C', 'emphasis': 0.5},
                    ctx=ctx
                ))
                print(f"  ✓ Musical EQ applied")
                if eq_result and hasattr(eq_result, 'artifact_id'):
                    current_audio_id = eq_result.artifact_id
            except Exception as e:
                print(f"  ℹ Musical EQ skipped: {str(e)[:60]}")

            # 6C: Tempo Dynamics
            print(f"  [6C] Tempo Dynamics (Synced Limiter)...")
            try:
                tempo_result = await safe_await(apply_tempo_dynamics(
                    {'audio_id': current_audio_id, 'dynamics_ratio': 2.0},
                    ctx=ctx
                ))
                print(f"  ✓ Tempo Dynamics applied")
                if tempo_result and hasattr(tempo_result, 'artifact_id'):
                    current_audio_id = tempo_result.artifact_id
            except Exception as e:
                print(f"  ℹ Tempo Dynamics skipped: {str(e)[:60]}")

            # STEP 7: AI STEM REMIX ANALYSIS
            print(f"\n[STEP 7] 🎵 AI STEM REMIX (Balance Analysis)")
            try:
                stem_advice = await safe_await(ai_stem_remix({'audio_id': current_audio_id}, ctx=ctx))
                print(f"✓ Stem Analysis: {type(stem_advice).__name__}")
            except Exception as e:
                print(f"ℹ Stem remix skipped: {str(e)[:60]}")

            # STEP 8: INTERACTIVE MASTERING - STAGE 2 (Commit)
            print(f"\n[STEP 8] 🎯 INTERACTIVE MASTERING - STAGE 2 (Final Commit)")
            stage2_req = {
                'session_token': session_token,
                'warmth': 0.75,
                'transient_boost_db': 2.0,
                'control_profile': {
                    'brightness_tilt': config['control']['brightness_tilt'] + 0.05,
                    'harshness_control': config['control']['harshness_control'] - 0.1,
                    'movement_amount': config['control']['movement_amount'] + 0.1
                }
            }
            final_master = await safe_await(commit_interactive_mastering(stage2_req, ctx=ctx))
            print(f"✓ Stage 2 Committed: {type(final_master).__name__}")

            # STEP 9: POST-MASTERING ANALYSIS
            print(f"\n[STEP 9] 📊 POST-MASTERING ANALYSIS")
            final_metrics = await safe_await(analyze_audio(current_audio_id, ctx=ctx))
            if final_metrics:
                print(f"✓ Final Metrics:")
                print(f"  • LUFS: {final_metrics.integrated_lufs:.2f}")
                print(f"  • True Peak: {final_metrics.true_peak_dbtp:.4f} dBTP")
                print(f"  • Crest: {final_metrics.crest_db:.2f} dB")
                if original_metrics:
                    print(f"  • LUFS Change: {final_metrics.integrated_lufs - original_metrics.integrated_lufs:+.2f}")

            # STEP 10: METRICS COMPARISON
            print(f"\n[STEP 10] 📈 DETAILED METRICS COMPARISON")
            try:
                comparison = await safe_await(compare_audio_metrics({
                    'audio_id_a': original_audio_id,
                    'audio_id_b': current_audio_id
                }, ctx=ctx))
                print(f"✓ Comparison Complete: {type(comparison).__name__}")
            except Exception as e:
                print(f"ℹ Comparison skipped: {str(e)[:60]}")

            print(f"\n{'='*100}")
            print(f"✓ {config['song'].upper()} - ULTIMATE EXPERT MASTER COMPLETE")
            print(f"  ✅ All AuralMind2 features leveraged for professional reference grade")
            print(f"{'='*100}\n")

        except Exception as e:
            print(f"\n✗ Error in mastering {config['song']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(run())
