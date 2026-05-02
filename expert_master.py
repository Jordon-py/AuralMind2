import asyncio
from server import register_audio_from_path, plan_mastering_strategy, start_interactive_mastering, commit_interactive_mastering, StrategyPlanIn

async def run():
    class MockContext:
        def __init__(self, session_id):
            self.session_id = session_id

    ctx = MockContext('expert_tier_master_2026')

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

    # EXPERT TIER MASTERING: Staged interactive refinement with semantic planning
    expert_tasks = [
        {
            'song': 'New Project (15)',
            'stage1_goal': 'Professional trap vocal master: aggressive competitive loudness exceeding -11.4 LUFS Spotify standard, surgical vocal upfront presence without sibilance fatigue, 808 bass locked in tight stereo mono-safe with surgical low-end control, modern hook-oriented production polish. Platform: Spotify competitive positioning.',
            'platform': 'spotify',
            'stage1_preset': 'competitive_trap',
            'stage1_control': {
                'brightness_tilt': 0.35,
                'movement_amount': 0.5,
                'low_end_focus': 0.65,
                'spatial_width': 0.3,
                'harshness_control': -0.4
            },
            'stage2_tweaks': {
                'warmth': 0.7,
                'transient_boost_db': 2.5,
                'control_profile': {
                    'brightness_tilt': 0.4,
                    'harshness_control': -0.5
                }
            }
        },
        {
            'song': 'Close to the edge',
            'stage1_goal': 'Mastered orchestral cinematic reference: immersive wide stereo imaging with pristine transient clarity, warm analog saturation character minimizing digital glare, dynamic macro-motion for emotional arc emphasis, articulate string definition with cello richness, spatial depth suggesting large ensemble hall acoustics without bloat. Platform: YouTube premium reference grade.',
            'platform': 'youtube',
            'stage1_preset': 'cinematic',
            'stage1_control': {
                'brightness_tilt': -0.25,
                'movement_amount': 0.3,
                'low_end_focus': -0.15,
                'spatial_width': 0.9,
                'harshness_control': -0.5
            },
            'stage2_tweaks': {
                'warmth': 0.85,
                'transient_boost_db': 1.2,
                'control_profile': {
                    'brightness_tilt': -0.3,
                    'spatial_width': 0.95,
                    'harshness_control': -0.6
                }
            }
        }
    ]

    results = []
    for task in expert_tasks:
        if task['song'] not in audio_map:
            continue

        print(f"\n{'='*80}")
        print(f"🏆 EXPERT TIER MASTER: {task['song']}")
        print(f"{'='*80}")

        try:
            # STAGE 1: Semantic planning into smart preset selection
            print(f"\n[STAGE 1] Semantic Planning & DSP Strategy")
            print(f"Goal: {task['stage1_goal'][:100]}...")

            plan_req = StrategyPlanIn(
                audio_id=audio_map[task['song']],
                goal=task['stage1_goal'],
                platform=task['platform'],
                control_profile=task['stage1_control']
            )

            plan_result = plan_mastering_strategy(plan_req, ctx=ctx)
            print(f"✓ Strategy planned: {plan_result}")

            # STAGE 2: Interactive first-pass render with session persistence
            print(f"\n[STAGE 2] Interactive Stage-1 Render (First Pass)")
            print(f"Preset: {task['stage1_preset']}")
            print(f"Control Profile applied with harshness mitigation")

            interactive_req = {
                'audio_id': audio_map[task['song']],
                'preset_name': task['stage1_preset'],
                'control_profile': task['stage1_control'],
                'stem_mode': 'auto'
            }

            interactive_result = start_interactive_mastering(interactive_req, ctx=ctx)
            print(f"✓ Stage-1 render complete: {interactive_result}")
            session_token = getattr(interactive_result, 'session_token', None)

            if not session_token:
                print(f"✗ No session token returned, checking result structure...")
                print(f"Result: {interactive_result}")
                session_token = interactive_result

            # STAGE 3: Expert-level interactive refinement commit
            print(f"\n[STAGE 3] Expert Refinement & Interactive Commit")
            print(f"Warmth: {task['stage2_tweaks']['warmth']}")
            print(f"Transient Boost: {task['stage2_tweaks']['transient_boost_db']} dB")
            print(f"Final control profile polishing...")

            commit_req = {
                'session_token': session_token,
                'warmth': task['stage2_tweaks']['warmth'],
                'transient_boost_db': task['stage2_tweaks']['transient_boost_db'],
                'control_profile': task['stage2_tweaks']['control_profile']
            }

            final_result = commit_interactive_mastering(commit_req, ctx=ctx)
            print(f"✓ Expert master committed: {final_result}")

            results.append({
                'song': task['song'],
                'plan': plan_result,
                'interactive': interactive_result,
                'final': final_result
            })

            print(f"\n✓ {task['song']} expert master complete!\n")

        except Exception as e:
            print(f"\n✗ Error in expert mastering {task['song']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"🏆 EXPERT TIER MASTERING SESSION COMPLETE")
    print(f"{'='*80}")
    return results

if __name__ == '__main__':
    asyncio.run(run())
