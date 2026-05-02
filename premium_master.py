import asyncio
from server import register_audio_from_path, master_closed_loop, StrategyPlanIn

async def run():
    class MockContext:
        def __init__(self, session_id):
            self.session_id = session_id

    ctx = MockContext('premium_creative_master_2026')

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

    # Creative premium mastering using deterministic closed-loop optimization
    premium_tasks = [
        {
            'song': 'New Project (15)',
            'goal': 'Premium streaming master with forward vocal presence and modern trap punch, ultra-polished and retail-ready. Maximize hook impact and vocal clarity.',
            'platform': 'spotify',
            'control': {'brightness_tilt': 0.3, 'movement_amount': 0.4, 'low_end_focus': 0.5}
        },
        {
            'song': 'Close to the edge',
            'goal': 'Cinematic orchestral master with expansive stereo width, warm analog character, and dynamic restraint for film/scoring use. Emphasize texture and space.',
            'platform': 'youtube',
            'control': {'brightness_tilt': -0.2, 'spatial_width': 0.8, 'harshness_control': -0.3}
        }
    ]

    results = []
    for task in premium_tasks:
        if task['song'] not in audio_map:
            continue
        try:
            print(f"\n🎵 PREMIUM MASTER: {task['song']}")
            print(f"   Goal: {task['goal']}")
            print(f"   Platform: {task['platform']}")

            req = StrategyPlanIn(
                audio_id=audio_map[task['song']],
                goal=task['goal'],
                platform=task['platform'],
                control_profile=task['control']
            )

            # Use master_closed_loop for auto-optimization
            result = master_closed_loop(req, ctx=ctx)

            print(f"   ✓ Premium master completed!")
            print(f"   Result: {result}")
            results.append(result)
        except Exception as e:
            print(f"   ✗ Error mastering {task['song']}: {e}")
            import traceback
            traceback.print_exc()

    return results

if __name__ == '__main__':
    asyncio.run(run())
