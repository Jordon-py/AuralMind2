#!/usr/bin/env python3
"""
AuralMind2 Master Tier UI Launcher
Enhanced dashboard for monitoring NextGen mastering chain
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import asyncio
from threading import Thread
import json

def start_master_tier_ui():
    """Start enhanced mastering UI with NextGen chain support"""

    print("\n" + "=" * 70)
    print(" AuralMind2 Premium  •  Master Tier Edition")
    print("=" * 70)
    print()

    # Check dependencies
    try:
        import flask
        import numpy
        import scipy
        import soundfile
        print("✓ All dependencies available")
        print("  - Flask (web server)")
        print("  - NumPy (numerics)")
        print("  - SciPy (signal processing)")
        print("  - Soundfile (audio I/O)")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        sys.exit(1)

    # Create necessary directories
    Path('static').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    Path('Album_Ignorance_is_bliss/masters').mkdir(parents=True, exist_ok=True)

    print()
    print("━" * 70)
    print(" STARTING PREMIUM MASTERING DASHBOARD")
    print("━" * 70)
    print()
    print("🎛️  Dashboard: http://localhost:5000")
    print("📊 Monitor real-time mastering visualization")
    print("🎚️  Control master-tier enhancement chain")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    print("Loading dashboard...")
    print()

    # Start Flask server
    try:
        from mastering_ui import app

        # Open browser after short delay
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open('http://localhost:5000')
                print("✓ Browser opened at http://localhost:5000\n")
            except:
                print("Note: Could not auto-open browser. Visit http://localhost:5000 manually\n")

        browser_thread = Thread(target=open_browser, daemon=True)
        browser_thread.start()

        # Run Flask app with master tier enhancements
        print("=" * 70)
        print(" DASHBOARD READY")
        print("=" * 70)
        print()

        # Import and patch for master tier features
        import mastering_ui

        # Run app
        app.run(
            debug=False,
            port=5000,
            host='127.0.0.1',
            use_reloader=False,
            threaded=True
        )

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(" Shutting down AuralMind2 Premium Dashboard")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)


def show_processed_logs():
    """Display NextGen mastering processing logs"""

    log_dir = Path('Album_Ignorance_is_bliss/masters')
    log_files = list(log_dir.glob('*NextGen_ProcessingLog.json'))

    if not log_files:
        print("No NextGen processing logs found.")
        return

    print("\n" + "=" * 70)
    print(" NextGen Master Tier Processing Logs")
    print("=" * 70)

    for log_file in sorted(log_files):
        print(f"\n📊 {log_file.name}")
        print("-" * 70)

        try:
            with open(log_file, 'r') as f:
                data = json.load(f)

            print(f"  Song: {data.get('song_name', 'Unknown')}")
            print(f"  Stages: {data.get('stages_completed', 0)} completed")
            print(f"  Timestamp: {data.get('timestamp', 'N/A')}")

            # Show stage summary
            stages = data.get('processing_log', [])
            stage_names = set()
            for entry in stages:
                stage = entry.get('stage')
                if stage and stage != 'CHAIN':
                    stage_names.add(stage)

            if stage_names:
                print(f"  Stages executed: {', '.join(sorted(stage_names))}")

            # Show stage results
            results = data.get('stage_results', {})
            if results:
                print("\n  Stage Results:")
                for stage, result in results.items():
                    print(f"    - {stage}: {type(result).__name__}")

        except Exception as e:
            print(f"  Error reading log: {e}")


def show_status():
    """Show AuralMind2 system status"""

    print("\n" + "=" * 70)
    print(" AuralMind2 System Status")
    print("=" * 70)
    print()

    # Check environment
    try:
        import server
        print("✓ MCP Server available")
    except:
        print("✗ MCP Server not found")

    try:
        import ai_mastering_tool
        print("✓ AI Mastering Tool available")
    except:
        print("✗ AI Mastering Tool not found")

    try:
        import nextgen_master_chain
        print("✓ NextGen Master Chain available")
    except:
        print("✗ NextGen Master Chain not found")

    # Check audio files
    audio_dir = Path('data')
    if audio_dir.exists():
        audio_files = list(audio_dir.glob('*.wav'))
        print(f"✓ {len(audio_files)} audio files available")
    else:
        print("✗ No audio directory")

    # Check masters
    masters_dir = Path('Album_Ignorance_is_bliss/masters')
    if masters_dir.exists():
        master_files = list(masters_dir.glob('*.wav')) + list(masters_dir.glob('*.json'))
        print(f"✓ {len(master_files)} items in Album/masters")

    print()


if __name__ == '__main__':

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == '--logs':
            show_processed_logs()
        elif arg == '--status':
            show_status()
        elif arg == '--help' or arg == '-h':
            print("""
AuralMind2 Master Tier UI Launcher

USAGE:
  python start_master_ui.py              Start dashboard (default)
  python start_master_ui.py --logs       Show NextGen processing logs
  python start_master_ui.py --status     Show system status
  python start_master_ui.py --help       Show this help

FEATURES:
  • Real-time mastering visualization
  • Master-tier chain monitoring
  • Multi-stage processing display
  • Live metrics (LUFS, Peak, Crest, Correlation)
  • Frequency spectrogram analysis

For more information, see MASTERING_UI_README.md
            """)
        else:
            start_master_tier_ui()
    else:
        start_master_tier_ui()
