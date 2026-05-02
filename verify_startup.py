#!/usr/bin/env python3
"""
AuralMind2 Startup Verification
Checks that all components are ready to run
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND: {filepath}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description} - {e}")
        return False

def main():
    print("\n" + "="*60)
    print("AuralMind2 STARTUP VERIFICATION")
    print("="*60 + "\n")

    base_dir = Path(__file__).parent
    all_good = True

    # Check core files
    print("Checking core files...")
    all_good &= check_file_exists(str(base_dir / 'run_ui.py'), "Launcher (run_ui.py)")
    all_good &= check_file_exists(str(base_dir / 'mastering_ui.py'), "Flask App (mastering_ui.py)")
    all_good &= check_file_exists(str(base_dir / 'server.py'), "MCP Server (server.py)")
    all_good &= check_file_exists(str(base_dir / 'mastering_ui_bridge.py'), "UI Bridge (mastering_ui_bridge.py)")
    print()

    # Check agent files
    print("Checking agent files...")
    all_good &= check_file_exists(str(base_dir / 'agents' / '__init__.py'), "Agents Package")
    all_good &= check_file_exists(str(base_dir / 'agents' / 'job_queue_manager.py'), "Job Queue Manager")
    all_good &= check_file_exists(str(base_dir / 'agents' / 'audio_analysis_engine.py'), "Audio Analysis Engine")
    all_good &= check_file_exists(str(base_dir / 'agents' / 'export_manager.py'), "Export Manager")
    print()

    # Check dependencies
    print("Checking Python dependencies...")
    all_good &= check_import('flask', "Flask")
    all_good &= check_import('numpy', "NumPy")
    all_good &= check_import('scipy', "SciPy")
    all_good &= check_import('soundfile', "Soundfile")
    all_good &= check_import('requests', "Requests")
    all_good &= check_import('asyncio', "AsyncIO (stdlib)")
    print()

    # Check agents can be imported
    print("Checking agent imports...")
    all_good &= check_import('agents', "Agents package")
    if all_good:
        try:
            from agents import JobQueueManager, AudioAnalysisEngine, ExportManager
            print(f"✓ JobQueueManager")
            print(f"✓ AudioAnalysisEngine")
            print(f"✓ ExportManager")
        except ImportError as e:
            print(f"✗ Agent imports failed: {e}")
            all_good = False
    print()

    # Summary
    print("="*60)
    if all_good:
        print("✓ ALL SYSTEMS READY")
        print("\nTo start AuralMind2, run:")
        print("  python run_ui.py")
        print("\nThe system will:")
        print("  1. Start MCP Server on http://127.0.0.1:8080/mcp (HTTP mode)")
        print("  2. Start Flask UI on http://localhost:5000")
        print("  3. Open browser automatically")
        print("\nPress Ctrl+C to stop both servers.")
        return 0
    else:
        print("✗ SOME ISSUES FOUND")
        print("\nPlease fix the above issues before starting.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
