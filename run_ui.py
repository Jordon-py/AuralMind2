#!/usr/bin/env python3
"""
AuralMind2 Premium UI Launcher
Starts the connected web dashboard and syncs completed masters from the live
UI session store.
"""

from __future__ import annotations

import asyncio
import sys
import time
import webbrowser
from importlib import import_module
from pathlib import Path
from threading import Thread
from typing import Any


def _load_mastering_ui() -> Any:
    """Import the connected Flask UI module on demand."""

    return import_module("mastering_ui")


def start_ui_server() -> None:
    """Start the connected Flask mastering UI server."""

    print("=" * 60)
    print("AuralMind2 Premium Mastering Dashboard")
    print("=" * 60)
    print()

    # Check if Flask is installed.
    try:
        import flask  # noqa: F401

        print("✓ Flask framework ready")
    except ImportError:
        print("✗ Flask not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "-q"])
        print("✓ Flask installed")

    # Check audio dependencies before booting the UI.
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import soundfile  # noqa: F401

        print("✓ Audio processing libraries available")
    except ImportError as exc:
        print(f"✗ Missing dependency: {exc}")
        sys.exit(1)

    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)

    print()
    print("Starting mastering UI server on http://127.0.0.1:5000")
    print("Connected mastering bridge is loaded in-process")
    print("Press Ctrl+C to stop")
    print()

    try:
        ui_module = _load_mastering_ui()
        app = ui_module.app

        def open_browser() -> None:
            time.sleep(2)
            webbrowser.open("http://127.0.0.1:5000")

        browser_thread = Thread(target=open_browser, daemon=True)
        browser_thread.start()

        app.run(debug=False, port=5000, host="127.0.0.1", use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)


async def export_pending_masters() -> None:
    """Refresh connected UI sessions and export completed masters."""

    print("\n" + "=" * 60)
    print("Master Export Manager")
    print("=" * 60)
    print()

    ui_module = _load_mastering_ui()
    sessions = getattr(ui_module, "mastering_sessions", {})

    if not sessions:
        print("No active mastering sessions found in the connected UI.")
        return

    print("Checking for completed mastering jobs...")
    print()

    exported_count = 0
    for session_id, session in sessions.items():
        job_id = getattr(session, "job_id", None)
        if not job_id:
            continue

        print(f"Checking: {session_id}")
        refreshed = ui_module._refresh_session(session)

        if getattr(refreshed, "output_file", None):
            print(f"  ✓ Exported to: {refreshed.output_file}")
            exported_count += 1
        elif getattr(refreshed, "status", None) in {"queued", "running"}:
            print("  ⏳ Job still processing")
        elif getattr(refreshed, "error", None):
            print(f"  ✗ {refreshed.error}")
        else:
            print("  ⏳ Job not ready for export")

        print()

    if exported_count == 0:
        print("No completed masters were ready for export.")


def show_help() -> None:
    """Show help information."""

    print(
        """
AuralMind2 Premium Mastering Dashboard
=======================================

USAGE:
  python run_ui.py [OPTIONS]

OPTIONS:
  --server         Start the connected web dashboard (default)
  --export         Refresh live UI sessions and export completed masters
  --help           Show this help

FEATURES:
  • Real-time mastering visualization
  • Live loudness metering (LUFS, Peak, Crest)
  • Frequency spectrogram analysis
  • Job progress monitoring
  • Automatic master export to Album folder

DASHBOARD:
  Open http://127.0.0.1:5000 in your browser

  1. Select an audio file to master
  2. Choose a mastering preset
  3. Click "Start Mastering"
  4. Watch real-time visualization of the mastering process
  5. Masters export through the connected UI bridge when complete

PRESETS AVAILABLE:
  • HiFi Streaming    - High fidelity for streaming platforms
  • Trap Competitive  - Aggressive trap mastering
  • Club Ready        - Dance/club floor optimization
  • Radio Loud        - AM/FM radio loudness
  • Cinematic         - Cinematic/film score treatment

KEYBOARD SHORTCUTS:
  Ctrl+C   Stop the server

For more information, see README.md
        """
    )


def main(argv: list[str] | None = None) -> int:
    """Run the requested launcher mode."""

    args = list(sys.argv[1:] if argv is None else argv)

    if args:
        arg = args[0].lower()
        if arg == "--export":
            asyncio.run(export_pending_masters())
            return 0
        if arg in {"--help", "-h"}:
            show_help()
            return 0

    start_ui_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
