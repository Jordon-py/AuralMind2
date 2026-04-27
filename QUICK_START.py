#!/usr/bin/env python3
"""
AuralMind2 Quick Start Card
One-page reference for getting started
"""

STARTUP_GUIDE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                         AURALMIND2 QUICK START                            ║
║                    Professional Audio Mastering System                     ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 PREREQUISITES
────────────────────────────────────────────────────────────────────────────
✓ Python 3.8+
✓ Virtual environment activated: .venv\Scripts\Activate.ps1
✓ All dependencies installed (NumPy, SciPy, SoundFile, Flask, Requests)
✓ Verification passed: python verify_startup.py


🚀 START THE SYSTEM
────────────────────────────────────────────────────────────────────────────

COMMAND:
    python run_ui.py

WHAT HAPPENS:
    1. ✓ MCP Server starts on http://127.0.0.1:8080/mcp (HTTP mode)
    2. ✓ Flask UI launches on http://localhost:5000
    3. ✓ Browser opens automatically to dashboard


🌐 ACCESS POINTS
────────────────────────────────────────────────────────────────────────────
    • Dashboard:         http://localhost:5000
    • MCP Server:        http://127.0.0.1:8080/mcp
    • API Endpoint:      http://127.0.0.1:8080/mcp (POST)


💡 MAIN FEATURES
────────────────────────────────────────────────────────────────────────────
    ✓ Real-time waveform visualization
    ✓ Animated glassmorphism design
    ✓ Smart preset recommender (analyzes audio → suggests preset)
    ✓ Quality score predictor (0-100 scale)
    ✓ Output folder picker (user-selected storage locations)
    ✓ Master history dashboard (re-export capability)


⚙️ MASTERING PRESETS
────────────────────────────────────────────────────────────────────────────
    • Premium HiFi        - High fidelity for streaming
    • Premium Punchy      - Competitive loudness
    • Premium Clean       - Subtle enhancement
    • Competitive Trap    - Aggressive trap mastering
    • Hi-Fi Streaming     - Optimized for Spotify/Apple Music


🎯 ASYNC AGENTS (BACKGROUND PROCESSING)
────────────────────────────────────────────────────────────────────────────

    1. JobQueueManager         - Queue & prioritize mastering jobs
       Max: 4 concurrent jobs
       Features: Retry logic, priority ordering, dead letter queue

    2. AudioAnalysisEngine     - Analyze audio → predict quality
       Metrics: LUFS, crest factor, dynamics, frequency, stereo
       Quality Score: 0-100 (higher = better for mastering)

    3. ExportManager           - Concurrent export to disk
       Max: 3 concurrent exports
       Formats: WAV, FLAC, MP3, AAC
       Features: Verification, metadata embedding, history tracking


📊 WORKFLOW
────────────────────────────────────────────────────────────────────────────

    1. UPLOAD
       Select audio file → system analyzes it

    2. ANALYZE (AudioAnalysisEngine)
       • Measures current LUFS, dynamics, frequency balance
       • Predicts quality score 0-100
       • Shows improvement potential

    3. RECOMMEND (Smart Preset)
       • Suggests best preset based on analysis
       • Shows why each preset was recommended
       • User can override selection

    4. QUEUE (JobQueueManager)
       • Job added to queue with priority
       • Real-time progress visualization
       • Can cancel if needed

    5. MASTER (MCP Server)
       • 8-stage mastering pipeline
       • Real-time metering (LUFS, Peak, Crest)
       • Frequency spectrogram display

    6. EXPORT (ExportManager)
       • Exports to user-selected folder
       • Multiple format support
       • Metadata embedding
       • Export history tracking


🛑 STOP THE SYSTEM
────────────────────────────────────────────────────────────────────────────
    Press Ctrl+C in the terminal
    Both MCP server and Flask UI will shut down gracefully


📁 KEY FILES
────────────────────────────────────────────────────────────────────────────
    run_ui.py                  - Main launcher (start this)
    mastering_ui.py            - Flask web server
    server.py                  - FastMCP audio mastering engine
    mastering_ui_bridge.py     - UI ↔ Server bridge

    agents/
    ├── job_queue_manager.py   - Job queuing agent
    ├── audio_analysis_engine.py - Audio analysis agent
    ├── export_manager.py       - Export agent
    └── __init__.py             - Agent imports


📖 DOCUMENTATION
────────────────────────────────────────────────────────────────────────────
    agents/README.md           - Agent overview & quick start
    agents/QUICK_REFERENCE.md  - Complete API reference
    agents/INTEGRATION_GUIDE.md - Integration examples


🔧 ENVIRONMENT VARIABLES (Optional)
────────────────────────────────────────────────────────────────────────────
    ACTIVE_TRANSPORT    - Set to 'streamable-http' (default already set)
    MCP_HOST           - MCP server host (default: 127.0.0.1)
    PORT               - MCP server port (default: 8080)
    MCP_PATH           - MCP endpoint path (default: /mcp)


⚡ PERFORMANCE TIPS
────────────────────────────────────────────────────────────────────────────
    • Use ChromeOS or Firefox for best dashboard performance
    • Analyze files <100MB for faster processing
    • Queue multiple jobs for batch processing efficiency
    • Clear export history periodically (Settings → Cleanup)


🐛 TROUBLESHOOTING
────────────────────────────────────────────────────────────────────────────
    Problem: "Port 5000 already in use"
    → Kill existing process: lsof -ti:5000 | xargs kill -9

    Problem: "MCP server connection error"
    → Check port 8080 is available: netstat -ano | findstr :8080

    Problem: "Missing dependency"
    → Reinstall: pip install -r requirements.txt

    Problem: "Audio analysis slow"
    → Clear cache: Clear browser cache or restart


📞 SUPPORT
────────────────────────────────────────────────────────────────────────────
    Check README.md for full documentation
    Review agent source code docstrings for implementation details
    See INTEGRATION_GUIDE.md for advanced usage patterns


╔════════════════════════════════════════════════════════════════════════════╗
║  READY? Run: python run_ui.py                                            ║
║  Questions? Check: agents/README.md or agents/QUICK_REFERENCE.md         ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

if __name__ == '__main__':
    print(STARTUP_GUIDE)
