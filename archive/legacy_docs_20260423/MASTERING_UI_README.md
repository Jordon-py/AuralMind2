# AuralMind2 Premium Mastering Dashboard

Professional real-time mastering visualization system with live levels and spectral analysis.

## Features

✨ **Real-Time Visualization**
- Live LUFS loudness metering with normalized display
- True Peak and Crest Factor monitoring
- Stereo correlation visualization
- Dynamic frequency spectrogram analysis

🎛️ **Professional Metering**
- LUFS: -23 to -11 dB range (streaming standard)
- True Peak: -3 to 0 dBTP (loudness ceiling)
- Crest Factor: 5-24 dB (dynamic range)
- Stereo Correlation: 0-1 (mono compatibility)

🎨 **Premium Minimalist UI**
- Dark theme optimized for audio work
- Accessible control panel
- Responsive design for all screen sizes
- Real-time status updates

## Installation

### Requirements
- Python 3.8+
- Flask
- NumPy, SciPy, Soundfile
- Plotly.js (loaded via CDN)

### Setup

1. **Install dependencies:**
```bash
pip install flask numpy scipy soundfile
```

2. **Ensure AuralMind2 MCP server is running:**
```bash
# In separate terminal
python server.py
```

3. **Run the UI:**
```bash
python run_ui.py
```

The dashboard will automatically open at `http://localhost:5000`

## Usage

### Starting a Mastering Session

1. **Open Dashboard**: Navigate to http://localhost:5000
2. **Select Audio File**: Click "📁 Select Audio File" button
3. **Choose Preset**: Select from available mastering profiles
4. **Start Mastering**: Click "▶ Start Mastering"

### Available Presets

| Preset | Use Case | Characteristics |
|--------|----------|-----------------|
| **HiFi Streaming** | Spotify, Apple Music, etc. | Natural, transparent, loud |
| **Trap Competitive** | Trap/Hip-hop tracks | Aggressive, bass-heavy, punchy |
| **Club Ready** | Dance floor | Extended bottom end, tight dyn. |
| **Radio Loud** | AM/FM broadcast | Maximum commercial loudness |
| **Cinematic** | Film/trailer scores | Dynamic, musical, theatrical |

### Monitoring Progress

The dashboard displays four key metrics in real-time:

**LUFS Loudness**
- Target: -14 to -12 LUFS (streaming)
- Shows overall perceived loudness
- Green fill indicates loudness level

**True Peak**
- Target: -1 to -0.1 dBTP (no clipping)
- Prevents digital clipping in playback
- Should not exceed -0.1 dBTP

**Crest Factor (dB)**
- Measures peak-to-average ratio
- Higher = more dynamic range
- Typical range: 12-18 dB for mastered audio

**Stereo Correlation**
- 1.0 = Fully mono (correlated)
- 0.0 = Maximum stereo width
- Typical: 0.8-1.0 for mono-safe masters

### Frequency Spectrogram

Real-time frequency analysis showing:
- **Time axis (X)**: Progression of audio
- **Frequency axis (Y)**: 0 Hz to 24 kHz
- **Color intensity**: Signal magnitude (brightness = louder)

Visual interpretation:
- **Bright areas** = Energy concentration (musical content)
- **Dark areas** = Quiet or missing frequencies
- **Smooth distribution** = Well-balanced mastering

## File Structure

```
AuralMind2/
├── mastering_ui.py           # Flask backend server
├── mastering_ui_bridge.py    # Integration with AI tool
├── run_ui.py                 # Launcher script
├── templates/
│   └── index.html            # Web interface
├── static/                   # CSS/JS assets (auto-created)
└── Album_Ignorance_is_bliss/
    └── masters/              # Output folder for exports
        ├── NewProject15_CompetitiveTrap_Master.wav
        ├── NewProject15_HiFiStreaming_Master.wav
        └── ...
```

## API Endpoints

For advanced users, the dashboard provides HTTP endpoints:

### Create Session
```
POST /api/session/new
Body: {"audio_path": "path/to/audio.wav"}
Response: {"session_id": "...", "duration": 229.5}
```

### Get Metrics
```
GET /api/session/<session_id>/metrics
Response: {
  "lufs": -23.0,
  "true_peak": -0.1,
  "crest_db": 19.33,
  "stereo_corr": 0.872
}
```

### Get Spectrogram
```
GET /api/session/<session_id>/spectrogram
Response: {
  "frequencies": [...],
  "times": [...],
  "magnitude": [...],
  "min_db": -80.0,
  "max_db": 0.0
}
```

### Update Session
```
POST /api/session/<session_id>/update
Body: {
  "is_mastering": true,
  "progress": 45,
  "lufs": -14.0,
  "true_peak": -0.1,
  "crest_db": 15.0,
  "stereo_corr": 0.88
}
```

## Exporting Masters

### Automatic Export
Masters are automatically exported to `Album_Ignorance_is_bliss/masters/` when mastering completes.

### Manual Export
```bash
python run_ui.py --export
```

### File Naming
Masters follow the naming convention:
```
{SongName}_{Preset}_Master.wav
```

Example outputs:
- `NewProject15_CompetitiveTrap_Master.wav`
- `CloseToEdge_HiFiStreaming_Master.wav`

## Command Line

### Start Dashboard
```bash
python run_ui.py
# or
python run_ui.py --server
```

### Export Completed Mastering Jobs
```bash
python run_ui.py --export
```

### Show Help
```bash
python run_ui.py --help
```

## Troubleshooting

### Dashboard Won't Load
- Ensure Flask is installed: `pip install flask`
- Check port 5000 is available: `netstat -an | grep 5000`
- Check server logs in terminal for errors

### Audio File Not Found
- Use absolute paths or place audio in working directory
- Check file format is WAV, MP3, or FLAC
- Verify file isn't already open in another application

### No Spectrogram Showing
- Wait for analysis to complete
- Check audio file has valid format
- Verify audio duration is >1 second

### Metrics Not Updating
- Check if mastering job is still running
- Verify AuralMind2 MCP server is connected
- Check browser developer console for errors (F12)

## Performance

- **Spectrogram computation**: <5 seconds for typical mastered audio
- **Metrics update interval**: Real-time streaming
- **Browser compatibility**: Chrome, Firefox, Safari, Edge (latest versions)
- **Memory usage**: ~200-500MB for session management

## Advanced Features

### Integration with AI Mastering Tool

The UI integrates with `AIIntegratedMasteringTool` for autonomous mastering optimization:

```python
from mastering_ui_bridge import MasteringUIBridge

async def start_smart_mastering():
    bridge = MasteringUIBridge()
    session = await bridge.start_mastering_session(
        'audio.wav',
        'hi_fi_streaming',
        'MyTrack'
    )
    await bridge.monitor_job(session)
    await bridge.fetch_results(session)
```

### Custom Metrics

Extend the metrics display by modifying `updateMetrics()` in `index.html`:

```javascript
// Add your custom metric
document.getElementById('customValue').textContent =
    metrics.your_metric.toFixed(2);
```

## License

AuralMind2 © 2025

## Support

For issues, feature requests, or documentation:
- Check the AI Mastering Tool guide
- Review server.py documentation
- Consult MCP protocol specifications
