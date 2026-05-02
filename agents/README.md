# AuralMind2 Expert Agents

Three powerful async agents for professional audio mastering pipeline.

## 🎯 Overview

Three specialist async worker agents handle the core operations of the mastering system in parallel:

### 1️⃣ **JobQueueManager**
Manages job queuing, prioritization, and retry logic for mastering operations.

- **Purpose**: Queue mastering jobs, manage concurrency, handle retries
- **Max Concurrent**: 4 jobs (configurable)
- **Features**:
  - Priority-based ordering
  - Retry with exponential backoff
  - Dead letter queue for failed jobs
  - Job status tracking
  - Performance metrics

### 2️⃣ **AudioAnalysisEngine**
Parallel audio analysis to predict quality and guide mastering parameters.

- **Purpose**: Analyze audio properties, predict quality scores
- **Features**:
  - Parallel analysis (loudness, dynamics, frequency, stereo, temporal)
  - Quality score prediction (0-100)
  - Results caching for performance
  - Batch processing support
  - 10+ audio metrics per file

### 3️⃣ **ExportManager**
Handles batched exports with concurrent file I/O and verification.

- **Purpose**: Export mastered files with concurrent processing
- **Max Concurrent**: 3 exports (configurable)
- **Features**:
  - Multiple format support (WAV, FLAC, MP3, AAC)
  - File integrity verification
  - Metadata embedding
  - Export history tracking
  - Storage analytics
  - Automatic cleanup

## 📁 File Structure

```
agents/
├── __init__.py                 # Agent imports
├── job_queue_manager.py        # Agent 1: Queue management
├── audio_analysis_engine.py    # Agent 2: Audio analysis
├── export_manager.py           # Agent 3: Export management
├── QUICK_REFERENCE.md          # Quick reference guide (START HERE)
├── INTEGRATION_GUIDE.md        # How to integrate with server
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

No additional dependencies beyond what's already in AuralMind2:
- `numpy` - Numeric operations
- `scipy` - Signal processing
- `soundfile` - Audio I/O
- `asyncio` - Async operations (standard library)

### Basic Usage

```python
from agents import JobQueueManager, AudioAnalysisEngine, ExportManager
import asyncio

async def main():
    # Initialize agents
    queue = JobQueueManager(max_concurrent_jobs=4)
    analyzer = AudioAnalysisEngine(cache_enabled=True)
    exporter = ExportManager(max_concurrent_exports=3)
    await exporter.initialize()

    # Analyze an audio file
    metrics = await analyzer.analyze_audio_file('track.wav')
    print(f"Quality Score: {metrics.quality_score:.0f}/100")

    # Queue a mastering job
    job_id = queue.submit_job(
        audio_id='aud_123',
        preset='Premium HiFi',
        session_id='sess_user1'
    )
    print(f"Queued: {job_id}")

    # Export a file
    export_id = exporter.submit_export(
        audio_path='master.wav',
        output_dir='deliverables/',
        filename='song_hifi'
    )
    print(f"Export: {export_id}")

asyncio.run(main())
```

## 📚 Documentation

1. **QUICK_REFERENCE.md** - Start here for quick API reference
2. **INTEGRATION_GUIDE.md** - How to integrate with server.py and Flask UI
3. **Docstrings** - Full documentation in each agent file

## 🔧 Configuration

### JobQueueManager
```python
queue = JobQueueManager(
    max_concurrent_jobs=4,      # Concurrence level
    checkpoint_file=None        # Optional state persistence
)
```

### AudioAnalysisEngine
```python
analyzer = AudioAnalysisEngine(
    cache_enabled=True          # Cache results for speed
)
```

### ExportManager
```python
exporter = ExportManager(
    max_concurrent_exports=3,   # Concurrent file operations
    verify_writes=True          # Verify file integrity
)
```

## 💡 Common Patterns

### Pattern 1: Analyze → Queue → Export
```python
# Analyze to get quality
metrics = await analyzer.analyze_audio_file(audio_path)

# Use quality to set priority
priority = 1 if metrics.quality_score < 50 else 0
job_id = queue.submit_job(audio_id, preset, session_id, priority)

# Export when done
export_id = exporter.submit_export(master_path, output_dir, filename)
```

### Pattern 2: Batch Processing
```python
# Queue multiple jobs at once
for track in tracks:
    queue.submit_job(track['id'], track['preset'], session_id)

# Process all concurrently
await queue.process_queue(mastering_processor)
```

### Pattern 3: Monitor Status
```python
while running:
    queue_status = queue.get_queue_status()
    export_stats = exporter.get_storage_stats()
    analyzer_stats = analyzer.get_stats()

    # Use for UI updates or logging
    await asyncio.sleep(5)
```

## 📊 Monitoring

Each agent provides statistics:

**JobQueueManager.get_queue_status()**
```python
{
    'active_jobs': 2,
    'queued_jobs': 5,
    'completed_jobs': 45,
    'failed_jobs': 2,
    'dead_letter_count': 2,
    'stats': {
        'total_jobs': 49,
        'avg_processing_time': 12.5
    }
}
```

**AudioAnalysisEngine.get_stats()**
```python
{
    'analyses_run': 12,
    'cache_hits': 8,
    'cache_size': 12,
    'cache_efficiency': 66.7
}
```

**ExportManager.get_storage_stats()**
```python
{
    'total_exported': 45,
    'total_bytes': 5368709120,
    'total_gb': 5.0,
    'failed_exports': 1,
    'success_rate': 97.8,
    'avg_export_time': 2.3
}
```

## 🔌 Integration Points

### For FastMCP Server (server.py)
```python
from agents import JobQueueManager, AudioAnalysisEngine, ExportManager

# Initialize globally
job_queue = JobQueueManager(max_concurrent_jobs=4)
audio_analyzer = AudioAnalysisEngine(cache_enabled=True)
export_mgr = ExportManager(max_concurrent_exports=3)

# Use in tools
@mcp.tool()
def start_mastering(audio_id, preset):
    job_id = job_queue.submit_job(audio_id, preset, session_id)
    return {'job_id': job_id}
```

### For Flask UI (mastering_ui.py)
```python
# Add API endpoints
@app.route('/api/queue_status')
def queue_status():
    status = job_queue.get_queue_status()
    return jsonify(status)

@app.route('/api/analyze', methods=['POST'])
async def analyze():
    metrics = await audio_analyzer.analyze_audio_file(path)
    return jsonify(metrics.to_dict())
```

## 📈 Performance Tips

1. **Tune Concurrency**
   - JobQueueManager: 4-8 jobs for CPU-bound mastering
   - ExportManager: 3-5 exports for disk I/O

2. **Use Caching**
   - Enable analysis caching for repeated files
   - Cache hits can be 60-80% for similar batch files

3. **Priority Scheduling**
   - Set higher priority for urgent jobs
   - Use quality analysis to guide priority

4. **Batch Operations**
   - Use `batch_analyze()` for multiple files
   - Use `export_batch()` for multiple exports

## 🐛 Debugging

### Check Queue Status
```python
status = queue.get_queue_status()
if status['failed_jobs'] > 0:
    dead_letter = queue.get_dead_letter_jobs()
    for job in dead_letter:
        print(f"Failed: {job['job_id']} - {job['error_message']}")
```

### Check Export Results
```python
history = exporter.get_export_history(limit=10)
for exp in history:
    if exp['status'] == 'failed':
        print(f"Export failed: {exp}")
```

### Check Analysis Cache
```python
stats = analyzer.get_stats()
print(f"Cache efficiency: {stats['cache_efficiency']:.1f}%")
if stats['cache_efficiency'] < 10:
    analyzer.clear_cache()
```

## 🎓 Learning Resources

- **QUICK_REFERENCE.md** - API reference for all 3 agents
- **INTEGRATION_GUIDE.md** - 4 integration examples
- **Source code** - Fully documented with docstrings
- **inline comments** - Explain key algorithms

## 📝 License

Part of AuralMind2 Premium Mastering System

---

**Next Steps:**
1. Read QUICK_REFERENCE.md for API details
2. Review INTEGRATION_GUIDE.md for integration examples
3. Check source code docstrings for detailed parameter docs
4. Test locally before integrating with server
