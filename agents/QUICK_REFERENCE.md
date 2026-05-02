"""
Expert Agents Quick Reference
3 Async Agents for AuralMind2 Mastering Pipeline
"""

# ============================================================================
# AGENT 1: JobQueueManager
# ============================================================================
"""
Purpose: Queue, prioritize, and process mastering jobs asynchronously

Key Features:
  • Priority-based job ordering (higher priority processed first)
  • Retry logic with exponential backoff (up to 3 retries by default)
  • Job status tracking (QUEUED → PROCESSING → COMPLETED/FAILED)
  • Dead letter queue for permanently failed jobs
  • Metrics: total jobs, completion time, retry statistics

Usage:

    from agents import JobQueueManager

    # Initialize
    queue = JobQueueManager(max_concurrent_jobs=4)

    # Submit a job
    job_id = queue.submit_job(
        audio_id='aud_abc123',
        preset='Premium HiFi',
        session_id='sess_user123',
        priority=1,  # Higher = processed first
        metadata={'song_name': 'example.wav'}
    )

    # Get job status
    status = queue.get_job_status(job_id)
    # Returns: {'job_id': 'job_xyz', 'status': 'processing', 'progress': 45.0, ...}

    # Get queue status
    queue_status = queue.get_queue_status()
    # Returns: {'active_jobs': 2, 'queued_jobs': 5, 'completed_jobs': 12, ...}

    # Process jobs (async)
    async def my_processor(job):
        # Your mastering logic here
        result = await do_mastering(job.audio_id, job.preset)
        return (True, result)  # (success, result)

    await queue.process_queue(my_processor)

    # Cancel a job
    queue.cancel_job(job_id)

    # Get failed jobs
    dead_letter = queue.get_dead_letter_jobs(limit=10)

Queue Status Fields:
  • active_jobs: Jobs currently processing
  • queued_jobs: Jobs waiting in queue
  • completed_jobs: Successfully completed
  • failed_jobs: Permanently failed (exhausted retries)
  • stats.total_retries: Total retry attempts
  • stats.avg_processing_time: Average job duration

"""


# ============================================================================
# AGENT 2: AudioAnalysisEngine
# ============================================================================
"""
Purpose: Analyze audio files in parallel and predict mastering quality

Key Features:
  • Parallel analysis of multiple dimensions:
    - Loudness: RMS, LUFS, True Peak
    - Dynamics: Crest Factor, Dynamic Range
    - Frequency: Peak Frequency, Spectral Centroid, Spectral Flatness
    - Stereo: Correlation, Phase Coherence
    - Temporal: Onset Detection, Zero Crossing Rate
  • Quality score prediction (0-100)
  • Results caching for faster re-analysis
  • Support for mono and stereo files
  • Batch analysis of multiple files concurrently

Usage:

    from agents import AudioAnalysisEngine

    # Initialize
    analyzer = AudioAnalysisEngine(cache_enabled=True)

    # Analyze a file
    metrics = await analyzer.analyze_audio_file(
        audio_path='path/to/audio.wav',
        cache_key='unique_id'  # Optional, for caching
    )

    # Access metrics
    print(metrics.lufs)                    # Loudness in LUFS
    print(metrics.crest_factor)            # Peak-to-RMS ratio in dB
    print(metrics.dynamic_range)           # Dynamic range in dB
    print(metrics.spectral_centroid)       # Center of mass in frequency
    print(metrics.quality_score)           # Predicted quality 0-100

    # Get as dict
    metrics_dict = metrics.to_dict()

    # Analyze from memory
    metrics = await analyzer.analyze_audio_data(
        audio=np.array([...]),  # NumPy array
        sr=44100                # Sample rate
    )

    # Batch analyze
    files = ['track1.wav', 'track2.wav', 'track3.wav']
    results = await analyzer.batch_analyze(files)

    # Check cache stats
    stats = analyzer.get_stats()
    # Returns: {'analyses_run': 5, 'cache_hits': 2, 'cache_efficiency': 40.0, ...}

    # Clear cache
    analyzer.clear_cache()

Quality Score Calculation:
  (Loudness: 35% × Dynamics: 25% × Frequency: 25% × Stereo: 15%)

Metrics Output:
  {
    'duration': float,
    'sample_rate': int,
    'channels': int,
    'loudness': {'rms': float, 'lufs': float, 'true_peak': float},
    'dynamics': {'crest_factor': float, 'dynamic_range': float},
    'frequency': {'peak_freq': float, 'spectral_centroid': float, ...},
    'stereo': {'correlation': float, 'phase_coherence': float},
    'temporal': {'onset_count': int, 'zero_crossing_rate': float},
    'quality_score': float  # 0-100
  }

"""


# ============================================================================
# AGENT 3: ExportManager
# ============================================================================
"""
Purpose: Export, organize, and manage delivery of mastered audio files

Key Features:
  • Concurrent export operations with retry logic
  • Multiple format support (WAV, FLAC, MP3, AAC)
  • Metadata embedding in exported files
  • File integrity verification after export
  • Export history tracking and analytics
  • Storage management and cleanup
  • Batch operations with concurrent processing

Usage:

    from agents import ExportManager, ExportFormat

    # Initialize
    exporter = ExportManager(max_concurrent_exports=3, verify_writes=True)
    await exporter.initialize()

    # Submit single export
    export_id = exporter.submit_export(
        audio_path='masters/track_mastered.wav',
        output_dir='Album_Ignorance_is_bliss/masters/',
        filename='track_hifi',
        format='wav',
        metadata={
            'preset': 'Premium HiFi',
            'artist': 'Artist Name',
            'album': 'Album Name'
        }
    )
    # Result: 'exp_abc123def456'

    # Get export status
    status = exporter.get_export_status(export_id)
    # Returns: {'export_id': 'exp_xyz', 'status': 'completed', 'file_size': 45670654, ...}

    # Batch export
    batch_list = [
        ('master1.wav', 'deliverables/', 'song1'),
        ('master2.wav', 'deliverables/', 'song2'),
        ('master3.wav', 'deliverables/', 'song3'),
    ]

    results = await exporter.export_batch(batch_list, preset_name='HiFi')
    # Returns: {'total': 3, 'successful': 3, 'failed': 0, 'exports': [...]}

    # Process all pending exports
    await exporter.process_exports()

    # Get export history
    history = exporter.get_export_history(limit=50)

    # Get storage statistics
    stats = exporter.get_storage_stats()
    # Returns: {
    #   'total_exported': 125,
    #   'total_bytes': 5368709120,
    #   'total_gb': 5.00,
    #   'avg_file_size_mb': 42.5,
    #   'success_rate': 98.4,
    #   'avg_export_time': 2.3
    # }

    # Organize by preset
    hifi_dir = await exporter.organize_by_preset('masters/', 'Premium HiFi')

    # Cleanup old exports (older than 7 days)
    cleanup = await exporter.cleanup_old_exports(
        days=7,
        dry_run=True  # Just report, don't delete
    )

Export Formats:
  • WAV (default, highest quality)
  • FLAC (lossless compression)
  • MP3 (lossy, smaller files)
  • AAC (lossy, Apple standard)

Export Status:
  • PENDING: Waiting to export
  • EXPORTING: Currently exporting
  • COMPLETED: Successfully exported
  • FAILED: Export failed
  • RETRYING: Retrying after failure

Storage Stats Output:
  {
    'total_exported': int,
    'total_bytes': int,
    'total_gb': float,
    'failed_exports': int,
    'avg_export_time': float,
    'active_exports': int,
    'pending_exports': int,
    'avg_file_size_mb': float,
    'success_rate': float  # 0-100
  }

"""


# ============================================================================
# AGENT COMPARISON TABLE
# ============================================================================
"""
┌─────────────────┬─────────────────────┬──────────────────┬────────────────────┐
│ Agent           │ Primary Task        │ Concurrency      │ Primary Metrics    │
├─────────────────┼─────────────────────┼──────────────────┼────────────────────┤
│ JobQueueManager │ Queue & process     │ Configurable     │ job count, retries │
│                 │ mastering jobs      │ (default: 4)     │ processing time    │
├─────────────────┼─────────────────────┼──────────────────┼────────────────────┤
│ AudioAnalysis   │ Analyze audio       │ Parallel task    │ LUFS, dynamics,    │
│ Engine          │ properties          │ per file         │ quality score      │
├─────────────────┼─────────────────────┼──────────────────┼────────────────────┤
│ ExportManager   │ Export mastered     │ Configurable     │ export count,      │
│                 │ files to disk       │ (default: 3)     │ file size, success │
└─────────────────┴─────────────────────┴──────────────────┴────────────────────┘

Typical Pipeline Flow:
  1. AudioAnalysisEngine analyzes incoming audio → quality metrics
  2. JobQueueManager queues mastering job with priority based on analysis
  3. Server processes queued job → produces mastered file
  4. ExportManager exports result to user-selected folder

Key Integration Points:
  • Use analyzer output to set job priority in queue
  • Use queue metrics for UI updates and status reports
  • Use export manager for all file outputs
  • Monitor agent stats for performance optimization
"""


# ============================================================================
# COMMON PATTERNS
# ============================================================================
"""

Pattern 1: Analyze Before Mastering
─────────────────────────────────────

    # Get analysis to decide preset
    metrics = await analyzer.analyze_audio_file(audio_path)

    # Higher quality → standard preset (less aggressive)
    # Lower quality → stronger preset (more aggressive)

    if metrics.quality_score > 70:
        preset = 'Premium Clean'  # Subtle enhancement
    else:
        preset = 'Premium Punchy'  # More aggressive

    # Queue with priority
    priority = 0 if metrics.quality_score > 70 else 1
    job_id = queue.submit_job(audio_id, preset, session_id, priority)


Pattern 2: Process Queue with Real Work
───────────────────────────────────────

    async def mastering_processor(job):
        try:
            # Your actual mastering logic
            master = await server.call_mastering_tool(
                audio_id=job.audio_id,
                preset=job.preset
            )

            # Queue export if successful
            export_id = exporter.submit_export(
                audio_path=master,
                output_dir='masters/',
                filename=f"{job_id}__{job.preset}"
            )

            return (True, export_id)
        except Exception as e:
            return (False, str(e))

    # Run queue processor
    await queue.process_queue(mastering_processor)


Pattern 3: Monitoring Queue and Exports
───────────────────────────────────────

    while True:
        queue_status = queue.get_queue_status()
        export_status = exporter.get_storage_stats()

        print(f"Queue: {queue_status['active_jobs']} active, "
              f"{queue_status['queued_jobs']} waiting")
        print(f"Exports: {export_status['active_exports']} active, "
              f"{export_status['success_rate']:.1f}% success")

        await asyncio.sleep(5)

"""


if __name__ == '__main__':
    print(__doc__)
