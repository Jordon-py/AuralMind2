"""
Agent Integration Guide for AuralMind2
How to use the 3 expert async agents with your mastering pipeline
"""

import asyncio
from agents import JobQueueManager, AudioAnalysisEngine, ExportManager

# ============================================================================
# INTEGRATION EXAMPLE: Using All 3 Agents Together
# ============================================================================

class MasteringPipeline:
    """
    Example of how to integrate all 3 expert agents into the mastering system
    """

    def __init__(self):
        # Initialize the 3 agents
        self.job_queue = JobQueueManager(max_concurrent_jobs=4)
        self.audio_analyzer = AudioAnalysisEngine(cache_enabled=True)
        self.export_mgr = ExportManager(max_concurrent_exports=3, verify_writes=True)

    async def initialize(self):
        """Initialize async components"""
        await self.export_mgr.initialize()

    async def process_mastering_request(self, audio_path: str, preset: str,
                                       session_id: str, output_dir: str):
        """
        Process a complete mastering request using all agents

        Flow:
        1. Queue the mastering job (JobQueueManager)
        2. Analyze audio properties (AudioAnalysisEngine)
        3. Register audio with server via MCP protocol
        4. Submit mastering job to FastMCP server
        5. Monitor job progress
        6. Export completed master (ExportManager)
        """

        print(f"\n[Pipeline] Starting master: {audio_path}")
        print(f"  Preset: {preset}")
        print(f"  Session: {session_id}")

        # Step 1: Analyze the audio first (helps with preset selection)
        print("\n[Step 1] Analyzing audio...")
        metrics = await self.audio_analyzer.analyze_audio_file(
            audio_path,
            cache_key=audio_path
        )

        print(f"  ✓ Analysis complete:")
        print(f"    LUFS: {metrics.lufs:.1f}")
        print(f"    Crest Factor: {metrics.crest_factor:.1f}dB")
        print(f"    Dynamic Range: {metrics.dynamic_range:.1f}dB")
        print(f"    Quality Score: {metrics.quality_score:.0f}/100")

        # Step 2: Queue the mastering job
        print("\n[Step 2] Queueing mastering job...")
        job_id = self.job_queue.submit_job(
            audio_id=audio_path,  # Would be actual artifact ID from MCP
            preset=preset,
            session_id=session_id,
            priority=1 if metrics.quality_score < 50 else 0,  # Priority if low quality
            metadata={
                'lufs': metrics.lufs,
                'crest_factor': metrics.crest_factor,
                'quality_score': metrics.quality_score
            }
        )

        print(f"  ✓ Queued as {job_id}")

        # Step 3: Later, when job completes (in actual implementation):
        # This would be called after FastMCP server completes the mastering
        print("\n[Step 3] [Simulated] Job complete, exporting...")
        export_id = self.export_mgr.submit_export(
            audio_path=audio_path,  # Would be actual master output from MCP
            output_dir=output_dir,
            filename=f"{audio_path.split('/')[-1]}__{preset}",
            format='wav',
            metadata={
                'preset': preset,
                'original_quality': metrics.quality_score,
                'source_lufs': metrics.lufs
            }
        )

        print(f"  ✓ Export queued as {export_id}")

        return {
            'job_id': job_id,
            'export_id': export_id,
            'metrics': metrics.to_dict()
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_1_single_mastering():
    """Example 1: Process a single audio file"""

    print("\n" + "="*70)
    print("EXAMPLE 1: Single File Mastering with All 3 Agents")
    print("="*70)

    pipeline = MasteringPipeline()
    await pipeline.initialize()

    result = await pipeline.process_mastering_request(
        audio_path='path/to/audio.wav',
        preset='Premium HiFi',
        session_id='sess_abc123',
        output_dir='masters/'
    )

    print(f"\n[Result] Job: {result['job_id']}")
    print(f"[Result] Export: {result['export_id']}")
    print(f"[Result] Quality: {result['metrics']['quality_score']:.0f}/100")


async def example_2_batch_processing():
    """Example 2: Batch process multiple files"""

    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing with Queue Management")
    print("="*70)

    pipeline = MasteringPipeline()
    await pipeline.initialize()

    # Submit 5 files to the queue
    files = [
        'song1.wav',
        'song2.wav',
        'song3.wav',
        'song4.wav',
        'song5.wav'
    ]

    presets = ['Competitive Trap', 'Premium Clean', 'Hi-Fi Streaming',
               'Premium Punchy', 'Competitive Trap']

    job_ids = []
    for file_path, preset in zip(files, presets):
        job_id = pipeline.job_queue.submit_job(
            audio_id=file_path,
            preset=preset,
            session_id='batch_001',
            priority=1
        )
        job_ids.append(job_id)

    print(f"\n[Batch] Submitted {len(job_ids)} jobs:")
    for jid in job_ids:
        print(f"  - {jid}")

    # Check queue status
    status = pipeline.job_queue.get_queue_status()
    print(f"\n[Queue Status]")
    print(f"  Active: {status['active_jobs']}")
    print(f"  Queued: {status['queued_jobs']}")
    print(f"  Completed: {status['completed_jobs']}")


async def example_3_analysis_caching():
    """Example 3: Using audio analysis with caching"""

    print("\n" + "="*70)
    print("EXAMPLE 3: Audio Analysis with Caching")
    print("="*70)

    analyzer = AudioAnalysisEngine(cache_enabled=True)

    # First analysis (no cache)
    print("\n[Analysis 1] First analysis...")
    metrics1 = await analyzer.analyze_audio_data(
        audio=__import__('numpy').random.randn(44100 * 10),  # 10 seconds
        sr=44100
    )
    print(f"  Quality Score: {metrics1.quality_score:.0f}/100")

    # Second analysis (uses cache)
    print("\n[Analysis 2] Cache check...")
    stats = analyzer.get_stats()
    print(f"  Cache Size: {stats['cache_size']}")
    print(f"  Cache Efficiency: {stats['cache_efficiency']:.1f}%")


async def example_4_export_batch():
    """Example 4: Batch export with concurrent processing"""

    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Export Operations")
    print("="*70)

    exporter = ExportManager(max_concurrent_exports=3)
    await exporter.initialize()

    # Submit batch exports
    exports = [
        ('masters/track1.wav', 'deliverables/', 'track1_hifi'),
        ('masters/track2.wav', 'deliverables/', 'track2_hifi'),
        ('masters/track3.wav', 'deliverables/', 'track3_hifi'),
    ]

    print("\n[Export] Submitting batch...")
    batch_result = await exporter.export_batch(
        exports,
        preset_name='Premium HiFi'
    )

    print(f"\n[Export Results]")
    print(f"  Total: {batch_result['total']}")
    print(f"  Successful: {batch_result['successful']}")
    print(f"  Failed: {batch_result['failed']}")

    # Check storage stats
    stats = exporter.get_storage_stats()
    print(f"\n[Storage Stats]")
    print(f"  Total Exported: {stats['total_exported']} files")
    print(f"  Total Size: {stats['total_gb']:.2f}GB")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")


# ============================================================================
# INTEGRATION POINTS FOR YOUR SERVER
# ============================================================================

"""
To integrate these agents into your FastMCP server (server.py):

1. In server.py __init__, add:

   from agents import JobQueueManager, AudioAnalysisEngine, ExportManager

   # Global agent instances
   job_queue = JobQueueManager(max_concurrent_jobs=4)
   audio_analyzer = AudioAnalysisEngine(cache_enabled=True)
   export_mgr = ExportManager(max_concurrent_exports=3)

2. In your mastering tool, use the job queue:

   @mcp.tool()
   def start_mastering(job):
       audio_id = job['audio_id']
       preset = job['preset']
       session_id = job['session_id']

       # Queue the job
       job_id = job_queue.submit_job(
           audio_id=audio_id,
           preset=preset,
           session_id=session_id
       )

       return {'job_id': job_id}

3. For Flask integration (mastering_ui.py), use resources to check status:

   @mcp.resource()
   def tool_resources():
       return {
           'job_queue_status': job_queue.get_queue_status(),
           'analyzer_stats': audio_analyzer.get_stats(),
           'export_stats': export_mgr.get_storage_stats()
       }

4. Add a separate async loop to process the queue:

   async def run_queue_processor():
       async def process_job(job):
           # Call actual mastering logic
           result = await do_mastering(job)
           return (True, result) if result else (False, None)

       await job_queue.process_queue(process_job)

5. For exports, after mastering completes:

   export_id = export_mgr.submit_export(
       audio_path=master_output_path,
       output_dir='Album_Ignorance_is_bliss/masters/',
       filename=f'{song_name}__{preset}',
       format='wav',
       metadata={'preset': preset}
   )
"""


if __name__ == '__main__':
    # Run examples
    asyncio.run(example_1_single_mastering())
    asyncio.run(example_2_batch_processing())
    asyncio.run(example_3_analysis_caching())
    asyncio.run(example_4_export_batch())
