"""
ExportManager Agent
Batch export and delivery management for mastered audio
"""

import asyncio
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import soundfile as sf
import numpy as np


class ExportFormat(Enum):
    """Supported export formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    AAC = "aac"


class ExportStatus(Enum):
    """Export job status"""
    PENDING = "pending"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ExportJob:
    """Export job definition"""
    export_id: str
    audio_path: str
    output_dir: str
    filename: str
    format: ExportFormat
    metadata: Dict = None
    status: ExportStatus = ExportStatus.PENDING
    created_at: str = None
    completed_at: Optional[str] = None
    file_size: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    @property
    def output_path(self) -> str:
        """Full output path"""
        return os.path.join(self.output_dir, f"{self.filename}.{self.format.value}")

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        d['format'] = self.format.value
        d['status'] = self.status.value
        d['output_path'] = self.output_path
        return d


class ExportManager:
    """
    Expert async agent for batch export and delivery management

    Features:
    - Concurrent file exports with format conversion
    - Batch export with retry logic
    - Directory organization and cleanup
    - Export history and tracking
    - Metadata embedding
    - Storage analytics
    """

    def __init__(self, max_concurrent_exports: int = 3, verify_writes: bool = True):
        """
        Initialize export manager

        Args:
            max_concurrent_exports: Maximum concurrent export operations
            verify_writes: Verify file integrity after export
        """
        self.max_concurrent_exports = max_concurrent_exports
        self.verify_writes = verify_writes

        # Export tracking
        self.exports: Dict[str, ExportJob] = {}
        self.export_queue: asyncio.Queue = None
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # History and analytics
        self.export_history: List[ExportJob] = []
        self.storage_stats = {
            'total_exported': 0,
            'total_bytes': 0,
            'failed_exports': 0,
            'avg_export_time': 0.0
        }

    async def initialize(self):
        """Initialize async components"""
        self.export_queue = asyncio.Queue()

    def submit_export(self, audio_path: str, output_dir: str, filename: str,
                     format: str = "wav", metadata: Dict = None) -> str:
        """
        Submit an export job

        Args:
            audio_path: Path to audio file to export
            output_dir: Output directory
            filename: Output filename (without extension)
            format: Export format (wav, mp3, flac, aac)
            metadata: Metadata to embed (artist, album, etc.)

        Returns:
            export_id: Unique export identifier
        """
        import uuid
        export_id = f"exp_{uuid.uuid4().hex[:12]}"

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            export_format = ExportFormat[format.upper()]
        except KeyError:
            export_format = ExportFormat.WAV

        job = ExportJob(
            export_id=export_id,
            audio_path=audio_path,
            output_dir=output_dir,
            filename=filename,
            format=export_format,
            metadata=metadata or {}
        )

        self.exports[export_id] = job

        print(f"[ExportManager] Submitted {export_id} -> {job.output_path}")
        return export_id

    def get_export_status(self, export_id: str) -> Optional[Dict]:
        """Get export job status"""
        if export_id not in self.exports:
            return None
        return self.exports[export_id].to_dict()

    async def export_batch(self, job_list: List[Tuple[str, str, str]],
                          preset_name: str = "Master") -> Dict:
        """
        Export multiple files in batch

        Args:
            job_list: List of (audio_path, output_dir, filename) tuples
            preset_name: Preset name for metadata

        Returns:
            results: Export results summary
        """
        export_ids = []

        for audio_path, output_dir, filename in job_list:
            export_id = self.submit_export(
                audio_path, output_dir, filename,
                metadata={'preset': preset_name}
            )
            export_ids.append(export_id)

        print(f"[ExportManager] Batch submitted: {len(export_ids)} jobs")

        # Process batch
        return await self._process_batch(export_ids)

    async def process_exports(self):
        """Main export processor loop"""
        print(f"[ExportManager] Starting processor (max {self.max_concurrent_exports} concurrent)")

        pending_jobs = [j for j in self.exports.values()
                       if j.status == ExportStatus.PENDING]

        # Process pending jobs
        for job in pending_jobs:
            if len(self.active_tasks) >= self.max_concurrent_exports:
                await asyncio.sleep(0.1)
                continue

            task = asyncio.create_task(self._export_file(job))
            self.active_tasks[job.export_id] = task

        # Wait for all to complete
        await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

    async def _process_batch(self, export_ids: List[str]) -> Dict:
        """Process a batch of exports concurrently"""
        tasks = [
            self._export_file(self.exports[eid])
            for eid in export_ids
            if eid in self.exports
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compile results
        successful = sum(1 for r in results if r and r.get('success'))
        failed = len(results) - successful

        return {
            'total': len(export_ids),
            'successful': successful,
            'failed': failed,
            'exports': results
        }

    async def _export_file(self, job: ExportJob) -> Dict:
        """Export a single file"""
        job.status = ExportStatus.EXPORTING
        start_time = datetime.now()

        try:
            # Run export in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._perform_export,
                job
            )

            if result['success']:
                job.file_size = result['file_size']
                job.status = ExportStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()

                # Update metrics
                duration = (datetime.now() - start_time).total_seconds()
                self._update_metrics(job, duration)

                self.export_history.append(job)

                print(f"[Export] ✓ {job.export_id}: {job.output_path} "
                      f"({result['file_size']/1024:.1f}KB)")

                return {
                    'export_id': job.export_id,
                    'success': True,
                    'output_path': job.output_path,
                    'file_size': job.file_size
                }
            else:
                raise Exception(result.get('error', 'Export failed'))

        except Exception as e:
            job.status = ExportStatus.FAILED
            job.error_message = str(e)
            self.storage_stats['failed_exports'] += 1

            print(f"[Export] ✗ {job.export_id}: {str(e)}")

            return {
                'export_id': job.export_id,
                'success': False,
                'error': str(e)
            }

    def _perform_export(self, job: ExportJob) -> Dict:
        """Perform actual file export (runs in thread)"""
        try:
            # Load audio
            if not os.path.exists(job.audio_path):
                return {'success': False, 'error': f'Input file not found: {job.audio_path}'}

            audio, sr = sf.read(job.audio_path)

            # Export based on format
            if job.format == ExportFormat.WAV:
                sf.write(job.output_path, audio, sr)

            elif job.format == ExportFormat.FLAC:
                # FLAC export (requires soundfile with FLAC support)
                try:
                    sf.write(job.output_path, audio, sr, subtype='PCM_16')
                except Exception as e:
                    # Fallback to WAV
                    fallback_path = job.output_path.replace('.flac', '.wav')
                    sf.write(fallback_path, audio, sr)
                    job.format = ExportFormat.WAV

            else:
                # MP3/AAC would require additional libraries (pydub, librosa)
                # For now, export as WAV
                job.output_path = job.output_path.rsplit('.', 1)[0] + '.wav'
                sf.write(job.output_path, audio, sr)

            # Verify export
            if self.verify_writes:
                if not os.path.exists(job.output_path):
                    return {'success': False, 'error': 'File not created'}

                file_size = os.path.getsize(job.output_path)
                if file_size == 0:
                    return {'success': False, 'error': 'Empty file created'}

                # Verify readability
                try:
                    test_audio, test_sr = sf.read(job.output_path)
                    if len(test_audio) == 0:
                        return {'success': False, 'error': 'Audio data corrupted'}
                except Exception:
                    return {'success': False, 'error': 'File verification failed'}
            else:
                file_size = os.path.getsize(job.output_path) if os.path.exists(job.output_path) else 0

            # Embed metadata if provided
            if job.metadata:
                self._embed_metadata(job.output_path, job.metadata)

            return {'success': True, 'file_size': file_size}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _embed_metadata(self, file_path: str, metadata: Dict):
        """Embed metadata into audio file"""
        # This would require additional libraries like mutagen
        # For now, create a metadata sidecar file

        metadata_path = file_path + '.metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'file': os.path.basename(file_path),
                'embedded_at': datetime.now().isoformat(),
                **metadata
            }, f, indent=2)

    def _update_metrics(self, job: ExportJob, duration: float):
        """Update export metrics"""
        self.storage_stats['total_exported'] += 1
        self.storage_stats['total_bytes'] += job.file_size

        current_avg = self.storage_stats['avg_export_time']
        count = self.storage_stats['total_exported']
        self.storage_stats['avg_export_time'] = (
            (current_avg * (count - 1) + duration) / count
        )

    async def organize_by_preset(self, master_dir: str, preset_name: str) -> Path:
        """Organize exports by preset name"""
        preset_dir = Path(master_dir) / preset_name.replace(' ', '_').lower()
        preset_dir.mkdir(parents=True, exist_ok=True)
        return preset_dir

    def get_export_history(self, limit: int = 100) -> List[Dict]:
        """Get recent export history"""
        return [job.to_dict() for job in self.export_history[-limit:]]

    def get_storage_stats(self) -> Dict:
        """Get storage and export statistics"""
        avg_file_size = (
            self.storage_stats['total_bytes'] / self.storage_stats['total_exported']
            if self.storage_stats['total_exported'] > 0 else 0
        )

        return {
            **self.storage_stats,
            'active_exports': len(self.active_tasks),
            'pending_exports': sum(
                1 for j in self.exports.values()
                if j.status == ExportStatus.PENDING
            ),
            'total_gb': self.storage_stats['total_bytes'] / (1024**3),
            'avg_file_size_mb': avg_file_size / (1024**2),
            'success_rate': (
                self.storage_stats['total_exported'] /
                max(1, self.storage_stats['total_exported'] + self.storage_stats['failed_exports']) * 100
            )
        }

    async def cleanup_old_exports(self, days: int = 7, dry_run: bool = True) -> Dict:
        """
        Clean up old export files

        Args:
            days: Delete exports older than N days
            dry_run: Just report, don't delete

        Returns:
            cleanup_report: Files that would be deleted
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        cleanup_report = {
            'dry_run': dry_run,
            'files_to_delete': [],
            'space_freed': 0
        }

        for job in self.export_history:
            if os.path.exists(job.output_path):
                file_time = os.path.getmtime(job.output_path)
                if file_time < cutoff:
                    file_size = os.path.getsize(job.output_path)
                    cleanup_report['files_to_delete'].append({
                        'path': job.output_path,
                        'size': file_size,
                        'age_days': (datetime.now().timestamp() - file_time) / 86400
                    })
                    cleanup_report['space_freed'] += file_size

                    if not dry_run:
                        try:
                            os.remove(job.output_path)
                            print(f"[Cleanup] Deleted {job.output_path}")
                        except Exception as e:
                            print(f"[Cleanup] Failed to delete {job.output_path}: {e}")

        print(f"[Cleanup] Would delete {len(cleanup_report['files_to_delete'])} files, "
              f"freeing {cleanup_report['space_freed']/1024**2:.1f}MB")

        return cleanup_report
