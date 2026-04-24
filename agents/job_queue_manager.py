"""
JobQueueManager Agent
Manages async mastering job queuing, processing, and retries
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import json


class JobStatus(Enum):
    """Job lifecycle states"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class MasteringJob:
    """Mastering job definition"""
    job_id: str
    audio_id: str
    preset: str
    session_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0  # Higher = processed first
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        d['status'] = self.status.value
        return d


class JobQueueManager:
    """
    Expert async agent for mastering job queue management

    Features:
    - Priority queue for job ordering
    - Retry logic with exponential backoff
    - Job status tracking and persistence
    - Async concurrent job processing
    - Dead letter queue for failed jobs
    """

    def __init__(self, max_concurrent_jobs: int = 4, checkpoint_file: str = None):
        """
        Initialize queue manager

        Args:
            max_concurrent_jobs: Maximum jobs to process concurrently
            checkpoint_file: Path to persist queue state
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.checkpoint_file = checkpoint_file

        # Job storage
        self.jobs: Dict[str, MasteringJob] = {}
        self.job_queue: deque = deque()  # Priority queue
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.dead_letter_queue: List[MasteringJob] = []

        # Metrics
        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'total_retries': 0,
            'avg_processing_time': 0.0
        }

    def submit_job(self, audio_id: str, preset: str, session_id: str,
                   priority: int = 0, metadata: Dict = None) -> str:
        """
        Submit a mastering job to the queue

        Args:
            audio_id: Reference to audio artifact
            preset: Mastering preset name
            session_id: Client session ID
            priority: Job priority (higher processed first)
            metadata: Custom job metadata

        Returns:
            job_id: Unique job identifier
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        job = MasteringJob(
            job_id=job_id,
            audio_id=audio_id,
            preset=preset,
            session_id=session_id,
            priority=priority,
            metadata=metadata or {}
        )

        self.jobs[job_id] = job
        self._enqueue_job(job)

        self.stats['total_jobs'] += 1

        print(f"[JobQueue] Submitted {job_id} (preset: {preset}, priority: {priority})")
        return job_id

    def _enqueue_job(self, job: MasteringJob):
        """Add job to priority queue"""
        self.job_queue.append(job)
        # Sort by priority (descending) and created_at (ascending)
        self.job_queue = deque(sorted(
            self.job_queue,
            key=lambda j: (-j.priority, j.created_at)
        ))

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status"""
        if job_id not in self.jobs:
            return None
        return self.jobs[job_id].to_dict()

    def get_queue_status(self) -> Dict:
        """Get queue statistics"""
        active = [j for j in self.jobs.values() if j.status == JobStatus.PROCESSING]
        queued = [j for j in self.jobs.values() if j.status == JobStatus.QUEUED]

        return {
            'active_jobs': len(active),
            'queued_jobs': len(queued),
            'completed_jobs': self.stats['completed_jobs'],
            'failed_jobs': self.stats['failed_jobs'],
            'dead_letter_count': len(self.dead_letter_queue),
            'stats': self.stats
        }

    async def process_queue(self, job_processor_fn):
        """
        Main async queue processor loop

        Args:
            job_processor_fn: Async function to process a job
                Takes MasteringJob and returns (success: bool, result: Any)
        """
        print(f"[JobQueue] Starting queue processor (max {self.max_concurrent_jobs} concurrent)")

        try:
            while True:
                # Process jobs while queue has items and we have capacity
                while self.job_queue and len(self.active_tasks) < self.max_concurrent_jobs:
                    job = self.job_queue.popleft()

                    # Check retry limits before processing
                    if job.retry_count >= job.max_retries and job.status == JobStatus.RETRYING:
                        self._move_to_dead_letter(job)
                        continue

                    # Create async task for this job
                    task = asyncio.create_task(
                        self._process_job(job, job_processor_fn)
                    )
                    self.active_tasks[job.job_id] = task

                    print(f"[JobQueue] Started processing {job.job_id}")

                # Check for completed tasks
                completed_ids = []
                for job_id, task in list(self.active_tasks.items()):
                    if task.done():
                        completed_ids.append(job_id)

                for job_id in completed_ids:
                    del self.active_tasks[job_id]

                # Wait before next check
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            print("[JobQueue] Queue processor cancelled, waiting for active tasks...")
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

    async def _process_job(self, job: MasteringJob, processor_fn):
        """Process a single job with retry logic"""
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now().isoformat()

        try:
            # Call the processor function
            success, result = await processor_fn(job)

            if success:
                job.status = JobStatus.COMPLETING
                job.completed_at = datetime.now().isoformat()
                job.progress = 100.0

                # Calculate processing time
                start = datetime.fromisoformat(job.started_at)
                end = datetime.fromisoformat(job.completed_at)
                processing_time = (end - start).total_seconds()

                # Update metrics
                self.stats['completed_jobs'] += 1
                self._update_avg_time(processing_time)

                print(f"[JobQueue] ✓ {job.job_id} completed in {processing_time:.1f}s")
                job.status = JobStatus.COMPLETED
            else:
                raise Exception(result or "Processing failed")

        except Exception as e:
            print(f"[JobQueue] ✗ {job.job_id} failed: {str(e)}")
            job.error_message = str(e)

            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.RETRYING
                self.stats['total_retries'] += 1

                # Exponential backoff before retry
                backoff_seconds = min(2 ** job.retry_count, 60)
                print(f"[JobQueue] Retrying {job.job_id} in {backoff_seconds}s "
                      f"(attempt {job.retry_count}/{job.max_retries})")

                await asyncio.sleep(backoff_seconds)
                self._enqueue_job(job)
            else:
                job.status = JobStatus.FAILED
                self.stats['failed_jobs'] += 1
                self._move_to_dead_letter(job)

    def _move_to_dead_letter(self, job: MasteringJob):
        """Move failed job to dead letter queue"""
        job.status = JobStatus.FAILED
        self.dead_letter_queue.append(job)
        print(f"[JobQueue] {job.job_id} moved to dead letter queue "
              f"(failed after {job.retry_count} retries)")

    def _update_avg_time(self, processing_time: float):
        """Update moving average of processing time"""
        completed = self.stats['completed_jobs']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (
            (current_avg * (completed - 1) + processing_time) / completed
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or processing job"""
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        if job.status in (JobStatus.QUEUED, JobStatus.RETRYING):
            job.status = JobStatus.CANCELLED
            print(f"[JobQueue] {job_id} cancelled")
            return True

        if job_id in self.active_tasks:
            task = self.active_tasks[job_id]
            task.cancel()
            print(f"[JobQueue] {job_id} task cancelled")
            return True

        return False

    def get_dead_letter_jobs(self, limit: int = 50) -> List[Dict]:
        """Get failed jobs for debugging"""
        return [job.to_dict() for job in self.dead_letter_queue[-limit:]]

    def export_checkpoint(self) -> Dict:
        """Export queue state for persistence"""
        return {
            'jobs': {jid: job.to_dict() for jid, job in self.jobs.items()},
            'dead_letter': [job.to_dict() for job in self.dead_letter_queue],
            'stats': self.stats,
            'checkpoint_time': datetime.now().isoformat()
        }
