#!/usr/bin/env python3
"""
Export Completed AuralMind2 Mastering Jobs
Retrieves results from completed jobs and saves to Album folder
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

from server import (
    job_status as mcp_get_job_status,
    job_result as mcp_fetch_job_result,
    read_artifact as mcp_read_artifact
)

# Create a mock FastMCP context for direct tool calls
class MockContext:
    def __init__(self, session_id='export_masters'):
        self.session_id = session_id


class MasterExporter:
    """Export mastering job results"""

    def __init__(self):
        self.ctx = MockContext()
        self.output_dir = Path('Album_Ignorance_is_bliss/masters')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_log = []

    async def check_job_status(self, job_id: str) -> dict:
        """Check status of a mastering job"""

        try:
            result = await mcp_get_job_status(job_id, self.ctx)
            return result
        except Exception as e:
            print(f"✗ Error checking job {job_id}: {e}")
            return None

    async def export_job(
        self,
        job_id: str,
        song_name: str = "Master",
        preset: str = "unknown"
    ) -> bool:
        """Export a mastering job to Album folder"""

        print(f"\n[Export] Processing job: {job_id}")

        try:
            # Check status
            status = await self.check_job_status(job_id)

            if not status:
                print(f"  ✗ Could not retrieve job status")
                return False

            print(f"  Status: {status.get('status', 'unknown')}")
            print(f"  Progress: {status.get('progress', 0)}%")

            # Check if complete
            if status.get('status') != 'done':
                print(f"  ⏳ Job not complete, skipping")
                return False

            # Fetch results
            print(f"  Fetching results...")
            result = await mcp_fetch_job_result(job_id, self.ctx)

            if not result:
                print(f"  ✗ No result returned")
                return False

            # Extract artifact
            artifacts = result.get('artifacts', [])
            if not artifacts:
                print(f"  ✗ No artifacts in result")
                return False

            artifact_id = artifacts[0].get('id')
            print(f"  Artifact ID: {artifact_id}")

            # Read artifact data
            print(f"  Reading audio data...")
            artifact_data = await mcp_read_artifact(artifact_id, self.ctx)

            if not artifact_data:
                print(f"  ✗ No data from artifact")
                return False

            # Save to file
            output_file = self.output_dir / f"{song_name}_{preset}_Master.wav"

            print(f"  Saving to: {output_file}")
            with open(output_file, 'wb') as f:
                f.write(artifact_data)

            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ Saved ({file_size:.1f} MB)")

            # Log export
            self.export_log.append({
                'timestamp': datetime.now().isoformat(),
                'job_id': job_id,
                'song_name': song_name,
                'preset': preset,
                'output_file': str(output_file),
                'file_size_mb': file_size,
                'status': 'success'
            })

            return True

        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.export_log.append({
                'timestamp': datetime.now().isoformat(),
                'job_id': job_id,
                'song_name': song_name,
                'preset': preset,
                'status': 'failed',
                'error': str(e)
            })
            return False

    async def export_batch(self, jobs: list) -> dict:
        """Export multiple jobs"""

        print("=" * 60)
        print("AuralMind2 Master Export Utility")
        print("=" * 60)
        print(f"\nExporting {len(jobs)} job(s) to: {self.output_dir}")

        results = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'exports': []
        }

        for job_info in jobs:
            success = await self.export_job(
                job_info['job_id'],
                job_info.get('song_name', 'Master'),
                job_info.get('preset', 'unknown')
            )

            if success:
                results['successful'] += 1
                results['exports'].append(job_info)
            else:
                results['failed'] += 1

        # Print summary
        print(f"\n" + "=" * 60)
        print("Export Summary")
        print("=" * 60)
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Exports saved to: {self.output_dir}")

        # List exported files
        exported_files = list(self.output_dir.glob('*.wav'))
        if exported_files:
            print(f"\nExported files ({len(exported_files)}):")
            for f in sorted(exported_files):
                size = f.stat().st_size / (1024 * 1024)
                print(f"  ✓ {f.name} ({size:.1f} MB)")

        # Save log
        self.save_export_log()

        return results

    def save_export_log(self):
        """Save export log to JSON"""

        log_file = self.output_dir / 'export_log.json'

        with open(log_file, 'w') as f:
            json.dump(self.export_log, f, indent=2)

        print(f"\nExport log saved to: {log_file}")


async def main():
    """Main execution"""

    if len(sys.argv) < 2:
        print("Usage: python export_masters.py <job_id> [song_name] [preset]")
        print("       python export_masters.py --batch job1,job2,job3")
        print()
        print("Examples:")
        print("  python export_masters.py job_b0dc4dc70408 NewProject15 competitive_trap")
        print("  python export_masters.py --batch job_abc123,job_def456,job_ghi789")
        sys.exit(1)

    exporter = MasterExporter()

    if sys.argv[1] == '--batch':
        # Batch export from comma-separated job IDs
        if len(sys.argv) < 3:
            print("Usage: python export_masters.py --batch job1,job2,job3")
            sys.exit(1)

        job_ids = sys.argv[2].split(',')
        jobs = [{'job_id': jid.strip(), 'song_name': 'Master', 'preset': 'unknown'}
                for jid in job_ids]

        await exporter.export_batch(jobs)

    else:
        # Single export
        job_id = sys.argv[1]
        song_name = sys.argv[2] if len(sys.argv) > 2 else 'Master'
        preset = sys.argv[3] if len(sys.argv) > 3 else 'unknown'

        jobs = [{'job_id': job_id, 'song_name': song_name, 'preset': preset}]
        await exporter.export_batch(jobs)


if __name__ == '__main__':
    asyncio.run(main())
