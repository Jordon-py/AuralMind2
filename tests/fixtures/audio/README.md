# Audio Fixtures

Top-level documentation: this folder is the only Git-allowlisted place for tiny audio files used by tests. Data shapes include short synthetic WAV/FLAC/MP3 clips, fixture notes, and optional future hashes. Important functions that may use these fixtures are `server.py:3787 run_master_job`, `server.py:3987 master_audio`, and `tools/auralmind_maestro.py:2683 load_audio`. Possible bugs: real songs or generated masters could be added here accidentally; overly large fixtures will slow CI and bloat history. Two extensions: add a deterministic fixture generator, and add a manifest with duration, sample rate, size, and hash.

Rules:

- Keep fixtures tiny, preferably under 256 KB and under 2 seconds.
- Use synthetic or clearly licensed audio only.
- Do not place Christopher's real source songs or rendered masters here.
- Document why each fixture exists before adding it.
