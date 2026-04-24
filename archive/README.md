# AuralMind2 Archive

Top-level documentation: this folder holds files moved out of the repo root during cleanup without deleting potentially useful history. Data shapes include Markdown delivery notes, notebooks, legacy Python engines, and stale config snippets. Important runtime functions remain outside this archive: `server.py:3724 run_master_job`, `server.py:3924 master_audio`, and `tools/auralmind_maestro.py:2804 master`. Possible bugs: archived scripts may have stale imports or assumptions; moving archived files back into root can reintroduce clutter. Two extensions: add an index with owner/status per archived file, and promote any reusable archived idea into `tools/` with tests before restoring it.

## 2026-04-23 Cleanup Buckets

- `legacy_docs_20260423/`: older delivery and quick-start documents that no longer need to occupy the repo root.
- `legacy_engines_20260423/`: duplicate or non-canonical mastering engines kept for reference.
- `notebooks_20260423/`: exploratory notebooks moved out of the source root.
- `stale_config_20260423/`: non-canonical config fragments, including the old duplicate `gitignore` file.
