# Artifact Hygiene

Top-level documentation: this file records the repository policy for source audio, generated masters, logs, manifests, and test fixtures. Data shapes include local audio assets under `data/`, generated master folders, runtime JSON/log files, Git-tracked docs, and tiny future test fixtures. Important functions affected by this policy: `server.py:3787 run_master_job`, `server.py:3987 master_audio`, and `tools/auralmind_maestro.py:2804 master`. Possible bugs: untracked audio still exists locally and can be lost outside backups; a broad ignore rule can hide a fixture if the fixture allowlist is not used. Two extensions: add a checked manifest generator with file hashes, and add external object-storage sync for source audio.

## Decision

Source audio and rendered masters should not live in Git history. They should remain on disk for local mastering work and be referenced by manifests or delivery notes.

This pass removed tracked `data/` audio and DAW sidecars from the Git index only. The files remain in `C:\Users\goku\Documents\AuralMind2\data` for local use.

Current decision: `data/` is a local source pool, not a versioned fixture folder. Future portable references should use checked-in manifests or external storage, not Git blobs.

The current local source pool is referenced in `docs/DATA-ASSET-MANIFEST.md`. That manifest is intentionally metadata-only and does not validate or copy audio.

## Boundaries

- Keep in Git: source code, docs, tests, small text manifests, and fixture policy files.
- Keep out of Git: `.wav`, `.mp3`, `.m4a`, `.flac`, `.aif`, `.aiff`, `.ogg`, `.asd`, `.reapeaks`, runtime DBs, generated masters, logs, and local environment files.
- Allow only curated tiny audio fixtures under `tests/fixtures/audio/`.

## Migration Pattern

1. Leave local audio files in place.
2. Keep `.gitignore` blocking new audio additions.
3. Store future source-set references in docs/manifests rather than Git blobs.
4. If a tiny audio fixture is needed for tests, place it under `tests/fixtures/audio/` and document why it exists.

## Verification

- `git ls-files data` should not list source audio or DAW sidecars after this cleanup.
- `git status --short` may show staged deletions for previously tracked audio; that is expected and means the files will be removed from Git on commit, not deleted from disk.
