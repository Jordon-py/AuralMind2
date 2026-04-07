# AI Project Memory

## Project Index
- [2026-04-02] AuralMind2 - MCP-driven audio mastering and orchestration repo focused on mastering workflows, async jobs, and audio analysis - active

## Active Project Snapshot
- Date: 2026-04-02
- Active project: AuralMind2
- Goal: Non-destructively group source songs, select the 2 newest versions per song, and create high-fidelity trap masters inside `./Ignorance Is Bliss` using the AuralMind2 MCP workflow first.
- Current phase: batch complete, QA summarized, and outputs staged for review
- Current blocker: several exported masters failed the stricter QA gate, mainly due to stereo correlation/image behavior on no-stems fallbacks or conservative loudness on a few melodic tracks
- Next milestone: review the flagged masters, decide whether to accept them as-is or rerun the failed set with narrower width / hotter targets / alternate presets

## Important Paths
- Project root: C:\Users\goku\Documents\AuralMind2
- Source audio path: C:\Users\goku\Documents\AuralMind2\data
- Mastering code path: C:\Users\goku\Documents\AuralMind2\mastering
- Output root: C:\Users\goku\Documents\AuralMind2\Ignorance Is Bliss
- Reports path: C:\Users\goku\Documents\AuralMind2\Ignorance Is Bliss\_reports
- Manifests path: C:\Users\goku\Documents\AuralMind2\Ignorance Is Bliss\_manifests

## Important Files
- C:\Users\goku\Documents\AuralMind2\server.py - main MCP server implementation and likely local orchestration entrypoint
- C:\Users\goku\Documents\AuralMind2\README.md - repo usage and operational context
- C:\Users\goku\Documents\AuralMind2\ai_memory.md - repo-local execution memory for this mastering run

## Architecture / Data Flow Notes
- [2026-04-02] Live MCP surface confirms async jobs, closed-loop mastering, semantic strategy planning, control profiles, safe filesystem access, server-side ingest, and chunked uploads.
- [2026-04-02] Core resources exposed by `AuralMind2`: `config://system-prompt`, `config://mcp-docs`, `config://maintainer-guide`, `config://server-info`, `auralmind://workflow`, `auralmind://metrics`, `auralmind://presets`, `auralmind://control-surface`, `auralmind://contracts`, `auralmind://connect-kit`.

## Progress Log
- [2026-04-02] completed: discovered live AuralMind2 tools, prompts, resources, and feature flags via `bootstrap`, `capabilities`, resource listing, preset listing, session-state listing, and audio-asset listing.
- [2026-04-02] discovered: repo-local `./ai_memory.md` was missing and was created for this run; a separate global memory file exists at `C:\Users\goku\Documents\Projects\ai_memory.md`.
- [2026-04-02] completed: inspected `server.py`, `resources/mcp_docs.md`, `resources/system_prompt.md`, `resources/maintainer_guide.md`, `tools/auralmind_maestro.py`, and the batch task runners to confirm the async job path, closed-loop path, and local fallback path.
- [2026-04-02] completed: created the `Ignorance Is Bliss` output root plus ready-state schemas for `_reports/run_log.md`, `_manifests/run_results.json`, and `_manifests/qa_summary.json`.
- [2026-04-02] next: inspect repo scripts, finalize manifests, and execute mastering jobs with deterministic batching.
- [2026-04-02] completed: scanned `./data` recursively and classified 98 audio files with provenance tags for source candidates, derived renders, reference tracks, and scratch/misc items.
- [2026-04-02] completed: wrote deterministic manifests to `C:\Users\goku\Documents\AuralMind2\Ignorance Is Bliss\_manifests\source_inventory.json`, `song_groups.json`, and `selected_versions.json`.
- [2026-04-02] verified: 16 eligible same-title groups were selected with an 8-group no-stems first half and an 8-group stems-if-available second half.
- [2026-04-02] discovered: `Last Time` needed explicit cleanup to merge `Last Time (13)` / `Last Time (14)` derivative families back into the same canonical title.
- [2026-04-03] completed: superseded the earlier provisional group count; final selected set is 14 eligible song groups / 28 source versions, sorted alphabetically by normalized title and hardlinked into `Ignorance Is Bliss/_sorted/`.
- [2026-04-03] completed: created `tools/run_ignorance_is_bliss_batch.py` to run the live AuralMind2 server module directly, cap concurrency at 2 jobs, preserve `movement_amount=0.32`, and export masters plus per-track summaries into `Ignorance Is Bliss/masters/`.
- [2026-04-03] completed: mastered all 28 selected versions with 0 execution failures using the MCP-first orchestration path backed by the local server module and session artifact storage.
- [2026-04-03] verified: final output set includes 14 per-song master folders, updated manifests, `strategy_notes.md`, `run_log.md`, and a repo-level `final_summary.md`.
- [2026-04-03] verified: planned stems mode covered the second-half 7 groups, but 7 of those 14 track-level jobs fell back to no-stems after runtime or quality checks (`I'm Him` x2, `In The Moment` x1, `Last Time` x2, `Project` x2).
- [2026-04-03] next: inspect the 14 QA-flagged exports and decide whether to rerun only those tracks or accept them as creative variants.
- [2026-04-03] completed: created `tools/remaster_difference10_creative_variants.py` for a focused single-track creative remaster workflow on `data/difference (10).wav`.
- [2026-04-03] completed: rendered 3 creative `Difference` variants into `Ignorance Is Bliss/masters/Difference/creative-remaster-3x/` with `movement_amount=0.35`.
- [2026-04-03] verified: all 3 creative variants were finished with local two-pass `ffmpeg loudnorm` after AuralMind shaping so the final outputs landed inside the requested loudness / ceiling window.
- [2026-04-03] completed: prepared a rerun-target subset for the 14 QA-failed tracks only, preserving the existing `Song::FileName` track-key convention used by `run_results.json` and `qa_summary.json`.
- [2026-04-03] note: the rerun subset is intentionally narrower than the earlier 14-pass/14-flag split in `qa_summary.json`; it follows the user-provided failed-track list exactly.

## Pain Points / Bugs / Risks
- [2026-04-02] issue: `./data` includes prior AuralMind renders, reports, compat files, and likely mastered derivatives mixed with raw source versions.
- impact: grouping can accidentally over-select derived masters instead of source candidates if normalization is too aggressive.
- suspected cause: the data folder is being used as both a source pool and an output scratch space.
- next step: classify files by provenance signals and preserve grouping confidence plus exclusions in manifests.
- [2026-04-03] issue: 14 of the 28 exported masters did not clear the stricter QA pass gate even though all 28 rendered successfully.
- impact: delivery is complete, but several files should be reviewed before being treated as final-approved masters.
- suspected cause: no-stems jobs on certain trap mixes widened or decorrelated the image more than desired, while a few melodic tracks stayed comparatively conservative in loudness.
- next step: rerun only the flagged tracks with reduced spatial width, slightly hotter target LUFS where musically safe, and possibly `radio_loud` or `competitive_trap` swaps depending on the source.

## Metrics / Quality Signals
- [2026-04-02] metric: live AuralMind2 preset count
- value: 6 presets discovered (`hi_fi_streaming`, `radio_loud`, `cinematic`, `club`, `competitive_trap`, `club_clean`)
- meaning: enough preset diversity for trap-focused semantic planning and A/B fallback if one direction underperforms
- trend/notes: `competitive_trap` and `radio_loud` look most likely to anchor final strategies, with `hi_fi_streaming` as the cleaner fallback
- [2026-04-03] metric: batch mastering completion
- value: 28 completed tracks, 0 failed jobs, 7 stems-to-no-stems fallbacks
- meaning: the orchestration path was stable and resumable across the full selected set
- trend/notes: stems jobs were treated as slower, approximately 7-minute operations for scheduling purposes, so concurrency stayed capped at 2 active jobs
- [2026-04-03] metric: aggregate final master metrics
- value: average integrated LUFS `-15.324`, average true peak `-1.0 dBTP`, average stereo correlation `0.768`
- meaning: loudness landed in a competitive but still conservative trap zone with consistent true-peak control
- trend/notes: the loudness average was pulled down by a few melodic / fallback tracks that should be the first rerun candidates
- [2026-04-03] metric: QA pass rate
- value: 14 passed / 14 flagged
- meaning: execution success was high, but the quality gate was intentionally stricter than simple render success
- trend/notes: failed items are concentrated in `Been Winning`, `Fall In Love`, `Fire`, `Got Too`, `Hot Shit`, `I'm Him`, `In The Moment`, `Last Time`, and `Project`
- [2026-04-03] metric: `Difference (10)` creative remaster variants
- value: `Sub Spine` LUFS `-13.83` TP `-0.75`; `Low-End Halo` LUFS `-13.91` TP `-0.75`; `Jet Fuel Lean` LUFS `-13.73` TP `-0.75`
- meaning: the creative trio hit the requested final delivery window while preserving distinct bass-shaping identities
- trend/notes: `Low-End Halo` was the only stems-based variant; the other two used no-stems and differentiated mostly through low-end shape, width discipline, and tonal lean

## AI Self Notes
- [2026-04-02] Keep all source audio untouched; use manifests, staged copies, or links only if needed.
- [2026-04-02] `movement_amount` must default to `0.32` for every delivered master unless a hard quality gate forces a documented reduction.
- [2026-04-02] First half of eligible groups must run no-stems; second half should attempt stems and clearly log any fallback.
- [2026-04-02] `config://server-info` confirms the live register roots are `data/` and `Downloads`, max upload bytes are 419,430,400, and the accepted stem modes are `off`, `auto`, and `on`.
- [2026-04-02] The local CLI and task runners already support deterministic `movement_amount`, stem toggles, and summary JSON copying, so `master_audio` is the cleanest fallback if async jobs stall.
- [2026-04-03] `semantic_a_b_mastering` and `master_closed_loop` were inspected and attempted during discovery, but they were not used for the full batch because the tool-layer timeout was too slow for a 28-track run.
- [2026-04-03] The direct batch runner copies mastered WAVs from the session artifact store instead of `read_artifact`, which is materially faster for large outputs.
- [2026-04-03] For exact final loudness / true-peak delivery on creative one-offs, AuralMind shaping plus a local two-pass `ffmpeg loudnorm` finish is a strong hybrid path.

## Christopher ↔ AI Notes
### Notes from Christopher
- [2026-04-02] Requested a 3-sub-agent execution-first run inside AuralMind2 to inventory `./data`, group same-title versions, pick the 2 newest per group, and master them into `./Ignorance Is Bliss` using AuralMind2 MCP as the primary execution path.

### Questions for Christopher
- [2026-04-02] None yet; proceed best-effort unless a destructive ambiguity or hard blocker appears.

### Tips / Suggestions
- [2026-04-02] Because `./data` mixes source and previously rendered material, the highest-value repo improvement after this run may be separating raw source pools from render outputs.
- [2026-04-03] For the next mastering pass, consider promoting the batch runner into a first-class repo utility with explicit QA rerun support so failed masters can be reprocessed automatically instead of manually curated.

## Next Best Actions
- [2026-04-02] priority 1: classify and group candidate songs with confidence scoring and provenance tracking
- [2026-04-02] priority 2: read the live MCP docs/resources and derive a trap-specific mastering plan per selected version
- [2026-04-02] priority 3: execute mastering in deterministic batches with QA metrics and fallback logging
- [2026-04-02] priority 4: consume the selected-version manifest and begin MCP job execution with no-stems/stems batching
- [2026-04-02] priority 5: keep the manifests as the source of truth for downstream mastering and avoid touching originals in `./data`
- [2026-04-03] priority 1: review the 14 QA-flagged masters and decide whether to rerun or approve them manually
- [2026-04-03] priority 2: keep `run_results.json` and `qa_summary.json` as the machine-readable truth for any follow-up reruns
- [2026-04-03] priority 3: if a second pass is requested, target stereo correlation and loudness issues first instead of reprocessing the whole batch
- [2026-04-03] priority 1: if Christopher wants more `Difference` explorations, branch from the new creative-remaster script instead of the batch runner
- [2026-04-03] priority 2: if the flagged batch tracks are still a priority, resume with the isolated pass-2 rerun manifests rather than touching pass-1 truth
- [2026-04-03] priority 4: if rerun manifests are created, store them under `Ignorance Is Bliss/_manifests/` with a narrow qa-failed-only scope
