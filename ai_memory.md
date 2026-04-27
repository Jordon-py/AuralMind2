# AI Project Memory

## Project Index
- [2026-04-02] AuralMind2 - MCP-driven audio mastering and orchestration repo focused on mastering workflows, async jobs, and audio analysis - active

## Active Project Snapshot
- Date: 2026-04-08
- Active project: AuralMind2
- Goal: Use AuralMind2 to create premium trap masters with explicit stem/no-stem variants and Desktop delivery for `New Project (8).wav` and `New Project (9).wav`.
- Current phase: 4-master premium delivery complete with AuralMind shaping, final loudness pass, and Desktop staging
- Current blocker: none for the requested 4-master delivery; only optional follow-up is taste-based revision after listening
- Next milestone: listen-check the four Desktop masters and revise only if Christopher wants alternate tone or louder/quieter variant targets

## Important Paths
- Project root: C:\Users\goku\Documents\AuralMind2
- Source audio path: C:\Users\goku\Documents\AuralMind2\data
- Mastering code path: C:\Users\goku\Documents\AuralMind2\mastering
- Output root: C:\Users\goku\Documents\AuralMind2\Ignorance Is Bliss
- Desktop delivery path: C:\Users\goku\Desktop\AuralMind2 Premium Masters\New Project 8-9
- Reports path: C:\Users\goku\Documents\AuralMind2\Ignorance Is Bliss\_reports
- Manifests path: C:\Users\goku\Documents\AuralMind2\Ignorance Is Bliss\_manifests

## Important Files
- C:\Users\goku\Documents\AuralMind2\docs\REPO-INFO.md - repo dossier for current runtime map, artifact boundaries, cleanup decisions, risks, and safe next actions.
- C:\Users\goku\Documents\AuralMind2\docs\ARTIFACT-HYGIENE.md - repo policy for keeping source audio and generated masters out of Git while preserving local assets.
- C:\Users\goku\Documents\AuralMind2\docs\DATA-ASSET-MANIFEST.md - metadata-only manifest for the current local `data/` audio pool after Git index cleanup.
- C:\Users\goku\Documents\AuralMind2\docs\SERVER-MODULARIZATION.md - safe modularization order for extracting `server.py` without touching job execution first.
- C:\Users\goku\Documents\AuralMind2\resources\premium_trap_workflow.md - MCP resource guiding connected AI clients toward premium trap/rap mastering decisions and quality gates.
- C:\Users\goku\Documents\AuralMind2\archive\README.md - archive index explaining why legacy docs, notebooks, stale config, and duplicate engines were moved out of the repo root.
- C:\Users\goku\Documents\AuralMind2\server.py - main MCP server implementation and likely local orchestration entrypoint
- C:\Users\goku\Documents\AuralMind2\tools\run_explicit_premium_hifi_trap_batch.py - MCP-only explicit 10-song runner that connects with `fastmcp.Client`, uses server tools for analyze/plan/render/phase-align/export, and writes a resume manifest plus run log.
- C:\Users\goku\Documents\AuralMind2\README.md - repo usage and operational context
- C:\Users\goku\Documents\AuralMind2\ai_memory.md - repo-local execution memory for this mastering run

## Architecture / Data Flow Notes
- [2026-04-24] Explicit premium trap batch flow is MCP-only: `fastmcp.Client(server.mcp)` -> `bootstrap` / resources / prompt -> `register_audio_from_path` -> `analyze_audio` -> `plan_mastering_strategy` / `propose_master_settings` -> `run_master_job` / `job_status` / `job_result` -> `premium_phase_align` -> `analyze_audio` -> `read_artifact` export.
- [2026-04-24] Premium phase alignment is now a first-class MCP tool, `premium_phase_align`, which applies zero-phase low-band isolation and material-aware mono centering to the chosen mastered artifact before export.
- [2026-04-23] MCP connect guidance now has three layers: `FastMCP(instructions=...)` for clients that honor server instructions, `on_connect` / `premium_trap_mastering_session` prompts for prompt-capable clients, and resources (`auralmind://connect-kit`, `auralmind://premium-trap-workflow`, `auralmind://contracts`, `auralmind://control-surface`) for clients that must fetch guidance explicitly.
- [2026-04-23] Source audio policy is now local-assets-first: `data/` audio remains on disk for runners and MCP registration, but audio blobs are removed from Git tracking and represented by `docs/DATA-ASSET-MANIFEST.md`.
- [2026-04-02] Live MCP surface confirms async jobs, closed-loop mastering, semantic strategy planning, control profiles, safe filesystem access, server-side ingest, and chunked uploads.
- [2026-04-02] Core resources exposed by `AuralMind2`: `config://system-prompt`, `config://mcp-docs`, `config://maintainer-guide`, `config://server-info`, `auralmind://workflow`, `auralmind://metrics`, `auralmind://presets`, `auralmind://control-surface`, `auralmind://contracts`, `auralmind://connect-kit`.

## Progress Log
- [2026-04-24] changed: added MCP tool `premium_phase_align` to `server.py`, updated `resources/premium_trap_workflow.md`, and added `tools/run_explicit_premium_hifi_trap_batch.py` so the requested 10-song batch uses the MCP server only for planning, mastering, phase alignment, analysis, and artifact export.
- [2026-04-24] verified: `python -m py_compile server.py tools\\run_explicit_premium_hifi_trap_batch.py`, `python -m pytest tests\\test_discovery_smoke.py -q`, and `python -m pytest -q` passed; FastMCP client discovery reported 37 tools including `premium_phase_align`; dry-run confirmed all 10 requested source files exist.
- [2026-04-24] fixed: first MCP-only batch launch exposed FastMCP in-process argument wrapping for `req: Model` tools; patched the runner to wrap `register_audio_from_path`, `analyze_audio`, `plan_mastering_strategy`, `run_master_job`, `job_status`, `job_result`, and `premium_phase_align` payloads under `req`, then verified register/analyze through the MCP client.
- [2026-04-24] fixed: second MCP-only batch retry exposed a FastMCP prompt-rendering `Platform` TypeAdapter issue for `premium_trap_mastering_session`; prompt loading is now non-fatal while resource guidance and all render/phase-align/analyze/export steps remain MCP tool calls.
- [2026-04-24] changed: repaired `premium_trap_mastering_session` prompt rendering by typing its `platform` argument as `str` and adding `premium_phase_align` to its required flow.
- [2026-04-23] completed: initiated repo-info cleanup for AuralMind2, created `docs/REPO-INFO.md`, expanded `.gitignore` for generated audio/runtime/cache outputs, removed generated cache folders outside `.venv`, and moved root bloat into `archive/` buckets instead of deleting potentially useful history.
- [2026-04-23] changed: moved exploratory notebooks to `archive/notebooks_20260423/`, older delivery docs to `archive/legacy_docs_20260423/`, duplicate legacy engine `auralmind_match_maestro_v7_3_expert1.py.py` to `archive/legacy_engines_20260423/`, and stale duplicate `gitignore` to `archive/stale_config_20260423/`.
- [2026-04-23] note: live AuralMind2 FastMCP stdio processes were detected during cleanup, so active runtime files, active output folders, source audio, and server/job code were intentionally left untouched.
- [2026-04-23] changed: fixed `server.py` preset override filtering so tests with smaller fake preset dataclasses do not receive full-engine-only fields like `enable_masking_eq`.
- [2026-04-23] verified: `python -m py_compile server.py tools\\auralmind_maestro.py mastering_ui_bridge.py mastering_ui.py` passed, and `python -m pytest -q` passed with `19 passed`.
- [2026-04-23] completed: used two Codex 5.4 expert assistants asynchronously: Archimedes reviewed artifact hygiene/tracked-audio risk, and Maxwell reviewed MCP prompt/resource compatibility.
- [2026-04-23] changed: removed 59 tracked `data/` audio/sidecar files from the Git index only; local files remain in `data/`, while `.gitignore` blocks future source/output audio additions and `tests/fixtures/audio/` is the only tiny-fixture allowlist.
- [2026-04-23] changed: added `docs/ARTIFACT-HYGIENE.md`, `docs/DATA-ASSET-MANIFEST.md`, `docs/SERVER-MODULARIZATION.md`, `resources/premium_trap_workflow.md`, and the `premium_trap_mastering_session` MCP prompt.
- [2026-04-23] changed: wired concise `FastMCP(instructions=...)` guidance in `server.py` so clients that honor server instructions are directed to contracts, control-surface, async jobs, and premium trap workflow guidance on connect.
- [2026-04-23] verified: full bootstrap surface now reports `36` tools, `11` resources, and `5` prompts; `auralmind://premium-trap-workflow` and `premium_trap_mastering_session` are published.
- [2026-04-23] verified: `python -m py_compile server.py tools\\auralmind_maestro.py mastering_ui_bridge.py mastering_ui.py` passed, and `python -m pytest -q` passed with `23 passed`.
- [2026-04-21] changed: cleaned up the AuralMind2 MCP surface in `server.py`, tightened prompt/resource wording, added typed request/response wrappers for key tools, and kept the async mastering flow centered on `run_master_job` / `job_status` / `job_result`.
- [2026-04-21] verified: `ruff check server.py tools\\auralmind_maestro.py`, `python -m py_compile server.py tools\\auralmind_maestro.py`, `python -c "import server; print(len(server.bootstrap().tools), len(server.bootstrap().resources), len(server.bootstrap().prompts))"`, and `python -m pytest tests\\test_discovery_smoke.py -q` all passed.
- [2026-04-21] note: semantic-planning tests still hit Windows tempdir permission cleanup failures in this environment, so the remaining verification gap is environment-specific rather than a code parse/lint issue.
- [2026-04-21] changed: cleaned up `server.py` Pylance/Pyright blockers around Pydantic defaults, dataclass job serialization, missing in-memory job registry globals, context default typing, `_get_maestro()` optional narrowing, and `hex_payload` narrowing.
- [2026-04-21] changed: removed a stray invalid `@mcp.tool` example block near the top of `server.py` that made the file fail parsing before the real FastMCP server instance was defined.
- [2026-04-21] verified: `python -m py_compile server.py`, `npx --yes pyright server.py`, an import/constructor smoke check, and `python -m pytest tests/test_mastering_ui_bridge.py -q` all passed.
- [2026-04-21] completed: used Harvey and Galileo sidecar agents for read-only inspection; Harvey confirmed the job/catalog/settings clusters and flagged `VISIBLE_MASTER_FIELDS`, while Galileo confirmed the context, maestro, and upload payload narrowing fixes.
- [2026-04-14] completed: used `AIIntegratedMasteringTool` to master the 2 newest source-like songs in `data/` (`Close to the edge.wav` and `New Project (15).wav`) into 8 exported WAVs under `masters/ai_integrated_latest_two_20260414`.
- [2026-04-14] changed: added `tools/run_ai_integrated_latest_two_songs.py`, which filters `data/` for the newest source candidates, registers/analyzes them through the AI-integrated path, launches 4 variants per song, monitors async jobs, copies mastered artifacts out of the session store, and writes `manifest.json` plus `summary.md`.
- [2026-04-14] verified: `python tools/run_ai_integrated_latest_two_songs.py` completed successfully after launching 8 async jobs and exporting 8 WAVs plus `masters/ai_integrated_latest_two_20260414/manifest.json` and `summary.md`.
- [2026-04-14] completed: used 3 expert sub-agents in parallel for the UI-connection pass, with one agent auditing frontend contract drift, one reviewing FastMCP session/risk handling, and one aligning the launcher plus focused tests.
- [2026-04-14] changed: connected both `templates/index.html` and `templates/mastery.html` to the real Flask mastering flow in `mastering_ui.py` so the browser now uploads audio, creates a session, launches `/api/session/<id>/start`, and polls `/api/session/<id>/status` instead of simulating progress.
- [2026-04-14] changed: rebuilt `mastering_ui_bridge.py` around stable per-session FastMCP contexts, chunked artifact reads, and direct mastered-file export so register/analyze/job/result/export all stay in the same server session.
- [2026-04-14] changed: hardened `mastering_ui.py` to keep uploaded/session audio inside `data/`, anchored UI paths to the repo root, and preserved the UI session store as the source of truth for live job polling.
- [2026-04-14] changed: simplified `run_ui.py` to launch the connected in-process Flask UI directly instead of booting an extra unused MCP HTTP subprocess.
- [2026-04-14] verified: `python -m pytest tests/test_mastering_ui_bridge.py -q` passed with `4 passed`.
- [2026-04-14] verified: a Flask test-client smoke path for `/api/upload` -> `/api/session/new` -> `/api/session/<id>/start` -> `/api/session/<id>/status` completed successfully with mocked bridge calls (`route-smoke-ok`).
- [2026-04-14] changed: standardized AuralMind2 local MCP startup on `stdio` by switching `.env`, `server.py`, `fastmcp.json`, `README.md`, `render.yaml`, and the VS Code user `mcp.json` entry; hosted HTTP remains opt-in through `ACTIVE_TRANSPORT=streamable-http`.
- [2026-04-13] completed: rendered a single alternate minimalist premium trap no-stem master for `Trapstar` into `masters/trapstar_alt_minimal_premium_trap_20260413/` with a hotter final true-peak request of `-0.23 dBTP`.
- [2026-04-13] completed: rendered a single `Trapstar` minimalistic hi-fidelity no-stem master into `masters/trapstar_minimal_hifi_20260413/` using a restrained custom `hi_fi_streaming`-based profile.
- [2026-04-13] completed: rendered a second `Trapstar` pass with more explicit custom trap-enhancement settings and exported the raw artifacts into `masters/trapstar_custom_trap_enhanced_20260413/`.
- [2026-04-13] completed: delivered 2 custom trap-focused finals named `Sub Anchor Punch` and `Wide Hook Lift`, both no-stem and both finished with a local two-pass loudness/TP pass.
- [2026-04-13] completed: ran an MCP-native two-version no-stem trap mastering workflow on `data/Trapstar.wav` using live AuralMind2 resources, analysis, async render jobs, and two explicit expert sub-agent workstreams.
- [2026-04-13] completed: mapped async jobs to artifacts (`Version A -> art_9c9fd5367304`, `Version B -> art_1f7c3659981a`) and exported both raw mastered WAVs into `masters/trapstar_premium_no_stem_20260413/`.
- [2026-04-13] completed: finished both `Trapstar` versions with a local two-pass `ffmpeg loudnorm` delivery pass after AuralMind2 shaping so the final files land close to the requested `-13.8 LUFS / -0.75 dBTP` spec.
- [2026-04-10] completed: inspected the render/QC path for the next batch and confirmed the best export flow is `tools/auralmind_maestro.py` as the primary engine, writing one 32-bit float master first and deriving the 24-bit WAV from that exact render.
- [2026-04-10] discovered: `server.py` MCP normalization only allows `bit_depth` values `float32` and `float64`, so the server job path alone cannot satisfy a hard 24-bit deliverable requirement.
- [2026-04-10] defined: QC gate should verify LUFS/TP on both 32-bit and 24-bit outputs, confirm low-band mono behavior with sub-band correlation or side-energy checks, and reject `_compat.wav` sidecars as deliverables.
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
- [2026-04-08] completed: created `tools/render_new_project_premium_masters.py` to render 4 custom masters for `New Project (8).wav` and `New Project (9).wav` with exactly 2 stems-on variants, 2 no-stems variants, and `movement_amount=0.23` locked across all renders.
- [2026-04-08] verified: Desktop delivery folder `C:\Users\goku\Desktop\AuralMind2 Premium Masters\New Project 8-9` now contains 4 mastered WAVs plus `render_manifest.json`.
- [2026-04-08] completed: preserved the raw AuralMind-only outputs in `Desktop\\AuralMind2 Premium Masters\\New Project 8-9\\raw_auralmind` and applied a final two-pass `ffmpeg loudnorm` finish to tighten loudness/true-peak delivery.
- [2026-04-08] completed: stopped no-longer-needed AuralMind `server.py` processes after render completion; editor language-server and app-server processes were left alone.
- [2026-04-08] discovered: requested expert sub-agent spawning was blocked by an external model-routing bug that kept forcing unsupported Gemini targets, so the same strategist/execution workstreams were completed locally instead.

## Pain Points / Bugs / Risks
- [2026-04-14] issue: the AI-integrated batch rendered conservatively loud for both newest-song runs, with outputs landing roughly in the `-17` to `-16.6 LUFS` zone rather than the hotter commercial lane Christopher often prefers.
- impact: the delivered versions are valid and unclipped, but they will likely sound quieter than prior premium-delivery folders that received a local final loudness pass.
- suspected cause: raw AuralMind2 async job outputs were exported directly without the repo's optional post-render `ffmpeg loudnorm` finish.
- next step: if Christopher wants these 8 versions competitively hotter, rerun only the selected keepers through a final-loudnorm finishing stage rather than remastering the whole batch first.
- [2026-04-02] issue: `./data` includes prior AuralMind renders, reports, compat files, and likely mastered derivatives mixed with raw source versions.
- impact: grouping can accidentally over-select derived masters instead of source candidates if normalization is too aggressive.
- suspected cause: the data folder is being used as both a source pool and an output scratch space.
- next step: classify files by provenance signals and preserve grouping confidence plus exclusions in manifests.
- [2026-04-03] issue: 14 of the 28 exported masters did not clear the stricter QA pass gate even though all 28 rendered successfully.
- impact: delivery is complete, but several files should be reviewed before being treated as final-approved masters.
- suspected cause: no-stems jobs on certain trap mixes widened or decorrelated the image more than desired, while a few melodic tracks stayed comparatively conservative in loudness.
- next step: rerun only the flagged tracks with reduced spatial width, slightly hotter target LUFS where musically safe, and possibly `radio_loud` or `competitive_trap` swaps depending on the source.
- [2026-04-08] issue: Codex sub-agent spawning ignored explicit `gpt-5.4` requests and still attempted unsupported `gemini-1.5-pro` routing.
- impact: the requested 2-expert-sub-agent pattern could not be executed through the platform runtime even though the task itself was completed.
- suspected cause: external model-routing bug in the sub-agent runtime, not repo code.
- next step: treat sub-agent orchestration as unavailable until the platform stops overriding model selection.

## Metrics / Quality Signals
- [2026-04-14] metric: `Close to the edge` AI-integrated 4-version set
- value: `cinematic` versions measured `-17.97 LUFS / -1.0027 dBTP / 14.05 dB crest / 0.845 corr`; `hi_fi_streaming` versions measured `-17.07 LUFS / -1.0033 dBTP / 13.33 dB crest / 0.841 corr`
- meaning: the preset choice changed the result, but stems-on/off did not materially change the exported metrics for this source in the current engine path
- trend/notes: both stem/no-stem pairs landed identically by metrics within each preset family, so the audible difference may be negligible unless Christopher hears a clear separation by ear
- [2026-04-14] metric: `New Project (15)` AI-integrated 4-version set
- value: `competitive_trap` stems `-16.59 LUFS`, `club_clean` stems `-16.62 LUFS`, `radio_loud` no-stems `-16.59 LUFS`, `hi_fi_streaming` no-stems `-17.31 LUFS`, all around `-1.0 dBTP`
- meaning: the source took all four variants cleanly, with the `hi_fi_streaming` no-stem version staying the most open/quiet and the other three clustering into a denser but still conservative loudness band
- trend/notes: stereo correlation opened from the source `0.872` down to roughly `0.796` to `0.825`, which is still usable but worth checking in mono on the more aggressive variants
- [2026-04-13] metric: `Trapstar` alternate minimal premium trap final
- value: final delivered WAV measured `-14.19 LUFS / -0.23 dBTP / LRA 10.30`
- meaning: the hotter true-peak ceiling was achieved exactly, while loudness landed slightly under target to preserve the minimal premium character and avoid over-limiting
- trend/notes: this pass is riskier for translation than the `-0.75 dBTP` deliveries, so it should be auditioned on bright playback chains before being treated as the default master
- [2026-04-13] metric: `Trapstar` minimal hi-fi final
- value: final delivered WAV measured `-13.94 LUFS / -0.75 dBTP / LRA 10.50`
- meaning: the minimal hi-fi pass stayed slightly under the target loudness while preserving a cleaner, more open, less trap-hyped envelope than the custom trap-enhanced pair
- trend/notes: raw AuralMind2 render measured `-17.93 LUFS / -1.00 dBTP` by the MCP analyzer before the local delivery finish
- [2026-04-13] metric: `Trapstar` custom trap-enhanced finals
- value: `Sub Anchor Punch` measured `-13.96 LUFS / -0.75 dBTP / LRA 10.10`; `Wide Hook Lift` measured `-13.95 LUFS / -0.75 dBTP / LRA 10.10`
- meaning: the more trap-forward custom pass kept the same controlled delivery window while shifting the internal AuralMind2 shaping toward stronger low-end focus, stronger transient emphasis, and a clearer punch-vs-width split
- trend/notes: this pass stayed close to the target without chasing extra loudness; both custom versions remained slightly under target rather than risking brittle overshoot
- [2026-04-13] metric: `Trapstar` premium no-stem finals
- value: both final delivered WAVs measured `input_i -13.96 LUFS / input_tp -0.75 dBTP / input_lra 10.10` by the final `ffmpeg loudnorm` verification pass
- meaning: the final delivery is slightly under the requested loudness target but inside a safe premium trap window with the requested true-peak ceiling held exactly
- trend/notes: raw AuralMind2 renders measured around `-11.7` to `-11.8 LUFS` by BS.1770 `ffmpeg` measurement even though AuralMind2's internal `analyze_audio` reported around `-17 LUFS`, so final delivery decisions should continue to trust the local loudness verifier
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
- [2026-04-08] metric: `New Project (8)` final premium pair
- value: `Trap Spine` final measured `input_i -10.86 / input_tp -1.00`; `Radio Gloss` final measured `input_i -11.05 / input_tp -1.00`
- meaning: both versions landed in a strong commercial trap zone after the final loudness pass while preserving separate stems/no-stems identities.
- trend/notes: AuralMind shaping alone landed around `-16` to `-17 LUFS`; the final `ffmpeg loudnorm` pass was the decisive delivery finish.
- [2026-04-08] metric: `New Project (9)` final premium pair
- value: `Dark Luxe` final measured `input_i -10.79 / input_tp -1.00`; `Night Cinema` final measured `input_i -11.39 / input_tp -1.00`
- meaning: the darker pair now translates as a premium stems/no-stems contrast without the underpowered loudness issue from the raw AuralMind render.
- trend/notes: `Night Cinema` was the weakest raw render (`~ -18.8 LUFS`) and benefited the most from the final delivery pass.

## AI Self Notes
- [2026-04-14] For newest-song auto-selection in `data/`, source filtering should exclude obvious derived markers like `AuralMind`, `_compat`, `_probe`, `mastered`, `analysis_master`, and double-underscore render names before sorting by modified time.
- [2026-04-14] The AI-integrated batch runner currently interprets “4 versions per song” as 2 stems-on variants plus 2 no-stems variants; for this run that produced 8 total WAVs across 2 songs.
- [2026-04-14] The connected UI path is now in-process: `run_ui.py` -> `mastering_ui.py` -> `MasteringUIBridge` -> direct `server.py` tool calls with the Flask `session_id` reused as the FastMCP session key.
- [2026-04-14] UI uploads must stay under `data/ui_uploads`; do not reintroduce arbitrary absolute path support in session creation because it breaks the repo-local safety boundary.
- [2026-04-14] For local VS Code or desktop MCP use, prefer the repo venv plus `ACTIVE_TRANSPORT=stdio`; only flip to `streamable-http` for hosted clients or manual HTTP testing.
- [2026-04-13] When Christopher requests a hotter ceiling like `-0.23 dBTP`, hold brightness and width tighter than normal; the safest move is to let LUFS land slightly under target rather than forcing density into the limiter.
- [2026-04-13] For a minimal hi-fi variant on this source, the cleanest working profile was `hi_fi_streaming` with low movement (`0.19`), low width (`0.10`), mild brightness (`0.02`), moderate low-end focus (`0.42`), light transient boost (`1.4`), no air motion, and strong harshness control.
- [2026-04-13] For trap-specific contrast on the same source, a strong split is: Version A with higher `low_end_focus` and tighter width for direct punch, Version B with slightly reduced low-end weight but increased `spatial_width` and `movement_amount` for hook spread.
- [2026-04-13] For AuralMind2 one-off premium deliveries, `job_result` is the cleanest way to map async render jobs back to mastered-audio artifact IDs before exporting binaries into `masters/`.
- [2026-04-13] `Trapstar` used `competitive_trap` as the shared preset anchor for both premium no-stem variants, with the version split driven mainly by control-profile width / brightness / movement choices rather than by preset swapping.
- [2026-04-10] For exact `-14 LUFS` / `-0.75 dBTP` plus dual `32-bit + 24-bit` delivery, do not rely on `server.run_master_job` alone; use `auralmind_maestro.master(...)` for the primary render, then derive the 24-bit file from the same finished waveform and QC both exports separately.
- [2026-04-02] Keep all source audio untouched; use manifests, staged copies, or links only if needed.
- [2026-04-02] `movement_amount` must default to `0.32` for every delivered master unless a hard quality gate forces a documented reduction.
- [2026-04-02] First half of eligible groups must run no-stems; second half should attempt stems and clearly log any fallback.
- [2026-04-02] `config://server-info` confirms the live register roots are `data/` and `Downloads`, max upload bytes are 419,430,400, and the accepted stem modes are `off`, `auto`, and `on`.
- [2026-04-02] The local CLI and task runners already support deterministic `movement_amount`, stem toggles, and summary JSON copying, so `master_audio` is the cleanest fallback if async jobs stall.
- [2026-04-03] `semantic_a_b_mastering` and `master_closed_loop` were inspected and attempted during discovery, but they were not used for the full batch because the tool-layer timeout was too slow for a 28-track run.
- [2026-04-03] The direct batch runner copies mastered WAVs from the session artifact store instead of `read_artifact`, which is materially faster for large outputs.
- [2026-04-03] For exact final loudness / true-peak delivery on creative one-offs, AuralMind shaping plus a local two-pass `ffmpeg loudnorm` finish is a strong hybrid path.
- [2026-04-08] For short premium-delivery jobs, keep the top-level Desktop folder clean with only the final WAVs, and move the raw AuralMind outputs into a sibling `raw_auralmind` backup folder.

## Christopher ↔ AI Notes
### Notes from Christopher
- [2026-04-02] Requested a 3-sub-agent execution-first run inside AuralMind2 to inventory `./data`, group same-title versions, pick the 2 newest per group, and master them into `./Ignorance Is Bliss` using AuralMind2 MCP as the primary execution path.

### Questions for Christopher
- [2026-04-02] None yet; proceed best-effort unless a destructive ambiguity or hard blocker appears.

### Tips / Suggestions
- [2026-04-02] Because `./data` mixes source and previously rendered material, the highest-value repo improvement after this run may be separating raw source pools from render outputs.
- [2026-04-03] For the next mastering pass, consider promoting the batch runner into a first-class repo utility with explicit QA rerun support so failed masters can be reprocessed automatically instead of manually curated.

## Next Best Actions
- [2026-04-21] priority 1: if Christopher wants the async workflow fully restart-safe, decide whether to keep the SQLite-backed job cache and make it the source of truth or delete the unused persistence scaffolding entirely.
- [2026-04-21] priority 2: if a follow-up pass is worthwhile, cache Demucs stem loading and consider an explicit phase/groove analysis surface only if it fits cleanly into the existing engine helpers.
- [2026-04-14] priority 1: listen to the 8 files in `masters/ai_integrated_latest_two_20260414` and pick one keeper per song before doing any loudness finishing.
- [2026-04-14] priority 2: if Christopher wants a hotter commercial delivery, apply the repo's final-loudnorm finish only to the chosen keepers instead of all 8 variants.
- [2026-04-14] priority 3: if `Close to the edge` stems/no-stems pairs sound identical by ear too, drop the redundant variant family on future runs and save render time.
- [2026-04-14] priority 1: run `python run_ui.py` and do a browser-level manual check against a real source file to validate the connected dashboard end-to-end with the live mastering engine, not just mocked bridge calls.
- [2026-04-14] priority 2: if the Master Tier page stays long-term, decide whether its “NextGen” workflow should remain a staged visualization over the standard async job or be upgraded to a real multi-pass chain endpoint.
- [2026-04-14] priority 3: if export routing needs to target user-selected folders later, add an explicit server-side export destination contract instead of relying on browser folder-picker placeholders.
- [2026-04-13] priority 1: compare the `premium_no_stem` and `custom_trap_enhanced` `Trapstar` folders by ear and decide whether the record wants the safer premium pair or the more trap-forward custom pair.
- [2026-04-13] priority 2: if Christopher wants an even harder trap cut, the next revision should increase movement on Version B only and leave Version A as the tighter translation-safe option.
- [2026-04-13] priority 1: do a human listening pass on the two `Trapstar` finals in `masters/trapstar_premium_no_stem_20260413/` to choose the keeper or request a brighter/darker/wider revision.
- [2026-04-13] priority 2: if Christopher wants repeatability, fold the async-artifact export plus two-pass finish logic into a reusable `Trap` one-off render helper instead of repeating it manually.
- [2026-04-13] priority 3: if an exact integrated target is non-negotiable on future one-offs, iterate the local finish pass against the verified BS.1770 measurement rather than the MCP analyzer's LUFS number.
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
- [2026-04-08] priority 1: do a listening pass on the 4 Desktop masters and decide if any version should be brighter, darker, wider, or more aggressive.
- [2026-04-08] priority 2: if Christopher wants this exact premium-delivery flow reusable, add a `--final-loudnorm` option to `tools/render_new_project_premium_masters.py` instead of relying on a one-off finish pass.

## 2026-04-24 MCP-Only Phase-Aligned Trap Batch
- completed: Rendered the explicit 10-song premium hi-fi trap queue through AuralMind2 MCP only, including server-side `premium_phase_align` for every final master.
- output: `masters/mcp_premium_hifi_trap_explicit_current/final` contains 10 phase-aligned WAV exports; `manifest.json` reports `done=10 error=0`.
- repo state: pushed through commit `81dd067 fix: use valid premium trap prompt intensity`; full test suite passed with `23 passed` and Git was clean afterward.

## 2026-04-24 FaceTime Same-Lane MCP Master
- completed: Rendered `data/FaceTime (6).wav` through the same MCP-only premium hi-fi trap lane: `competitive_trap`, no stems, target `-12.2 LUFS`, and server-side `premium_phase_align`.
- output: `masters/mcp_premium_hifi_trap_facetime_6_current` contains the MCP phase-aligned master plus 24-bit PCM and 32-bit float delivery WAVs.
- metrics: final master measured `-15.01 LUFS`, `-0.97 dBTP`, and low-band phase correlation improved `0.9972 -> 0.9987`.
- changed: `tools/run_explicit_premium_hifi_trap_batch.py` now supports `--source` for one-off renders and `--delivery-formats 24,32` for delivery encodes from the MCP artifact.
- verified: `python -m pytest -q` passed with `23 passed`; delivery probes confirmed 24-bit PCM and 32-bit float stereo WAVs at 48 kHz.
