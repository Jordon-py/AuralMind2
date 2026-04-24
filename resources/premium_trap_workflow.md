# Premium Trap Workflow

Top-level documentation: this resource guides MCP clients through premium trap and rap mastering with AuralMind2. Data shapes include `aud_*` source handles, `art_*` artifacts, `job_*` async jobs, `MasterSettings`, `MasteringControlProfile`, metrics JSON, and rendered audio artifacts. Important functions: `server.py:3775 plan_mastering_strategy`, `server.py:3781 propose_master_settings`, `server.py:3787 run_master_job`, `server.py:3815 job_status`, `server.py:3837 job_result`, and `tools/auralmind_maestro.py:2804 master`. Possible bugs: clients may chase loudness before checking harshness, mono low-end discipline, or vocal presence; stale job IDs can look active if not confirmed through `job_status`. Two extensions: add a dedicated premium-trap QC report model, and expose a delivery-finish tool for exact 24-bit/32-bit release exports.

## Intent

Use this workflow when the user asks for trap, rap, hip-hop, 808-heavy, premium, industry-standard, release-ready, radio-ready, club-ready, or melodic-vocal trap masters.

## Required Client Flow

1. Discover or resume with `bootstrap`, `get_connect_packet`, and `list_session_state`.
2. Read `auralmind://contracts`, `auralmind://control-surface`, and this resource before building payloads.
3. Register or upload the source, then run `analyze_audio`.
4. Use `plan_mastering_strategy` for natural-language goals.
5. Use `propose_master_settings` before execution when the client composes or modifies settings.
6. Use `run_master_job` for normal renders and poll with `job_status`.
7. Fetch `job_result`, then evaluate with `analyze_audio` or `compare_audio_metrics`.
8. Apply one intervention at a time only when the pass is close.

## Premium Trap Defaults

- Preset anchors: `competitive_trap` for 808 punch, `radio_loud` for vocal-forward commercial impact, `club_clean` for cleaner dense club energy, `hi_fi_streaming` for brittle or already-loud sources.
- Platform defaults: `spotify` for streaming masters, `soundcloud` for louder rap uploads, `club` only when the user explicitly wants club density.
- Stem policy: start with `stem_mode="auto"` unless the user requires no-stems or stems-on.
- Control bias: positive `low_end_focus`, moderate `movement_amount`, moderate `harshness_control`, restrained `spatial_width` when bass or mono translation is risky.
- Bit depth: use `float32` for MCP artifacts unless a repo-local delivery runner is handling exact 24-bit/32-bit exports.

## Quality Gates

- Vocals should stay present after low-end enhancement.
- 808/sub energy should feel centered and mono-safe.
- Stereo width should not push correlation into unsafe negative territory.
- Highs should be bright enough for premium sheen without brittle harshness.
- Crest should preserve punch; do not flatten transients just to chase loudness.
- True peak should remain controlled before final delivery decisions.

## Intervention Rules

- Preset uncertainty: use `semantic_a_b_mastering`.
- Almost finished but needs more punch, warmth, or polish: use `start_interactive_mastering` then `commit_interactive_mastering`.
- Loudness versus punch conflict: use `analyze_and_optimize_governor`.
- Buried vocal, bloated bass, or bed-balance complaint: use `ai_stem_remix` and feed justified `stem_gains_db` back into planning.
- Tone or color issue: use one creative DSP tool, re-analyze, then stop or finalize.

## Finalization

Return the winning artifact handle, key metrics, selected preset/settings, and any tradeoff that remains. Do not claim a release-ready file exists until the server returns the artifact or a repo-local delivery runner verifies the exported WAV.
