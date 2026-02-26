# AuralMind Maestro MCP Usage

This document provides quick examples for MCP resources, prompts, and tools.
All tool calls return the standard envelope:

```json
{ "ok": true, "result": {}, "error": null }
```

## Resources

Available resources:
- `config://system-prompt`
- `config://mcp-docs`
- `config://server-info`

Example read:

```json
{
  "method": "resources/read",
  "params": { "uri": "config://server-info" }
}
```

## Prompt

Prompt name: `generate-mastering-strategy`

Arguments:
- `lufs` (float)
- `crest` (float)
- `platform` (`spotify` | `apple_music` | `youtube` | `soundcloud` | `club`)

Example render:

```json
{
  "method": "prompts/render",
  "params": {
    "name": "generate-mastering-strategy",
    "arguments": {
      "lufs": "-13.2",
      "crest": "9.4",
      "platform": "spotify"
    }
  }
}
```

## Tools

### upload_audio_to_session

```json
{
  "method": "tools/call",
  "params": {
    "name": "upload_audio_to_session",
    "arguments": {
      "filename": "song.wav",
      "payload_b64": "<base64-audio>"
    }
  }
}
```

### analyze_audio

```json
{
  "method": "tools/call",
  "params": {
    "name": "analyze_audio",
    "arguments": { "audio_id": "aud_1a2b3c4d5e6f" }
  }
}
```

### list_presets

```json
{
  "method": "tools/call",
  "params": { "name": "list_presets", "arguments": {} }
}
```

### propose_master_settings

```json
{
  "method": "tools/call",
  "params": {
    "name": "propose_master_settings",
    "arguments": {
      "preset_name": "hi_fi_streaming",
      "target_lufs": -12.0,
      "warmth": 0.45,
      "transient_boost_db": 1.2,
      "enable_harshness_limiter": true,
      "enable_air_motion": true,
      "bit_depth": "float32"
    }
  }
}
```

### run_master_job

```json
{
  "method": "tools/call",
  "params": {
    "name": "run_master_job",
    "arguments": {
      "audio_id": "aud_1a2b3c4d5e6f",
      "preset_name": "hi_fi_streaming",
      "target_lufs": -12.0,
      "warmth": 0.45,
      "transient_boost_db": 1.2,
      "enable_harshness_limiter": true,
      "enable_air_motion": true,
      "bit_depth": "float32"
    }
  }
}
```

### job_status

```json
{
  "method": "tools/call",
  "params": {
    "name": "job_status",
    "arguments": { "job_id": "job_1a2b3c4d5e6f" }
  }
}
```

### job_result

```json
{
  "method": "tools/call",
  "params": {
    "name": "job_result",
    "arguments": { "job_id": "job_1a2b3c4d5e6f" }
  }
}
```

### read_artifact

```json
{
  "method": "tools/call",
  "params": {
    "name": "read_artifact",
    "arguments": {
      "artifact_id": "art_1a2b3c4d5e6f",
      "offset": 0,
      "length": 2097152
    }
  }
}
```
