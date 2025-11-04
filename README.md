# Laban TTS

Deterministic, file-first audiobook production pipeline: **normalize → cue → synthesize**. Every stage is an explicit CLI command that reads and writes well-defined artifacts so engineers, directors, and narrators can audit, edit, and rerun any step without hidden state.

```
parts/ → normalize/ → cues/ → audio/ (→ finalize/)
```

## Requirements

- Python 3.11 (ships with the provided PyTorch CUDA image).
- `uv` package manager (already available in the container).
- OpenAI project/API key exported as `OPENAI_API_KEY` (used for normalization + cueing via `gpt-5-mini-2025-08-07`).
- ffmpeg in the system path (needed by `pydub` to mux WAVs).
- Optional GPU (8 GiB VRAM works). The toolchain auto-falls back to CPU if chatterbox-tts cannot load on CUDA.

## Installation

```bash
# From the repository root
uv pip install -e .

# (Optional) confirm Torch + CUDA availability
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
PY
```

Project depends on `torchaudio`; install TorchCodec if your environment lacks the built-in encoder support.

## Workspace Layout

Workspaces live under `$WORKSPACE_DIR/tts/<text_name>/` (defaults to `/data/workspace`). Running the pipeline on `republic.txt` produces:

```
/data/workspace/tts/republic/
  manifest.json
  parts/       # text fragments emitted by partition_text()
  normalize/   # NormalizedPart XML (one per part)
  cues/        # CuedScript XML
  audio/       # <text>_pNNN_####_<speaker>.wav + .json metadata
```

Voice references are resolved from `$WORKSPACE_DIR/voices/` (e.g., `voices/enoch.wav`). Provide `speaker:voice` pairs through the CLI when synthesizing.

## Quickstart

1. Place your EPUB, PDF, or plain-text source in `/data/workspace/in/`.
2. Export your OpenAI key: `export OPENAI_API_KEY=...`.
3. Run the toolchain end-to-end:

   ```bash
   python -m laban_tts.workflow run \
       --text_file /data/workspace/in/republic.txt \
       --auto \
       --voice_files default:enoch \
       --prepare_conditionals=true
   ```

4. Inspect artifacts between stages (XML, JSON, WAV). Re-run individual stages with `--force` if you edit upstream files.

## Stage Reference

| Stage | Command | Inputs | Outputs | Notes |
| --- | --- | --- | --- | --- |
| `parts` | `run` (without `--auto`) | Source text | `parts/*.xml` | Partitioned slices sized for LLM context. |
| `normalize` | `normalize <workspace>` | `parts/*.xml` | `normalize/*-normalized.xml` | Cleans text, classifies speakers + discourse; skips existing files unless `--force`. |
| `cue` | `cue <workspace>` | `normalize/*.xml` | `cues/*-cues.xml` | LLM produces chunk, rhetoric, profile, emphasis metadata; reuses existing scripts unless `--force`. |
| `synthesize` | `synthesize <workspace>` | `cues/*.xml`, voice WAVs | `audio/*.wav` + `.json` | Uses chatterbox-tts; reuses existing audio unless `--force`; handles GPU OOM with CPU fallback. |
| `finalize` | `finalize <workspace>` | `audio/*.wav` | Logs inventory | Placeholder for later concatenation/mixing. |

Invoke stages individually:

```bash
python -m laban_tts.workflow normalize /data/workspace/tts/republic
python -m laban_tts.workflow cue /data/workspace/tts/republic
python -m laban_tts.workflow synthesize /data/workspace/tts/republic --voice_files default:enoch
```

## Regenerating After Edits

Need to tweak a section?

1. Edit `parts/republic-part007.xml` directly.
2. Re-run `normalize` (only that part is regenerated).
3. Re-run `cue` and `synthesize`; downstream files for other parts stay untouched.

Use `--force` to allow stages to overwrite existing directories.

## Voices & Prompt Conditioning

- Supply a comma-separated list: `default:enoch,glaucon:enoch`. Speaker identifiers are lowercase.
- Per-chunk overrides in `CuedScript` (`audio-prompt` attribute) take precedence over CLI flags.
- `--prepare_conditionals` primes chatterbox-tts with the default voice when supported.

## Testing & QA

Project policy requires:

```bash
black --check .
pyright
pytest -q
```

Integration smoke test (replace the OpenAI key with your own):

```bash
export OPENAI_API_KEY=...
python -m laban_tts.workflow run \
    --text_file /data/workspace/in/republic.txt \
    --auto \
    --voice_files default:enoch
```

This yields 59 WAV chunks for the sample Republic excerpt and records chunk timing metadata alongside the audio.

## Troubleshooting

- **CUDA out of memory:** the loader automatically retries on CPU (`tts.load_cuda_failed` warning). Expect longer synthesis times but stable output.
- **Missing voice files:** ensure `<voice_name>.wav` exists under `$WORKSPACE_DIR/voices/`. Set `--voice_files default:<voice>` to avoid per-speaker resolution errors.
- **ffmpeg errors:** install binary utilities (`apt-get install ffmpeg`) so `pydub` can decode WAV buffers.
- **LLM quota/timeouts:** the pipeline uses structured outputs; failing calls abort the stage. Re-run once quotas recover.

For deeper context, the project scratchpad (`scratchpad.md`) logs the latest pipeline run, token usage, and validation notes.
