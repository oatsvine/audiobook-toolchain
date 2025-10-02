# Audiobook Toolchain

*File-first audiobook synthesis workflow anchored in formal acting literature (Normalize → Cue → Synthesize).*

## TL;DR

Instead of orchestrating an opaque graph, we expose a **deterministic, file-first toolchain**: a **sequence of explicit CLI stages** that read/write **self-contained data directories by convention**. Humans can pause between stages, inspect/change files, and resume. Each stage:

* **Declares preconditions** (required input dirs/files),
* **Split and Normalize** ebook into chunks and normalize (strip non-performable paratext, adapt to spoken word),
* **Generate Cue Scripts** from speaker Laban profile, rhetoric tag, emphasis,
* **Manual Editing** edit stage outputs as normal text files,
* **Synthesize** tuning tables convert cues to precise `chatterbox-tts` for engaging delivery.

This document specifies the general framework and then grounds it in the audiobook workflow:

```
parts/ → normalize/ → cues/ → audio/  (→ finalize/)
```

---

## Core Concepts

### Workspace

A **workspace** is a folder named after the source text (book or input file). Every stage reads/writes subdirectories under this workspace:

```
<workspace>/
  <EBOOK_FILE>    # original inputs (moved/copied here)
  manifest.json   # optional manifest (indexed view over parts & artifacts)
  parts/          # manageable slices of source
  normalize/      # cleaned text + NormalizedOutput JSON
  cues/           # cued chunks + CuedScript JSON
  audio/          # per-chunk WAV + JSON sidecars
```

## Stage Contracts

### A. `parts/` (preparation)

* **Inputs:** `book/<file>.epub` **or** `text/<file>.txt`.
* **Process:** partition into token-bounded parts (or accept a single part).
* **Outputs:** `parts/<name>-partNNN.txt` + `<name>-partNNN.json` (PartMeta).
* **Human step:** freely edit any `parts/*.txt` before normalization.

### B. `normalize/`

* **Inputs:** `parts/…txt` + `parts/…json` (PartMeta).
* **Process:** run LLM with **structured outputs** → `NormalizedOutput`.
* **Outputs:** text: `…-normalized.txt`; JSON: `…-normalized.json` without the large `cleaned_text` field (kept in `.txt` for review).
* **Policy:** **fail** if `normalize/` exists (prevent silent clobber); allow `--force`.

### C. `cues/`

* **Inputs:** `normalize/*.json` + `normalize/*.txt`.
* **Process:** run LLM with **structured outputs** → `CuedScript`.
* **Outputs:** `…-cues.json` + `…-review.txt` (human-readable summary).
* **Checks:** coverage heuristic (total chunk text vs normalized length).

### D. `audio/` (synthesize)

* **Inputs:** `cues/*.json` + voice assets in `voices/`.
* **Process:** TTS per chunk; write per-chunk WAV + sidecar JSON.
* **Outputs:** `audio/*.wav` + `audio/*.json`.

### E. `finalize/` (optional, reserved)

* **Inputs:** `audio/*.wav`.
* **Process:** concatenate/mix/ducking/etc. (future).
* **Outputs:** final deliverables (e.g., a chapter WAV).

---

## Execution Examples

```bash
# 0) Initialize a workspace from an EPUB
python -m audiobook_toolchain.book --book_file ~/in/GospelOfThomas.epub --chunk_tokens 3500
# -> creates: /data/workspace/tts/GospelOfThomas/{GospelOfThomas.epub,manifest.json,parts}

# 1) Human edits the parts/*.txt

# 2) Normalize (fails if normalize/ already exists unless --force)
python -m audiobook_toolchain.normalize /data/workspace/tts/GospelOfThomas

# 3) Cue from normalized outputs
python -m audiobook_toolchain.cue /data/workspace/tts/GospelOfThomas

# 4) Synthesize audio with available voices
python -m audiobook_toolchain.synthesize /data/workspace/tts/GospelOfThomas --prepare_conditionals=true

# (Optional) finalize
python -m audiobook_toolchain.finalize /data/workspace/tts/GospelOfThomas

# ---- OR - run all at once:
python -m audiobook_toolchain.book --book_file ~/in/GospelOfThomas.epub --auto
```

**Partial re-runs:**
Fix `parts/<name>-part007.txt` → `normalize` (will regenerate `normalize/*part007*`) → `cue` → `synthesize`. Other parts remain untouched.

---

## Voice Assets

* Convention: `voices/` directory adjacent to workspaces **or** a global `voices_dir`.
* `speaker → voice_path` resolution:

  * Prefer exact `speaker` name match.
  * Fallback to `default.wav`.
* Expose `--voices_dir` and allow **per-chunk override** (e.g., `params.audio_prompt_path` in `CuedScript`).

