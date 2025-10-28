# Laban TTS

*File-first audiobook synthesis workflow anchored in formal acting literature (Normalize → Cue → Synthesize).*

## TL;DR

Instead of orchestrating an opaque graph, we expose a **deterministic, file-first toolchain**: a **sequence of explicit CLI stages** that read/write **self-contained data directories by convention**. Humans can pause between stages, inspect/change files, and resume. Each stage:

```
parts/ → normalize/ → cues/ → audio/  (→ finalize/)
```

* **Declares preconditions** (required input dirs/files).
* **Splits and normalizes** the ebook into chunks with the LLM (strip non-performable paratext, adapt to spoken word).
* **Generates cue scripts** with the LLM from the speaker's Laban profile, rhetoric tag, emphasis.
* **Keeps stage outputs editable** as normal text files.
* **Synthesizes audio** with TTS tuning tables that convert cues to precise `chatterbox-tts` delivery.

Tested with `gpt-5-mini`; using `langchain` makes it easy to swap the LLM. Married to `chatterbox-tts` until standard exaggeration knobs emerge.

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
  cues/           # cued chunks + CuedScript XML
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
* **Outputs:** `…-cues.xml` (editable cue script).
* **Checks:** coverage heuristic (total chunk text vs normalized length).

### D. `audio/` (synthesize)

* **Inputs:** `cues/*.xml` + voice assets in `voices/`.
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
python -m laban_tts.book --book_file ~/in/GospelOfThomas.epub --chunk_tokens 3500
# -> creates: /data/workspace/tts/GospelOfThomas/{GospelOfThomas.epub,manifest.json,parts}

# 1) Human edits the parts/*.txt

# 2) Normalize (fails if normalize/ already exists unless --force)
python -m laban_tts.normalize /data/workspace/tts/GospelOfThomas

# 3) Cue from normalized outputs
python -m laban_tts.cue /data/workspace/tts/GospelOfThomas

# 4) Synthesize audio with available voices
python -m laban_tts.synthesize /data/workspace/tts/GospelOfThomas --prepare_conditionals=true

# (Optional) finalize
python -m laban_tts.finalize /data/workspace/tts/GospelOfThomas

# ---- OR - run all at once:
python -m laban_tts.book --book_file ~/in/GospelOfThomas.epub --auto
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
