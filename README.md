# Audiobook Toolchain

*File-first audiobook synthesize workflow (Normalize → Cue → Synthesize).*

## TL;DR

Instead of orchestrating an opaque graph, we expose a **deterministic, file-first toolchain**: a **sequence of explicit CLI stages** that read/write **self-contained data directories by convention**. Humans can pause between stages, inspect/change files, and resume. Each stage:

* **Declares preconditions** (required input dirs/files),
* **Produces versioned, typed artifacts** (Pydantic-shaped JSON + companion text/audio),
* **Fails fast** on unmet dependencies or risk of clobbering,
* **Is idempotent** when outputs are identical, and explicit about overwrite policy otherwise.

This document specifies the general framework and then grounds it in the audiobook workflow:

```
parts/ → normalize/ → cues/ → audio/  (→ finalize/)
```

---

## Core Concepts

### 1) Stage

A **Stage** is a command with:

* **Inputs (preconditions):** directories and files it expects.
* **Outputs (postconditions):** directories and files it produces.
* **Contract:** Pydantic v2 models define the JSON sidecar schemas.
* **Policy:** idempotence, overwrite behavior, and logging.

### 2) Workspace

A **workspace** is a folder named after the source text (book or input file). Every stage reads/writes subdirectories under this workspace:

```
<workspace>/
  <EBOOK_FILE>    # original inputs (moved/copied here)
  manifest.json   # optional manifest (indexed view over parts & artifacts)
  parts/          # maneagable slices of source
  normalize/      # cleaned text + NormalizedOutput JSON
  cues/           # cued chunks + CuedScript JSON
  audio/          # per-chunk WAV + JSON sidecars
```

### 3) Artifacts: “human text + machine JSON”

Each stage emits **human-readable artifacts** (e.g., `.txt`, `.md`) **and** **typed sidecars** (`.json`) that encode the structured state (Pydantic models).

**Overwrite policy:**

* `parts/` is **human-owned**.
* `normalize/`, `cues/`, `audio/` are **stage-owned**. Stages **refuse to run** if their output dir exists **unless** `--force` is passed (or you `tool clean`).

---

## Stage Contracts (General Pattern)

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

## Models

> **Rule:** All structured domain data is a Pydantic model. No raw dicts.

### Normalization output (example only; defined in `tts.normalize`)

* `NormalizedOutput`: cleaned text + heuristics (`input_chars`, `output_chars`, …) + classification (`category: TextType`) + `speakers: list[SpeakerCandidate]`.
* **Persist policy:** write `cleaned_text` as a `.txt` and write the **rest** to `.json` to keep the JSON small/diff-friendly.

### Cues output (example only; defined in `tts.cues`)

* `CuedScript`: `chunks: list[Chunk]` with per-chunk `speaker`, `text`, `params` (TTS engine knobs), `pre_pause_ms`, `post_pause_ms`, etc.
* Provide `CuedScript.format()` to render `review.txt`.

### (Optional) Workspace manifest

A read-optimized index (for UIs/ops):

```python
class PartIndex(BaseModel):
    part: int
    part_path: Path
    normalized_json: Optional[Path] = None
    cues_json: Optional[Path] = None
    audio_files: list[Path] = []

class Manifest(BaseModel):
    text_name: str
    workspace: Path
    parts: list[PartIndex]
```

---

## Fire CLI Surface (Generalized)

Expose **one class per toolchain**. Constructor receives **config**; methods are **small, typed** steps.

```python
import fire
from pathlib import Path
from loguru import logger

class Toolchain:
    """
    Deterministic, file-first pipeline.
    Subcommands are independent stages; run them in order.
    """

    def __init__(
        self,
        in_dir: str | Path = Path("/data/workspace/in"),
        voices_dir: str | Path = Path("/data/workspace/voices"),
        workspace_dir: str | Path = Path("/data/workspace/tts"),
        debug: bool = False,
        force: bool = False,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.in_dir = Path(in_dir)
        self.voices_dir = Path(voices_dir)
        self.workspace_dir = Path(workspace_dir)
        self.debug = debug
        self.force = force
        # configure logger once here

    # --- Initialization ---
    def book(self, book_file: str = "", chunk_tokens: int = 3500) -> str:
        """Partition a book (EPUB) into parts/ (creates workspace)."""
        ...

    def text(self, text_file: str, part: int = 1) -> str:
        """Register a single text as parts/part{part:03d} (creates workspace)."""
        ...

    # --- Stages ---
    def normalize(self, work_dir: str) -> None:
        """parts/ → normalize/"""
        ...

    def cue(self, work_dir: str) -> None:
        """normalize/ → cues/"""
        ...

    def synthesize(self, work_dir: str, prepare_conditionals: bool = False) -> None:
        """cues/ → audio/"""
        ...

    def finalize(self, work_dir: str) -> None:
        """audio/ → final renders (placeholder)"""
        ...

    # --- Utilities ---
    def status(self, work_dir: str) -> None: ...
    def clean(self, work_dir: str, target: str) -> None: ...
    def tree(self, work_dir: str) -> None: ...
```

**Notes:**

* Each stage **accepts `work_dir`** (the workspace root).
* A **global `--force`** flag enables overwrite of stage outputs (or per-stage `--force` if preferable).
* Utilities like `status`/`tree`/`clean` help ops.

---

## Execution Examples

```bash
# 0) Initialize a workspace from an EPUB
python -m toolchain.book --book_file ~/in/GospelOfThomas.epub --chunk_tokens 3500
# -> creates: /data/workspace/tts/GospelOfThomas/{book,parts}

# 1) Human edits the parts/*.txt

# 2) Normalize (fails if normalize/ already exists unless --force)
python -m toolchain.normalize /data/workspace/tts/GospelOfThomas

# 3) Cue from normalized outputs
python -m toolchain.cue /data/workspace/tts/GospelOfThomas

# 4) Synthesize audio with available voices
python -m toolchain.synthesize /data/workspace/tts/GospelOfThomas --prepare_conditionals=true

# (Optional) finalize
python -m toolchain.finalize /data/workspace/tts/GospelOfThomas
```

**Partial re-runs:**
Fix `parts/<name>-part007.txt` → `normalize` (will regenerate `normalize/*part007*`) → `cue` → `synthesize`. Other parts remain untouched.

---

## Preconditions & Failure Policy

* **Preconditions explicitly asserted** (clear error messages):

  * `parts/` must exist before `normalize/`.
  * `normalize/` must exist before `cues/`, etc.
  * Voice files must exist for requested speakers (or a `default.wav` must be present).
* **Overwrite rules**:

  * If a stage’s output dir **exists**, **fail** unless `--force`.
  * If `--force`, either clear and rewrite the output dir or generate **only missing** outputs (choose and document).
* **Idempotence**:

  * Emitting identical artifacts should be a no-op (e.g., diff then skip) when feasible; otherwise overwrite is explicit.


---

## LLM Usage (Structured Outputs)

* Use **OpenAI Structured Outputs** to **directly produce Pydantic instances**:

  * `NormalizedOutput` for Stage *normalize*.
  * `CuedScript` for Stage *cue*.
* Keep prompts **short**; rely on **field docs** for guidance.
* Serialize with `.model_dump_json(...)` and avoid embedding large text blobs (e.g., `cleaned_text`) in JSON; put those in `.txt` for diff-friendly review.

---

## Voice Assets

* Convention: `voices/` directory adjacent to workspaces **or** a global `voices_dir`.
* `speaker → voice_path` resolution:

  * Prefer exact `speaker` name match.
  * Fallback to `default.wav`.
* Expose `--voices_dir` and allow **per-chunk override** (e.g., `params.audio_prompt_path` in `CuedScript`).

---

## Guardrails & Principles (Org-wide)

* **Pydantic v2 maximalism:** all structured data are models; `.model_dump()`/`.model_dump_json()`.
* **No fallback:** for instance, if `sk` is used in starting point, `sk` is always available, period. NEVER implement fallbacks unless explicitely done by the user. 
* **Fire CLI (class-based):** one class, small typed methods, explicit I/O.
* **Loguru once:** brace formatting; log inputs/outputs, decisions, and heuristics.
* **Strict typing:** Python ≥ 3.11, Pyright strict.
* **Pathlib only:** no raw string paths.
* **Over-document:** public APIs with docstrings, code comments where logic is non-obvious, and reasoning notes for trade-offs.

## Why a “toolchain” (not a graph)?

Creative pipelines benefit from **manual review and iteration**. Graph schedulers are excellent for distributed automation, but can hide state and blur boundaries. A toolchain instead:

* **Surfaces boundaries** as directories and files humans see and can edit.
* **Encodes contracts** via on-disk schemas, not in-memory nodes.
* **Improves debuggability**: every step has a working folder and sidecars.
* **Allows focused re-runs**: e.g., fix `parts/part003.txt` then run only `normalize → cue` for that part.

You can still automate end-to-end (e.g., a `make`/`just`/CI job), but **the primary interface is human-driven, stepwise, and file-centric**.

