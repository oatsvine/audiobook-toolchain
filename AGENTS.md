## 0. Canonical comments conventions

* You must **never** author `TODO`, `NOTE`, or `IGNORE`. These are **authoritative commands to you**.
    You may write only `# REVIEW:` or `# REJECT:` — these are **from you to the user** and must include a brief justification.
* `# TODO:` — Mandatory instruction. Resolve **exactly** as written, then **remove the comment**.
    If contradictory or non-actionable, replace with `# REJECT:` and state the contradiction.
    If ambiguous but actionable, apply the **smallest compliant change**, then add `# REVIEW:` summarizing what changed and why.
* `# NOTE:` — Authoritative context from the user. Keep it. Unless stated otherwise, it applies to the **pattern**, not just this occurrence.
* `# PATTERN:` — Like `NOTE` but always denotes a generalizable non-negotialble pattern in the statement below. Never override or change. When you notice conflicting patterns occurences, fix them immidiately.  
* `# IGNORE:` — Explicit exception to a rule here. When unsure, assume the **narrowest scope** (line/file over module/project).

## 1. Runtime Environment
- Agent executes **inside a PyTorch Docker image** (CUDA-enabled, preinstalled `torch`).
- **Never reinstall or upgrade Torch/CUDA**; query their versions to confirm availability:
  - `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()`
  - `nvidia-smi` should return driver information.
- The runtime GPU memory target is **12 GB VRAM** (you have 8 GB VRAM in your runtime environment). Training and inference must remain within that constraint.
- The agent assumes Linux paths, UTF-8 locale, and POSIX shell semantics.

## 2. Project Context
- Repository: `laban-tts/`
- Dependencies are managed via **uv**; the project is self-contained (use `uv pip install -e .`).
- All code must run successfully under existing Torch runtime and verify CUDA before proceeding.
- Your python 3.11 environment is preinstalled in your pytorch container. After having installed with `uv pip install -e .`, use `python -m`. Do NOT attempt do create virtual environments.

## 3. Code Conventions
- **Fail fast.** Use `assert` for invariants; no layered error handling.
- **Minimalism first.** Fewer lines > abstractions; no helpers or wrappers unless unavoidable.
- **Deterministic structure.** Always read/write within `$WORKSPACE_DIR` tree; never rely on CWD.
- **Imports.** Keep standard order: stdlib → third-party → local. Group logically with a blank line.
- **Logging.** Use `loguru` for single-line start/finish messages per operation.
- **Typing.** Enforce `pyright --strict`; all public functions must have explicit type hints.
- **Formatting.** `black` with default settings; line length ≤ 88 chars.
- **Testing.** `pytest` used for all red→green scenarios. No mocks; real minimal assets.
- **Configuration.** Only via env vars or minimal `pydantic` models; no YAML/INI.

## 4. Agent Behavior (Codex CLI)
- Treat each feature as an independent research-build-verify loop.
- **Research before code:** inspect upstream repo examples/tests; run one REPL proof.
- Record the source file:line reference as a comment above each derived implementation.
- Keep commands idempotent; if preconditions fail, exit immediately.
- Maintain minimal commit granularity: one atomic feature per commit with its test.
- Do not progress past any stage until its test passes and completion criteria are logged.

## 5. GPU & Quantization Discipline
- Must use real chatterbox-tts model

## 6. Security & Compliance
- Must manually export `OPENAI_API_KEY` before testing.

## 7. Deliverables & Checks
- A complete build is defined by:
  - `pytest -q` all green
  - `pyright` zero errors
  - `black --check .` clean
- Commits must include the REPL proof logs for every new integration.

---

**Codex CLI agents must adhere strictly to these conventions to ensure reproducibility and minimalism within the Torch Docker runtime.**
