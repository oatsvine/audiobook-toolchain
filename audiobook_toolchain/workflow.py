"""
Audiobook TTS Toolchain (Normalize → Cue → Synthesize)

This module implements a deterministic, file-first toolchain for audiobook
production. Stages are exposed as explicit CLI commands:

parts/ → normalize/ → cues/ → audio/ (→ finalize/)

Each stage validates its preconditions, produces typed artifacts (Pydantic v2
models), and fails fast on policy violations. Humans can inspect or edit
artifacts between stages and rerun stages with ``--force`` when they need to
regenerate outputs.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
import json
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Sequence, Tuple

import fire
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from audiobook_toolchain.normalize import NORMALIZE_PROMPT, NormalizedOutput  # type: ignore[import]  # noqa: E402
from audiobook_toolchain.cues import CUE_PRIMER, CUE_PROMPT, CuedChunk, CuedScript  # type: ignore[import]  # noqa: E402

import torch
import torchaudio  # type: ignore[import]
from chatterbox.tts import ChatterboxTTS  # type: ignore[import]
from pydub import AudioSegment

tts_model = ChatterboxTTS.from_pretrained(
    device="cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(
    "torchaudio.version={version} backends={backends}",
    version=torchaudio.__version__,
    backends=torchaudio.list_audio_backends(),
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts (concise). We rely on the Pydantic field descriptions for formatting.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic helper models for on-disk artifacts
# ─────────────────────────────────────────────────────────────────────────────


class Manifest(BaseModel):
    text_name: str
    workspace: Path
    kind: Literal["text", "book"] = "book"


class PartMeta(BaseModel):
    """Sidecar describing one text part produced during preparation."""

    model_config = ConfigDict(extra="forbid")

    text_name: str = Field(min_length=1)
    part: int = Field(ge=1)
    source: str = Field(description="Original source path or URL for this part.")
    last_excerpt: Optional[str] = Field(default=None)
    next_excerpt: Optional[str] = Field(default=None)


class PartEntry(BaseModel):
    """In-memory representation of a part ready for normalization."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    meta: PartMeta
    text_path: Path
    meta_path: Path
    text: str


class NormalizedEntry(BaseModel):
    """Part material paired with normalization outputs."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text_name: str
    part: int
    text_path: Path
    json_path: Path
    normalized: NormalizedOutput


class CueEntry(BaseModel):
    """Cue artifacts ready for synthesis."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text_name: str
    part: int
    script: CuedScript
    json_path: Path


class ChunkEntry(CuedChunk):
    duration: float
    position: float
    phrases: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning helpers
# ─────────────────────────────────────────────────────────────────────────────


def partition_text(
    path_or_url: Path | str, chunk_tokens: int = 5000
) -> Generator[Tuple[PartMeta, str], None, None]:
    """Read a local file or URL and yield (metadata, text) pairs for each part."""
    from unstructured.partition.auto import partition  # lazy import to keep CLI snappy

    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        text_name = path_or_url.split("/")[-1].split(".")[0]
        elements = partition(url=path_or_url)
    else:
        path = Path(path_or_url)
        assert path.exists(), f"File {path} does not exist."
        text_name = path.stem
        elements = partition(filename=str(path.resolve()))

    logger.info(
        "Partitioned {count} elements from {source}",
        count=len(elements),
        source=path_or_url,
    )
    text = "\n\n".join([e.text for e in elements if e.text and e.text.strip()])

    # Chunk by token budget for LLM calls
    part_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=chunk_tokens,
        chunk_overlap=0,
    )
    excerpt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    texts = part_splitter.split_text(text)
    logger.info("Split into {count} parts", count=len(texts))
    for idx, content in enumerate(texts, start=1):
        meta = PartMeta(text_name=text_name, part=idx, source=str(path_or_url))
        if idx > 1:
            meta.last_excerpt = "…" + excerpt_splitter.split_text(texts[idx - 2])[-1]
        if idx < len(texts):
            meta.next_excerpt = excerpt_splitter.split_text(texts[idx])[:1][0] + "…"
        yield meta, content


# ─────────────────────────────────────────────────────────────────────────────
# Toolchain implementation
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/data/workspace"))


class Toolchain:
    """Text-to-speech pipeline exposed as sequential CLI stages."""

    def __init__(
        self,
        debug: bool = False,
        in_dir: Path | str = WORKSPACE_DIR / "in",
        voices_dir: Path | str = WORKSPACE_DIR / "voices",
        workspace_dir: Path | str = WORKSPACE_DIR / "tts",
        force: bool = False,
        llm=None,
        model_name: str = "gpt-5-mini",
        language_id: Optional[str] = None,
    ) -> None:
        self.debug = debug
        self.force = force
        self.language_id = language_id
        self.in_dir = Path(in_dir)
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm or ChatOpenAI(
            model=model_name,
            temperature=0,
            max_retries=3,
        )

    # —————————————————— Utilities ——————————————————

    def voices_from_spec(self, voice_specs: Sequence[str]) -> Dict[str, Path]:
        """Map speaker:name specs to voice files in `voices_dir`."""
        voices: Dict[str, Path] = {}
        for spec in voice_specs:
            if not spec:
                continue
            parts = spec.split(":", maxsplit=1)
            assert (
                len(parts) == 2
            ), f"Invalid voice format: {spec}. Expected speaker:voice_name."
            speaker = parts[0].strip().lower()
            voices[speaker] = self.voice_file(parts[1].strip())
        return voices

    def voice_file(self, voice_name: str) -> Path:
        """Resolve a voice WAV path and assert existence."""
        voice_file = self.voices_dir / f"{voice_name}.wav"
        assert voice_file.exists(), f"Voice file {voice_file} does not exist."
        return voice_file

    @staticmethod
    def _choose_voice(
        speaker: str, voices: Dict[str, Path]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (voice_path, voice_name) using speaker match or default."""
        normalized = speaker.lower()
        if normalized in voices:
            voice_path = voices[normalized]
            return str(voice_path), voice_path.stem
        if "default" in voices:
            voice_path = voices["default"]
            return str(voice_path), voice_path.stem
        return None, None

    def choose_book(self) -> Path:
        """Interactive selection of an EPUB from input directory using `sk`."""
        books = list(self.in_dir.rglob("*.epub")) + list(self.in_dir.rglob("*.pdf"))
        if not books:
            raise ValueError("No book files found in the input directory.")
        choices = [str(path.relative_to(self.in_dir)) for path in books]
        result = subprocess.run(
            ["sk"], input="\n".join(choices), text=True, capture_output=True, check=True
        )
        file_path = result.stdout.strip()
        return self.in_dir / file_path

    def _prepare_output_dir(self, path: Path, stage: str) -> None:
        """Ensure an output directory is writable, honoring the `--force` policy."""
        if path.exists():
            if not self.force:
                raise FileExistsError(
                    f"Stage {stage} refuses to overwrite existing directory: {path}"
                )
            logger.warning(
                "Overwriting existing directory for stage {stage}: {path}",
                stage=stage,
                path=path,
            )
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _resolve_workspace(self, work_dir: Path | str) -> Path:
        """Resolve a workspace path, accepting relative names under `workspace_dir`."""
        work_path = Path(work_dir)
        if not work_path.is_absolute():
            work_path = self.workspace_dir / work_path
        if not work_path.exists():
            raise FileNotFoundError(
                f"Workspace {work_path} does not exist. Run `book` or `text` first."
            )
        return work_path

    # —————————————————— Stage helpers ——————————————————

    @staticmethod
    def _iter_parts(parts_dir: Path) -> List[PartEntry]:
        entries: List[PartEntry] = []
        for text_path in sorted(parts_dir.glob("*.txt")):
            meta_path = text_path.with_suffix(".json")
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file missing for part: {meta_path}")
            meta = PartMeta.model_validate_json(meta_path.read_text())
            entries.append(
                PartEntry(
                    meta=meta,
                    text_path=text_path,
                    meta_path=meta_path,
                    text=text_path.read_text(),
                )
            )
        entries.sort(key=lambda entry: entry.meta.part)
        return entries

    @staticmethod
    def _parse_part_filename(stem: str, suffix: str) -> Tuple[str, int]:
        """Extract (text_name, part) from a stage artifact stem."""
        if not stem.endswith(suffix):
            raise ValueError(f"Stem {stem} does not end with expected suffix {suffix}.")
        base = stem[: -len(suffix)]
        if base.endswith("-"):
            base = base[:-1]
        if "-part" not in base:
            raise ValueError(f"Stem {stem} is missing -partNNN.")
        text_name, part_str = base.rsplit("-part", 1)
        return text_name, int(part_str)

    def _iter_normalized(self, normalize_dir: Path) -> List[NormalizedEntry]:
        entries: List[NormalizedEntry] = []
        for json_path in sorted(normalize_dir.glob("*-normalized.json")):
            text_path = json_path.with_suffix(".txt")
            if not text_path.exists():
                raise FileNotFoundError(f"Normalized text file missing: {text_path}")
            text_name, part = self._parse_part_filename(json_path.stem, "-normalized")
            payload = json.loads(json_path.read_text())
            cleaned_text = text_path.read_text()
            normalized = NormalizedOutput.model_validate(
                {**payload, "cleaned_text": cleaned_text}
            )
            entries.append(
                NormalizedEntry(
                    text_name=text_name,
                    part=part,
                    text_path=text_path,
                    json_path=json_path,
                    normalized=normalized,
                )
            )
        entries.sort(key=lambda entry: (entry.text_name, entry.part))
        return entries

    def _iter_cues(self, cue_dir: Path) -> List[CueEntry]:
        entries: List[CueEntry] = []
        for json_path in sorted(cue_dir.glob("*-cues.json")):
            text_name, part = self._parse_part_filename(json_path.stem, "-cues")
            script = CuedScript.model_validate_json(json_path.read_text())
            entries.append(
                CueEntry(
                    text_name=text_name,
                    part=part,
                    script=script,
                    json_path=json_path,
                )
            )
        entries.sort(key=lambda entry: (entry.text_name, entry.part))
        return entries

    def _run_full_pipeline(
        self,
        work_dir: Path | str,
        voice_files: str,
        prepare_conditionals: bool,
    ) -> None:
        workspace = self._resolve_workspace(work_dir)
        logger.info("auto.pipeline.start work_dir={work_dir}", work_dir=workspace)
        self.normalize(workspace)
        self.cue(workspace)
        self.synthesize(
            workspace,
            voice_files=voice_files,
            prepare_conditionals=prepare_conditionals,
        )
        self.finalize(workspace)
        logger.info("auto.pipeline.done work_dir={work_dir}", work_dir=workspace)

    # —————————————————— Stage commands ——————————————————

    def book(
        self,
        book_file: Path | str = "",
        chunk_tokens: int = 3500,
        auto: bool = False,
        voice_files: str = "default:enoch",
        prepare_conditionals: bool = False,
    ) -> Path:
        """Partition an EPUB into workspace parts/ for subsequent stages."""
        if not book_file:
            book_file = self.choose_book()
        else:
            book_file = Path(book_file)

        assert book_file.exists(), f"Book file {book_file} does not exist."
        logger.info(
            "book.start source={source} chunk_tokens={chunk_tokens}",
            source=book_file,
            chunk_tokens=chunk_tokens,
        )

        work_dir = self.workspace_dir / book_file.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest = Manifest(text_name=book_file.stem, workspace=work_dir, kind="book")
        manifest_file = work_dir / "manifest.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        logger.debug("book.workspace work_dir={work_dir}", work_dir=work_dir)

        parts_dir = work_dir / "parts"
        self._prepare_output_dir(parts_dir, stage="book.parts")

        parts_payload = list(partition_text(book_file, chunk_tokens))
        if not parts_payload:
            raise ValueError("Partitioning produced no documents; check input file.")

        for meta, content in parts_payload:
            stem = f"{meta.text_name}-part{meta.part:03d}"
            text_path = parts_dir / f"{stem}.txt"
            json_path = parts_dir / f"{stem}.json"
            text_path.write_text(content)
            json_path.write_text(meta.model_dump_json(indent=2, exclude_none=True))
            logger.debug(
                "book.part_saved part={part} path={path}",
                part=meta.part,
                path=text_path,
            )

        logger.info(
            "book.done work_dir={work_dir} parts={count}",
            work_dir=work_dir,
            count=len(parts_payload),
        )
        if auto:
            self._run_full_pipeline(
                work_dir,
                voice_files=voice_files,
                prepare_conditionals=prepare_conditionals,
            )
        return work_dir

    def text(
        self,
        text_file: Path | str,
        part: int = 1,
        auto: bool = False,
        voice_files: str = "default:enoch",
        prepare_conditionals: bool = False,
    ) -> Path:
        """Register a single text file as parts/part{part:03d} in a new workspace."""
        text_file = Path(text_file)
        assert text_file.exists(), f"Text file {text_file} does not exist."
        logger.info(
            "text.start source={source} part={part}",
            source=text_file,
            part=part,
        )

        work_dir = self.workspace_dir / text_file.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest = Manifest(text_name=text_file.stem, workspace=work_dir, kind="text")
        manifest_file = work_dir / "manifest.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        logger.debug("text.workspace work_dir={work_dir}", work_dir=work_dir)

        parts_dir = work_dir / "parts"
        self._prepare_output_dir(parts_dir, stage="text.parts")

        content = text_file.read_text()
        meta = PartMeta(
            text_name=text_file.stem,
            part=part,
            source=str(text_file),
            last_excerpt=None,
            next_excerpt=None,
        )
        stem = f"{meta.text_name}-part{meta.part:03d}"
        text_path = parts_dir / f"{stem}.txt"
        json_path = parts_dir / f"{stem}.json"
        text_path.write_text(content)
        json_path.write_text(meta.model_dump_json(indent=2, exclude_none=True))
        logger.debug(
            "text.part_saved part={part} path={path}",
            part=meta.part,
            path=text_path,
        )
        logger.info(
            "text.done work_dir={work_dir} parts=1",
            work_dir=work_dir,
        )
        if auto:
            self._run_full_pipeline(
                work_dir,
                voice_files=voice_files,
                prepare_conditionals=prepare_conditionals,
            )
        return work_dir

    def normalize(self, work_dir: Path | str) -> None:
        """Stage A — produce NormalizedOutput from parts/*.txt."""
        workspace = self._resolve_workspace(work_dir)
        parts_dir = workspace / "parts"
        if not parts_dir.exists():
            raise FileNotFoundError(f"Parts directory missing: {parts_dir}")

        logger.info("normalize.start work_dir={work_dir}", work_dir=workspace)
        normalize_dir = workspace / "normalize"
        self._prepare_output_dir(normalize_dir, stage="normalize")

        part_entries = self._iter_parts(parts_dir)
        if not part_entries:
            raise ValueError("No parts found to normalize.")

        previous: Optional[NormalizedOutput] = None
        for entry in part_entries:
            prev_summary = (
                previous.model_dump(exclude={"cleaned_text"}, exclude_unset=True)
                if previous
                else None
            )
            metadata_block = json.dumps(
                {
                    "text_name": entry.meta.text_name,
                    "part": entry.meta.part,
                    "source": entry.meta.source,
                    "last_excerpt": entry.meta.last_excerpt,
                    "next_excerpt": entry.meta.next_excerpt,
                    "previous_normalized": prev_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            messages = [
                SystemMessage(content=NORMALIZE_PROMPT),
                HumanMessage(
                    content=(
                        "== METADATA ==\n"
                        f"{metadata_block}\n\n"
                        "== INPUT TEXT ==\n"
                        f"{entry.text}"
                    )
                ),
            ]
            logger.info(
                "normalize.part text_name={name} part={part} chars={chars}",
                name=entry.meta.text_name,
                part=entry.meta.part,
                chars=len(entry.text),
            )
            callback = UsageMetadataCallbackHandler()
            res = self.llm.with_structured_output(
                NormalizedOutput, include_raw=True
            ).invoke(messages, config=RunnableConfig(callbacks=[callback]))
            normalized: NormalizedOutput = res["parsed"]
            logger.debug(
                "normalize.tokens usage={usage}", usage=callback.usage_metadata
            )
            stem = f"{entry.meta.text_name}-part{entry.meta.part:03d}-normalized"
            text_path = normalize_dir / f"{stem}.txt"
            json_path = normalize_dir / f"{stem}.json"
            text_path.write_text(normalized.cleaned_text)
            json_path.write_text(
                normalized.model_dump_json(
                    indent=2,
                    exclude={"cleaned_text"},
                    exclude_unset=True,
                )
            )

            heuristics = normalized.heuristics
            logger.debug(
                (
                    "normalize.heuristics input={input_chars} output={output_chars} "
                    "removed={removed_chars} removal_ratio={ratio:.3f} paragraphs={paragraphs} "
                    "speakers={speakers}"
                ),
                input_chars=heuristics.input_chars,
                output_chars=heuristics.output_chars,
                removed_chars=heuristics.removed_chars,
                ratio=heuristics.removal_ratio,
                paragraphs=heuristics.paragraph_count,
                speakers=heuristics.speaker_candidate_count,
            )
            previous = normalized

        logger.info(
            "normalize.done work_dir={work_dir} parts={count}",
            work_dir=workspace,
            count=len(part_entries),
        )

    def cue(self, work_dir: Path | str) -> None:
        """Stage B — produce CuedScript from normalize/*.json."""
        workspace = self._resolve_workspace(work_dir)
        normalize_dir = workspace / "normalize"
        if not normalize_dir.exists():
            raise FileNotFoundError(f"Normalized directory missing: {normalize_dir}")
        logger.info("cue.start work_dir={work_dir}", work_dir=workspace)
        cue_dir = workspace / "cues"
        self._prepare_output_dir(cue_dir, stage="cue")

        normalized_entries = self._iter_normalized(normalize_dir)
        if not normalized_entries:
            raise ValueError("No normalized outputs found; run `normalize` first.")

        previous_script: Optional[CuedScript] = None
        for entry in normalized_entries:
            prev_summary = (
                previous_script.model_dump(exclude={"chunks"}, exclude_unset=True)
                if previous_script
                else None
            )
            metadata_block = json.dumps(
                {
                    "text_name": entry.text_name,
                    "part": entry.part,
                    "category": entry.normalized.category,
                    "speakers": [speaker.name for speaker in entry.normalized.speakers],
                    "previous_script": prev_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            messages = [
                SystemMessage(content=CUE_PRIMER),
                SystemMessage(content=CUE_PROMPT),
                HumanMessage(
                    content=(
                        "== CONTEXT ==\n"
                        f"{metadata_block}\n\n"
                        "== CLEANED TEXT ==\n"
                        f"{entry.normalized.cleaned_text}"
                    )
                ),
            ]
            logger.info(
                "cue.part text_name={name} part={part}",
                name=entry.text_name,
                part=entry.part,
            )
            callback = UsageMetadataCallbackHandler()
            res = self.llm.with_structured_output(CuedScript, include_raw=True).invoke(
                messages, config=RunnableConfig(callbacks=[callback])
            )
            script: CuedScript = res["parsed"]
            logger.debug("cue.tokens usage={usage}", usage=callback.usage_metadata)

            # Ensure speaker registry covers all chunk speakers even if the LLM omits narrator/default entries.
            chunk_speakers = [chunk.speaker for chunk in script.chunks]
            merged_speakers = list(dict.fromkeys([*script.speakers, *chunk_speakers]))
            if merged_speakers != script.speakers:
                script = script.model_copy(update={"speakers": merged_speakers})
                logger.debug("cue.speakers merged={speakers}", speakers=merged_speakers)

            stem = f"{entry.text_name}-part{entry.part:03d}-cues"
            json_path = cue_dir / f"{stem}.json"
            review_path = cue_dir / f"{stem}-review.txt"
            json_path.write_text(
                script.model_dump_json(indent=2, exclude_none=True, exclude_unset=True)
            )
            review_path.write_text(script.format())

            norm_len = len(entry.normalized.cleaned_text)
            chunk_len = sum(len(chunk.text) for chunk in script.chunks)
            coverage = chunk_len / max(1, norm_len)
            logger.debug(
                "cue.coverage norm_chars={norm} chunk_chars={chunk} ratio={ratio:.3f}",
                norm=norm_len,
                chunk=chunk_len,
                ratio=coverage,
            )
            if coverage < 0.5:
                logger.warning(
                    "cue.coverage_low ratio={coverage:.3f} norm_chars={norm} chunk_chars={chunks}",
                    coverage=coverage,
                    norm=norm_len,
                    chunks=chunk_len,
                )
            previous_script = script

        logger.info(
            "cue.done work_dir={work_dir} scripts={count}",
            work_dir=workspace,
            count=len(normalized_entries),
        )

    def synthesize(
        self,
        work_dir: Path | str,
        voice_files: str = "default:enoch",
        prepare_conditionals: bool = False,
    ) -> None:
        """Stage C — synthesize audio for each chunk in cues/*.json."""
        workspace = self._resolve_workspace(work_dir)
        cue_dir = workspace / "cues"
        if not cue_dir.exists():
            raise FileNotFoundError(f"Cue directory missing: {cue_dir}")
        logger.info("synthesize.start work_dir={work_dir}", work_dir=workspace)
        audio_dir = workspace / "audio"
        self._prepare_output_dir(audio_dir, stage="synthesize")

        cue_entries = self._iter_cues(cue_dir)
        if not cue_entries:
            raise ValueError("No cue scripts found; run `cue` first.")

        voice_specs = [spec.strip() for spec in voice_files.split(",") if spec.strip()]
        voices = self.voices_from_spec(voice_specs)
        if voices:
            logger.info("synthesize.voices count={count}", count=len(voices))

        if prepare_conditionals and hasattr(tts_model, "prepare_conditionals"):
            default_voice = voices.get("default")
            chosen = default_voice or (next(iter(voices.values())) if voices else None)
            if chosen is not None:
                try:
                    tts_model.prepare_conditionals(str(chosen))  # type: ignore[attr-defined]
                    logger.debug("synthesize.prepared voice={voice}", voice=chosen)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "synthesize.prepare_conditionals_failed voice={voice} error={error}",
                        voice=chosen,
                        error=exc,
                    )

        total_saved = 0
        position = 0.0
        for entry in cue_entries:
            logger.info(
                "synthesize.script text_name={name} part={part} chunks={chunks}",
                name=entry.text_name,
                part=entry.part,
                chunks=len(entry.script.chunks),
            )
            script = entry.script
            for chunk in script.chunks:
                voice_path, voice_name = self._choose_voice(chunk.speaker, voices)
                audio_prompt_path = chunk.audio_prompt_path or voice_path

                logger.info(
                    "Chunk {idx} / {count}: size={size} speaker={speaker} pre_ms={pre}ms post_ms={post}ms voice={voice}",
                    idx=chunk.idx,
                    count=len(script.chunks),
                    size=len(chunk.text),
                    speaker=chunk.speaker,
                    pre=chunk.pre_pause_ms,
                    post=chunk.post_pause_ms,
                    voice=voice_name,
                )
                phrases = chunk.split_text()
                out_file = (
                    audio_dir
                    / f"{entry.text_name}_p{entry.part:03d}_{chunk.idx:04d}_{chunk.speaker}.wav"
                )
                base_audio = AudioSegment.silent(duration=chunk.pre_pause_ms)
                sample_rate = int(tts_model.sr)

                spoken_phrases: List[str] = []
                for idx, phrase in enumerate(phrases, start=1):
                    clean_phrase = phrase.strip()
                    if not clean_phrase:
                        continue
                    logger.debug("Phrase {} / {}: {}", idx, len(phrases), clean_phrase)
                    params = chunk.engine_params()
                    wav = tts_model.generate(
                        text=clean_phrase,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=params.exaggeration,
                        cfg_weight=params.cfg_weight,
                        temperature=params.temperature,
                        repetition_penalty=params.repetition_penalty,
                        min_p=params.min_p,
                        top_p=params.top_p,
                    )
                    buf = BytesIO()
                    torchaudio.save(buf, wav, sample_rate, format="wav")
                    buf.seek(0)
                    base_audio += AudioSegment.from_file(buf, format="wav")
                    spoken_phrases.append(clean_phrase)

                if chunk.post_pause_ms:
                    base_audio += AudioSegment.silent(duration=chunk.post_pause_ms)

                base_audio.export(out_file, format="wav")
                position += base_audio.duration_seconds
                logger.debug(
                    "Saved '{}' duration={:.1f}s position={:.1f}s",
                    out_file,
                    base_audio.duration_seconds,
                    position,
                )
                meta = ChunkEntry.model_validate(
                    {
                        **chunk.model_dump(),
                        "duration": base_audio.duration_seconds,
                        "position": position,
                        "phrases": spoken_phrases or phrases,
                    }
                )
                out_file.with_suffix(".json").write_text(
                    meta.model_dump_json(
                        indent=2, exclude_none=True, exclude_unset=True
                    )
                )
                total_saved += 1

        logger.info(
            "synthesize.done work_dir={work_dir} files={count}",
            work_dir=workspace,
            count=total_saved,
        )

    def finalize(self, work_dir: Path | str) -> None:
        """Reserved for concatenation/mix; currently just logs existing assets."""
        workspace = self._resolve_workspace(work_dir)
        audio_dir = workspace / "audio"
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory missing: {audio_dir}")
        wav_files = list(audio_dir.glob("*.wav"))
        logger.info("finalize.start work_dir={work_dir}", work_dir=workspace)
        logger.debug(
            "finalize.files wav_files={files}",
            files=[file.name for file in wav_files],
        )
        logger.info(
            "finalize.done work_dir={work_dir} wav_count={count}",
            work_dir=workspace,
            count=len(wav_files),
        )


if __name__ == "__main__":
    fire.Fire(Toolchain)
